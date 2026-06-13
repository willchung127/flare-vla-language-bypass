"""velocity_divergence.py — velocity-level and action-chunk-level divergence.

After mechanism_probe localized cross-attention bypass to expert L17
(cos=0.993 vs paligemma L17 cos=0.846), this script tests whether the bypass
extends to the *outputs* — the velocity field v_t and the final action chunk.

Captures:
  - v_t  at each denoise step via monkey-patched model.denoise_step
  - final action chunk via policy.infer()["actions"]

The single most important number is cos(v_A, v_B) averaged over denoise steps:
  cos ≈ 1.000        : total bypass at velocity → CFG dies; need activation steering
  cos ∈ [0.95, 0.99] : residual signal           → CFG has something to amplify
  cos < 0.95         : surprising signal         → bypass story is incomplete

Outputs in --out-dir:
  velocities_a.pt, velocities_b.pt   list of v_t tensors per prompt
  actions_a.pt,    actions_b.pt      final action chunks per prompt
  divergence.json                    per-step + summary divergence metrics
  sanity_checks.json                 PASS/FAIL on each built-in correctness check
  velocity_divergence_vs_t.png       (if --plot) cos & L2 vs denoising time
  velocity_per_dim_heatmap.png       (if --plot) (denoise_step × action_dim) |Δv|
  action_chunk_overlay.png           (if --plot) actions_A vs actions_B per dim
  action_chunk_divergence.png        (if --plot) (chunk_step × action_dim) |Δa|
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Reuse small helpers from the attention probe.
sys.path.insert(0, str(Path(__file__).parent))
from mechanism_probe import (  # noqa: E402
    set_all_seeds,
    force_eager_attention,
    _check_observation_parity,
    _check_prompts_differ,
)


# =============================================================================
# VELOCITY CAPTURE — monkey-patches model.denoise_step
# =============================================================================

class VelocityCapture:
    """Context manager: captures v_t at each call to model.denoise_step.

    patch_pi0_sde.py's sample_actions_sde calls:
        v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
    The original sample_actions calls it from inside its own integration loop.
    Both paths are caught — we wrap the bound method on the instance.

    Captured: list of v_t tensors, one per denoise step.
    Shape: (B, action_horizon, action_dim_internal). For pi0 LIBERO,
    typically (1, 50, 32) — action_dim is padded to 32.
    """

    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.captured: List[torch.Tensor] = []
        self._original = None
        self._had_instance_attr = False

    def __enter__(self):
        if not hasattr(self.model, "denoise_step"):
            raise RuntimeError(
                f"model {type(self.model).__name__} has no denoise_step method"
            )
        self._original = self.model.denoise_step  # bound method via descriptor
        self._had_instance_attr = "denoise_step" in self.model.__dict__
        captured = self.captured
        orig = self._original

        def wrapped(*args, **kwargs):
            v = orig(*args, **kwargs)
            if not isinstance(v, torch.Tensor):
                raise RuntimeError(
                    f"denoise_step returned {type(v).__name__}, expected Tensor"
                )
            captured.append(v.detach().to("cpu"))
            return v

        self.model.denoise_step = wrapped
        if self.verbose:
            print(f"[VelocityCapture] Hooked denoise_step on "
                  f"{type(self.model).__name__}")
        return self.captured

    def __exit__(self, *_):
        if self._had_instance_attr and self._original is not None:
            self.model.denoise_step = self._original
        else:
            if "denoise_step" in self.model.__dict__:
                del self.model.__dict__["denoise_step"]
        if self.verbose:
            print(f"[VelocityCapture] Restored denoise_step; "
                  f"captured {len(self.captured)} velocities")


# =============================================================================
# DIVERGENCE METRICS
# =============================================================================

def cos_sim_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Cosine similarity between two tensors after flattening."""
    af = a.flatten().float()
    bf = b.flatten().float()
    n = (af.norm() * bf.norm()).clamp(min=eps)
    return float((af * bf).sum() / n)


def l2_dist_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """L2 distance between two flattened tensors."""
    return float((a.flatten().float() - b.flatten().float()).norm())


def per_dim_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-element |a - b|, returned with original shape (caller may reduce)."""
    return (a.float() - b.float()).abs()


# =============================================================================
# SANITY CHECKS
# =============================================================================

def _check_velocities_nonempty(captured: List[torch.Tensor]) -> Tuple[bool, str]:
    if len(captured) == 0:
        return False, "no velocities captured (denoise_step never called?)"
    return True, f"captured {len(captured)} velocities"


def _check_velocity_shapes(va: List[torch.Tensor], vb: List[torch.Tensor]
                            ) -> Tuple[bool, str]:
    if len(va) != len(vb):
        return False, f"step count mismatch: A={len(va)} B={len(vb)}"
    for i, (a, b) in enumerate(zip(va, vb)):
        if a.shape != b.shape:
            return False, f"step {i}: shapes {tuple(a.shape)} vs {tuple(b.shape)}"
    return True, f"all {len(va)} step shapes match"


def _check_velocity_finite(captured: List[torch.Tensor]) -> Tuple[bool, str]:
    for i, v in enumerate(captured):
        if not torch.isfinite(v).all():
            n_nan = int((~torch.isfinite(v)).sum())
            return False, f"step {i} has {n_nan} non-finite values"
    return True, "all velocities are finite"


def _check_velocity_self_consistency(va: List[torch.Tensor],
                                      va_dup: List[torch.Tensor],
                                      tol: float = 1e-6) -> Tuple[bool, str, float]:
    if len(va) != len(va_dup):
        return False, f"step counts differ: {len(va)} vs {len(va_dup)}", float("inf")
    max_diff = 0.0
    where = -1
    for i, (a, b) in enumerate(zip(va, va_dup)):
        d = (a - b).abs().max().item()
        if d > max_diff:
            max_diff, where = d, i
    msg = f"max abs diff = {max_diff:.2e} at step {where}"
    return (max_diff <= tol), msg, max_diff


def _check_action_finite(actions: torch.Tensor) -> Tuple[bool, str]:
    if not torch.isfinite(actions).all():
        n = int((~torch.isfinite(actions)).sum())
        return False, f"{n} non-finite action values"
    return True, f"action chunk shape={tuple(actions.shape)} all finite"


def run_sanity_checks(va, vb, actions_a, actions_b, prompt_a, prompt_b,
                       obs_a, obs_b, va_dup=None) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    ok, msg = _check_prompts_differ(prompt_a, prompt_b)
    out["prompts_differ"] = {"pass": ok, "message": msg}
    ok, msg = _check_observation_parity(obs_a, obs_b)
    out["observation_parity"] = {"pass": ok, "message": msg}
    ok, msg = _check_velocities_nonempty(va)
    out["A_velocities_nonempty"] = {"pass": ok, "message": msg}
    ok, msg = _check_velocities_nonempty(vb)
    out["B_velocities_nonempty"] = {"pass": ok, "message": msg}
    ok, msg = _check_velocity_finite(va)
    out["A_velocities_finite"] = {"pass": ok, "message": msg}
    ok, msg = _check_velocity_finite(vb)
    out["B_velocities_finite"] = {"pass": ok, "message": msg}
    ok, msg = _check_velocity_shapes(va, vb)
    out["AB_velocity_shapes_match"] = {"pass": ok, "message": msg}
    ok, msg = _check_action_finite(actions_a)
    out["A_actions_finite"] = {"pass": ok, "message": msg}
    ok, msg = _check_action_finite(actions_b)
    out["B_actions_finite"] = {"pass": ok, "message": msg}
    if va_dup is not None:
        ok, msg, max_diff = _check_velocity_self_consistency(va, va_dup)
        out["A_self_consistency"] = {
            "pass": ok, "message": msg, "max_diff": max_diff
        }
    else:
        out["A_self_consistency"] = {
            "pass": True, "message": "skipped (use --self-consistency-check)"
        }
    return out


# =============================================================================
# DIVERGENCE COMPUTATION
# =============================================================================

def compute_velocity_divergence(va: List[torch.Tensor], vb: List[torch.Tensor]
                                  ) -> Dict[str, list]:
    """Per-step cos and L2 between v_A[t] and v_B[t]. Lists indexed by step."""
    n = min(len(va), len(vb))
    cos_per_step: List[float] = []
    l2_per_step: List[float] = []
    # Per-dim aggregated diff: average over batch and horizon, leave action_dim
    per_dim_diff_per_step: List[List[float]] = []
    for i in range(n):
        a, b = va[i], vb[i]
        cos_per_step.append(cos_sim_flat(a, b))
        l2_per_step.append(l2_dist_flat(a, b))
        # Per-dim breakdown: mean(|a - b|) over batch and horizon dims, by action_dim
        diff = per_dim_abs_diff(a, b)  # shape (B, H, D)
        per_dim_diff_per_step.append(diff.mean(dim=(0, 1)).tolist())
    return {
        "cos_per_step": cos_per_step,
        "l2_per_step": l2_per_step,
        "per_dim_diff_per_step": per_dim_diff_per_step,
        "cos_mean": float(np.mean(cos_per_step)) if cos_per_step else None,
        "cos_min": float(np.min(cos_per_step)) if cos_per_step else None,
        "l2_mean": float(np.mean(l2_per_step)) if l2_per_step else None,
    }


def compute_action_chunk_divergence(actions_a: torch.Tensor, actions_b: torch.Tensor
                                     ) -> Dict[str, object]:
    """Action chunks are typically (chunk_horizon, action_dim) e.g. (50, 7)."""
    cos_total = cos_sim_flat(actions_a, actions_b)
    l2_total = l2_dist_flat(actions_a, actions_b)
    # Per-chunk-step: cos and L2 per (timestep, dim) — for visualization
    diff = per_dim_abs_diff(actions_a, actions_b)  # shape (T, D)
    # Per-timestep cos (treat each timestep as a vector)
    cos_per_t = []
    if actions_a.dim() == 2:
        for t in range(actions_a.shape[0]):
            cos_per_t.append(cos_sim_flat(actions_a[t], actions_b[t]))
    return {
        "cos_total": cos_total,
        "l2_total": l2_total,
        "cos_per_t": cos_per_t,
        "diff_grid": diff.tolist(),  # for heatmap
        "shape": list(actions_a.shape),
    }


# =============================================================================
# PLOTS
# =============================================================================

def make_plots(div_v: dict, div_a: dict, va: List[torch.Tensor], vb: List[torch.Tensor],
                actions_a: torch.Tensor, actions_b: torch.Tensor, out_dir: Path,
                title_suffix: str = "") -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[probe] matplotlib not available — skipping plots")
        return []
    written: List[Path] = []

    # ---------- Plot 1: velocity divergence vs denoising time ----------
    cos_per_step = div_v["cos_per_step"]
    l2_per_step = div_v["l2_per_step"]
    steps = list(range(len(cos_per_step)))
    # Denoising "time" in pi0 goes from 1.0 down to 0.0 across num_steps.
    # The first call is at t≈1.0 (pure noise), the last at t≈0.0 (clean).
    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()
    ax_left.plot(steps, cos_per_step, "o-", color="C0", label="cos(v_A, v_B)")
    ax_left.axhline(1.0, color="grey", linestyle="--", alpha=0.5,
                    label="identical")
    ax_left.set_xlabel("denoising step (early → late)")
    ax_left.set_ylabel("cosine similarity", color="C0")
    ax_left.tick_params(axis="y", labelcolor="C0")
    ax_left.set_ylim(min(cos_per_step + [0.85]) - 0.02, 1.005)
    ax_left.grid(True, alpha=0.3)
    ax_right.plot(steps, l2_per_step, "s-", color="C1", label="L2 dist")
    ax_right.set_ylabel("L2 distance", color="C1")
    ax_right.tick_params(axis="y", labelcolor="C1")
    fig.suptitle(f"Velocity divergence per denoising step{title_suffix}")
    # combine legends
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    p = out_dir / "velocity_divergence_vs_t.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # ---------- Plot 2: per-dim velocity heatmap ----------
    diff_grid_v = np.array(div_v["per_dim_diff_per_step"])  # (steps, action_dim)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(diff_grid_v.T, aspect="auto", cmap="viridis")
    ax.set_xlabel("denoising step (early → late)")
    ax.set_ylabel("action dim (model-internal, 0..D_padded-1)")
    ax.set_title(f"|v_A - v_B| averaged over batch & horizon{title_suffix}")
    fig.colorbar(im, ax=ax, label="mean |Δv|")
    fig.tight_layout()
    p = out_dir / "velocity_per_dim_heatmap.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # ---------- Plot 3: action chunk overlay ----------
    # actions are (T, D). Plot one panel per action dim.
    if actions_a.dim() == 2:
        T, D = actions_a.shape
        n_cols = min(4, D)
        n_rows = (D + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows),
                                 sharex=True)
        axes = np.atleast_2d(axes)
        for d in range(D):
            r, c = d // n_cols, d % n_cols
            ax = axes[r, c]
            ax.plot(actions_a[:, d].numpy(), label="prompt A", alpha=0.85)
            ax.plot(actions_b[:, d].numpy(), label="prompt B", alpha=0.85,
                    linestyle="--")
            ax.set_title(f"action dim {d}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if d == 0:
                ax.legend(fontsize=8)
        # Hide unused panels
        for d in range(D, n_rows * n_cols):
            r, c = d // n_cols, d % n_cols
            axes[r, c].axis("off")
        fig.suptitle(f"Action chunks overlaid{title_suffix}")
        fig.tight_layout()
        p = out_dir / "action_chunk_overlay.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # ---------- Plot 4: action chunk divergence heatmap ----------
    diff_grid_a = np.array(div_a["diff_grid"])  # (T, D)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(diff_grid_a.T, aspect="auto", cmap="viridis")
    ax.set_xlabel("chunk step (action horizon)")
    ax.set_ylabel("action dim")
    ax.set_title(f"|actions_A - actions_B| per (step, dim){title_suffix}")
    fig.colorbar(im, ax=ax, label="|Δa|")
    fig.tight_layout()
    p = out_dir / "action_chunk_divergence.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    return written


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-config", default="pi0_libero")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--bddl-override", default=None)
    p.add_argument("--prompt-a", required=True)
    p.add_argument("--prompt-b", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=7000)
    p.add_argument("--init-idx", type=int, default=0)
    p.add_argument("--self-consistency-check", action="store_true")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[probe] importing openpi/libero ...", flush=True)
    from openpi.training import config as _c
    from openpi.policies import policy_config
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    sys.path.insert(0, str(Path.home() / "flare"))
    from patch_pi0_sde import apply_patch  # noqa: E402
    from remote_multimode_matrix import (  # type: ignore  # noqa: E402
        format_obs, NUM_STEPS_WAIT, LIBERO_DUMMY_ACTION,
    )

    print(f"[probe] applying SDE patch + loading policy "
          f"(config={args.policy_config})...", flush=True)
    Pi0 = apply_patch()
    cfg = _c.get_config(args.policy_config)

    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    elif args.policy_config == "pi0_libero":
        ckpt_dir = str(Path.home() / "flare/checkpoints/pi0_libero_pt")
    else:
        from openpi.shared import download as openpi_download
        gs = f"gs://openpi-assets/checkpoints/{args.policy_config}"
        ckpt_dir = str(Path(openpi_download.maybe_download(gs)))
    print(f"[probe] checkpoint dir: {ckpt_dir}", flush=True)
    policy = policy_config.create_trained_policy(cfg, ckpt_dir)
    model = None
    for attr in ("_model", "model", "_policy", "policy"):
        if hasattr(policy, attr) and isinstance(getattr(policy, attr), Pi0):
            model = getattr(policy, attr)
            print(f"[probe] found model at policy.{attr}", flush=True)
            break
    if model is None:
        raise RuntimeError(
            f"Could not find a {Pi0.__name__} instance on the policy. "
            f"This probe requires a PyTorch pi0 (not JAX-only)."
        )

    # Pure ODE inference — no SDE, no guidance.
    model.flare_eta = 0.0
    model.flare_verifier_fn = None
    model.flare_alpha = 0.0
    model.flare_obs_state = None
    model.flare_eta_high = None
    model.flare_eta_low = None
    model.flare_noise_bias_direction = None
    model.flare_noise_bias_strength = 0.0
    model.eval()
    force_eager_attention(model, verbose=False)

    # Env setup.
    print(f"[probe] env setup: suite={args.task_suite} task_id={args.task_id}",
          flush=True)
    bench = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bench.get_task(args.task_id)
    if args.bddl_override:
        bddl = args.bddl_override
    else:
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
    init_states = bench.get_task_init_states(args.task_id)
    if args.init_idx >= len(init_states):
        raise ValueError(f"--init-idx={args.init_idx} >= {len(init_states)}")
    init_state = init_states[args.init_idx]

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(args.env_seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    obs_a = format_obs(obs, args.prompt_a)
    obs_b = format_obs(obs, args.prompt_b)

    # Inference passes
    print(f"[probe] PROMPT A: {args.prompt_a!r}", flush=True)
    set_all_seeds(args.seed)
    t0 = time.time()
    with VelocityCapture(model, verbose=True) as va:
        out_a = policy.infer(obs_a)
    actions_a = torch.as_tensor(out_a["actions"], dtype=torch.float32)
    print(f"[probe]   done in {time.time()-t0:.1f}s; "
          f"velocities={len(va)} action_chunk={tuple(actions_a.shape)}",
          flush=True)

    print(f"[probe] PROMPT B: {args.prompt_b!r}", flush=True)
    set_all_seeds(args.seed)
    t0 = time.time()
    with VelocityCapture(model, verbose=True) as vb:
        out_b = policy.infer(obs_b)
    actions_b = torch.as_tensor(out_b["actions"], dtype=torch.float32)
    print(f"[probe]   done in {time.time()-t0:.1f}s", flush=True)

    va_dup = None
    if args.self_consistency_check:
        print("[probe] self-consistency re-run on PROMPT A ...", flush=True)
        set_all_seeds(args.seed)
        with VelocityCapture(model, verbose=False) as va_dup:
            _ = policy.infer(obs_a)

    # Sanity
    print("\n[probe] === SANITY CHECKS ===", flush=True)
    sanity = run_sanity_checks(va, vb, actions_a, actions_b,
                                args.prompt_a, args.prompt_b,
                                obs_a, obs_b, va_dup)
    for k, v in sanity.items():
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  [{status}] {k}: {v['message']}", flush=True)
    n_pass = sum(1 for v in sanity.values() if v["pass"])
    n_total = len(sanity)
    print(f"[probe] sanity: {n_pass}/{n_total} passing\n", flush=True)

    # Divergence
    div_v = compute_velocity_divergence(va, vb)
    div_a = compute_action_chunk_divergence(actions_a, actions_b)

    print("\n[probe] === HEADLINE ===")
    print(f"  velocity cos (mean over steps): {div_v['cos_mean']:.4f}")
    print(f"  velocity cos (min over steps):  {div_v['cos_min']:.4f}")
    print(f"  velocity L2  (mean):            {div_v['l2_mean']:.4f}")
    print(f"  action chunk cos (overall):     {div_a['cos_total']:.4f}")
    print(f"  action chunk L2  (overall):     {div_a['l2_total']:.4f}")
    if div_v["cos_min"] is not None:
        if div_v["cos_min"] > 0.999:
            verdict = "TOTAL BYPASS — velocity prompt-independent (no signal to amplify)"
        elif div_v["cos_min"] > 0.95:
            verdict = "RESIDUAL SIGNAL — CFG viable"
        else:
            verdict = "SURPRISING SIGNAL — bypass story incomplete"
        print(f"\n  Verdict: {verdict}")

    # Save
    print(f"\n[probe] saving to {out_dir} ...")
    torch.save({"prompt": args.prompt_a, "velocities": va},
               out_dir / "velocities_a.pt")
    torch.save({"prompt": args.prompt_b, "velocities": vb},
               out_dir / "velocities_b.pt")
    torch.save({"prompt": args.prompt_a, "actions": actions_a},
               out_dir / "actions_a.pt")
    torch.save({"prompt": args.prompt_b, "actions": actions_b},
               out_dir / "actions_b.pt")
    if va_dup is not None:
        torch.save({"prompt": args.prompt_a, "velocities": va_dup},
                   out_dir / "velocities_a_dup.pt")

    summary = {
        "policy_config": args.policy_config,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "n_denoise_steps": len(va),
        "velocity_shape": list(va[0].shape) if va else None,
        "action_chunk_shape": list(actions_a.shape),
        "velocity_divergence": div_v,
        "action_chunk_divergence": div_a,
    }
    (out_dir / "divergence.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2))

    if args.plot:
        suffix = f"\nA: {args.prompt_a[:60]} | B: {args.prompt_b[:60]}"
        paths = make_plots(div_v, div_a, va, vb, actions_a, actions_b, out_dir, suffix)
        for pp in paths:
            print(f"[probe] wrote {pp}")

    if n_pass < n_total:
        print(f"[probe] WARN: {n_total-n_pass} sanity check(s) FAILED — "
              f"do NOT trust the divergence numbers.")
        sys.exit(1)


if __name__ == "__main__":
    main()
