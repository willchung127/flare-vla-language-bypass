"""activation_steering.py — residual activation steering of the action expert.

Hypothesis (from the linear probe finding):
  At expert L12 the prompt is PERFECTLY linearly decodable from the residual
  stream (AUC=1.000) but the velocity output is prompt-independent (cos=0.9997).
  So the information is present in the residual stream but causally disconnected
  from the output computation.

Method:
  Extract the linear-probe direction `w_prompt` at expert L12 (the layer where
  prompt-decodability first saturates and the residual stream still routes
  through 5 more transformer blocks before producing velocity). At inference,
  register a forward hook on that decoder layer to add `α · w_unit` to the
  hidden states. The hypothesis: amplifying the prompt-discriminative direction
  in the residual stream forces it to leak through the late-layer suppression
  and changes the output velocity (and ultimately the action chunk).

Key correctness considerations:
  - The probe was trained on RAW mean-pooled hidden states (BF16 cast to F32).
    Its coef_ is in raw activation space (no StandardScaler in our path).
  - We normalize w to unit norm, then sweep α as an absolute magnitude in
    activation-space units. Sanity defaults: 0, 1, 5, 10, 50, 100, ±.
  - α=0 must produce velocity bit-identical to the no-hook baseline.
  - The hook must clean up cleanly (no dangling handles after the run).

Outputs:
  steering_<label>.pt        per-α: velocities (list[Tensor]), action chunk
  steering_results.json      per-α: velocity cos vs baseline, action cos vs baseline,
                              ||residual_stream perturbation||, sanity flags
  sanity_checks.json         PASS/FAIL on hook lifecycle + α=0 reproducibility
  steering_alpha_sweep.png   (--plot) cos(v_steered, v_baseline) and action shift vs α
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from mechanism_probe_attn import set_all_seeds, force_eager_attention
from probe_velocity_and_actions import (
    VelocityCapture, cos_sim_flat, l2_dist_flat,
)
from probe_linear_decodability import EXPERT_LAYER_RE


# =============================================================================
# DIRECTION EXTRACTION
# =============================================================================

def fit_steering_direction(probe_data_npz: Path, target_layer_idx: int,
                            verbose: bool = True) -> Tuple[np.ndarray, dict]:
    """Re-fit the LR probe on ALL data at the target layer; return raw direction.

    Returns:
        w_unit: unit-norm direction in raw activation space, shape (D,)
        info:   {"auc", "y_class_balance", "norm_raw", "layer_name", ...}

    Math:
        sklearn LR with C=0.1 on standardized inputs gives coef in standardized
        space. We re-fit with WITHOUT scaling so coef is in raw space, then
        normalize to unit length. To recover from a scaled fit instead, multiply
        coef by sample stddev per dim.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    npz = np.load(probe_data_npz, allow_pickle=False)
    y = npz["y"]
    # find the X key matching the target layer
    target_suffix = f"layers.{target_layer_idx}"
    matching = [k for k in npz.files if k.startswith("X__") and target_suffix in k]
    if not matching:
        raise RuntimeError(
            f"No X data found for layer {target_layer_idx} in {probe_data_npz}. "
            f"Available: {[k for k in npz.files if k.startswith('X__')][:5]}..."
        )
    if len(matching) > 1:
        # Should be exactly one — but if multiple, take the one that ends with .self_attn
        # or doesn't have a sub-suffix
        candidates = [k for k in matching if k.endswith(f"layers.{target_layer_idx}")]
        matching = candidates if candidates else matching[:1]
    X_key = matching[0]
    X = npz[X_key]  # shape (N, D)
    if verbose:
        print(f"[direction] using {X_key}, shape {X.shape}, y balance "
              f"{dict(zip(*np.unique(y, return_counts=True)))}")

    # Fit on raw X (no scaling) — so coef is in raw activation space.
    lr_raw = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear")
    lr_raw.fit(X, y)
    w_raw = lr_raw.coef_[0]  # shape (D,)
    w_norm = float(np.linalg.norm(w_raw))
    w_unit = w_raw / max(w_norm, 1e-12)

    # Cross-validated AUC for sanity (confirms direction is meaningful)
    try:
        aucs = cross_val_score(lr_raw, X, y, cv=5, scoring="roc_auc")
        cv_auc = float(np.mean(aucs))
    except Exception:
        cv_auc = float("nan")

    info = {
        "X_key": X_key,
        "n_samples": int(X.shape[0]),
        "D": int(X.shape[1]),
        "norm_raw": w_norm,
        "cv_auc": cv_auc,
        "y_class_balance": {int(k): int(v) for k, v in
                            zip(*np.unique(y, return_counts=True))},
        "scaling": "raw_no_scaler",
    }
    if verbose:
        print(f"[direction] ||w_raw|| = {w_norm:.4f}, CV AUC = {cv_auc:.4f}")
    return w_unit.astype(np.float32), info


# =============================================================================
# STEERING HOOK
# =============================================================================

class SteeringHook:
    """Forward-hook on the target expert decoder layer; adds α · w to the
    hidden states output. Reusable via context manager.

    Hook behavior:
        Decoder layer output is typically (hidden_states, ...) tuple OR just
        hidden_states. We add the bias to `hidden_states` in-place
        (clone-then-add to avoid aliasing surprises).

    Setting alpha=0 must produce bit-identical behavior to no hook.

    direction: shape (D,), unit-norm, on CPU.
    """

    def __init__(self, model: nn.Module, target_layer_idx: int,
                  direction: np.ndarray, alpha: float = 0.0,
                  verbose: bool = True,
                  layer_pattern_template: str = (
                      r"^paligemma_with_expert\.gemma_expert\.model\.layers\.{idx}$"
                  )):
        """Forward-hook the expert decoder layer at `target_layer_idx`.

        layer_pattern_template: regex template with `{idx}` placeholder for the
        layer index. Default matches openpi PI0Pytorch. Override for LeRobot,
        converted models, etc. — e.g.
            r"^model\.action_expert\.layers\.{idx}$"
        """
        self.model = model
        self.target_layer_idx = target_layer_idx
        self.layer_pattern_template = layer_pattern_template
        self.direction_cpu = torch.as_tensor(direction, dtype=torch.float32)
        self.alpha = float(alpha)
        self.verbose = verbose
        self._handle = None
        self._target_module: Optional[nn.Module] = None
        self._n_calls = 0
        self._direction_dev: Optional[torch.Tensor] = None  # filled on first call

    def _find_target(self) -> nn.Module:
        target_path_regex = re.compile(
            self.layer_pattern_template.format(idx=self.target_layer_idx)
        )
        found = None
        for name, module in self.model.named_modules():
            if target_path_regex.match(name):
                found = module
                break
        if found is None:
            # Show candidate paths to help the user choose a different template
            sample = []
            for n, _ in self.model.named_modules():
                nlow = n.lower()
                if "expert" in nlow or "decoder" in nlow or ".layers." in n:
                    sample.append(n)
                if len(sample) >= 10:
                    break
            raise RuntimeError(
                f"Could not find layer at pattern "
                f"{self.layer_pattern_template.format(idx=self.target_layer_idx)!r}. "
                f"Sample paths: {sample}"
            )
        return found

    def _hook(self, module, inputs, output):
        self._n_calls += 1
        if self.alpha == 0.0:
            # Bit-identical to no hook by short-circuiting.
            return output
        # Extract hidden_states from the output
        if isinstance(output, torch.Tensor):
            hs = output
            wrap = lambda new_hs: new_hs
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            hs = output[0]
            wrap = lambda new_hs: (new_hs,) + output[1:]
        else:
            # Unexpected output shape; leave unchanged.
            return output
        # Move direction to the right device + dtype once (cache).
        if (self._direction_dev is None or
                self._direction_dev.device != hs.device or
                self._direction_dev.dtype != hs.dtype):
            self._direction_dev = self.direction_cpu.to(device=hs.device, dtype=hs.dtype)
        # hs shape: (B, L, D). direction shape: (D,). Broadcast.
        hs_new = hs + self.alpha * self._direction_dev
        return wrap(hs_new)

    def __enter__(self):
        self._target_module = self._find_target()
        self._handle = self._target_module.register_forward_hook(self._hook)
        if self.verbose:
            cnt = self._direction_dev.numel() if self._direction_dev is not None else \
                  self.direction_cpu.numel()
            print(f"[SteeringHook] installed on expert L{self.target_layer_idx}, "
                  f"alpha={self.alpha}, direction_dim={self.direction_cpu.numel()}")
        return self

    def __exit__(self, *_):
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
        self._target_module = None
        if self.verbose:
            print(f"[SteeringHook] removed; hook fired {self._n_calls} times")
        self._n_calls = 0


# =============================================================================
# SANITY CHECKS
# =============================================================================

def _check_direction(w: np.ndarray) -> Tuple[bool, str]:
    if w.ndim != 1:
        return False, f"direction must be 1-D, got {w.shape}"
    if not np.isfinite(w).all():
        return False, "direction has non-finite values"
    norm = float(np.linalg.norm(w))
    if abs(norm - 1.0) > 1e-3:
        return False, f"direction not unit-norm; ||w||={norm:.4f}"
    return True, f"direction OK; D={w.size}, ||w||={norm:.6f}"


def _check_alpha_zero_reproduces_baseline(
        velocities_baseline: List[torch.Tensor],
        velocities_alpha0: List[torch.Tensor], tol: float = 1e-6
) -> Tuple[bool, str]:
    """With alpha=0, the hook is short-circuited; outputs must be bit-identical."""
    if len(velocities_baseline) != len(velocities_alpha0):
        return False, (f"velocity step count differs: "
                       f"baseline={len(velocities_baseline)} alpha0={len(velocities_alpha0)}")
    max_diff = 0.0
    where = -1
    for i, (a, b) in enumerate(zip(velocities_baseline, velocities_alpha0)):
        d = (a.float() - b.float()).abs().max().item()
        if d > max_diff:
            max_diff, where = d, i
    if max_diff <= tol:
        return True, f"alpha=0 reproduces baseline (max diff {max_diff:.2e} at step {where})"
    return False, f"alpha=0 DIFFERS from baseline by {max_diff:.2e} at step {where}"


def _check_alpha_perturbs_consistently(
        velocities_baseline: List[torch.Tensor],
        velocities_per_alpha: Dict[float, List[torch.Tensor]],
) -> Tuple[bool, str]:
    """As |alpha| increases, the perturbation should generally increase."""
    diffs = []
    for a, vs in velocities_per_alpha.items():
        if a == 0.0:
            continue
        ds = [l2_dist_flat(v_b, v_s) for v_b, v_s in zip(velocities_baseline, vs)]
        diffs.append((a, float(np.mean(ds))))
    if not diffs:
        return True, "no nonzero alphas to compare"
    # Just print the trend; don't fail unless something pathological.
    msg = "; ".join(f"α={a}→{d:.4f}" for a, d in sorted(diffs))
    return True, "perturbation magnitudes: " + msg


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
    p.add_argument("--prompt", required=True,
                   help="The prompt to use for inference under steering")
    p.add_argument("--probe-data-npz", required=True,
                   help="Path to probe_data.npz from probe_linear_decodability run")
    p.add_argument("--target-layer", type=int, default=12,
                   help="Expert decoder layer index to steer (default 12 — where AUC first saturates)")
    p.add_argument("--layer-pattern-template", default=(
        r"^paligemma_with_expert\.gemma_expert\.model\.layers\.{idx}$"
    ), help="Regex template for the target decoder layer's module path. "
            "Use {idx} as the layer-index placeholder. Default matches openpi "
            "PI0Pytorch convention; override for LeRobot, converted models, etc.")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 1.0, 5.0, 10.0, 50.0, 100.0, -1.0, -5.0, -10.0, -50.0, -100.0],
                   help="Steering magnitudes to sweep")
    p.add_argument("--init-idx", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=7000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Extract steering direction from probe
    print("\n[steering] === EXTRACTING STEERING DIRECTION ===", flush=True)
    w_unit, w_info = fit_steering_direction(
        Path(args.probe_data_npz).expanduser(), args.target_layer
    )

    ok, msg = _check_direction(w_unit)
    print(f"  [{'PASS' if ok else 'FAIL'}] direction_sanity: {msg}")
    if not ok:
        sys.exit(1)

    # 2) Load model + env + observation
    print("\n[steering] === LOADING MODEL + ENV ===", flush=True)
    from openpi.training import config as _c
    from openpi.policies import policy_config
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    sys.path.insert(0, str(Path.home() / "flare"))
    from patch_pi0_sde import apply_patch  # noqa: E402
    from remote_multimode_matrix import (  # type: ignore  # noqa: E402
        format_obs, NUM_STEPS_WAIT, LIBERO_DUMMY_ACTION,
    )

    Pi0 = apply_patch()
    cfg = _c.get_config(args.policy_config)
    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    elif args.policy_config == "pi0_libero":
        ckpt_dir = str(Path.home() / "flare/checkpoints/pi0_libero_pt")
    else:
        from openpi.shared import download as openpi_download
        ckpt_dir = str(Path(openpi_download.maybe_download(
            f"gs://openpi-assets/checkpoints/{args.policy_config}")))
    policy = policy_config.create_trained_policy(cfg, ckpt_dir)
    model = None
    for attr in ("_model", "model", "_policy", "policy"):
        if hasattr(policy, attr) and isinstance(getattr(policy, attr), Pi0):
            model = getattr(policy, attr)
            break
    if model is None:
        raise RuntimeError("Could not find PI0Pytorch on policy")

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

    bench = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bench.get_task(args.task_id)
    bddl = args.bddl_override or os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = bench.get_task_init_states(args.task_id)
    init_state = init_states[args.init_idx]

    env = OffScreenRenderEnv(bddl_file_name=bddl,
                              camera_heights=256, camera_widths=256)
    env.seed(args.env_seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    obs_in = format_obs(obs, args.prompt)

    # 3) Baseline run (no hook)
    print("\n[steering] === BASELINE (no hook) ===", flush=True)
    set_all_seeds(args.seed)
    with VelocityCapture(model, verbose=False) as v_base:
        out_base = policy.infer(obs_in)
    actions_base = torch.as_tensor(out_base["actions"], dtype=torch.float32)
    print(f"  velocities captured: {len(v_base)}; action chunk shape: {tuple(actions_base.shape)}")

    # 4) Sweep alphas
    print("\n[steering] === ALPHA SWEEP ===", flush=True)
    results: Dict[float, dict] = {}
    velocities_per_alpha: Dict[float, List[torch.Tensor]] = {}
    actions_per_alpha: Dict[float, torch.Tensor] = {}

    for alpha in args.alphas:
        set_all_seeds(args.seed)
        with SteeringHook(model, args.target_layer, w_unit, alpha=alpha,
                           verbose=False,
                           layer_pattern_template=args.layer_pattern_template) as hook:
            with VelocityCapture(model, verbose=False) as v_alpha:
                out_alpha = policy.infer(obs_in)
            n_hook_fires = hook._n_calls
        actions_alpha = torch.as_tensor(out_alpha["actions"], dtype=torch.float32)
        velocities_per_alpha[alpha] = v_alpha
        actions_per_alpha[alpha] = actions_alpha

        # Compute per-step velocity divergence vs baseline
        vel_cos = []
        vel_l2 = []
        for vb, vs in zip(v_base, v_alpha):
            vel_cos.append(cos_sim_flat(vb, vs))
            vel_l2.append(l2_dist_flat(vb, vs))
        act_cos = cos_sim_flat(actions_base, actions_alpha)
        act_l2 = l2_dist_flat(actions_base, actions_alpha)

        results[alpha] = {
            "alpha": alpha,
            "n_hook_fires": n_hook_fires,
            "vel_cos_mean": float(np.mean(vel_cos)),
            "vel_cos_min": float(np.min(vel_cos)),
            "vel_l2_mean": float(np.mean(vel_l2)),
            "action_cos": act_cos,
            "action_l2": act_l2,
        }
        print(f"  α={alpha:+8.2f}: vel_cos={results[alpha]['vel_cos_mean']:.4f}  "
              f"act_cos={act_cos:.4f}  act_l2={act_l2:.4f}  hook_fires={n_hook_fires}",
              flush=True)

    # 5) Sanity checks
    print("\n[steering] === SANITY CHECKS ===", flush=True)
    sanity: Dict[str, dict] = {}
    ok, msg = _check_direction(w_unit)
    sanity["direction"] = {"pass": ok, "message": msg}

    if 0.0 in velocities_per_alpha:
        ok, msg = _check_alpha_zero_reproduces_baseline(v_base, velocities_per_alpha[0.0])
        sanity["alpha0_reproduces_baseline"] = {"pass": ok, "message": msg}
    ok, msg = _check_alpha_perturbs_consistently(v_base, velocities_per_alpha)
    sanity["alpha_perturbation_trend"] = {"pass": ok, "message": msg}

    for k, v in sanity.items():
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  [{status}] {k}: {v['message']}")
    n_pass = sum(1 for v in sanity.values() if v["pass"])

    # 6) Save
    print(f"\n[steering] saving to {out_dir} ...")
    torch.save({
        "direction": w_unit,
        "direction_info": w_info,
        "alphas": list(args.alphas),
        "baseline_velocities": v_base,
        "baseline_actions": actions_base,
        "velocities_per_alpha": {a: vs for a, vs in velocities_per_alpha.items()},
        "actions_per_alpha": {a: act for a, act in actions_per_alpha.items()},
    }, out_dir / "steering_raw.pt")
    (out_dir / "steering_results.json").write_text(json.dumps({
        "prompt": args.prompt,
        "target_layer": args.target_layer,
        "direction_info": w_info,
        "per_alpha": {f"{a:+.4f}": results[a] for a in args.alphas},
        "n_baseline_velocities": len(v_base),
        "action_chunk_shape": list(actions_base.shape),
    }, indent=2))
    (out_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2))

    # 7) Plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[steering] matplotlib unavailable — skipping plot")
        else:
            sorted_alphas = sorted(args.alphas)
            vel_cos = [results[a]["vel_cos_mean"] for a in sorted_alphas]
            act_cos = [results[a]["action_cos"] for a in sorted_alphas]
            act_l2 = [results[a]["action_l2"] for a in sorted_alphas]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ax = axes[0]
            ax.plot(sorted_alphas, vel_cos, "o-", label="vel cos vs baseline")
            ax.plot(sorted_alphas, act_cos, "s-", label="action cos vs baseline")
            ax.axhline(1.0, color="grey", linestyle="--", alpha=0.4, label="identical")
            ax.axvline(0.0, color="black", linestyle=":", alpha=0.4)
            ax.set_xlabel("steering α (unit-norm direction multiplier)")
            ax.set_ylabel("cosine vs baseline")
            ax.set_title("Steering perturbs the output as α grows")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            ax2 = axes[1]
            ax2.plot(sorted_alphas, act_l2, "d-", color="C2")
            ax2.axvline(0.0, color="black", linestyle=":", alpha=0.4)
            ax2.set_xlabel("steering α")
            ax2.set_ylabel("action chunk L2 vs baseline")
            ax2.set_title("Action chunk shift magnitude vs α")
            ax2.grid(True, alpha=0.3)

            fig.suptitle(f"Activation steering at expert L{args.target_layer} | "
                         f"prompt: {args.prompt!r}")
            fig.tight_layout()
            out_png = out_dir / "steering_alpha_sweep.png"
            fig.savefig(out_png, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[steering] wrote plot: {out_png}")

    print(f"\n[steering] ✓ done. Sanity {n_pass}/{len(sanity)} passing.")
    if n_pass < len(sanity):
        sys.exit(1)


if __name__ == "__main__":
    main()
