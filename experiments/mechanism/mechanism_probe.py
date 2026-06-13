"""mechanism_probe.py — cross-attention divergence probe for pi0 (PyTorch).

Hypothesis (FLARE language-bypass thesis): SFT-fine-tuned flow VLAs preserve
attention to language tokens at early layers but the action expert decouples
downstream. Two-prompt-same-observation inference + attention-weight capture
locates the bypass at the layer where attention divergence drops to ~0.

This script runs pi0 on ONE observation with TWO prompts using the SAME noise
seed, captures attention weights at every attention layer via monkey-patched
forward methods, and saves them to disk for divergence analysis.

REQUIRED: pi0 (or any pi0_pytorch variant). For JAX-only pi0.5 checkpoints,
the model extraction will fail loudly — see mechanism_probe_velocity.py for
the JAX-compatible path.

Outputs in --out-dir:
    captured_a.pt          {"prompt": str, "captures": {layer_name: [tensor or None, ...]}}
    captured_b.pt          same for prompt B
    captured_a_dup.pt      (if --self-consistency-check) prompt A re-run
    metadata.json          model class, prompt strings, shapes, layer order, env config
    sanity_checks.json     PASS/FAIL on each built-in correctness check

If any sanity check fails, the script exits with status 1 so wrapper scripts
won't mistakenly treat a broken run as success.

Usage:
    python3 mechanism_probe.py \\
        --policy-config pi0_libero \\
        --checkpoint-dir ~/flare/checkpoints/pi0_libero_pt \\
        --task-suite libero_10 --task-id 0 \\
        --prompt-a "put both the alphabet soup and the tomato sauce in the basket" \\
        --prompt-b "put both the tomato sauce and the alphabet soup in the basket" \\
        --self-consistency-check \\
        --out-dir ~/flare/results/mechanism_probe_T0

After running, feed the directory to analyze_mechanism_probe.py.
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


# =============================================================================
# ATTENTION CAPTURE
# =============================================================================

class AttentionCapture:
    """Context manager: captures attention weights at every attention layer.

    Strategy: for each module whose class name ends with "Attention" and which
    has q/k/v projection submodules (real attention, not container), replace
    its `forward` method with a wrapper that (a) forces `output_attentions=True`
    and (b) captures the returned attention tensor into a dict.

    Captured dict structure:
        { layer_name: [attn_tensor_call_0, attn_tensor_call_1, ...] }

    Each call corresponds to one forward pass through that layer (i.e., one
    denoising step in the integration loop, plus the prefix-encoding call).

    Tensors are .detach().cpu()-d and shaped (batch, num_heads, q_len, k_len).
    If a particular call returns no attention (e.g., the impl ignored the kwarg),
    the list contains None at that index.
    """

    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.captured: Dict[str, List[Optional[torch.Tensor]]] = {}
        self._orig: Dict[str, callable] = {}
        self._modules: Dict[str, nn.Module] = {}

    @staticmethod
    def _is_attention(module: nn.Module) -> bool:
        cn = module.__class__.__name__
        if not cn.endswith("Attention"):
            return False
        # Real attention has q/k/v projections; container modules don't.
        # Accepting any of (q_proj, qkv_proj, wq) — covers Gemma, Llama, custom.
        return any(hasattr(module, a) for a in ("q_proj", "qkv_proj", "wq"))

    def _make_wrapper(self, name: str, original):
        captured = self.captured

        def wrapped(*args, **kwargs):
            kwargs["output_attentions"] = True
            try:
                out = original(*args, **kwargs)
            except TypeError:
                # Attention class doesn't accept output_attentions — retry without.
                kwargs.pop("output_attentions", None)
                out = original(*args, **kwargs)
                captured.setdefault(name, []).append(None)
                return out
            # HF eager returns (attn_output, attn_weights, past_kv) or similar.
            # attn_weights is the 4-d (B, H, Q, K) tensor.
            attn = None
            if isinstance(out, tuple):
                for item in out[1:]:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        attn = item
                        break
            captured.setdefault(name, []).append(
                attn.detach().to("cpu") if attn is not None else None
            )
            return out

        return wrapped

    def __enter__(self):
        for name, module in self.model.named_modules():
            if self._is_attention(module):
                self._modules[name] = module
        if not self._modules:
            seen = sorted({m.__class__.__name__ for _, m in self.model.named_modules()})
            raise RuntimeError(
                f"No attention modules found in {self.model.__class__.__name__}.\n"
                f"Module classes seen (sample): {seen[:20]}"
            )
        for name, module in self._modules.items():
            # Save the bound method (calling the descriptor's __get__).
            # We use this for wrapped() to call the real attention forward.
            self._orig[name] = module.forward
            # Track whether 'forward' existed as an INSTANCE attribute before
            # we patched it — so we restore correctly on exit.
            self._had_instance_attr = getattr(self, "_had_instance_attr", {})
            self._had_instance_attr[name] = "forward" in module.__dict__
            module.forward = self._make_wrapper(name, self._orig[name])
        if self.verbose:
            print(f"[AttentionCapture] Hooked {len(self._modules)} attention modules.")
        return self.captured

    def __exit__(self, *_):
        had_instance = getattr(self, "_had_instance_attr", {})
        for name, original in self._orig.items():
            module = self._modules[name]
            if had_instance.get(name, False):
                # 'forward' was a real instance attribute before; restore it.
                module.forward = original
            else:
                # 'forward' was inherited from the class via descriptor; the
                # cleanest restoration is to delete our instance shadow.
                if "forward" in module.__dict__:
                    del module.__dict__["forward"]
        if self.verbose:
            print(f"[AttentionCapture] Restored {len(self._orig)} forwards.")
        self._orig.clear()
        self._had_instance_attr = {}


# =============================================================================
# EAGER ATTENTION ENFORCEMENT
# =============================================================================

def force_eager_attention(model: nn.Module, verbose: bool = True) -> List[str]:
    """Set _attn_implementation='eager' on every config we find.

    Necessary so attention weights are actually computed (not fused away) and
    can be captured. patch_pi0_sde.py sets this on paligemma.language_model
    but the action expert (and other sub-modules) may use a different config.
    """
    changed: List[str] = []
    for name, module in model.named_modules():
        cfg = getattr(module, "config", None)
        if cfg is None:
            continue
        if hasattr(cfg, "_attn_implementation"):
            old = getattr(cfg, "_attn_implementation", None)
            if old != "eager":
                cfg._attn_implementation = "eager"
                changed.append(f"{name}: {old!r} -> 'eager'")
    if verbose:
        print(f"[force_eager_attention] Modified {len(changed)} configs.")
        for c in changed:
            print(f"  {c}")
    return changed


# =============================================================================
# SANITY CHECKS
# =============================================================================

def _check_nonempty(captured: Dict[str, List]) -> Tuple[bool, str]:
    if not captured:
        return False, "no layers captured"
    n_with = sum(1 for v in captured.values() if any(t is not None for t in v))
    return (n_with > 0), f"{n_with}/{len(captured)} layers captured >=1 non-None tensor"


def _check_distribution(captured: Dict[str, List], tol: float = 1e-2
                        ) -> Tuple[bool, str]:
    """Each attention row should sum to ~1 (active query) or ~0 (masked query)."""
    bad: List[str] = []
    for name, lst in captured.items():
        for i, a in enumerate(lst):
            if a is None:
                continue
            if a.dim() != 4:
                bad.append(f"{name}[{i}] dim={a.dim()} (expected 4)")
                continue
            s = a.sum(dim=-1)  # (B, H, Q)
            near_one = (s - 1).abs() < tol
            near_zero = s.abs() < tol
            if not (near_one | near_zero).all().item():
                err = (s[~(near_one | near_zero)] - 1).abs().max().item()
                bad.append(f"{name}[{i}] worst row-sum err = {err:.4f}")
                if len(bad) >= 5:
                    break
        if len(bad) >= 5:
            break
    if bad:
        return False, f"{len(bad)} violations: " + " | ".join(bad[:3])
    return True, "all attention rows sum to ~0 or ~1"


def _check_shapes_match(ca: Dict[str, List], cb: Dict[str, List]
                        ) -> Tuple[bool, str]:
    if set(ca) != set(cb):
        return False, (f"layer keys differ: "
                       f"only_a={list(set(ca)-set(cb))[:3]}, "
                       f"only_b={list(set(cb)-set(ca))[:3]}")
    diffs: List[str] = []
    for name in ca:
        la, lb = ca[name], cb[name]
        if len(la) != len(lb):
            diffs.append(f"{name}: n_calls {len(la)}!={len(lb)}")
            continue
        for i, (a, b) in enumerate(zip(la, lb)):
            if (a is None) != (b is None):
                diffs.append(f"{name}[{i}]: None mismatch")
                continue
            if a is None:
                continue
            if a.shape != b.shape:
                diffs.append(f"{name}[{i}]: {tuple(a.shape)} vs {tuple(b.shape)}")
                if len(diffs) >= 5:
                    break
    if diffs:
        return False, f"{len(diffs)} mismatches: " + " | ".join(diffs[:3])
    return True, "all layer keys, call counts, and shapes match"


def _check_self_consistency(ca: Dict[str, List], ca_dup: Dict[str, List],
                             tol: float = 1e-4) -> Tuple[bool, str, float]:
    """If A was captured twice with same seed+prompt, both should be bit-identical.

    If this FAILS, downstream divergence numbers are not trustworthy (there's
    nondeterminism in the inference path we haven't controlled for).
    """
    max_diff = 0.0
    where = ""
    n = 0
    for name in ca:
        if name not in ca_dup:
            continue
        la, lb = ca[name], ca_dup[name]
        if len(la) != len(lb):
            continue
        for a, b in zip(la, lb):
            if a is None or b is None:
                continue
            d = (a.float() - b.float()).abs().max().item()
            n += 1
            if d > max_diff:
                max_diff, where = d, name
    msg = f"max abs diff = {max_diff:.2e} at '{where}' (over {n} tensor pairs)"
    return (max_diff <= tol), msg, max_diff


def _check_prompts_differ(prompt_a: str, prompt_b: str) -> Tuple[bool, str]:
    if prompt_a == prompt_b:
        return False, "prompts are identical — at least one of --prompt-a/--prompt-b must change"
    return True, f"prompts differ (a={len(prompt_a)}ch, b={len(prompt_b)}ch)"


def _check_observation_parity(obs_a: dict, obs_b: dict) -> Tuple[bool, str]:
    """Confirm only the prompt differs between obs_a and obs_b."""
    problems = []
    for k in ("observation/image", "observation/wrist_image"):
        if k in obs_a and k in obs_b:
            if not np.array_equal(obs_a[k], obs_b[k]):
                problems.append(f"{k} differs")
    if "observation/state" in obs_a and "observation/state" in obs_b:
        if not np.allclose(obs_a["observation/state"], obs_b["observation/state"]):
            problems.append("observation/state differs")
    if obs_a.get("prompt") == obs_b.get("prompt"):
        problems.append("prompt is identical")
    if problems:
        return False, "; ".join(problems)
    return True, "images & state identical; only prompt differs"


def run_sanity_checks(captured_a, captured_b, prompt_a, prompt_b,
                       obs_a, obs_b, captured_a_dup=None
                       ) -> Dict[str, dict]:
    """Run all sanity checks; return dict of {check_name: {pass, message, ...}}."""
    out: Dict[str, dict] = {}

    ok, msg = _check_prompts_differ(prompt_a, prompt_b)
    out["prompts_differ"] = {"pass": ok, "message": msg}

    ok, msg = _check_observation_parity(obs_a, obs_b)
    out["observation_parity"] = {"pass": ok, "message": msg}

    ok, msg = _check_nonempty(captured_a)
    out["A_nonempty"] = {"pass": ok, "message": msg}
    ok, msg = _check_nonempty(captured_b)
    out["B_nonempty"] = {"pass": ok, "message": msg}

    ok, msg = _check_distribution(captured_a)
    out["A_valid_distribution"] = {"pass": ok, "message": msg}
    ok, msg = _check_distribution(captured_b)
    out["B_valid_distribution"] = {"pass": ok, "message": msg}

    ok, msg = _check_shapes_match(captured_a, captured_b)
    out["A_B_shape_match"] = {"pass": ok, "message": msg}

    if captured_a_dup is not None:
        ok, msg, max_diff = _check_self_consistency(captured_a, captured_a_dup)
        out["A_self_consistency"] = {
            "pass": ok, "message": msg, "max_diff": max_diff
        }
    else:
        out["A_self_consistency"] = {
            "pass": True, "message": "skipped (use --self-consistency-check to enable)"
        }

    return out


# =============================================================================
# DETERMINISTIC SEEDING
# =============================================================================

def set_all_seeds(seed: int) -> None:
    """Set torch + numpy + python seeds for reproducible noise sampling."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN determinism (slows things down a bit but removes a noise source).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# MAIN PROBE
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-config", default="pi0_libero",
                   help="openpi config name (pi0_libero / pi05_libero / ...)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Checkpoint directory; default = ~/flare/checkpoints/pi0_libero_pt")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--bddl-override", default=None,
                   help="Optional BDDL path to override the task's default")
    p.add_argument("--prompt-a", required=True)
    p.add_argument("--prompt-b", required=True)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed reset before each policy.infer call (noise reproducibility)")
    p.add_argument("--env-seed", type=int, default=7000)
    p.add_argument("--init-idx", type=int, default=0)
    p.add_argument("--self-consistency-check", action="store_true",
                   help="Re-run prompt A a second time; verify capture is bit-identical")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Imports that need the openpi env (delayed for cleaner error messages) -----
    print("[probe] importing openpi/libero ...", flush=True)
    from openpi.training import config as _c
    from openpi.policies import policy_config
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    # Reuse format_obs + constants from the matrix runner.
    sys.path.insert(0, str(Path.home() / "flare"))
    from patch_pi0_sde import apply_patch  # noqa: E402
    from remote_multimode_matrix import (  # type: ignore  # noqa: E402
        format_obs, NUM_STEPS_WAIT, LIBERO_DUMMY_ACTION,
    )

    # ----- Load policy + extract underlying nn.Module -----
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
        print(f"[probe] no --checkpoint-dir; downloading from {gs} ...", flush=True)
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
            f"This probe requires a PyTorch pi0 (not JAX-only). "
            f"Top-level policy attrs: "
            f"{[a for a in dir(policy) if not a.startswith('_')][:20]}"
        )

    # Disable any SDE/guidance side-effects — pure ODE inference, deterministic up to noise.
    model.flare_eta = 0.0
    model.flare_verifier_fn = None
    model.flare_alpha = 0.0
    model.flare_obs_state = None
    model.flare_eta_high = None
    model.flare_eta_low = None
    model.flare_noise_bias_direction = None
    model.flare_noise_bias_strength = 0.0

    # eval mode (no dropout, no batchnorm-training behavior).
    model.eval()

    # Force eager attention everywhere so attention weights are produced.
    force_eager_attention(model, verbose=True)

    # ----- Setup environment + observation -----
    print(f"[probe] env setup: suite={args.task_suite} task_id={args.task_id}",
          flush=True)
    bench = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bench.get_task(args.task_id)
    if args.bddl_override:
        bddl = args.bddl_override
        print(f"[probe]   using BDDL override: {bddl}", flush=True)
    else:
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        print(f"[probe]   default BDDL: {bddl}", flush=True)

    init_states = bench.get_task_init_states(args.task_id)
    if args.init_idx >= len(init_states):
        raise ValueError(f"--init-idx={args.init_idx} >= len(init_states)={len(init_states)}")
    init_state = init_states[args.init_idx]

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(args.env_seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    obs_a = format_obs(obs, args.prompt_a)
    obs_b = format_obs(obs, args.prompt_b)

    # ----- Run inference with capture -----
    print(f"[probe] PROMPT A: {args.prompt_a!r}", flush=True)
    set_all_seeds(args.seed)
    t0 = time.time()
    with AttentionCapture(model, verbose=True) as captured_a:
        out_a = policy.infer(obs_a)
    print(f"[probe]   done in {time.time()-t0:.1f}s; "
          f"{len(captured_a)} layers; "
          f"{sum(len(v) for v in captured_a.values())} total tensors captured",
          flush=True)
    # Sanity: actions should be present
    if "actions" not in out_a:
        print(f"[probe]   WARN: policy.infer output keys = {list(out_a.keys())}",
              flush=True)

    print(f"[probe] PROMPT B: {args.prompt_b!r}", flush=True)
    set_all_seeds(args.seed)
    t0 = time.time()
    with AttentionCapture(model, verbose=True) as captured_b:
        _ = policy.infer(obs_b)
    print(f"[probe]   done in {time.time()-t0:.1f}s", flush=True)

    captured_a_dup = None
    if args.self_consistency_check:
        print("[probe] re-running PROMPT A for self-consistency check ...", flush=True)
        set_all_seeds(args.seed)
        with AttentionCapture(model, verbose=False) as captured_a_dup:
            _ = policy.infer(obs_a)

    # ----- Sanity checks -----
    print("\n[probe] === SANITY CHECKS ===", flush=True)
    sanity = run_sanity_checks(
        captured_a, captured_b, args.prompt_a, args.prompt_b,
        obs_a, obs_b, captured_a_dup
    )
    for k, v in sanity.items():
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  [{status}] {k}: {v['message']}", flush=True)
    n_pass = sum(1 for v in sanity.values() if v["pass"])
    n_total = len(sanity)
    print(f"[probe] sanity: {n_pass}/{n_total} passing\n", flush=True)

    # ----- Save -----
    print(f"[probe] saving to {out_dir} ...", flush=True)
    torch.save({"prompt": args.prompt_a, "captures": dict(captured_a)},
               out_dir / "captured_a.pt")
    torch.save({"prompt": args.prompt_b, "captures": dict(captured_b)},
               out_dir / "captured_b.pt")
    if captured_a_dup is not None:
        torch.save({"prompt": args.prompt_a, "captures": dict(captured_a_dup)},
                   out_dir / "captured_a_dup.pt")

    layer_summary = {
        name: {
            "n_calls": len(lst),
            "shapes": [tuple(t.shape) if t is not None else None for t in lst],
        }
        for name, lst in captured_a.items()
    }
    meta = {
        "policy_config": args.policy_config,
        "checkpoint_dir": ckpt_dir,
        "model_class": model.__class__.__name__,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "bddl": bddl,
        "init_idx": args.init_idx,
        "env_seed": args.env_seed,
        "seed": args.seed,
        "n_wait_steps": NUM_STEPS_WAIT,
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "obs_state": obs_a["observation/state"].tolist(),
        "n_attention_layers": len(captured_a),
        "layer_summary": layer_summary,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2))

    print(f"[probe] ✓ done. Files in {out_dir}", flush=True)
    if n_pass < n_total:
        print(f"[probe] WARN: {n_total - n_pass} sanity check(s) FAILED — "
              f"do NOT trust the analysis until these are resolved.", flush=True)
        sys.exit(1)


# =============================================================================
# MANUAL VERIFICATION CHECKLIST (after a successful run)
# =============================================================================
# 1. Open sanity_checks.json — every check should be "pass": true.
#    Of particular importance:
#      - prompts_differ              — confirms the two prompts are actually different.
#      - observation_parity          — confirms only language differs.
#      - A_self_consistency          — confirms inference is deterministic
#                                       (max_diff should be 0.0 or near-machine-epsilon).
#      - A_valid_distribution        — confirms captured tensors really are attention.
# 2. Open metadata.json:
#      - n_attention_layers should be plausible (~30-50 for a paligemma+expert).
#      - layer_summary should show n_calls > 1 for action-expert layers
#        (one prefix call + N denoising calls).
#      - Shapes: prefix call has Q_len = prefix_len; denoising calls have
#        Q_len = action_horizon. Different across the two.
# 3. Eyeball captured_a.pt — sum(attn[0,0,0,:]) ~ 1.0 for active queries.
# =============================================================================


if __name__ == "__main__":
    main()
