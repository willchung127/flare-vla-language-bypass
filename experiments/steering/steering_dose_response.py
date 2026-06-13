#!/usr/bin/env python3
"""steering_dose_response.py — test-time (no fine-tuning) recovery of language use via residual steering.

The keystone experiment. Answers two of Unnat's questions at once:
  (1) "Is there a test-time-only optimization instead of LoRA?"  -> yes: residual steering.
  (2) "Is it the data or is the VLA incapable?"  -> if writing a per-object language DIRECTION into the
      action expert's residual makes it SELECT that object, the CAPACITY is there (a data/usage problem,
      not a capability limit) — and it moves behavior where forcing ATTENTION (eval_forced_attention.py)
      provably could not ("steerable-but-not-decodable").

Method:
  BANK   : v = mean(resid | "...tomato...") - mean(resid | "...alphabet...") at expert layer L, action rows.
  DIAG   : steer the AMBIGUOUS prompt by alpha*v; does its velocity move toward the tomato/alphabet prompt's?
  ROLLOUT: closed-loop, ambiguous prompt + SteeringHook(alpha); does the FIRST-moved object follow alpha?
           (+alpha -> tomato, -alpha -> alphabet). Uses eval_grounding's temporal-first + in-basket logic.

USAGE (remote, openpi venv):
  ~/flare/openpi/.venv/bin/python ~/flare/eval_steering_bank.py \
     --checkpoint ~/flare/checkpoints/pi0_libero_pt --layer 16 --out-dir ~/flare/results/steering_bank
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

FLARE = Path.home() / "flare"
sys.path.insert(0, str(FLARE))

from audit_lora_attention import (  # noqa: E402
    load_model, VelocityCapture, setup_env as setup_env_obs, N_DENOISE_REAL_DIMS,
)

PROMPT_A = ("tomato", "pick up the tomato sauce and put it in the basket")
PROMPT_B = ("alphabet", "pick up the alphabet soup and put it in the basket")
AMBIG_PROMPT = "put both the alphabet soup and the tomato sauce in the basket"   # which gets grabbed first?


class ResidualCapture:
    """Forward hook on the expert decoder layer; capture the residual at action-query rows."""
    def __init__(self, model, layer_idx):
        import re
        self.captured = []
        self._h = None
        pat = re.compile(rf"^paligemma_with_expert\.gemma_expert\.model\.layers\.{layer_idx}$")
        self.module = next((m for n, m in model.named_modules() if pat.match(n)), None)
        if self.module is None:
            raise RuntimeError(f"no expert layer {layer_idx} module found")

    def _hook(self, module, inp, out):
        import torch
        hs = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(hs) and hs.dim() == 3 and hs.shape[1] >= 50:
            self.captured.append(hs[0, -50:].mean(0).detach().cpu().float().numpy())   # (D,)

    def __enter__(self):
        self._h = self.module.register_forward_hook(self._hook); return self

    def __exit__(self, *a):
        if self._h: self._h.remove()


def _resid_mean(policy, model, obs, layer, seed=42):
    from mechanism_probe_attn import set_all_seeds
    set_all_seeds(seed)
    with ResidualCapture(model, layer) as cap:
        _ = policy.infer(obs)
    return np.mean(np.stack(cap.captured, 0), 0) if cap.captured else None


def build_direction(policy, model, env_obs, layer):
    """v = mean(resid|tomato-prompt) - mean(resid|alphabet-prompt), mean over denoise steps. unit dir + raw norm."""
    a = _resid_mean(policy, model, env_obs["format_obs"](env_obs["obs"], PROMPT_A[1]), layer)
    b = _resid_mean(policy, model, env_obs["format_obs"](env_obs["obs"], PROMPT_B[1]), layer)
    raw = a - b
    norm = float(np.linalg.norm(raw))
    return raw / (norm + 1e-9), norm


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def velocity_steps(policy, model, obs, layer=None, direction=None, alpha=0.0):
    from mechanism_probe_attn import set_all_seeds
    from activation_steering import SteeringHook
    set_all_seeds(42)
    cm = SteeringHook(model, layer, direction, alpha=alpha, verbose=False) if (direction is not None and alpha != 0.0) else _Null()
    with cm, VelocityCapture(model) as vels:
        _ = policy.infer(obs)
    return [v.mean(0).float().numpy() for v in vels] if vels else None


def cos_real7(va, vb):
    R = N_DENOISE_REAL_DIMS
    return float(np.mean([float(np.dot(A[..., :R].ravel(), B[..., :R].ravel()) /
                                (np.linalg.norm(A[..., :R]) * np.linalg.norm(B[..., :R]) + 1e-12))
                          for A, B in zip(va, vb)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(FLARE / "checkpoints/pi0_libero_pt"))
    p.add_argument("--config-name", default="pi0_libero")
    p.add_argument("--layer", type=int, default=16, help="expert layer to steer (try 12 / 14 / 16)")
    # NOTE: direction is UNIT-normalized; the raw contrastive norm is ~76, so natural
    # alphas are O(10-100). The old default (+-800/+-1500) was ~10x too large and
    # collapsed the policy (0 success at every nonzero alpha) — an INCONCLUSIVE run,
    # not a negative result. argparse gotcha: leading '-' needs '--alphas="-150,..."'.
    p.add_argument("--alphas", default="-150,-100,-60,-40,-20,0,20,40,60,100,150",
                   help="steering magnitudes to sweep (natural scale)")
    p.add_argument("--direction-mode", default="contrastive",
                   choices=["contrastive", "random", "shuffle"],
                   help="contrastive = tomato-minus-alphabet (the real direction); "
                        "random = seeded random unit vector (placebo); "
                        "shuffle = permuted contrastive direction (norm/stat-matched placebo)")
    p.add_argument("--placebo-seed", type=int, default=1234)
    p.add_argument("--rollout-trials", type=int, default=8, help="closed-loop trials per alpha (0 = skip rollouts)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]

    policy, model = load_model(args.config_name, args.checkpoint)
    if model is None:
        raise RuntimeError("need a PyTorch checkpoint")
    env_obs = setup_env_obs("libero_10", 0, env_seed=7000, init_idx=0)   # obs + format_obs (for bank/diag)

    print(f"[bank] building direction at expert L{args.layer}  ({PROMPT_A[0]} - {PROMPT_B[0]}) ...", flush=True)
    direction, raw_norm = build_direction(policy, model, env_obs, args.layer)
    print(f"[bank] ||raw direction|| = {raw_norm:.3f}", flush=True)

    # Placebo controls: same norm (unit), same alphas — only the CONTENT differs.
    # If the contrastive direction flips first-moved but placebos don't, the effect
    # is direction-specific (the Tan-et-al. brittleness defense).
    if args.direction_mode == "random":
        rng = np.random.default_rng(args.placebo_seed)
        v = rng.standard_normal(direction.shape).astype(direction.dtype)
        direction = v / (np.linalg.norm(v) + 1e-9)
        print(f"[bank] PLACEBO: random unit direction (seed {args.placebo_seed})", flush=True)
    elif args.direction_mode == "shuffle":
        rng = np.random.default_rng(args.placebo_seed)
        direction = rng.permutation(direction)
        print(f"[bank] PLACEBO: shuffled contrastive direction (seed {args.placebo_seed})", flush=True)

    # --- DIAGNOSTIC: does steering the AMBIGUOUS prompt toward tomato make its velocity look like the tomato prompt's? ---
    obs_s = env_obs["format_obs"](env_obs["obs"], AMBIG_PROMPT)
    v_A = velocity_steps(policy, model, env_obs["format_obs"](env_obs["obs"], PROMPT_A[1]))
    v_B = velocity_steps(policy, model, env_obs["format_obs"](env_obs["obs"], PROMPT_B[1]))
    diag = {"baseline_cos_tomato_vs_alphabet": cos_real7(v_A, v_B)}
    for alpha in alphas:
        v_s = velocity_steps(policy, model, obs_s, args.layer, direction, alpha)
        if v_s:
            diag[f"alpha={alpha:+.0f}"] = {"cos_to_tomato": cos_real7(v_s, v_A), "cos_to_alphabet": cos_real7(v_s, v_B)}
            d = diag[f"alpha={alpha:+.0f}"]
            print(f"  [diag] a={alpha:+.0f}: cos(steered,tomato)={d['cos_to_tomato']:.4f}  "
                  f"cos(steered,alphabet)={d['cos_to_alphabet']:.4f}", flush=True)
    summary = {"layer": args.layer, "raw_norm": raw_norm,
               "direction_mode": args.direction_mode,
               "placebo_seed": args.placebo_seed if args.direction_mode != "contrastive" else None,
               "diagnostic": diag}

    # --- ROLLOUT: does the FIRST-moved object follow the steering? (the behavioral keystone) ---
    if args.rollout_trials > 0:
        from eval_lora_behavioral import setup_env as setup_env_full
        from eval_grounding import grounding_trial, temporal_first_object
        from activation_steering import SteeringHook
        env_full = setup_env_full("libero_10", 0)        # task/bddl/init_states/mmm (for rollouts)
        n = min(args.rollout_trials, env_full["n_inits"])
        roll = {}
        for alpha in alphas:
            rows = []
            for i in range(n):
                cm = SteeringHook(model, args.layer, direction, alpha=alpha, verbose=False) if alpha != 0 else _Null()
                with cm:
                    r = grounding_trial(policy, model, env_full, env_full["init_states"][i], 7000 + i, AMBIG_PROMPT, 5)
                tf = temporal_first_object(r["init_pos"], r["waypoints"])
                rows.append({"first_moved": tf, "tomato_first": "tomato" in tf.lower(),
                             "alphabet_first": "alphabet" in tf.lower(), "success": r["done"]})
            tk = sum(x["tomato_first"] for x in rows); ak = sum(x["alphabet_first"] for x in rows)
            roll[f"alpha={alpha:+.0f}"] = {"n": len(rows), "tomato_first": tk, "alphabet_first": ak,
                                           "tomato_first_rate": tk / max(len(rows), 1),
                                           "alphabet_first_rate": ak / max(len(rows), 1),
                                           "success": sum(x["success"] for x in rows)}
            print(f"  [rollout] a={alpha:+.0f}: tomato-first {tk}/{len(rows)}  alphabet-first {ak}/{len(rows)}", flush=True)
        summary["rollout"] = roll

    (out / "steering_bank.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[done] {out}/steering_bank.json")
    print("  KEY: if first-moved FOLLOWS alpha (tomato at +alpha, alphabet at -alpha), residual steering "
          "RECOVERS object selection at TEST TIME -> capacity is there (data/usage problem, not a VLA limit), "
          "and it moves behavior where forcing attention could not. Sweep --layer 12/14/16 if flat.")


if __name__ == "__main__":
    main()
