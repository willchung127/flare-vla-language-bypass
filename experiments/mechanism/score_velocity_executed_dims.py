"""score_velocity_executed_dims.py — verify the cos(v_A, v_B) headline is not
inflated by either (1) zero-padded action dims, or (2) the discarded tail of
the predicted action chunk.

pi0_libero internal action_dim = 32 (padded for multi-robot training).
LIBERO real action_dim = 7 (3 xyz + 3 rpy + 1 gripper).
pi0_libero chunk horizon H = 50 actions per forward pass.
**Only the first 5 actions per chunk are EXECUTED before replan; the remaining
45 are computed and discarded.**

The original velocity_divergence output computed cos over the entire
flattened (1, H, D) tensor. That averages over:
  - 25 zero-padded dims (a known artifact of multi-robot training)
  - 45 abandoned future-action positions (computed but never rolled out)

Both could inflate the headline cos≈0.999 above what the EXECUTED actions
actually look like. This script reports cos restricted to:

  PART 1 — by action dim:
    cos(all 32) — what the original probe reported
    cos(real 7) — restricted to LIBERO's actual action dims
    cos(pad 25) — sanity check on padded artifacts

  PART 2 — by action position (within real 7 dims):
    cos(pos[0:N]) for N ∈ {5, 10, 15, 25, 50}
    cos(pos[N:50]) for N ∈ {5, 10, 15, 25}  (the discarded-tail complement)

The critical number for behavior is **cos(pos[0:5], real 7 dims)** — that's
what the model actually executes.

Usage:
    python3 score_velocity_executed_dims.py \\
        --velocity-dirs ~/flare/results/probe_velocity_T0_v1 \\
                        ~/flare/results/probe_velocity_T0_P1_single \\
                        ~/flare/results/probe_velocity_T0_P2_order \\
                        ~/flare/results/probe_velocity_T0_P3_unrelated \\
                        ~/flare/results/probe_velocity_T1_v1 \\
        --real-action-dim 7 \\
        --position-slices 5 10 15 25 50 \\
        --out-json ~/flare/results/velocity_real_dims_extended.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Flatten-then-cosine. Treats the tensor as a single vector."""
    af = a.flatten().float()
    bf = b.flatten().float()
    n = (af.norm() * bf.norm()).clamp(min=eps)
    return float((af * bf).sum() / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--velocity-dirs", nargs="+", required=True)
    p.add_argument("--real-action-dim", type=int, default=7)
    p.add_argument("--internal-action-dim", type=int, default=32)
    p.add_argument("--position-slices", nargs="+", type=int,
                   default=[5, 10, 15, 25, 50],
                   help="Action-position prefixes (first N actions). "
                        "Default {5,10,15,25,50}. 5 = pi0_libero replan_steps.")
    p.add_argument("--action-horizon", type=int, default=50,
                   help="Total action chunk horizon (pi0_libero=50)")
    p.add_argument("--out-json", default=None,
                   help="Optional path to save full per-probe results as JSON")
    args = p.parse_args()

    R = args.real_action_dim
    D = args.internal_action_dim
    H = args.action_horizon
    slices = sorted(set(args.position_slices))

    # ========================================================================
    # PART 1 header — cos by action DIMENSION
    # ========================================================================
    print(f"\n{'='*100}")
    print(f"  PART 1: cos restricted by ACTION DIMENSION")
    print(f"  (real action dims 0..{R-1} = LIBERO's xyz+rpy+gripper; "
          f"padded {R}..{D-1} are unused-robot slots)")
    print(f"{'='*100}")
    print(f"  {'probe':<42s} {'cos(all)':>10s} {'cos(real 7)':>12s} "
          f"{'cos(pad 25)':>12s} {'max|pad|':>10s}")
    print(f"  {'-'*42} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    # Combined pass: compute both dim-slice (Part 1) AND position-slice (Part 2)
    all_results = {}

    for d in args.velocity_dirs:
        d = Path(d).expanduser()
        try:
            va_payload = torch.load(d / "velocities_a.pt", weights_only=False)
            vb_payload = torch.load(d / "velocities_b.pt", weights_only=False)
        except Exception as e:
            print(f"  {d.name:<42s} FAIL to load: {type(e).__name__}: {e}")
            continue
        va = va_payload["velocities"]  # list of (1, H, D) tensors
        vb = vb_payload["velocities"]
        if not va or not vb or len(va) != len(vb):
            print(f"  {d.name:<42s} no/mismatched velocities "
                  f"(len_a={len(va)}, len_b={len(vb)})")
            continue

        # Per-step accumulators
        cos_all = []
        cos_real = []
        cos_pad = []
        max_pad_abs = 0.0
        prefix_cos = {n: [] for n in slices}            # cos over pos[0:n] × real R
        suffix_cos = {n: [] for n in slices if n < H}   # cos over pos[n:H] × real R

        captured_horizon = None
        for a, b in zip(va, vb):
            assert a.shape[-1] == D and b.shape[-1] == D, \
                f"expected action_dim={D}, got a={a.shape[-1]}, b={b.shape[-1]}"
            if captured_horizon is None:
                captured_horizon = a.shape[1]
                if captured_horizon != H:
                    print(f"  WARNING: {d.name} horizon={captured_horizon}, "
                          f"expected {H}. Slices will use H from data.")

            # PART 1 — dim slices (over all positions)
            cos_all.append(cos_sim(a, b))
            cos_real.append(cos_sim(a[..., :R], b[..., :R]))
            a_pad = a[..., R:]
            b_pad = b[..., R:]
            cos_pad.append(cos_sim(a_pad, b_pad))
            max_pad_abs = max(max_pad_abs,
                              a_pad.abs().max().item(),
                              b_pad.abs().max().item())

            # PART 2 — position slices (within real R dims)
            h_actual = a.shape[1]  # in case it doesn't match args.action_horizon
            for n in slices:
                n_eff = min(n, h_actual)
                a_pre = a[..., :n_eff, :R]
                b_pre = b[..., :n_eff, :R]
                prefix_cos[n].append(cos_sim(a_pre, b_pre))
                if n < h_actual:
                    a_suf = a[..., n_eff:h_actual, :R]
                    b_suf = b[..., n_eff:h_actual, :R]
                    suffix_cos[n].append(cos_sim(a_suf, b_suf))

        # Print Part 1 row
        print(f"  {d.name:<42s} "
              f"{np.mean(cos_all):>10.4f} "
              f"{np.mean(cos_real):>12.4f} "
              f"{np.mean(cos_pad):>12.4f} "
              f"{max_pad_abs:>10.4e}")

        all_results[d.name] = {
            "horizon": captured_horizon,
            "n_denoise_steps": len(va),
            "cos_all_32_dims": float(np.mean(cos_all)),
            "cos_real_7_dims": float(np.mean(cos_real)),
            "cos_pad_25_dims": float(np.mean(cos_pad)),
            "max_pad_abs": float(max_pad_abs),
            "prefix_cos_real7": {
                f"pos[0:{n}]": float(np.mean(prefix_cos[n])) for n in slices
            },
            "suffix_cos_real7": {
                f"pos[{n}:{H}]": float(np.mean(suffix_cos[n]))
                for n in slices if n < H
            },
        }

    if not all_results:
        print("\n  No probes loaded successfully. Exiting.")
        return

    # ========================================================================
    # PART 2 — cos by action POSITION (prefix = executed-ish, suffix = discarded)
    # ========================================================================
    print(f"\n{'='*100}")
    print(f"  PART 2: cos restricted by ACTION POSITION (within real {R} dims)")
    print(f"  pi0_libero EXECUTES positions 0..4 then replans; "
          f"positions 5..{H-1} are COMPUTED-then-DISCARDED.")
    print(f"  Question: is the bypass headline (cos≈0.999) inflated by the discarded tail?")
    print(f"{'='*100}")

    # Prefix table (cumulative first-N positions)
    header_cells = [f"pos[0:{n}]" for n in slices]
    print(f"\n  --- PREFIX: cos over FIRST-N action positions (within real {R} dims) ---")
    print(f"  {'probe':<42s} " + " ".join(f"{h:>10s}" for h in header_cells))
    print(f"  {'-'*42} " + " ".join("-"*10 for _ in slices))
    for probe_name, res in all_results.items():
        cells = [f"{res['prefix_cos_real7'][f'pos[0:{n}]']:>10.4f}" for n in slices]
        print(f"  {probe_name:<42s} " + " ".join(cells))

    # Suffix table (complement: positions N..H)
    suffix_slices = [n for n in slices if n < H]
    if suffix_slices:
        print(f"\n  --- SUFFIX: cos over DISCARDED-TAIL positions [N:{H}] (within real {R} dims) ---")
        header_cells = [f"pos[{n}:{H}]" for n in suffix_slices]
        print(f"  {'probe':<42s} " + " ".join(f"{h:>10s}" for h in header_cells))
        print(f"  {'-'*42} " + " ".join("-"*10 for _ in suffix_slices))
        for probe_name, res in all_results.items():
            cells = [f"{res['suffix_cos_real7'][f'pos[{n}:{H}]']:>10.4f}"
                     for n in suffix_slices]
            print(f"  {probe_name:<42s} " + " ".join(cells))

    # ========================================================================
    # DELTA TABLE: how much does cos drop going from full chunk to executed-only?
    # ========================================================================
    print(f"\n  --- DELTA: cos(pos[0:5]) − cos(pos[0:{H}]) — "
          f"negative = executed actions DIVERGE more than the full chunk ---")
    print(f"  {'probe':<42s} {'cos[0:5]':>10s} {'cos[0:'+str(H)+']':>10s} {'delta':>10s} "
          f"{'inflation?':>14s}")
    print(f"  {'-'*42} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")
    for probe_name, res in all_results.items():
        c5 = res['prefix_cos_real7'].get('pos[0:5]', float('nan'))
        cH = res['prefix_cos_real7'].get(f'pos[0:{H}]', float('nan'))
        delta = c5 - cH
        # If cos[0:5] is meaningfully lower than cos[0:H], the headline is inflated
        if delta < -0.05:
            label = "STRONG"
        elif delta < -0.01:
            label = "moderate"
        elif delta < -0.001:
            label = "slight"
        else:
            label = "negligible"
        print(f"  {probe_name:<42s} {c5:>10.4f} {cH:>10.4f} {delta:>+10.4f} {label:>14s}")

    # ========================================================================
    # Interpretation guide
    # ========================================================================
    print(f"\n{'='*100}")
    print(f"  INTERPRETATION GUIDE")
    print(f"{'='*100}")
    print(f"  pi0_libero rolls out positions 0..4 (the first 5 actions of each chunk)")
    print(f"  before getting a new observation. Positions 5..{H-1} are computed but discarded.")
    print(f"")
    print(f"  The headline 'cos(v_A,v_B) ≈ 0.999' from velocity_divergence averages")
    print(f"  over ALL {H} positions. Three possible outcomes:")
    print(f"")
    print(f"  (A) cos[0:5] is close to cos[0:{H}] (delta ≥ -0.001, 'negligible')")
    print(f"      → bypass extends to executed actions. Headline holds.")
    print(f"      → steering at positions 0..4 only is unlikely to help.")
    print(f"")
    print(f"  (B) cos[0:5] is meaningfully lower (delta < -0.01, 'moderate' or 'STRONG')")
    print(f"      → executed actions DO differ between prompts; headline is partially")
    print(f"        inflated by the abandoned tail.")
    print(f"      → Reframe: bypass affects N% of the chunk but executed actions partially")
    print(f"        follow the prompt. Selective steering at pos 0..4 has a fighting chance.")
    print(f"")
    print(f"  (C) cos[5:{H}] ≈ 1.0000 across all probes (suffix table)")
    print(f"      → the discarded tail collapses to a default trajectory regardless of prompt")
    print(f"        — common in flow-matching with one trained trajectory per scene.")
    print(f"      → strengthens the 'one trajectory mode per scene' interpretation.")
    print(f"")
    print(f"  cos[0:5] is the SINGLE MOST IMPORTANT NUMBER for the behavioral claim.")

    if args.out_json:
        out = Path(args.out_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2))
        print(f"\n  [saved] full per-probe results → {out}")


if __name__ == "__main__":
    main()
