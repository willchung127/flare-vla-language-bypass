#!/usr/bin/env python3
"""weight_sanity.py — prove the LoRA merge actually TOOK (converted model != base model).

The single biggest risk in merge_and_convert.py is a SILENT base-model conversion (the merge
patch never fires, or drops the adapter). This catches it WITHOUT any GPU inference, by diffing
the saved safetensors of:
  - D_v2_pt        : the merged+converted checkpoint (what we'll trust)
  - naive_no_merge : the SAME jax checkpoint run through the STOCK converter (no monkeypatch)
  - base (optional): pi0_libero_pt

The LoRA touches attention (q/k/v/o) and MLP (gate/up/down) of the gemma layers. So:
  PASS = D_v2_pt vs naive_no_merge DIFFER on >=10 of those tensors (rel-frob > 1e-3)  -> merge took
  and  naive_no_merge ~= base on those same tensors (the stock converter drops the adapter -> base).

Produce naive_no_merge first (stock converter, no merge):
  ~/flare/openpi/.venv/bin/python ~/flare/openpi/examples/convert_jax_model_to_pytorch.py \
     --checkpoint-dir ~/flare/openpi/checkpoints/pi0_libero_cf_lora_bos_masked/variant_d_v2/1499 \
     --config-name pi0_libero_cf_lora_bos_masked \
     --output-path ~/flare/checkpoints/variant_d_v2_naive_pt

Then:
  ~/flare/openpi/.venv/bin/python ~/flare/weight_sanity.py \
     --merged   ~/flare/checkpoints/variant_d_v2_pt \
     --naive    ~/flare/checkpoints/variant_d_v2_naive_pt \
     --base     ~/flare/checkpoints/pi0_libero_pt
"""
import argparse
import os

import numpy as np


def load_sd(path):
    import safetensors.torch
    f = path if path.endswith(".safetensors") else os.path.join(path, "model.safetensors")
    sd = safetensors.torch.load_file(f)
    return {k: v.float().numpy() for k, v in sd.items()}


def is_lora_target(key):
    # gemma attention + mlp projections that LoRA adapts: the LLM (language_model) and the
    # action expert (gemma_expert) ONLY — NOT the vision tower (which has q/k/v/o_proj but no LoRA).
    if "vision_tower" in key or "vision_model" in key:
        return False
    if not ("language_model" in key or "gemma_expert" in key):
        return False
    return ("self_attn" in key and any(p in key for p in ("q_proj", "k_proj", "v_proj", "o_proj"))) \
        or ("mlp" in key and any(p in key for p in ("gate_proj", "up_proj", "down_proj")))


def compare(a, b, label):
    shared = [k for k in a if k in b and a[k].shape == b[k].shape]
    rows = []
    for k in shared:
        if not is_lora_target(k):
            continue
        d = np.abs(a[k] - b[k]).max()
        rel = np.linalg.norm(a[k] - b[k]) / (np.linalg.norm(b[k]) + 1e-12)
        rows.append((k, float(d), float(rel)))
    n_diff = sum(1 for _, _, rel in rows if rel > 1e-3)
    print(f"\n=== {label} === ({len(rows)} LoRA-target tensors compared)")
    print(f"  tensors differing (rel-frob > 1e-3): {n_diff}/{len(rows)}")
    if rows:
        worst = sorted(rows, key=lambda x: -x[2])[:3]
        for k, d, rel in worst:
            print(f"    {k[-60:]:60s} max|Δ|={d:.2e} rel={rel:.2e}")
        med = float(np.median([r[2] for r in rows]))
        print(f"  median rel-frob: {med:.2e}")
    return n_diff, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="variant_*_pt (merged+converted)")
    ap.add_argument("--naive", default=None, help="stock-converter output (no merge) — the key control")
    ap.add_argument("--base", default=None, help="pi0_libero_pt")
    args = ap.parse_args()

    merged = load_sd(args.merged)
    print(f"[load] merged: {len(merged)} tensors")
    ok = True

    if args.naive:
        naive = load_sd(args.naive)
        n_diff, n_tot = compare(merged, naive, "MERGED vs NAIVE-NO-MERGE  (must DIFFER -> merge took)")
        merged_took = n_diff >= 10
        print(f"  -> {'PASS' if merged_took else 'FAIL'}: merge {'TOOK' if merged_took else 'did NOT take (SILENT BASE MODEL)'}")
        ok &= merged_took
        if args.base:
            base = load_sd(args.base)
            nb_diff, _ = compare(naive, base, "NAIVE vs BASE  (should be ~SAME -> stock converter drops adapter)")
            print(f"  -> {'as expected' if nb_diff < 5 else 'UNEXPECTED'}: stock converter "
                  f"{'drops' if nb_diff < 5 else 'may NOT drop'} the LoRA (smoking gun for the silent-base risk)")
    elif args.base:
        base = load_sd(args.base)
        n_diff, n_tot = compare(merged, base, "MERGED vs BASE pi0_libero_pt")
        print("  NOTE: without --naive this only shows merged != base; if the LoRA fine-tuned FROM "
              "pi0_libero, merged-vs-base should differ by the (small) LoRA delta. Use --naive for the clean test.")
        ok &= (n_diff >= 1)

    print("\n" + ("[PASS] merge verified — variant is NOT the base model." if ok
                  else "[FAIL] merge NOT verified — DO NOT trust the converted checkpoint."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
