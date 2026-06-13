#!/usr/bin/env python3
"""merge_and_convert.py — merge LoRA into base weights, then JAX→PyTorch convert.

The openpi converter (examples/convert_jax_model_to_pytorch.py) has no LoRA handling,
so converting a LoRA checkpoint directly would silently drop the adapter and yield the
BASE model. This wrapper fixes that by monkeypatching the converter's loader
(`slice_initial_orbax_checkpoint`) to merge LoRA into the base weights BEFORE the
existing slicing runs, then calls the converter unchanged.

Merge math (verified against openpi/models/lora.py):
  Einsum/FeedForward effective weight  W_eff = W + (lora_a @ lora_b) * (alpha/rank)
  For these configs alpha==rank (attn 16/16, ffn 16/16, expert 32/32) → scale = 1.0.
  Both experts live in the same flat tree: `..._einsum` = PaliGemma-2B, `..._einsum_1` = action expert.

Output is a plain PI0Pytorch checkpoint → loadable under config `pi0_libero` →
hookable (AttentionCapture) AND rollout-compatible (find_model_attr finds a PI0Pytorch).

Usage (remote, openpi venv):
  ~/flare/openpi/.venv/bin/python ~/flare/merge_and_convert.py \
    --checkpoint-dir ~/flare/openpi/checkpoints/pi0_libero_cf_lora_bos_masked/variant_d_v2/1499 \
    --config-name   pi0_libero_cf_lora_bos_masked \
    --output-path   ~/flare/checkpoints/variant_d_v2_pt
"""
import argparse
import importlib.util
import os

import numpy as np

CONV_PATH = os.path.expanduser("~/flare/openpi/examples/convert_jax_model_to_pytorch.py")


def _delta_matmul(a, b):
    """ΔW = a @ b (batched over leading dims). Correct when the LoRA'd axis is BATCHED
    in the output (q/kv have head in the output; mlp has no head)."""
    return np.matmul(a, b)


def _delta_attn_vec(a, b):
    """ΔW for the o_proj (attn_vec_einsum). The output projection CONTRACTS the head
    dimension (output = model-dim, no head index), so in openpi's lora.py the second
    einsum (eqn_b) sums w_b over its head axis independently of w_a's head axis. The
    effective delta is therefore NOT a per-head matmul:
        a = (layer, n_head, head_dim, rank)   b = (layer, m_head, rank, out)
        ΔW[p,n,h,d] = sum_{r,m} a[p,n,h,r] * b[p,m,r,d]
    Verified bit-exact (float64, err ~1e-12) against jnp.einsum(eqn_b, einsum(eqn_a,x,w_a), w_b).
    A plain matmul here is WRONG (err ~2e3) and would corrupt every output-projection weight."""
    return np.einsum("pnhr,pmrd->pnhd", a, b)


def merge_lora_flat(pg: dict) -> None:
    """Merge LoRA into the flat PaliGemma param dict, in place. scale = alpha/rank = 1.0."""
    n_merged = 0

    def merge(base_key, a_key, b_key, delta_fn):
        nonlocal n_merged
        if a_key not in pg or b_key not in pg:
            print(f"  [skip] {base_key}: no lora siblings ({a_key})")
            return
        a = pg.pop(a_key).astype(np.float32)
        b = pg.pop(b_key).astype(np.float32)
        w = pg[base_key]
        # Orientation guards: rank is the shared contraction dim and must be << the weight dims,
        # so a transposed/orientation bug can't slip through a pure product-shape check.
        r = a.shape[-1]
        assert b.shape[-2] == r, f"{base_key}: rank mismatch a[-1]={r} vs b[-2]={b.shape[-2]}"
        assert r < min(w.shape[-2], w.shape[-1]), f"{base_key}: rank {r} not << dims {tuple(w.shape[-2:])}"
        delta = delta_fn(a, b)
        assert delta.shape == tuple(w.shape), f"{base_key}: delta {delta.shape} != w {tuple(w.shape)}"
        assert np.isfinite(delta).all(), f"{base_key}: non-finite delta"
        pg[base_key] = (w.astype(np.float32) + delta).astype(w.dtype)
        n_merged += 1
        print(f"  [merge] {base_key}: w{tuple(w.shape)} += {delta_fn.__name__} a{tuple(a.shape)} b{tuple(b.shape)}")

    # q/kv: head is BATCHED in the output -> per-head matmul is exact.
    for X in ["q_einsum", "q_einsum_1", "kv_einsum", "kv_einsum_1"]:
        merge(f"llm/layers/attn/{X}/w", f"llm/layers/attn/{X}/lora_a",
              f"llm/layers/attn/{X}/lora_b", _delta_matmul)
    # attn_vec (o_proj): head is CONTRACTED in the output -> w_b head axis summed (NOT matmul).
    for X in ["attn_vec_einsum", "attn_vec_einsum_1"]:
        merge(f"llm/layers/attn/{X}/w", f"llm/layers/attn/{X}/lora_a",
              f"llm/layers/attn/{X}/lora_b", _delta_attn_vec)
    # MLP: no heads -> matmul. base at .../<M>/<name>, lora FLAT sibling .../<M>/<name>_lora_a.
    for M in ["mlp", "mlp_1"]:
        for name in ["gating_einsum", "linear"]:
            merge(f"llm/layers/{M}/{name}", f"llm/layers/{M}/{name}_lora_a",
                  f"llm/layers/{M}/{name}_lora_b", _delta_matmul)

    leftover = [k for k in pg if "lora" in k.lower()]
    if n_merged == 0:
        raise RuntimeError(
            "merge_lora_flat merged 0 weight groups — flat keys did not match. "
            "Converting now would silently yield the BASE model. Abort and check key names.")
    if n_merged < 8:
        raise RuntimeError(
            f"merge_lora_flat merged only {n_merged} groups (expected ~10) — partial merge would "
            "ship a half-adapted model. Abort and check key names.")
    if leftover:
        raise RuntimeError(f"{len(leftover)} lora keys NOT merged (would corrupt conversion): {leftover[:8]}")
    print(f"  [ok] merged {n_merged} weight groups; all lora keys removed.")
    return n_merged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--precision", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    # Import the converter module by path (it lives in examples/, not on sys.path).
    spec = importlib.util.spec_from_file_location("openpi_convert", CONV_PATH)
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)

    if not hasattr(conv, "slice_initial_orbax_checkpoint"):
        raise RuntimeError("converter has no slice_initial_orbax_checkpoint — inspect it; "
                           "the merge-injection point changed.")

    # Monkeypatch the loader: merge LoRA right after the flat param dict is restored.
    # _state tracks whether the patch ACTUALLY fired — if convert_pi0_checkpoint loads params
    # some other way, the merge never runs and the converter would silently emit the BASE model.
    _orig_loader = conv.slice_initial_orbax_checkpoint
    _state = {"fired": False, "n_merged": 0}

    def _patched_loader(checkpoint_dir, restore_precision=None):
        d = _orig_loader(checkpoint_dir, restore_precision)
        print("[merge] merging LoRA into base weights (scale=1.0) ...")
        _state["n_merged"] = merge_lora_flat(d["paligemma_params"])
        _state["fired"] = True
        return d

    conv.slice_initial_orbax_checkpoint = _patched_loader

    # Build the model config from the (lora) config name and run the converter unchanged.
    import openpi.training.config as _config
    cfg = _config.get_config(args.config_name)
    model_config = cfg.model

    print(f"[convert] {args.checkpoint_dir}")
    print(f"[convert]   -> {args.output_path}  (precision={args.precision})")
    conv.convert_pi0_checkpoint(args.checkpoint_dir, args.precision,
                                os.path.expanduser(args.output_path), model_config)

    # HARD GATE: prove the merge ran. A base-model conversion would otherwise look like success.
    if not _state["fired"]:
        raise RuntimeError(
            "FATAL: the LoRA-merge loader NEVER FIRED — convert_pi0_checkpoint did not call the "
            "patched slice_initial_orbax_checkpoint, so the output is the UN-MERGED BASE model. "
            "DO NOT USE it. Inspect convert_pi0_checkpoint's load path on the remote and re-point the patch.")
    print(f"[done] merged {_state['n_merged']} groups + converted -> {args.output_path}. "
          "Load under config 'pi0_libero'. Verify with weight_sanity.py before trusting.")


if __name__ == "__main__":
    main()
