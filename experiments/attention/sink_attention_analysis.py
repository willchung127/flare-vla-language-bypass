"""sink_attention_analysis.py — pull per-key L17 attention mass for several prompts.

Saves a small JSON (per-key arrays are 867 floats) so the figure scripts don't
reload the 489 MB attention tensors.
"""
from __future__ import annotations
from pathlib import Path
import warnings, json
warnings.filterwarnings("ignore")
import torch
import numpy as np

FL = Path.home() / "flare/results"
KEY = "paligemma_with_expert.gemma_expert.model.layers.17.self_attn"

# (probe_dir, side, short_label)
SOURCES = [
    ("mechanism_probe_T0_v1",          "a", "soup + sauce"),
    ("mechanism_probe_T0_v1",          "b", "sauce + soup (reorder)"),
    ("mechanism_probe_T0_P3_unrelated","a", "tomato sauce only"),
    ("mechanism_probe_T0_P3_unrelated","b", "stack the mugs"),
    ("mechanism_probe_T1_v1",          "a", "cream cheese + butter"),
]

IMAGE_END = 768
BOS = 768
LANG_START, LANG_END = 769, 816
ACTION_START = 816


def per_key_mass(cap_path: Path):
    d = torch.load(cap_path, map_location="cpu", weights_only=False)
    prompt = d.get("prompt", "?")
    calls = [c for c in d["captures"][KEY] if c is not None]
    attn = calls[-1].float()[0]            # (heads, q, k)
    pk = attn.mean(dim=(0, 1)).numpy()      # (867,)
    pk = pk / pk.sum()
    return prompt, pk


def main():
    out = {"image_end": IMAGE_END, "bos": BOS, "lang": [LANG_START, LANG_END],
           "action_start": ACTION_START, "key": KEY, "entries": []}
    for d, side, label in SOURCES:
        cap = FL / d / f"captured_{side}.pt"
        if not cap.exists():
            print(f"  MISSING {cap}")
            continue
        prompt, pk = per_key_mass(cap)
        entry = {
            "label": label,
            "prompt": prompt,
            "bos": float(pk[BOS]),
            "instruction": float(pk[LANG_START:LANG_END].sum()),
            "image": float(pk[:IMAGE_END].sum()),
            "image_sink_303": float(pk[303]),
            "action_self": float(pk[ACTION_START:].sum()),
            "per_key": [round(float(x), 5) for x in pk],
        }
        out["entries"].append(entry)
        print(f"{label:28s} bos={entry['bos']:.3f} instr={entry['instruction']:.3f} "
              f"img303={entry['image_sink_303']:.3f} img={entry['image']:.3f} "
              f"act={entry['action_self']:.3f}")
    (FL / "sink_per_key_multi.json").write_text(json.dumps(out))
    print(f"\n[wrote] {FL / 'sink_per_key_multi.json'}")


if __name__ == "__main__":
    main()
