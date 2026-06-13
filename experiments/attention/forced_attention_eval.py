#!/usr/bin/env python3
"""forced_attention_eval.py — does FORCING attention onto the instruction change the action?

The differentiator from IGAR (arXiv 2603.06001) and the load-bearing test for "the sink is a
SYMPTOM, the weights are the cause": if we force the action expert to attend to the instruction
(IGAR-style attention redistribution) and the velocity field barely moves, then attention is NOT
the bottleneck — the value/MLP weights never learned to USE the instruction. That explains why
inference-time attention fixes (IGAR) plateau (their pi0.5 LGS 1.4) and motivates a TRAINING fix.

Conditions (on baseline pi0_libero), reusing attention_boost.AttentionBoost (pre-softmax mask):
  baseline           : no intervention
  force_instruction  : +boost on the real instruction content tokens, ALL expert layers
  igar               : -inf on sinks (BOS 768, image-corner 303) + boost instruction, first 16 layers
                       (a STRONGER version of IGAR's 0.6 scale-down → full suppression; an upper bound
                        on what attention-redistribution can achieve)

For each condition we measure, on the tomato_only vs alphabet_only prompts:
  - L17 instruction attention mass  (did the intervention actually move attention onto the instruction?)
  - per-step velocity cos (real-7) between the two prompts  (did the ACTION become prompt-conditional?)
PREDICTION: force_instruction/igar RAISE instruction attention a lot, but velocity cos stays ~0.999
  -> attention isn't the bottleneck; the weights are.  (Confirm-or-falsify in one cheap inference run.)

USAGE (remote, openpi venv):
  ~/flare/openpi/.venv/bin/python ~/flare/forced_attention_eval.py \
     --checkpoint ~/flare/checkpoints/pi0_libero_pt --out-dir ~/flare/results/forced_attention
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

FLARE = Path.home() / "flare"
sys.path.insert(0, str(FLARE))

from audit_lora_attention import (  # noqa: E402
    load_model, VelocityCapture, expert_layer_profiles, region_masses,
    tokenize_positions, setup_env, LANG_START, LANG_END, KEY_BOS, KEY_IMG_CORNER, N_DENOISE_REAL_DIMS,
)

PROMPTS = [("T0_tomato_only", "pick up the tomato sauce and put it in the basket"),
           ("T0_alphabet_only", "pick up the alphabet soup and put it in the basket")]


def real_instruction_positions(prompt):
    """Non-pad instruction content token positions (769..815), excluding BOS, for boosting."""
    labels = tokenize_positions(prompt)
    out = []
    for pos in range(LANG_START, LANG_END):
        t = labels.get(pos, "").strip()
        if t and t not in ("<pad>", "<eos>", "<bos>") and not t.startswith("\x00"):
            out.append(pos)
    return out


def run(policy, model, obs_for, prompt, condition, env_pack):
    from mechanism_probe_attn import AttentionCapture, set_all_seeds
    from attention_boost import AttentionBoostHook
    obs = env_pack["format_obs"](env_pack["obs"], prompt)
    instr = real_instruction_positions(prompt)
    if condition == "baseline":
        layers, sinks, boosts, bval = [], [], [], 0.0
    elif condition == "force_instruction":
        layers, sinks, boosts, bval = list(range(18)), [], instr, 10.0
    elif condition == "igar":
        layers, sinks, boosts, bval = list(range(16)), [KEY_BOS, KEY_IMG_CORNER], instr, 10.0
    else:
        raise ValueError(condition)

    set_all_seeds(42)
    booster = AttentionBoostHook(model, layers, sink_positions=sinks, boost_positions=boosts,
                             boost_value=bval, verbose=False) if layers else None
    cm = booster if booster is not None else _NullCtx()
    with cm, AttentionCapture(model, verbose=False) as cap, VelocityCapture(model) as vels:
        _ = policy.infer(obs)
    prof = expert_layer_profiles(cap)
    instr_mass = region_masses(prof[17])["instruction"] if 17 in prof else float("nan")
    bos_mass = region_masses(prof[17])["bos"] if 17 in prof else float("nan")
    vsteps = [v.mean(0).float().numpy() for v in vels] if vels else None
    return {"instr_mass": instr_mass, "bos_mass": bos_mass, "n_instr_tokens": len(instr),
            "velocity_steps": vsteps}


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def cos_real7(va_steps, vb_steps):
    R = N_DENOISE_REAL_DIMS
    cs = []
    for A, B in zip(va_steps, vb_steps):
        a, b = A[..., :R].ravel(), B[..., :R].ravel()
        cs.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return float(np.mean(cs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(FLARE / "checkpoints/pi0_libero_pt"))
    p.add_argument("--config-name", default="pi0_libero")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    args = p.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    policy, model = load_model(args.config_name, args.checkpoint)
    if model is None:
        raise RuntimeError("need a PyTorch checkpoint (got JAX) for attention intervention")
    env_pack = setup_env(args.task_suite, args.task_id, env_seed=7000, init_idx=0)

    conditions = ["baseline", "force_instruction", "igar"]
    res = {c: {} for c in conditions}
    for c in conditions:
        for plabel, prompt in PROMPTS:
            res[c][plabel] = run(policy, model, env_pack["obs"], prompt, c, env_pack)
            r = res[c][plabel]
            print(f"[{c}/{plabel}] L17 instr_mass={r['instr_mass']:.4f} bos={r['bos_mass']:.4f} "
                  f"(boosted {r['n_instr_tokens']} instr tokens)", flush=True)

    # Summary: did attention move? did the action move?
    summary = {}
    for c in conditions:
        ra, rb = res[c]["T0_tomato_only"], res[c]["T0_alphabet_only"]
        instr_mean = float(np.nanmean([ra["instr_mass"], rb["instr_mass"]]))
        vcos = (cos_real7(ra["velocity_steps"], rb["velocity_steps"])
                if ra["velocity_steps"] and rb["velocity_steps"] else float("nan"))
        summary[c] = {"L17_instruction_mass": instr_mean, "velocity_cos_tomato_vs_alphabet_real7": vcos}
    # how much did the action change vs baseline (per prompt)?
    for c in ("force_instruction", "igar"):
        d = []
        for plabel in [p[0] for p in PROMPTS]:
            vb = res["baseline"][plabel]["velocity_steps"]; vc = res[c][plabel]["velocity_steps"]
            if vb and vc:
                d.append(cos_real7(vb, vc))   # cos(baseline action, intervened action) per prompt
        summary[c]["velocity_cos_vs_baseline_real7"] = float(np.nanmean(d)) if d else float("nan")

    (out_dir / "forced_attention.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "summary": summary,
         "raw": {c: {pl: {k: v for k, v in r.items() if k != "velocity_steps"}
                     for pl, r in res[c].items()} for c in conditions}}, indent=2, default=str))

    # Figure: instruction attention (did intervention work) vs velocity cos (did action change)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(conditions))
    ax1.bar(x, [summary[c]["L17_instruction_mass"] * 100 for c in conditions], color="#2ca02c")
    ax1.set_xticks(x); ax1.set_xticklabels(conditions, rotation=10, fontsize=8)
    ax1.set_ylabel("L17 instruction attention mass (%)")
    ax1.set_title("Did the intervention MOVE attention onto the instruction?\n(force/igar should be >> baseline)")
    ax2.bar(x, [summary[c]["velocity_cos_tomato_vs_alphabet_real7"] for c in conditions], color="#1f77b4")
    ax2.set_ylim(0.95, 1.001); ax2.set_xticks(x); ax2.set_xticklabels(conditions, rotation=10, fontsize=8)
    ax2.set_ylabel("velocity cos (tomato vs alphabet, real-7)")
    ax2.set_title("Did the ACTION change?\n(stays ~0.999 ⇒ attention isn't the bottleneck — the WEIGHTS are)")
    fig.tight_layout(); fig.savefig(out_dir / "fig_forced_attention.png", dpi=150); plt.close(fig)

    print(f"\n[done] {out_dir}/forced_attention.json")
    print("  HEADLINE: if instruction attention rises sharply but velocity cos stays ~0.999, "
          "forcing attention does NOT make the model read language — the bypass is weight-level, "
          "which is why IGAR-style inference fixes plateau. (Differentiates us from IGAR.)")


if __name__ == "__main__":
    main()
