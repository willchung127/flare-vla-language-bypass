#!/usr/bin/env python3
"""score_grounding.py — language-grounded success, scored PER PROMPT (not the original BDDL goal).

Fixes the metric confound: the standard behavioral eval scores `success=bool(done)`
against the ORIGINAL two-object task goal for EVERY prompt, so "pick up the tomato sauce" reads 0%
even when the policy correctly places the tomato (the goal still wants BOTH). Here, success is defined
RELATIVE TO THE PROMPT: did the NAMED object(s) get placed in the basket?

Per prompt we report:
  - grounded_placed : named object(s) moved AND ended inside the basket region   <- the real metric
  - named_moved     : named object(s) displaced > min_disp (manipulated at all)
  - temporal_first  : the FIRST object to cross threshold IN TIME == the named one (from 25-step waypoints;
                      a temporal fix for the broken displacement-based "first_moved")
  - original_success: bool(done) from the env (the confounded metric, for reference/contrast)
with Wilson 95% CIs. Reuses remote_multimode_matrix.rollout (returns init/final object positions +
waypoints) and the proven loaders from eval_lora_behavioral.

Runs on a PyTorch checkpoint (baseline pi0_libero_pt or a converted variant_*_pt) under config pi0_libero.

USAGE (remote, openpi venv):
  ~/flare/openpi/.venv/bin/python ~/flare/score_grounding.py \
     --checkpoint ~/flare/checkpoints/variant_d_v2_pt --config-name pi0_libero \
     --variant-label Dv2_pt --n-trials 50 --out-dir ~/flare/results/grounding/Dv2_pt
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "flare"))

# Which physical object(s) each prompt asks to put in the basket (keyword stems matched to obj keys).
PROMPT_TARGETS = {
    "T0_canonical":     ["alphabet", "tomato"],     # both
    "T0_reordered":     ["tomato", "alphabet"],     # both
    "T0_tomato_only":   ["tomato"],
    "T0_alphabet_only": ["alphabet"],
    "T0_milk_test":     ["milk"],
    "T0_irrelevant":    [],                          # "stack the mugs" — no basket target here
}
BASKET_STEMS = ("basket", "tray", "bin", "container")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z/d) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, max(0.0, c-h), min(1.0, c+h)


def find_key(pos: dict, stems) -> str | None:
    for k in pos:
        kl = k.lower()
        if any(s in kl for s in (stems if isinstance(stems, (list, tuple)) else [stems])):
            return k
    return None


def placed_in_basket(init_pos, final_pos, obj_stem, basket_key,
                     min_disp=0.08, radius=0.13):
    """Named object was manipulated AND ended within `radius` (xy) of the basket."""
    ok = find_key(init_pos, [obj_stem])
    if ok is None or basket_key is None or ok not in final_pos or basket_key not in final_pos:
        return None  # object/basket not found -> undefined
    o0 = np.asarray(init_pos[ok][:3]); o1 = np.asarray(final_pos[ok][:3])
    b1 = np.asarray(final_pos[basket_key][:3])
    disp = float(np.linalg.norm(o1 - o0))
    horiz = float(np.linalg.norm(o1[:2] - b1[:2]))
    return bool(disp > min_disp and horiz < radius)


def temporal_first_object(init_pos, waypoints, thresh=0.05):
    """First object (by name stem) to cross `thresh` displacement IN TIME (fixes max-displacement first_moved)."""
    for wp in waypoints:
        op = wp.get("object_positions", {})
        crossed = []
        for k, p0 in init_pos.items():
            if k in op:
                d = float(np.linalg.norm(np.asarray(op[k][:3]) - np.asarray(p0[:3])))
                if d > thresh:
                    crossed.append((d, k))
        if crossed:
            crossed.sort(key=lambda x: -x[0])
            return crossed[0][1]   # most-displaced AT THE FIRST TIME anything crossed
    return "<none>"


def grounding_trial(policy, model, env_pack, init_state, env_seed, prompt, replan_steps):
    mmm = env_pack["mmm"]
    cond_ode = {"name": "ode_single", "eta": 0.0, "K": 1, "guidance": False, "alpha": 0.0}
    if model is not None:
        for a, v in dict(flare_eta=0.0, flare_verifier_fn=None, flare_alpha=0.0, flare_obs_state=None,
                         flare_eta_high=None, flare_eta_low=None, flare_noise_bias_direction=None,
                         flare_noise_bias_strength=0.0).items():
            if hasattr(model, a):
                setattr(model, a, v)
    out = mmm.rollout(policy=policy, model=model, task=env_pack["task"], task_id=env_pack["task_id"],
                      bddl=env_pack["bddl"], init_state=init_state, env_seed=env_seed, base_seed=42,
                      cond=cond_ode, scorer_fn=None, guidance_callable=None, obs_state_for_guidance=None,
                      replan_steps=replan_steps, video_mode="off", language_override=prompt)
    done, term, n_steps, init_p, final_p, waypoints, moved, frames = out
    return dict(done=bool(done), n_steps=int(n_steps), init_pos=init_p, final_pos=final_p, waypoints=waypoints)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config-name", default="pi0_libero")
    p.add_argument("--variant-label", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--env-seed", type=int, default=7000)
    p.add_argument("--prompts", default=None, help="comma-separated subset of prompt labels")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    from eval_lora_behavioral import load_lora_policy, setup_env, find_model_attr, EVAL_PROMPTS

    policy = load_lora_policy(args.config_name, Path(args.checkpoint).expanduser().resolve())
    model = find_model_attr(policy)
    if model is None:
        print("[warn] no PyTorch model found — rollouts require a PT checkpoint (mmm.rollout sets model.flare_*). "
              "Convert JAX checkpoints first.", flush=True)
    env_pack = setup_env(args.task_suite, args.task_id)
    n = min(args.n_trials, env_pack["n_inits"])

    prompts = EVAL_PROMPTS
    if args.prompts:
        want = {s.strip() for s in args.prompts.split(",")}
        prompts = [(l, t) for l, t in EVAL_PROMPTS if l in want]

    summary = {}
    all_trials = {}
    for plabel, prompt in prompts:
        targets = PROMPT_TARGETS.get(plabel, [])
        print(f"\n[grounding] {plabel}: {prompt!r}  targets={targets}", flush=True)
        rows = []
        for i in range(n):
            t0 = time.time()
            r = grounding_trial(policy, model, env_pack, env_pack["init_states"][i],
                                args.env_seed + i, prompt, args.replan_steps)
            basket = find_key(r["final_pos"], BASKET_STEMS)
            placed = [placed_in_basket(r["init_pos"], r["final_pos"], s, basket) for s in targets]
            placed_named = (len(targets) > 0 and all(x is True for x in placed))
            moved = [placed_in_basket(r["init_pos"], r["final_pos"], s, basket, radius=1e9) for s in targets]
            named_moved = (len(targets) > 0 and all(x is True for x in moved))
            tfirst = temporal_first_object(r["init_pos"], r["waypoints"])
            tfirst_named = any(find_key({tfirst: 0}, [s]) for s in targets) if targets else False
            rows.append(dict(grounded_placed=placed_named, named_moved=named_moved,
                             temporal_first=tfirst, temporal_first_named=tfirst_named,
                             original_success=r["done"], n_steps=r["n_steps"]))
            print(f"    trial {i+1}/{n}: placed={placed_named} moved={named_moved} "
                  f"tfirst={tfirst} orig_succ={r['done']} ({time.time()-t0:.1f}s)", flush=True)
        all_trials[plabel] = rows
        def rate(key):
            k = sum(1 for x in rows if x[key]); pp, lo, hi = wilson(k, len(rows))
            return {"k": k, "n": len(rows), "rate": pp, "ci95": [lo, hi]}
        summary[plabel] = {"targets": targets,
                           "grounded_placed": rate("grounded_placed"),
                           "named_moved": rate("named_moved"),
                           "temporal_first_named": rate("temporal_first_named"),
                           "original_success": rate("original_success")}
        s = summary[plabel]
        print(f"  => grounded_placed={s['grounded_placed']['rate']:.2f} "
              f"vs original_success={s['original_success']['rate']:.2f}  (n={len(rows)})", flush=True)

    (out_dir / "grounding_eval.json").write_text(json.dumps(
        {"variant": args.variant_label, "checkpoint": str(args.checkpoint),
         "config": args.config_name, "n_trials": n, "summary": summary, "all_trials": all_trials},
        indent=2, default=str))
    print(f"\n[done] {out_dir}/grounding_eval.json")
    print("  KEY: compare grounded_placed (per-prompt language goal) vs original_success (orig BDDL goal). "
          "On single-object prompts the gap is the metric artifact your insight predicted.")


if __name__ == "__main__":
    main()
