"""Exp C: Behavioral rollouts under instruction manipulations (pi0-LIBERO).

Conditions (per task):
  orig      - normal eval (baseline / sanity vs published numbers)
  empty     - "" prompt. High success on Spatial/Object = language bypass.
  scramble  - shuffled words. Tests string- vs semantics-keying.
  para      - rule-based paraphrase. Drop here + no drop under empty = task-ID lookup.
  swap      - env of task i, instruction of task j (j = i+1, i+5 mod n).
              Success is measured against ENV TASK i's goal:
              high success while told j = memorized-behavior / instruction ignored
              (most meaningful on libero_goal where scenes are shared).

Episode loop mirrors openpi/examples/libero/main.py (replan every 5 steps).
Output: results/exp_c_<suite>.json with per-(task, cond) success rates.
"""

import argparse
import collections
import json
import pathlib

import numpy as np

import libero_utils as U

REPLAN_STEPS = 5


def run_episode(policy, env, suite, task_id, init_idx, prompt, max_steps):
    obs = U.reset_to_init_state(env, suite, task_id, init_idx)
    plan = []
    t = 0
    while t < max_steps:
        if not plan:
            element = U.build_element(obs, prompt)
            actions = np.asarray(policy.infer(element)["actions"])
            plan = list(actions[:REPLAN_STEPS])
        obs, _r, done, _info = env.step(plan.pop(0).tolist())
        if done:
            return True
        t += 1
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--conds", default="orig,empty,para,swap")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--max-tasks", type=int, default=10)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as _config

    ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
    policy = policy_config.create_trained_policy(_config.get_config("pi0_libero"), ckpt)

    suite = U.get_suite(args.suite)
    n_tasks = min(suite.n_tasks, args.max_tasks)
    tasks = [suite.get_task(i) for i in range(n_tasks)]
    instrs = [t.language for t in tasks]
    max_steps = U.MAX_STEPS.get(args.suite, 300)
    conds = args.conds.split(",")

    results = collections.defaultdict(list)  # (task, cond_label) -> [0/1]
    out_path = pathlib.Path(args.out) / f"exp_c_{args.suite}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ti, task in enumerate(tasks):
        prompts = []
        if "orig" in conds:
            prompts.append(("orig", instrs[ti]))
        if "empty" in conds:
            prompts.append(("empty", ""))
        if "scramble" in conds:
            prompts.append(("scramble", U.scramble(instrs[ti], seed=ti)))
        if "para" in conds:
            for k, p in enumerate(U.make_paraphrases(instrs[ti])):
                prompts.append((f"para{k}", p))
        if "swap" in conds:
            for tj in {(ti + 1) % n_tasks, (ti + 5) % n_tasks} - {ti}:
                prompts.append((f"swap_from_{tj}", instrs[tj]))

        env = U.make_env(task, seed=0)
        for cond, prompt in prompts:
            for trial in range(args.trials):
                ok = run_episode(policy, env, suite, ti, trial, prompt, max_steps)
                results[f"{ti}|{cond}"].append(int(ok))
            sr = float(np.mean(results[f"{ti}|{cond}"]))
            print(f"[exp_c] task {ti} ({instrs[ti][:40]!r}) cond={cond:<14} "
                  f"success={sr:.2f}", flush=True)
            # checkpoint results after every condition so partial runs are usable
            with open(out_path, "w") as f:
                json.dump({k: v for k, v in results.items()}, f, indent=2)
        env.close()

    # Aggregate summary
    agg = collections.defaultdict(list)
    for key, vals in results.items():
        cond = key.split("|")[1].split("_from_")[0].rstrip("0123456789") or key.split("|")[1]
        agg[cond].extend(vals)
    print(f"\n[exp_c] ===== {args.suite} aggregate ({args.trials} trials/cell) =====")
    for cond, vals in sorted(agg.items()):
        print(f"  {cond:<10} success={np.mean(vals):.3f}  (n={len(vals)})")
    print("[exp_c] interpretation: empty ~= orig on this suite => bypass; "
          "swap success high (env task done while told otherwise) => memorization; "
          "para << orig while empty ~= orig => language used as task-ID string only.")


if __name__ == "__main__":
    main()
