"""Exp D follow-up: keyword-vs-semantics probes + swap videos.

Conditions (rollouts, libero_goal + libero_spatial):
  nounsyn   - noun synonyms ("bowl"->"dish", ...). Breaks => lexical keyword
              lookup confirmed; survives => real lexical semantics.
  keywords  - content words only ("put bowl stove"). Survives => syntax-free
              retrieval confirmed from the deletion side.
  verbose   - instruction embedded in distractor text. Tests robustness of the
              keyword channel to padding.

Also: --video N saves mp4s of N swap episodes (libero_goal) to results/videos/
to disambiguate "did the other task" vs "flailed", and to make the blog GIF.
"""

import argparse
import collections
import json
import pathlib

import imageio.v2 as imageio
import numpy as np

import libero_utils as U
from rollout_prompt_conditions import REPLAN_STEPS

NOUN_SYNONYMS = [
    ("bowl", "dish"),
    ("plate", "platter"),
    ("drawer", "compartment"),
    ("cabinet", "cupboard"),
    ("stove", "burner"),
    ("wine bottle", "wine flask"),
    ("cream cheese", "cheese block"),
    ("ramekin", "small cup"),
    ("cookie box", "biscuit carton"),
    ("rack", "stand"),
]
STOPWORDS = {"the", "a", "an", "of", "and", "to", "it", "is", "please"}


def nounsyn(instr):
    p = instr
    for a, b in NOUN_SYNONYMS:
        p = p.replace(a, b)
    return p


def keywords_only(instr):
    return " ".join(w for w in instr.split() if w.lower() not in STOPWORDS)


def verbose_wrap(instr):
    return ("this is a robot manipulation task in a simulated kitchen and the "
            f"goal for the robot arm right now is to {instr} as accurately as possible")


def run_episode(policy, env, suite, ti, init_idx, prompt, max_steps, video_path=None):
    obs = U.reset_to_init_state(env, suite, ti, init_idx)
    frames, plan, t = [], [], 0
    while t < max_steps:
        if not plan:
            actions = np.asarray(policy.infer(U.build_element(obs, prompt))["actions"])
            plan = list(actions[:REPLAN_STEPS])
        obs, _r, done, _info = env.step(plan.pop(0).tolist())
        if video_path is not None:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        if done:
            break
        t += 1
    if video_path is not None and frames:
        imageio.mimsave(str(video_path), frames[::2], fps=30)
    return bool(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="libero_goal,libero_spatial")
    ap.add_argument("--conds", default="nounsyn,keywords,verbose")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--max-tasks", type=int, default=10)
    ap.add_argument("--video", type=int, default=3, help="N swap episodes to record (goal suite)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as _config

    ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
    policy = policy_config.create_trained_policy(_config.get_config("pi0_libero"), ckpt)

    out_dir = pathlib.Path(args.out)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    transforms = {"nounsyn": nounsyn, "keywords": keywords_only, "verbose": verbose_wrap}

    for suite_name in args.suites.split(","):
        suite = U.get_suite(suite_name)
        n_tasks = min(suite.n_tasks, args.max_tasks)
        tasks = [suite.get_task(i) for i in range(n_tasks)]
        instrs = [t.language for t in tasks]
        max_steps = U.MAX_STEPS.get(suite_name, 300)
        results = collections.defaultdict(list)
        out_path = out_dir / f"exp_d_{suite_name}.json"

        for ti, task in enumerate(tasks):
            env = U.make_env(task, seed=0)
            for cond in args.conds.split(","):
                prompt = transforms[cond](instrs[ti])
                if ti == 0:
                    print(f"[exp_d] {suite_name} example {cond}: {prompt!r}")
                for trial in range(args.trials):
                    ok = run_episode(policy, env, suite, ti, trial, prompt, max_steps)
                    results[f"{ti}|{cond}"].append(int(ok))
                print(f"[exp_d] {suite_name} task {ti} {cond:<9} "
                      f"success={np.mean(results[f'{ti}|{cond}']):.2f}", flush=True)
                with open(out_path, "w") as f:
                    json.dump(dict(results), f, indent=2)
            env.close()

        agg = collections.defaultdict(list)
        for k, v in results.items():
            agg[k.split("|")[1]].extend(v)
        print(f"\n[exp_d] ===== {suite_name} aggregate =====")
        for c, v in sorted(agg.items()):
            print(f"  {c:<9} success={np.mean(v):.3f} (n={len(v)})")

    # --- swap videos on libero_goal: env task i, instruction from task j ---
    if args.video > 0:
        suite = U.get_suite("libero_goal")
        tasks = [suite.get_task(i) for i in range(min(suite.n_tasks, 10))]
        instrs = [t.language for t in tasks]
        for k in range(args.video):
            ti, tj = k, (k + 1) % len(tasks)
            env = U.make_env(tasks[ti], seed=0)
            vp = out_dir / "videos" / f"swap_env{ti}_instr{tj}.mp4"
            ok = run_episode(policy, env, suite, ti, 0, instrs[tj],
                             U.MAX_STEPS["libero_goal"], video_path=vp)
            print(f"[exp_d] video env={ti} ({instrs[ti][:35]!r}) told={instrs[tj][:35]!r} "
                  f"env-task-success={ok} -> {vp}")
            env.close()


if __name__ == "__main__":
    main()
