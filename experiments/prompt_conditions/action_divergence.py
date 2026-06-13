"""Exp A: Instruction-sensitivity of action outputs (forward passes only, no rollouts).

For each LIBERO-Goal task we grab K settled initial frames, then query the policy
with: the original instruction, every other task's instruction (swap), two
paraphrases, the empty string, and the SAME instruction repeated (noise floor for
the stochastic flow-matching sampler).

Metric: mean L2 distance between action chunks, reported as a ratio over the
noise floor. Ratio ~1.0 => language causally inert; ratio >> 1.0 => language
moves the actions.

Runs for pi0-LIBERO (finetuned) and optionally pi0-base with LIBERO norm stats
(--include-base), so the ONLY difference between the two runs is the weights.
If base-vs-finetuned ratios differ, finetuning changed language sensitivity.

Output: results/exp_a_<model>.npz + a printed summary table.
"""

import argparse
import json
import pathlib
import shutil

import numpy as np

import libero_utils as U


def build_hybrid_base_dir(out_dir: pathlib.Path) -> pathlib.Path:
    """pi0_base params + pi0_libero assets (norm stats), so preprocessing is identical.

    NOTE: fragile spot. Verify the downloaded checkpoint layout has `params/` and
    `assets/` subdirs; adjust if openpi changed its layout.
    """
    from openpi.shared import download

    base = pathlib.Path(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base"))
    lib = pathlib.Path(download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero"))
    hybrid = out_dir / "hybrid_pi0base_liberostats"
    if not hybrid.exists():
        hybrid.mkdir(parents=True)
        (hybrid / "params").symlink_to(base / "params")
        if (lib / "assets").exists():
            shutil.copytree(lib / "assets", hybrid / "assets")
    return hybrid


def load_policy(which: str, results_dir: pathlib.Path):
    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as _config

    cfg = _config.get_config("pi0_libero")
    if which == "finetuned":
        ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
    elif which == "base":
        ckpt = build_hybrid_base_dir(results_dir)
    else:
        raise ValueError(which)
    return policy_config.create_trained_policy(cfg, ckpt)


def chunk_dist(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    T = min(len(a), len(b))
    return float(np.linalg.norm(a[:T] - b[:T], axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--frames-per-task", type=int, default=3)
    ap.add_argument("--max-tasks", type=int, default=10)
    ap.add_argument("--include-base", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.out)
    results_dir.mkdir(parents=True, exist_ok=True)

    suite = U.get_suite(args.suite)
    n_tasks = min(suite.n_tasks, args.max_tasks)
    tasks = [suite.get_task(i) for i in range(n_tasks)]
    instrs = [t.language for t in tasks]
    print(f"[exp_a] {args.suite}: {n_tasks} tasks")
    for i, s in enumerate(instrs):
        print(f"  task {i}: {s}")

    # Collect frames once (env work is policy-independent).
    frames = []  # list of (task_id, obs)
    for ti, task in enumerate(tasks):
        env = U.make_env(task, seed=0)
        for k in range(args.frames_per_task):
            obs = U.reset_to_init_state(env, suite, ti, k)
            frames.append((ti, obs))
        env.close()
    print(f"[exp_a] collected {len(frames)} frames")

    models = ["finetuned"] + (["base"] if args.include_base else [])
    for which in models:
        print(f"\n[exp_a] ===== model: {which} =====")
        policy = load_policy(which, results_dir)

        conds_per_task = {}
        for ti in range(n_tasks):
            paras = U.make_paraphrases(instrs[ti])
            conds_per_task[ti] = (
                [("orig", instrs[ti]), ("repeat", instrs[ti]), ("empty", "")]
                + [("para", p) for p in paras]
                + [("swap", instrs[tj]) for tj in range(n_tasks) if tj != ti]
            )

        rows = []  # (task_id, frame_idx, cond, dist_to_orig)
        for fi, (ti, obs) in enumerate(frames):
            acts = {}
            for ci, (cond, prompt) in enumerate(conds_per_task[ti]):
                element = U.build_element(obs, prompt)
                acts[ci] = np.asarray(policy.infer(element)["actions"])
            ref = acts[0]  # 'orig'
            for ci, (cond, _p) in enumerate(conds_per_task[ti]):
                if ci == 0:
                    continue
                rows.append((ti, fi, cond, chunk_dist(ref, acts[ci])))
            if fi % 5 == 0:
                print(f"  frame {fi + 1}/{len(frames)} done")

        conds = sorted({r[2] for r in rows})
        summary = {}
        noise = np.mean([r[3] for r in rows if r[2] == "repeat"]) or 1e-9
        print(f"\n[exp_a:{which}] noise floor (repeat same instr): {noise:.4f}")
        print(f"{'condition':<10} {'L2 dist':>10} {'ratio/noise':>12}")
        for c in conds:
            d = float(np.mean([r[3] for r in rows if r[2] == c]))
            summary[c] = {"dist": d, "ratio": d / noise}
            print(f"{c:<10} {d:>10.4f} {d / noise:>12.2f}")

        np.savez(
            results_dir / f"exp_a_{which}.npz",
            rows=np.array(rows, dtype=object),
            instrs=np.array(instrs),
        )
        with open(results_dir / f"exp_a_{which}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        del policy  # free GPU memory before next model

    print("\n[exp_a] done. Key read: swap/empty ratio ~1 => language inert; "
          "compare ratios between finetuned and base.")


if __name__ == "__main__":
    main()
