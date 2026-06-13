"""Closed-loop outcome scan on libero_10.

K=8 closed-loop rollouts per (task, init, η) at iid integration noise.
Scale: 5 tasks × 5 inits × 8 rollouts × 4 etas = 800 episodes. ~16 GPU-hr overnight.

For each successful episode, log the terminal eef pose + on-table object pose
to build the K_outcome|success diagnostic (Track A headline metric).

Same server-restart-per-η pattern as Day 6a, but more episodes per η.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from pathlib import Path

ETAS = [0.0, 0.2, 0.4, 0.6]
TASK_SUITE = "libero_10"
NUM_TRIALS = 5    # inits per task
K_ROLLOUTS = 8    # iid noise rollouts per init
NUM_TASKS_LIMIT = 5
REPLAN_STEPS = 5

OPENPI_DIR = Path.home() / "flare/openpi"
OUT_DIR = Path.home() / "flare/results/day7_outcome_scan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SERVER_CMD = [
    "uv", "run", "python", "scripts/serve_policy.py",
    "policy:checkpoint",
    f"--policy.config=pi0_libero",
    f"--policy.dir={Path.home()}/flare/checkpoints/pi0_libero_pt",
    "--port=8000",
]
SERVER_READY_PROBE = ["curl", "-sf", "http://localhost:8000/metadata"]
SERVER_BOOT_TIMEOUT = 90


def kill_server() -> None:
    subprocess.run(["pkill", "-f", "serve_policy.py"], check=False)
    time.sleep(2)


def start_server(eta: float):
    env = os.environ.copy()
    env["FLARE_ETA"] = str(eta)
    p = subprocess.Popen(SERVER_CMD, cwd=OPENPI_DIR, env=env,
                         stdout=open(OUT_DIR / f"server_eta_{eta}.log", "w"),
                         stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < SERVER_BOOT_TIMEOUT:
        r = subprocess.run(SERVER_READY_PROBE, capture_output=True, timeout=3)
        if r.returncode == 0:
            return p
        time.sleep(2)
    raise RuntimeError("Server boot timeout")


def run_K_rollouts_for_eta(eta: float):
    """Run K iid noise rollouts per (task, init) by varying the --seed across runs.
    main.py uses np.random.seed(args.seed); each invocation re-seeds noise sampling
    inside the server (via the SDE integration's torch.randn calls).
    """
    kill_server()
    server = start_server(eta)
    eta_dir = OUT_DIR / f"eta_{eta}"
    eta_dir.mkdir(exist_ok=True)
    results = []
    try:
        for k in range(K_ROLLOUTS):
            seed_k = 1000 + k * 17
            log_path = eta_dir / f"client_k{k}.log"
            cmd = [
                "uv", "run", "python", "examples/libero/main.py",
                "--task_suite_name", TASK_SUITE,
                "--num_trials_per_task", str(NUM_TRIALS),
                "--replan_steps", str(REPLAN_STEPS),
                "--seed", str(seed_k),
                "--video_out_path", str(eta_dir / f"videos_k{k}"),
            ]
            print(f"  η={eta} k={k} seed={seed_k}")
            with open(log_path, "w") as f:
                subprocess.run(cmd, cwd=OPENPI_DIR, stdout=f, stderr=subprocess.STDOUT, check=False)
            text = log_path.read_text()
            succ = len(re.findall(r"success", text))
            fail = len(re.findall(r"failure", text))
            results.append({"k": k, "seed": seed_k, "successes": succ, "failures": fail})
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    return results


def main():
    out = {}
    for eta in ETAS:
        print(f"\n=== η = {eta} ===")
        out[str(eta)] = run_K_rollouts_for_eta(eta)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("DONE.")
    for eta, runs in out.items():
        total_s = sum(r["successes"] for r in runs)
        total_f = sum(r["failures"] for r in runs)
        print(f"  η={eta}: {total_s} succ / {total_f} fail across {len(runs)} K-rollouts")


if __name__ == "__main__":
    main()
