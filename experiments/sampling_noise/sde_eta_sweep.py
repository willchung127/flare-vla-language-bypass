"""Closed-loop SDE single-sample eta sweep on libero_10.

η ∈ {0.0, 0.2, 0.4, 0.6} (ODE included as control)
Scale: 5 tasks × 5 trials = 25 eps per η, ~100 eps total. Light scan (~4 GPU-hr).

Requires the openpi server to support reading η from a control file (or env var
on restart). This wrapper handles server restarts between η values.

Server side prerequisites (one-time):
1. patch_pi0_sde.py applied so `pi0.sample_actions` reads `self.flare_eta` (η=0
   delegates to original ODE).
2. scripts/serve_policy.py modified to honor FLARE_ETA env var:
       model.flare_eta = float(os.environ.get("FLARE_ETA", "0.0"))

This script:
- For each η, kills any old server, starts a new one with FLARE_ETA=η,
  waits for it to load (~45s), runs the eval client, kills server.
- Aggregates per-(task, seed, η) success into a JSON.
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
NUM_TRIALS = 5  # per task
NUM_TASKS_LIMIT = 5  # libero_10 has 10; cap to first 5 for the ~light scan
SEED = 42
REPLAN_STEPS = 5  # closed-loop default

OPENPI_DIR = Path.home() / "flare/openpi"
OUT_DIR = Path.home() / "flare/results/day6a_sde_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Adjust these to match your environment
SERVER_CMD = [
    "uv", "run", "python", "scripts/serve_policy.py",
    "policy:checkpoint",
    f"--policy.config=pi0_libero",
    f"--policy.dir={Path.home()}/flare/checkpoints/pi0_libero_pt",
    "--port=8000",
]
SERVER_READY_PROBE = ["curl", "-sf", "http://localhost:8000/metadata"]
SERVER_BOOT_TIMEOUT = 90  # seconds

CLIENT_CMD = lambda eta, log_path, video_dir: [
    "uv", "run", "python", "examples/libero/main.py",
    "--task_suite_name", TASK_SUITE,
    "--num_trials_per_task", str(NUM_TRIALS),
    "--replan_steps", str(REPLAN_STEPS),
    "--seed", str(SEED),
    "--video_out_path", str(video_dir),
]


def kill_server() -> None:
    subprocess.run(["pkill", "-f", "serve_policy.py"], check=False)
    time.sleep(2)


def start_server(eta: float):
    env = os.environ.copy()
    env["FLARE_ETA"] = str(eta)
    print(f"  starting server (FLARE_ETA={eta})")
    p = subprocess.Popen(SERVER_CMD, cwd=OPENPI_DIR, env=env,
                         stdout=open(OUT_DIR / f"server_eta_{eta}.log", "w"),
                         stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < SERVER_BOOT_TIMEOUT:
        try:
            r = subprocess.run(SERVER_READY_PROBE, capture_output=True, timeout=3)
            if r.returncode == 0:
                print(f"  server ready in {time.time() - t0:.0f}s")
                return p
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server didn't come up in {SERVER_BOOT_TIMEOUT}s")


def parse_main_log(text: str):
    """Pull aggregate success from main.py output. Falls back to counting 'success' lines."""
    # main.py doesn't print a structured summary by default; we count via env vars or videos
    succ = len(re.findall(r"success", text))
    fail = len(re.findall(r"failure", text))
    return {"raw_success_lines": succ, "raw_failure_lines": fail,
            "estimated_success_rate": succ / max(succ + fail, 1)}


def run_eta(eta: float):
    kill_server()
    server = start_server(eta)
    log_path = OUT_DIR / f"client_eta_{eta}.log"
    video_dir = OUT_DIR / f"videos_eta_{eta}"
    video_dir.mkdir(exist_ok=True)
    print(f"  running client → {log_path}")
    t0 = time.time()
    try:
        with open(log_path, "w") as f:
            subprocess.run(CLIENT_CMD(eta, log_path, video_dir), cwd=OPENPI_DIR,
                           stdout=f, stderr=subprocess.STDOUT, check=False)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    elapsed = time.time() - t0
    text = log_path.read_text()
    metrics = parse_main_log(text)
    metrics["wallclock_sec"] = elapsed
    return metrics


def main():
    results = {}
    for eta in ETAS:
        print(f"\n=== η = {eta} ===")
        results[str(eta)] = run_eta(eta)
        print(f"  η={eta}: {results[str(eta)]}")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSUMMARY at {OUT_DIR / 'summary.json'}")
    for eta, m in results.items():
        print(f"  η={eta}: est_succ={m['estimated_success_rate']:.2%} "
              f"({m['raw_success_lines']} succ / {m['raw_failure_lines']} fail in {m['wallclock_sec']:.0f}s)")


if __name__ == "__main__":
    main()
