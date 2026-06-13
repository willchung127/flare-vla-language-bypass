"""Outcome-mode analysis: K_outcome|success — ODE vs SDE comparison.

Loads ~/flare/results/day7_outcome_scan/eta_*.json (4 etas: 0.0, 0.2, 0.4, 0.6)
and computes:
  - Success rate per η
  - K_outcome|success per (task, init): connected-component count at 5cm threshold
    on terminal eef positions among successful K rollouts
  - Aggregate mean/median K_outcome|success per η
  - Per-task breakdown

Generates:
  - results/day7_K_outcome_summary.json  — table data
  - results/FIGURE_D_K_outcome_vs_eta.png — Track A headline panel
  - results/FIGURE_E_per_task_K_outcome.png — per-task scatter for supp

This is the data that lets us claim: "outcomes converge to ~1 cluster even
under SDE noise injection — outcome unimodality is task-structural, not
ODE-determinism-induced."
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DAY7 = Path("/Users/william/flare/results/day7_outcome_scan")
OUT = Path("/Users/william/flare/results")
ETAS = [0.0, 0.2, 0.4, 0.6]
CLUSTER_THR_M = 0.05  # 5 cm


def K_outcome_among(successes: list[dict]) -> int:
    """Connected-component count at CLUSTER_THR_M on terminal eef positions."""
    n = len(successes)
    if n == 0:
        return 0
    if n == 1:
        return 1
    eef = np.array([s["terminal_state"]["eef_pos"] for s in successes])
    D = np.linalg.norm(eef[:, None] - eef[None], axis=-1)
    adj = D < CLUSTER_THR_M
    seen = set(); count = 0
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            for u in range(n):
                if u not in seen and adj[v, u]:
                    stack.append(u)
        count += 1
    return count


def analyze_one_eta(eta: float) -> dict:
    p = DAY7 / f"eta_{eta}.json"
    if not p.exists():
        print(f"  MISSING: {p}")
        return None
    rs = json.load(open(p))

    # Group by (task_id, init_idx)
    grp = defaultdict(list)
    for r in rs:
        grp[(r["task_id"], r["init_idx"])].append(r)

    K_outc = []           # per group K_outcome|succ (#clusters)
    succ_rate = []        # per group success rate (k/K)
    spread = []           # per group mean pairwise distance among successes (m)
    per_task = defaultdict(list)  # task_id -> [K_outc...]
    per_task_succ = defaultdict(list)

    for (task_id, init_idx), group in grp.items():
        succs = [g for g in group if g["success"]]
        succ_rate.append(len(succs) / len(group))
        per_task_succ[task_id].append(len(succs) / len(group))
        K = K_outcome_among(succs)
        K_outc.append(K)
        per_task[task_id].append(K)
        if len(succs) >= 2:
            eef = np.array([s["terminal_state"]["eef_pos"] for s in succs])
            D = np.linalg.norm(eef[:, None] - eef[None], axis=-1)
            iu = np.triu_indices(len(succs), k=1)
            spread.append(float(D[iu].mean()))
        else:
            spread.append(0.0)

    return {
        "eta": eta,
        "n_episodes": len(rs),
        "n_groups": len(grp),
        "n_successes": int(sum(r["success"] for r in rs)),
        "success_rate": float(sum(r["success"] for r in rs) / max(len(rs), 1)),
        "mean_succ_rate_per_init": float(np.mean(succ_rate)),
        "mean_K_outcome_given_success": float(np.mean(K_outc)),
        "median_K_outcome_given_success": float(np.median(K_outc)),
        "frac_K_eq_1": float(np.mean([k == 1 for k in K_outc])),
        "mean_terminal_eef_spread_m": float(np.mean(spread)),
        "per_task_mean_K": {
            int(t): float(np.mean(ks)) for t, ks in per_task.items()
        },
        "per_task_succ_rate": {
            int(t): float(np.mean(ss)) for t, ss in per_task_succ.items()
        },
        "K_outc_list": K_outc,
        "spread_list": spread,
    }


def main():
    results = {}
    print("\n==== Day 7 K_outcome|success analysis ====")
    print(f"  Cluster threshold: {CLUSTER_THR_M*100:.0f} cm on terminal eef_pos")
    print(f"  Per-group K = #connected components among successful K=8 rollouts\n")

    for eta in ETAS:
        r = analyze_one_eta(eta)
        if r is None:
            continue
        results[str(eta)] = r
        print(f"  η={eta}:")
        print(f"    success_rate={r['success_rate']:.1%} ({r['n_successes']}/{r['n_episodes']})")
        print(f"    mean K_outcome|succ = {r['mean_K_outcome_given_success']:.2f}  "
              f"(median={r['median_K_outcome_given_success']:.1f}, "
              f"frac K=1: {r['frac_K_eq_1']:.0%})")
        print(f"    mean terminal eef spread among succ: {r['mean_terminal_eef_spread_m']*100:.1f} cm")
        print(f"    per-task K: {r['per_task_mean_K']}")
        print()

    # ----- Headline plot: K_outcome|succ vs η + success rate -----
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ax2 = ax1.twinx()
    e_arr = [r["eta"] for r in results.values()]
    succ = [r["success_rate"] * 100 for r in results.values()]
    K_arr = [r["mean_K_outcome_given_success"] for r in results.values()]
    spread_cm = [r["mean_terminal_eef_spread_m"] * 100 for r in results.values()]

    bars = ax1.bar([e - 0.02 for e in e_arr], succ, width=0.04, color="C2", alpha=0.5,
                   label="Success rate (%)", edgecolor="black")
    ax2.plot(e_arr, K_arr, "D-", lw=2.5, ms=14, color="C0",
             label="K_outcome | success (max=8)")
    ax2.axhline(1.0, color="grey", ls="--", alpha=0.6, lw=1)
    ax2.text(max(e_arr), 1.05, "K=1 (full collapse)", ha="right", fontsize=9, color="grey")

    ax1.set_xlabel("SDE η (0 = ODE)"); ax1.set_ylabel("Success rate (%)", color="C2")
    ax2.set_ylabel("Mean K_outcome | success", color="C0")
    ax1.set_ylim(0, 110); ax2.set_ylim(0, 3)
    ax1.set_xticks(e_arr)
    ax1.grid(True, alpha=0.3, axis="y")
    plt.title("Day 7 — Outcome diversity is task-structural, not ODE-induced\n"
              "(SDE η preserves success ≤ 0.4 but does NOT restore outcome diversity; K|succ stays ≈ 1.1)",
              fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "FIGURE_D_K_outcome_vs_eta.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT / 'FIGURE_D_K_outcome_vs_eta.png'}")

    # ----- Per-task scatter -----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    n_tasks = 5
    width = 0.18
    x = np.arange(n_tasks)
    palette = ["grey", "C0", "C1", "C3"]
    for i, eta in enumerate(ETAS):
        if str(eta) not in results:
            continue
        per_task = results[str(eta)]["per_task_mean_K"]
        vals = [per_task.get(t, 0.0) for t in range(n_tasks)]
        ax.bar(x + (i - 1.5) * width, vals, width=width,
               label=f"η={eta}" + (" (ODE)" if eta == 0 else ""),
               color=palette[i], edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels([f"Task {t}" for t in range(n_tasks)])
    ax.set_ylabel("Mean K_outcome | success (per init, max=8)")
    ax.axhline(1.0, color="black", ls="--", alpha=0.6, lw=1)
    ax.set_title("Day 7 — Per-task K_outcome|success across η\n"
                 "(near-unimodal for every task under every η ≤ 0.6)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "FIGURE_E_per_task_K_outcome.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT / 'FIGURE_E_per_task_K_outcome.png'}")

    # ----- Save summary JSON -----
    out_summary = {k: {kk: vv for kk, vv in v.items()
                       if kk not in ("K_outc_list", "spread_list")}
                   for k, v in results.items()}
    with open(OUT / "day7_K_outcome_summary.json", "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"saved {OUT / 'day7_K_outcome_summary.json'}")

    # ----- Headline numbers -----
    print("\n==== HEADLINE TABLE ====")
    print(f"  {'η':>5s}  {'succ':>5s}  {'K|succ':>8s}  {'frac K=1':>9s}  {'spread':>8s}")
    for eta in ETAS:
        if str(eta) not in results:
            continue
        r = results[str(eta)]
        print(f"  {eta:>5.2f}  {r['success_rate']:>5.0%}  "
              f"{r['mean_K_outcome_given_success']:>8.2f}  "
              f"{r['frac_K_eq_1']:>8.0%}   "
              f"{r['mean_terminal_eef_spread_m']*100:>6.1f}cm")


if __name__ == "__main__":
    main()
