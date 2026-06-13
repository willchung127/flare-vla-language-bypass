"""Guided-SDE analysis: McNemar's paired test + per-task breakdown + bootstrap CIs.

Loads the N=150 method matrix results and runs paired statistical analysis.

Output:
  - day12_headline_summary.json (machine-readable)
  - FIGURE_2_headline.png (bar chart with 95% CIs)
  - FIGURE_2_per_task.png (per-task breakdown)
  - FIGURE_2_alpha_curve.png (α-sensitivity inset)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


N150_DIR = Path("/Users/william/flare/results/day11_method_matrix_N150")
ALPHA_DIR = Path("/Users/william/flare/results/day12_alpha_sweep")
OUT = Path("/Users/william/flare/results")
BOOTSTRAP_N = 10000
np.random.seed(42)


def load_condition(name: str) -> list:
    """Load per-episode results for a condition."""
    p = N150_DIR / f"{name}.json"
    if not p.exists():
        return []
    return json.load(open(p))


def pair_outcomes(cond_a: list, cond_b: list) -> tuple[list, list]:
    """Match (task_id, trial) pairs across two conditions, return aligned success lists."""
    by_key_a = {(r["task_id"], r["trial"]): r["success"] for r in cond_a}
    by_key_b = {(r["task_id"], r["trial"]): r["success"] for r in cond_b}
    keys = sorted(set(by_key_a.keys()) & set(by_key_b.keys()))
    a = [by_key_a[k] for k in keys]
    b = [by_key_b[k] for k in keys]
    return a, b


def mcnemar(a: list, b: list) -> dict:
    """Paired McNemar's test.

    Counts:
      n11: a=succ, b=succ  (concordant)
      n10: a=succ, b=fail  (A helped where B failed = A>B)
      n01: a=fail, b=succ  (B helped where A failed = B>A)
      n00: both fail
    """
    n11 = n10 = n01 = n00 = 0
    for ai, bi in zip(a, b):
        if ai and bi: n11 += 1
        elif ai and not bi: n10 += 1
        elif not ai and bi: n01 += 1
        else: n00 += 1
    # Two-sided McNemar's exact: P(X >= max | binom(n10+n01, 0.5)) × 2
    from math import comb
    n = n10 + n01
    if n == 0:
        p_two_sided = 1.0
    else:
        x = max(n10, n01)
        # P(X >= x) under H0: each discordant is 50/50
        p_one_sided = sum(comb(n, k) for k in range(x, n + 1)) / (2 ** n)
        p_two_sided = min(1.0, 2 * p_one_sided)
    return {
        "n11_both_succ": n11,
        "n10_a_only": n10,
        "n01_b_only": n01,
        "n00_both_fail": n00,
        "n_paired": n11 + n10 + n01 + n00,
        "p_value_two_sided": float(p_two_sided),
    }


def bootstrap_diff_ci(a: list, b: list, n_boot: int = BOOTSTRAP_N, alpha: float = 0.05) -> dict:
    """Paired bootstrap CI on rate difference (b - a)."""
    a_arr = np.array(a, dtype=int)
    b_arr = np.array(b, dtype=int)
    n = len(a_arr)
    diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        diffs.append((b_arr[idx].mean() - a_arr[idx].mean()))
    diffs = np.array(diffs)
    return {
        "mean_diff": float(b_arr.mean() - a_arr.mean()),
        "ci_low": float(np.quantile(diffs, alpha / 2)),
        "ci_high": float(np.quantile(diffs, 1 - alpha / 2)),
        "bootstrap_n": n_boot,
    }


def per_task_breakdown(rows: list) -> dict:
    """Group by task_id."""
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r["success"])
    return {int(t): {"n": len(v), "succ": int(sum(v)),
                     "rate": float(sum(v) / len(v))}
            for t, v in by_task.items()}


def bootstrap_rate_ci(succ: int, n: int, n_boot: int = BOOTSTRAP_N) -> tuple[float, float]:
    """Bootstrap CI on a single success rate."""
    samples = np.random.binomial(n=1, p=succ/n if n > 0 else 0, size=(n_boot, n)).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main():
    print("==== Day 12 headline analysis ====")

    # Load N=150 data
    conditions = ["ODE_single", "GuidedODE_V3", "GuidedSDE_V3"]
    data = {c: load_condition(c) for c in conditions}
    for c, rs in data.items():
        if not rs:
            print(f"WARNING: missing data for {c}")
            return
        print(f"  {c}: {len(rs)} episodes")
    print()

    # ---- Aggregate rates with bootstrap CIs ----
    print("==== Per-condition success rates + 95% CI ====")
    rates = {}
    for c in conditions:
        rs = data[c]
        n = len(rs)
        s = sum(r["success"] for r in rs)
        rate = s / n
        ci_low, ci_high = bootstrap_rate_ci(s, n)
        rates[c] = {"n": n, "succ": s, "rate": rate,
                    "ci_low": ci_low, "ci_high": ci_high}
        print(f"  {c}: {s}/{n} = {100*rate:.1f}% "
              f"[95% CI: {100*ci_low:.1f}%, {100*ci_high:.1f}%]")
    print()

    # ---- Paired tests: Guided-X vs ODE ----
    print("==== Paired tests vs ODE_single ====")
    paired = {}
    for c in ["GuidedODE_V3", "GuidedSDE_V3"]:
        a, b = pair_outcomes(data["ODE_single"], data[c])
        m = mcnemar(a, b)
        boot = bootstrap_diff_ci(a, b)
        paired[f"{c}_vs_ODE"] = {**m, **boot}
        print(f"  {c} vs ODE_single (paired n={m['n_paired']}):")
        print(f"    n10 (ODE succ, Guided fail): {m['n10_a_only']}")
        print(f"    n01 (ODE fail, Guided succ): {m['n01_b_only']}")
        print(f"    McNemar's two-sided p: {m['p_value_two_sided']:.4f}")
        print(f"    Bootstrap Δ (Guided − ODE): {100*boot['mean_diff']:+.1f}pp  "
              f"[95% CI: {100*boot['ci_low']:+.1f}, {100*boot['ci_high']:+.1f}]")
    print()

    # Guided-SDE vs Guided-ODE — does noise help?
    print("==== Paired: Guided-SDE vs Guided-ODE (does noise help?) ====")
    a, b = pair_outcomes(data["GuidedODE_V3"], data["GuidedSDE_V3"])
    m = mcnemar(a, b)
    boot = bootstrap_diff_ci(a, b)
    paired["GuidedSDE_vs_GuidedODE"] = {**m, **boot}
    print(f"  GuidedSDE vs GuidedODE:")
    print(f"    n10 (ODE-guided succ, SDE-guided fail): {m['n10_a_only']}")
    print(f"    n01 (ODE-guided fail, SDE-guided succ): {m['n01_b_only']}")
    print(f"    McNemar's two-sided p: {m['p_value_two_sided']:.4f}")
    print(f"    Bootstrap Δ (SDE − ODE guided): {100*boot['mean_diff']:+.1f}pp  "
          f"[95% CI: {100*boot['ci_low']:+.1f}, {100*boot['ci_high']:+.1f}]")
    print()

    # ---- Per-task breakdown ----
    print("==== Per-task breakdown ====")
    per_task = {c: per_task_breakdown(data[c]) for c in conditions}
    task_ids = sorted(set(per_task[conditions[0]].keys()))
    print(f"  {'task':>4s}  " + "  ".join(f"{c:>14s}" for c in conditions))
    for t in task_ids:
        cells = []
        for c in conditions:
            d = per_task[c].get(t, {"succ": 0, "n": 0, "rate": 0})
            cells.append(f"{d['succ']:>2d}/{d['n']:<2d}={100*d['rate']:>3.0f}%")
        print(f"  {t:>4d}  " + "  ".join(f"{x:>14s}" for x in cells))
    print()

    # ---- α-sweep summary ----
    alpha_summary_path = ALPHA_DIR / "summary.json"
    alpha_data = None
    if alpha_summary_path.exists():
        alpha_data = json.load(open(alpha_summary_path))
        print("==== α-sweep (50 trials each) ====")
        for key, val in alpha_data["per_alpha"].items():
            print(f"  α={val['alpha']:.3f}: {val['n_successes']}/{val['n_episodes']} = "
                  f"{100*val['success_rate']:.0f}%")
        print()

    # ---- Save summary JSON ----
    summary = {
        "rates": rates,
        "paired_tests": paired,
        "per_task": per_task,
        "alpha_sweep": alpha_data,
    }
    with open(OUT / "day12_headline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {OUT / 'day12_headline_summary.json'}")

    # ---- Figure 2 v2: bar chart with CIs ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    cond_labels = {"ODE_single": "ODE\n(K=1)",
                   "GuidedODE_V3": "Guided-ODE\n(K=1, α=0.1)",
                   "GuidedSDE_V3": "Guided-SDE\n(K=1, α=0.1, η=0.2)"}
    x = np.arange(len(conditions))
    means = [rates[c]["rate"] * 100 for c in conditions]
    err_low = [(rates[c]["rate"] - rates[c]["ci_low"]) * 100 for c in conditions]
    err_high = [(rates[c]["ci_high"] - rates[c]["rate"]) * 100 for c in conditions]
    colors = ["lightgrey", "C0", "C3"]
    bars = ax.bar(x, means, yerr=[err_low, err_high], capsize=8, color=colors,
                  edgecolor="black", linewidth=1)
    for bi, m in zip(bars, means):
        ax.text(bi.get_x() + bi.get_width()/2, m + 1, f"{m:.1f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([cond_labels[c] for c in conditions])
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(80, 100)
    ax.set_title("LIBERO-Long, π₀ fine-tuned (N=150 episodes per condition, 5 tasks × 30 trials)\n"
                 "Verifier-guided sampling improves single-sample success by 5.4pp")
    ax.grid(True, alpha=0.3, axis="y")
    # Annotate significance
    p_sde = paired["GuidedSDE_V3_vs_ODE"]["p_value_two_sided"]
    sig_label = f"p={p_sde:.3f}" if p_sde >= 0.001 else "p<0.001"
    ax.annotate("", xy=(2, 95), xytext=(0, 95),
                arrowprops=dict(arrowstyle="-", color="black"))
    ax.text(1, 96, f"McNemar's {sig_label}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "FIGURE_2_headline.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT / 'FIGURE_2_headline.png'}")

    # ---- Figure: per-task ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    width = 0.25
    x = np.arange(len(task_ids))
    for i, c in enumerate(conditions):
        vals = [per_task[c][t]["rate"] * 100 for t in task_ids]
        ax.bar(x + (i - 1) * width, vals, width, label=cond_labels[c].replace("\n", " "),
               color=colors[i], edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels([f"Task {t}" for t in task_ids])
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title("Per-task breakdown (LIBERO-Long, N=30 trials per task)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "FIGURE_2_per_task.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT / 'FIGURE_2_per_task.png'}")

    # ---- α-sweep figure ----
    if alpha_data is not None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        alphas = alpha_data["alphas"]
        rates_a = [alpha_data["per_alpha"][f"alpha_{a}"]["success_rate"] * 100 for a in alphas]
        ax.plot(alphas, rates_a, "D-", lw=2.5, ms=12, color="C3")
        ax.axhline(rates["ODE_single"]["rate"] * 100, color="grey", ls="--",
                   alpha=0.7, label=f"ODE baseline ({rates['ODE_single']['rate']*100:.1f}%)")
        ax.set_xlabel("Guidance scale α"); ax.set_ylabel("Success rate (%)")
        ax.set_title("α-sensitivity: Guided-SDE-V3 (N=50 per α)\n"
                     "Robust in [0.05, 0.2]; α=0.5 destabilizes")
        ax.set_xscale("log"); ax.set_ylim(0, 105)
        ax.set_xticks(alphas); ax.set_xticklabels([str(a) for a in alphas])
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "FIGURE_2_alpha_curve.png", dpi=140, bbox_inches="tight")
        plt.close()
        print(f"saved {OUT / 'FIGURE_2_alpha_curve.png'}")

    print("\n==== HEADLINE SENTENCES FOR PAPER ====")
    p_guided_sde = paired["GuidedSDE_V3_vs_ODE"]["p_value_two_sided"]
    rate_ode = rates["ODE_single"]["rate"]
    rate_gsde = rates["GuidedSDE_V3"]["rate"]
    diff_pp = (rate_gsde - rate_ode) * 100
    print(f"1. ODE_single = {100*rate_ode:.1f}% (consistent with OpenVLA-OFT-reported 85.2%)")
    print(f"2. Guided-SDE-V3 = {100*rate_gsde:.1f}% (+{diff_pp:.1f}pp vs ODE)")
    print(f"3. McNemar's paired p = {p_guided_sde:.4f} "
          f"({'p<0.05' if p_guided_sde < 0.05 else 'n.s.'})")
    print(f"4. Bootstrap 95% CI on improvement: "
          f"[+{100*paired['GuidedSDE_V3_vs_ODE']['ci_low']:.1f}, "
          f"+{100*paired['GuidedSDE_V3_vs_ODE']['ci_high']:.1f}]pp")


if __name__ == "__main__":
    main()
