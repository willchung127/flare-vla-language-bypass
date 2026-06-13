"""make_figures.py — regenerate the seven figures in the report (docs/index.html).

Every number is loaded from a result file; nothing is hand-entered. The result
JSONs are not shipped with this repository (see README, "Data availability");
set RESULTS / SWEEP_RESULTS below to your local copy to reproduce the figures.

Output: docs/assets/figures/figN_*.png

Run:  python3 figures/make_figures.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Result JSONs live outside the repo. Point these at your local copy.
RESULTS = Path.home() / "flare" / "results"
SWEEP_RESULTS = Path.home() / "vla_sink_experiments" / "remote_results"

# Reuse the strict-compliance scorer that backs Figure 2.
sys.path.insert(0, str(ROOT / "experiments" / "instruction_sweep"))

RED, BLUE, GREY, LGREY, DARK = "#c23b22", "#2d6a9f", "#9a9a9a", "#cfcfcf", "#1a1a1a"
plt.rcParams.update({
    "font.size": 12.5, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/name}")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


# ===================== FIG 1 — instruction as a task label ===================
def fig_task_label():
    R = SWEEP_RESULTS

    def agg(path):
        raw = json.loads(path.read_text())
        pools = {}
        for key, vals in raw.items():
            cond = key.split("|")[1]
            if cond.startswith("para"):
                cond = "para"
            if cond.startswith("swap_from_"):
                cond = "swap"
            pools.setdefault(cond, []).extend(vals)
        return {c: float(np.mean(v)) for c, v in pools.items()}

    goal = {**agg(R / "exp_c_libero_goal.json"), **agg(R / "exp_d_libero_goal.json")}
    spat = {**agg(R / "exp_c_libero_spatial.json"), **agg(R / "exp_d_libero_spatial.json")}
    fa = json.loads((R / "exp_a_finetuned_summary.json").read_text())
    ba = json.loads((R / "exp_a_base_summary.json").read_text())

    conds = ["orig", "para", "scramble", "keywords", "nounsyn", "verbose", "empty", "swap"]
    labels = ["original", "paraphrase", "scrambled", "keywords\nonly", "noun\nsynonyms",
              "buried in\nfiller text", "empty", "swapped"]
    x = np.arange(len(conds))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.2),
                                   gridspec_kw={"width_ratios": [1.8, 1]})
    ax1.bar(x - w / 2, [goal.get(c, np.nan) for c in conds], w, color=BLUE, label="LIBERO-Goal")
    ax1.bar(x + w / 2, [spat.get(c, np.nan) for c in conds], w, color="#9cc2dd", label="LIBERO-Spatial")
    # distinguish "condition not run on this suite" from a true zero
    for dx, data in ((-w / 2, goal), (w / 2, spat)):
        for xi, c in zip(x, conds):
            if c not in data:
                ax1.text(xi + dx, 0.015, "n.t.", ha="center", va="bottom",
                         fontsize=7.5, color=GREY, rotation=90)
            elif data[c] < 0.005:
                ax1.text(xi + dx, 0.015, "0", ha="center", va="bottom",
                         fontsize=9, color=RED, fontweight="bold")
    ax1.text(0.015, 0.975, "n.t. = not tested on this suite", transform=ax1.transAxes,
             fontsize=8, color=GREY, va="top")
    ax1.axvspan(-0.5, 4.5, color="#3a9d5d", alpha=0.05)
    ax1.axvspan(4.5, 7.5, color=RED, alpha=0.05)
    ax1.set_xticks(x, labels, fontsize=9.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("rollout success rate")
    ax1.set_title("Success vs prompt manipulation")
    ax1.legend(loc="upper right", fontsize=9.5)

    ratios = ["repeat", "para", "empty", "swap"]
    rlabels = ["repeat\n(noise floor)", "paraphrase", "empty", "swapped"]
    xr = np.arange(len(ratios))
    ax2.bar(xr - w / 2, [fa[r]["ratio"] for r in ratios], w, color="#b3501a", label="fine-tuned")
    ax2.bar(xr + w / 2, [ba[r]["ratio"] for r in ratios], w, color="#e8c49f", label="base")
    ax2.axhline(1.0, color=GREY, lw=1, ls="--")
    ax2.set_xticks(xr, rlabels, fontsize=9.5)
    ax2.set_ylabel("action change / sampling noise")
    ax2.set_title("Forward-pass prompt sensitivity")
    ax2.legend(loc="upper left", fontsize=9.5)
    save(fig, "fig1_task_label.png")


# ===================== FIG 2 — strict vs lenient compliance ==================
def fig_strict_compliance():
    from score_strict_compliance import REQ_FORB, analyze_one
    agg = {}
    for stem, (_req, forb) in REQ_FORB.items():
        cat = stem.split("_")[1][0]   # "T0_A1_reorder_both" -> "A"
        path = RESULTS / f"multimode_matrix_promptsweep_pi0_n15_{stem}" / "ODE_single.json"
        if not path.exists():
            continue
        d = analyze_one(path, forb)
        a = agg.setdefault(cat, {"N": 0, "lenient": 0, "strict": 0})
        a["N"] += d["N"]; a["lenient"] += d["lenient"]; a["strict"] += d["strict"]

    cats = sorted(agg)
    x = np.arange(len(cats))
    lenient = [100 * agg[c]["lenient"] / agg[c]["N"] for c in cats]
    strict = [100 * agg[c]["strict"] / agg[c]["N"] for c in cats]
    labels = {"A": "reorder", "B": "single\ntarget", "C": "explicit\norder",
              "D": "negation", "E": "cross-\ntask"}

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(x - 0.2, lenient, 0.38, color=LGREY, label="lenient (standard success bit)")
    ax.bar(x + 0.2, strict, 0.38, color=RED, label="strict (forbidden object untouched)")
    for xi, s in zip(x, strict):
        ax.text(xi + 0.2, s + 2.5, f"{s:.0f}", ha="center", fontsize=11,
                fontweight="bold", color=RED)
    ax.set_xticks(x, [labels[c] for c in cats], fontsize=10.5)
    ax.set_ylabel("% of trials")
    ax.set_ylim(0, 108)
    ax.set_title("Compliance: lenient metric vs strict metric")
    ax.legend(loc="upper center", fontsize=9.5, ncol=1)
    save(fig, "fig2_strict_compliance.png")


# ===================== FIG 3 — attention budget ==============================
def fig_attention_anatomy():
    s = json.loads((RESULTS / "sink_per_key_multi.json").read_text())
    e = s["entries"]
    rows = [
        ("action tokens\n(self-attention)", np.mean([x["action_self"] for x in e]), GREY),
        ("image tokens", np.mean([x["image"] for x in e]), "#b8b8b8"),
        ("BOS token\n(attention sink)", np.mean([x["bos"] for x in e]), DARK),
        ("entire instruction", np.mean([x["instruction"] for x in e]), RED),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [100 * r[1] for r in rows], 0.62, color=[r[2] for r in rows])
    for yi, r in zip(y, rows):
        ax.text(100 * r[1] + 1, yi, f"{100*r[1]:.1f}%", va="center", fontsize=11,
                fontweight="bold", color=r[2] if r[2] != "#b8b8b8" else GREY)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=10.5)
    ax.set_xlim(0, 63)
    ax.set_xlabel("share of the action expert's attention (final layer, %)")
    ax.grid(axis="y", alpha=0)
    save(fig, "fig3_attention_anatomy.png")


# ===================== FIG 4 — velocity similarity ===========================
def fig_velocity():
    d = json.loads((RESULTS / "velocity_real_dims_extended.json").read_text())
    rows = [
        ("probe_velocity_T0_P3_unrelated", 'unrelated task ("stack the mugs")'),
        ("probe_velocity_T1_v1", "reorder (task 2)"),
        ("probe_velocity_T0_P1_single", "single object (tomato vs alphabet)"),
        ("probe_velocity_T0_v1", "reorder (task 1)"),
        ("probe_velocity_T0_P2_order", "word order swap"),
    ]
    rows = [(k, lab) for k, lab in rows if k in d]
    vals = [d[k]["cos_real_7_dims"] for k, _ in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ax.barh(y, vals, 0.62, color=RED)
    for yi, v in zip(y, vals):
        ax.text(v - 0.0003, yi, f"{v:.4f}", ha="right", va="center",
                color="white", fontsize=10.5, fontweight="bold")
    ax.set_yticks(y, [lab for _, lab in rows], fontsize=10)
    ax.set_xlim(0.99, 1.0008)
    ax.axvline(1.0, color=GREY, lw=1, ls=":")
    ax.set_xlabel("velocity field similarity between the two prompts (1.0 means identical)")
    ax.set_title("Changing the prompt barely changes the action computation")
    save(fig, "fig4_velocity.png")


# ===================== FIG 5 — linear probe ==================================
def fig_probe():
    per = json.loads((RESULTS / "probe_decode_v2b" / "probe_v2_results.json").read_text())["per_layer"]
    loci = {"expert": [], "paligemma": []}
    for d in per.values():
        loci[d["locus"]].append(d)
    for k in loci:
        loci[k].sort(key=lambda d: d["depth"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.0),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    style = {"expert": (RED, "action expert"), "paligemma": (BLUE, "VLM backbone")}
    for locus, rows in loci.items():
        c, lab = style[locus]
        axA.plot([d["depth"] for d in rows], [d["auc_mean"] for d in rows],
                 "-o", ms=4, lw=2, color=c, label=lab)
    axA.set_xlabel("layer")
    axA.set_ylabel("decoding AUC (held-out wordings)")
    axA.set_ylim(0.65, 1.0)
    axA.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axA.set_title("Which object the prompt names is readable")
    axA.legend(loc="lower left", fontsize=10)

    for locus, rows in loci.items():
        c, lab = style[locus]
        axB.plot([d["depth"] for d in rows],
                 [100 * d["neg_transfer_frac_tomato"] for d in rows],
                 "-o", ms=4, lw=2, color=c, label=lab)
    axB.set_ylim(-3, 100)
    axB.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axB.set_xlabel("layer")
    axB.set_ylabel("negation prompts grouped by meaning (%)")
    axB.text(8.5, 50, "0–3% at every layer\nof both towers", ha="center",
             fontsize=11, color=DARK)
    axB.set_title("What the sentence asks for never forms")
    axB.legend(loc="upper right", fontsize=10)
    save(fig, "fig5_probe.png")


# ===================== FIG 6 — steering dose response ========================
def fig_steering():
    L16 = json.loads((RESULTS / "steering_bank_L16_fine" / "steering_bank.json").read_text())
    PLA = json.loads((RESULTS / "steering_bank_L16_placebo" / "steering_bank.json").read_text())

    def series(block, field):
        ks = sorted([k for k in block if k.startswith("alpha=")],
                    key=lambda k: float(k.split("=")[1]))
        return ([float(k.split("=")[1]) for k in ks], [block[k][field] for k in ks])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.0))
    a, ct = series(L16["diagnostic"], "cos_to_tomato")
    _, ca = series(L16["diagnostic"], "cos_to_alphabet")
    _, pt = series(PLA["diagnostic"], "cos_to_tomato")
    axA.plot(a, ct, "-o", ms=4, lw=2, color=RED, label="toward tomato (steered)")
    axA.plot(a, ca, "-o", ms=4, lw=2, color=BLUE, label="toward alphabet soup (steered)")
    axA.plot(a, pt, "--", lw=1.6, color=GREY, label="toward tomato (random placebo)")
    axA.axvline(0, color=GREY, lw=0.8, alpha=0.5)
    axA.set_xlabel(r"steering strength $\alpha$")
    axA.set_ylabel("velocity similarity to object prompt")
    axA.set_title("The intervention works: velocity shifts")
    axA.legend(loc="lower left", fontsize=9.5)

    a, succ = series(L16["rollout"], "success")
    _, tom = series(L16["rollout"], "tomato_first")
    n = L16["rollout"]["alpha=+0"]["n"]
    axB.axvspan(min(a), -45, color=GREY, alpha=0.10)
    axB.axvspan(45, max(a), color=GREY, alpha=0.10)
    axB.plot(a, [s / n for s in succ], "-o", ms=4, lw=2, color=DARK, label="task success")
    axB.plot(a, [t / n for t in tom], "-s", ms=5, lw=2.2, color=RED,
             label="picks commanded object first")
    axB.axvline(0, color=GREY, lw=0.8, alpha=0.5)
    axB.set_xlabel(r"steering strength $\alpha$")
    axB.set_ylabel(f"rate (n={n} rollouts per $\\alpha$)")
    axB.set_ylim(-0.05, 1.05)
    axB.set_title("The behavior never flips")
    axB.legend(loc="center right", fontsize=9.5)
    save(fig, "fig6_steering.png")


# ===================== FIG 7 — LoRA ablation =================================
def fig_lora():
    variants = [("variant_a_v2", "plain LoRA"), ("variant_c_v2", "+ counterfactual\ndata"),
                ("variant_d_v2", "+ CF data\n+ sink mask")]
    tomato, canon = [], []
    for v, _ in variants:
        h = json.loads((RESULTS / "lora_ablation" / v / "headline_metrics.json").read_text())
        tomato.append(h["tomato_rate_on_tomato_prompt"])
        canon.append(h["alphabet_rate_on_canonical"])
    n = 15
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for rates, off, col, lab in [(tomato, -0.2, RED, "goes for the tomato first (tomato-only prompt)"),
                                 (canon, 0.2, LGREY, "original two-object task still works")]:
        ks = [round(r * n) for r in rates]
        cis = [wilson(k, n) for k in ks]
        ax.bar(x + off, [100 * r for r in rates], 0.38, color=col, label=lab)
        ax.errorbar(x + off, [100 * c[0] for c in cis],
                    yerr=[[100 * (c[0] - c[1]) for c in cis], [100 * (c[2] - c[0]) for c in cis]],
                    fmt="none", ecolor=DARK, capsize=3, lw=1)
        # value labels sit ABOVE the CI whisker, never on it
        for xi, r, c in zip(x + off, rates, cis):
            ax.text(xi, 100 * c[2] + 2.5, f"{100*r:.0f}%", ha="center", fontsize=11,
                    fontweight="bold", color=RED if col == RED else GREY)
    ax.set_xticks(x, [v[1] for v in variants], fontsize=10.5)
    ax.set_ylabel("% of 15 rollouts (95% CI)")
    ax.set_ylim(0, 132)   # headroom so the legend clears the 100% label
    ax.set_title("Counterfactual data moves the first reach toward the commanded object")
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, "fig7_lora.png")


if __name__ == "__main__":
    fig_task_label()
    fig_strict_compliance()
    fig_attention_anatomy()
    fig_velocity()
    fig_probe()
    fig_steering()
    fig_lora()
    print("done")
