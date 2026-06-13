"""score_strict_compliance.py — strict re-score of the 20-prompt sweep.

The standard sweep scorer reports the lenient "Succ%" (goal object placed),
which is INFLATED: a model that ignores the language but replays the memorized "put both"
trajectory still scores ~100% on a single-target / negation prompt, because the named object
incidentally ends up in the basket while the FORBIDDEN object is also moved.

This re-score adds, per prompt, the language-compliant metric:
  - LENIENT  : goal object(s) placed              (== standard Succ%)
  - STRICT   : goal object(s) placed AND no FORBIDDEN object was moved
               (for single-target / negation prompts, moving the OTHER object is a violation)
  - 1st-moved: which object moved first (displacement proxy)
  - BothMoved: fraction of trials where BOTH default objects (alphabet+tomato) moved
               -> the "replays the whole task regardless of the words" signature

Reads ~/flare/results/multimode_matrix_<out_suffix>_<variant>/ODE_single.json
Usage:  python3 score_strict_compliance.py --out-suffix promptsweep
"""
import argparse
import json
from collections import Counter
from pathlib import Path

# Category letter -> human-readable name (for the summary printout).
CATEGORY_NAMES = {
    "A": "Reordering ('put both X and Y')",
    "B": "Single-target paraphrase",
    "C": "Explicit ordering ('first X then Y')",
    "D": "Negative phrasing ('not Y')",
    "E": "Cross-task template",
}

# Object stems (substring-matched against moved_objects[*]["name"], e.g. "alphabet_soup_1_pos").
ALPHA, TOMATO, CREAM, BUTTER, MILK = "alphabet", "tomato", "cream_cheese", "butter", "milk"

# Per-variant (required_to_place, forbidden_to_move). Forbidden = what the LANGUAGE excludes.
#   A / C(both)  : both named -> nothing forbidden
#   B / C2 / D   : only tomato named (D explicitly negates alphabet) -> moving alphabet violates
#   E1/E2        : cream cheese + butter named -> moving the default pair violates
#   E3           : milk named -> moving the default pair violates
REQ_FORB = {
    "T0_A1_reorder_both": ([ALPHA, TOMATO], []),
    "T0_A2_reorder_nob":  ([ALPHA, TOMATO], []),
    "T0_A3_reorder_can":  ([ALPHA, TOMATO], []),
    "T0_A4_reorder_nothe":([ALPHA, TOMATO], []),
    "T0_A5_reorder_passive":([ALPHA, TOMATO], []),
    "T0_B1_single_put":   ([TOMATO], [ALPHA]),
    "T0_B2_single_place": ([TOMATO], [ALPHA]),
    "T0_B3_single_pickput":([TOMATO], [ALPHA]),
    "T0_B4_single_move":  ([TOMATO], [ALPHA]),
    "T0_B5_single_grab":  ([TOMATO], [ALPHA]),
    "T0_C1_order_firstthen":([ALPHA, TOMATO], []),
    "T0_C2_order_startwith":([TOMATO], [ALPHA]),
    "T0_C3_order_goesfirst":([ALPHA, TOMATO], []),
    "T0_C4_order_before": ([ALPHA, TOMATO], []),
    "T0_D1_neg_not":      ([TOMATO], [ALPHA]),
    "T0_D2_neg_avoid":    ([TOMATO], [ALPHA]),
    "T0_D3_neg_dontmove": ([TOMATO], [ALPHA]),
    "T0_E1_cross_t1":     ([CREAM, BUTTER], [ALPHA, TOMATO]),
    "T0_E2_cross_t1rev":  ([CREAM, BUTTER], [ALPHA, TOMATO]),
    "T0_E3_cross_milk":   ([MILK], [ALPHA, TOMATO]),
}

# Variant stem -> category letter (the letter is encoded in the stem: T0_A1_... -> A).
PROMPT_CATEGORIES = {stem: stem.split("_")[1][0] for stem in REQ_FORB}


def moved_stems(trial):
    names = [m.get("name", "") for m in trial.get("moved_objects", [])]
    stems = set()
    for s in (ALPHA, TOMATO, CREAM, BUTTER, MILK):
        if any(s in n for n in names):
            stems.add(s)
    return stems, names


def classify_first(trial):
    mo = trial.get("moved_objects", [])
    if not mo:
        return "<none>"
    n = mo[0].get("name", "")
    for s, lbl in ((TOMATO, "tomato"), (ALPHA, "alphabet"), (CREAM, "cream"),
                   (BUTTER, "butter"), (MILK, "milk")):
        if s in n:
            return lbl
    return "other"


def analyze_one(path, forbidden):
    trials = json.load(open(path))
    if not trials:
        return None
    N = len(trials)
    lenient = strict = both = 0
    firsts = Counter()
    all_names = set()
    for t in trials:
        stems, names = moved_stems(t)
        all_names.update(names)
        succ = bool(t.get("success", False))
        violated = any(f in stems for f in forbidden)
        lenient += int(succ)
        strict += int(succ and not violated)
        both += int(ALPHA in stems and TOMATO in stems)
        firsts[classify_first(t)] += 1
    return {"N": N, "lenient": lenient, "strict": strict, "both": both,
            "firsts": firsts, "names": all_names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(Path.home() / "flare/results"))
    ap.add_argument("--out-suffix", default="promptsweep")
    args = ap.parse_args()
    rd = Path(args.results_dir).expanduser()

    print("=" * 100)
    print(f"  STRICT RE-SCORE — 20-prompt sweep [{args.out_suffix}]  "
          f"(STRICT = goal placed AND forbidden object NOT moved)")
    print("=" * 100)
    fmt = "  {:<22s} | {:>3s} | {:>7s} | {:>7s} | {:>5s} {:>5s} {:>5s} | {:>9s}"
    print(fmt.format("Variant", "N", "Lenient", "STRICT", "1stT", "1stA", "1stO", "BothMoved"))
    print("  " + "-" * 96)

    per_cat = {c: [] for c in CATEGORY_NAMES}
    seen_names = set()
    for stem, cat in PROMPT_CATEGORIES.items():
        path = rd / f"multimode_matrix_{args.out_suffix}_{stem}" / "ODE_single.json"
        if not path.exists():
            print(fmt.format(stem.replace("T0_", ""), "-", "(missing)", "-", "-", "-", "-", "-"))
            continue
        _, forb = REQ_FORB.get(stem, ([], []))
        d = analyze_one(path, forb)
        if d is None:
            continue
        seen_names.update(d["names"])
        per_cat[cat].append((stem, d))
        f = d["firsts"]
        print(fmt.format(
            stem.replace("T0_", ""), str(d["N"]),
            f"{100*d['lenient']/d['N']:.0f}%", f"{100*d['strict']/d['N']:.0f}%",
            str(f.get("tomato", 0)), str(f.get("alphabet", 0)),
            str(d["N"] - f.get("tomato", 0) - f.get("alphabet", 0)),
            f"{100*d['both']/d['N']:.0f}%"))

    print("\n" + "=" * 100)
    for cat in sorted(per_cat):
        rows = per_cat[cat]
        if not rows:
            continue
        N = sum(d["N"] for _, d in rows)
        Ln = sum(d["lenient"] for _, d in rows)
        St = sum(d["strict"] for _, d in rows)
        print(f"  {cat} {CATEGORY_NAMES[cat]:42s}  lenient {100*Ln/N:5.1f}%   STRICT {100*St/N:5.1f}%   (n={N})")
    tot_n = sum(d["N"] for rows in per_cat.values() for _, d in rows)
    tot_s = sum(d["strict"] for rows in per_cat.values() for _, d in rows)
    if tot_n:
        print("  " + "-" * 96)
        print(f"  HEADLINE language-compliant (STRICT): {tot_s}/{tot_n} = {100*tot_s/tot_n:.1f}%")
    print("=" * 100)
    # sanity: surface the real object-name strings so the stem map can be verified/fixed
    print("\n  [sanity] object names seen in moved_objects:",
          ", ".join(sorted(seen_names)) or "(none)")


if __name__ == "__main__":
    main()
