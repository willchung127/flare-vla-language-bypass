"""generate_instruction_sweep.py — generates 20 BDDLs for T0 with prompts targeting mode-B
(tomato_sauce first), with GOALS ALIGNED to what each prompt actually asks for.

Output: ~/flare/custom_bddls/prompt_sweep/T0_<name>.bddl

The 20 variants probe whether pi0.5 attends to language at the level needed to flip
mode. They are grouped into 5 categories so the result is interpretable.

Goal alignment philosophy:
  - If the prompt names only one target object → goal requires only that object
    (so success = mode-B specifically).
  - If the prompt mentions both objects → goal requires both (success = task complete,
    first-moved tells us which was picked first).
  - Cross-task prompts (E) target objects from a different LIBERO task that ALSO exist
    in this scene; success is judged on those objects.
"""
from pathlib import Path
import subprocess


# Each entry: (variant_name, language, required_objects_for_success)
PROMPTS = [
    # =================================================================
    # A. Object REORDERING within the "put both X and Y" template (5)
    #    Tests: positional attention within preserved template.
    # =================================================================
    ("A1_reorder_both",
     "put both the tomato sauce and the alphabet soup in the basket",
     ["alphabet_soup_1", "tomato_sauce_1"]),
    ("A2_reorder_nob",
     "put the tomato sauce and the alphabet soup in the basket",
     ["alphabet_soup_1", "tomato_sauce_1"]),
    ("A3_reorder_can",
     "put both the tomato sauce can and the alphabet soup in the basket",
     ["alphabet_soup_1", "tomato_sauce_1"]),
    ("A4_reorder_nothe",
     "put both tomato sauce and alphabet soup in the basket",
     ["alphabet_soup_1", "tomato_sauce_1"]),
    ("A5_reorder_passive",
     "the tomato sauce and the alphabet soup should be in the basket",
     ["alphabet_soup_1", "tomato_sauce_1"]),

    # =================================================================
    # B. Single-target paraphrases (5)
    #    Tests: does pi0.5 follow simpler prompts? Goal aligned: tomato_sauce only.
    # =================================================================
    ("B1_single_put",
     "put the tomato sauce in the basket",
     ["tomato_sauce_1"]),
    ("B2_single_place",
     "place the tomato sauce in the basket",
     ["tomato_sauce_1"]),
    ("B3_single_pickput",
     "pick up the tomato sauce and put it in the basket",
     ["tomato_sauce_1"]),
    ("B4_single_move",
     "move the tomato sauce to the basket",
     ["tomato_sauce_1"]),
    ("B5_single_grab",
     "grab the tomato sauce",
     ["tomato_sauce_1"]),

    # =================================================================
    # C. Explicit ordering language (4)
    #    Tests: does pi0.5 understand "first/then" or "before/after"?
    # =================================================================
    ("C1_order_firstthen",
     "first put the tomato sauce in the basket then put the alphabet soup",
     ["alphabet_soup_1", "tomato_sauce_1"]),
    ("C2_order_startwith",
     "start with the tomato sauce",
     ["tomato_sauce_1"]),
    ("C3_order_goesfirst",
     "tomato sauce goes first then alphabet soup",
     ["alphabet_soup_1", "tomato_sauce_1"]),
    ("C4_order_before",
     "before picking up the alphabet soup put the tomato sauce in the basket",
     ["alphabet_soup_1", "tomato_sauce_1"]),

    # =================================================================
    # D. Negative / exclusionary phrasing (3)
    #    Tests: does pi0.5 understand "not", "avoid", "don't"?
    # =================================================================
    ("D1_neg_not",
     "put the tomato sauce in the basket not the alphabet soup",
     ["tomato_sauce_1"]),
    ("D2_neg_avoid",
     "avoid the alphabet soup place the tomato sauce in the basket",
     ["tomato_sauce_1"]),
    ("D3_neg_dontmove",
     "the alphabet soup should not be moved put the tomato sauce in the basket",
     ["tomato_sauce_1"]),

    # =================================================================
    # E. Cross-task templates (3) — the known-working pattern
    #    Tests: cross-task prompts referring to OTHER objects in same scene.
    # =================================================================
    ("E1_cross_t1",
     "put both the cream cheese box and the butter in the basket",
     ["cream_cheese_1", "butter_1"]),
    ("E2_cross_t1rev",
     "put both the butter and the cream cheese box in the basket",
     ["cream_cheese_1", "butter_1"]),
    ("E3_cross_milk",
     "put the milk in the basket",
     ["milk_1"]),
]


INPUT = Path.home() / (
    "flare/openpi/third_party/libero/libero/libero/bddl_files/libero_10/"
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl"
)
OUTDIR = Path.home() / "flare/custom_bddls/prompt_sweep"
MAKE_OR_BDDL = Path.home() / "flare/make_or_bddl.py"


def make_goal(objects):
    """Build a (And (In ...) (In ...)) goal from a list of object names.

    Single object: returns `(In obj region)` without And wrapper.
    Multiple:      returns `(And (In obj1 region) (In obj2 region) ...)`.
    """
    preds = [f"(In {o} basket_1_contain_region)" for o in objects]
    if len(preds) == 1:
        return preds[0]
    return f"(And {' '.join(preds)})"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Source BDDL not found: {INPUT}")
    if not MAKE_OR_BDDL.exists():
        raise FileNotFoundError(f"make_or_bddl.py not found at: {MAKE_OR_BDDL}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(PROMPTS)} BDDLs in {OUTDIR}/\n")

    paths = []
    for name, lang, objects in PROMPTS:
        out_path = OUTDIR / f"T0_{name}.bddl"
        goal = make_goal(objects)
        try:
            subprocess.run([
                "python3", str(MAKE_OR_BDDL),
                "--input", str(INPUT),
                "--output", str(out_path),
                "--new-language", lang,
                "--new-goal", goal,
            ], capture_output=True, text=True, check=True)
            print(f"  ✓ {name}")
            print(f"      lang: '{lang}'")
            print(f"      goal: {goal}")
            print(f"      targets ({len(objects)} obj): {objects}")
            print()
            paths.append(str(out_path))
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED {name}: {e.stderr}")

    print(f"\nGenerated {len(paths)}/{len(PROMPTS)} BDDLs.")
    print(f"\nFor the matrix runner --bddl-sweep-list flag, the JSON is:")
    import json as _j
    print(_j.dumps(paths))


if __name__ == "__main__":
    main()
