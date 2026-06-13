"""linear_probe_decoding.py — CORRECTED decodability probe (replaces the broken AUC=1.000 run).

WHY V1 WAS BROKEN (probe_decode_wrapup, 2026-06-06): two fixed prompts +
StratifiedKFold over samples = the probe memorizes PROMPT IDENTITY, which is
trivially separable from layer 0 (token embeddings). AUC=1.000 at every layer
with zero variance — leakage, not signal.

V2 DESIGN — decode the COMMANDED TARGET, not the prompt string:
  Classes (17 prompts from the sweep, E excluded):
    class 1 "tomato-only commanded": B1-B5, C2, D1-D3   (9 prompts)
    class 0 "both commanded":        A1-A5, C1, C3, C4  (8 prompts)
  The scene is IDENTICAL for every prompt (same BDDL, same inits), so any
  decodability is language information by construction.

  ANALYSIS 1 (main): GroupKFold over PROMPTS (train on some wordings, test on
    held-out wordings) -> AUC per layer. The probe must generalize across
    surface forms, so it cannot memorize prompt identity.
    + PERMUTATION NULL at the prompt level (reassign class labels to prompts,
    preserving the 9/8 split) -> p-value per layer.
  ANALYSIS 2 (negation transfer — the headline): train on {B*, C2} (tomato word
    only) vs {A*, C1, C3, C4} (both words); test on D1-D3 (BOTH words present,
    tomato commanded). If D classifies as tomato-only -> the representation
    encodes the COMMANDED target (semantic parse, incl. negation). If D
    classifies as both -> it only encodes MENTIONED objects (bag-of-words).
  LOCI: every PaliGemma language-model layer (backbone; instruction rows) AND
    every action-expert layer (action rows) -> backbone-vs-head answer for
    Unnat's frozen-backbone question without training anything.

KNOWN CONFOUND (documented, handled): prompt length / pad-row count correlates
with class for the L0 bag-of-words baseline. The negation transfer is immune:
D prompts are long and mention both objects (like class 0), so classifying
them as tomato-only cannot be explained by words or length.

USAGE (remote, openpi venv, ~15-25 min GPU + ~15 min CPU):
  ~/flare/openpi/.venv/bin/python ~/flare/linear_probe_decoding.py --out-dir ~/flare/results/probe_decode_v2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

FLARE = Path.home() / "flare"
sys.path.insert(0, str(FLARE))

from probe_linear_decodability import HiddenStateCapture  # noqa: E402
from mechanism_probe_attn import set_all_seeds, force_eager_attention  # noqa: E402

# ---------------------------------------------------------------------------
# The 17 probe prompts (verbatim from prompt_sweep_gen.PROMPTS; E excluded).
# label: 1 = tomato-only commanded, 0 = both commanded.
# train_group: "single_word" (tomato word only), "both_word" (both words,
#              both commanded), "negation" (both words, tomato commanded).
# ---------------------------------------------------------------------------
PROBE_PROMPTS = [
    ("A1_reorder_both",   "put both the tomato sauce and the alphabet soup in the basket", 0, "both_word"),
    ("A2_reorder_nob",    "put the tomato sauce and the alphabet soup in the basket",      0, "both_word"),
    ("A3_reorder_can",    "put both the tomato sauce can and the alphabet soup in the basket", 0, "both_word"),
    ("A4_reorder_nothe",  "put both tomato sauce and alphabet soup in the basket",         0, "both_word"),
    ("A5_reorder_passive","the tomato sauce and the alphabet soup should be in the basket",0, "both_word"),
    ("B1_single_put",     "put the tomato sauce in the basket",                            1, "single_word"),
    ("B2_single_place",   "place the tomato sauce in the basket",                          1, "single_word"),
    ("B3_single_pickput", "pick up the tomato sauce and put it in the basket",             1, "single_word"),
    ("B4_single_move",    "move the tomato sauce to the basket",                           1, "single_word"),
    ("B5_single_grab",    "grab the tomato sauce",                                         1, "single_word"),
    ("C1_order_firstthen","first put the tomato sauce in the basket then put the alphabet soup", 0, "both_word"),
    ("C2_order_startwith","start with the tomato sauce",                                   1, "single_word"),
    ("C3_order_goesfirst","tomato sauce goes first then alphabet soup",                    0, "both_word"),
    ("C4_order_before",   "before picking up the alphabet soup put the tomato sauce in the basket", 0, "both_word"),
    ("D1_neg_not",        "put the tomato sauce in the basket not the alphabet soup",      1, "negation"),
    ("D2_neg_avoid",      "avoid the alphabet soup place the tomato sauce in the basket",  1, "negation"),
    ("D3_neg_dontmove",   "the alphabet soup should not be moved put the tomato sauce in the basket", 1, "negation"),
]

# Hook BOTH loci. Group (2) captures depth; locus inferred from the name.
# PaliGemma path verified against audit_lora_attention.py:219 —
# it is paligemma.model.language_model.layers.N (NOT language_model.model.layers).
COMBINED_LAYER_PATTERN = (
    r"^paligemma_with_expert\."
    r"(gemma_expert\.model|paligemma\.model\.language_model)\.layers\.(\d+)$"
)

N_IMG_TOKENS = 768   # prefix layout: 768 image tokens, then BOS + instruction (padded)


def locus_of(name: str) -> str:
    return "expert" if "gemma_expert" in name else "paligemma"


def depth_of(name: str) -> int:
    return int(name.rsplit(".", 1)[-1])


def pool_locus_aware(hs: torch.Tensor, locus: str) -> np.ndarray:
    """(B, S, D) -> (D,). PaliGemma prefix pass: mean over instruction rows
    (768:). Expert denoise pass: mean over the last 50 rows (action queries,
    same convention as eval_steering_bank). Fallback: mean over all rows."""
    hs = hs.to(torch.float32)[0]                       # (S, D)
    if locus == "paligemma" and hs.shape[0] > N_IMG_TOKENS:
        return hs[N_IMG_TOKENS:].mean(dim=0).numpy()
    if locus == "expert" and hs.shape[0] >= 50:
        return hs[-50:].mean(dim=0).numpy()
    return hs.mean(dim=0).numpy()


def grouped_cv_auc(X, y, groups, n_splits, seed=0):
    """GroupKFold CV AUC; folds with single-class test sets are skipped."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    aucs = []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler()
        clf = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear")
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return aucs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-config", default="pi0_libero")
    p.add_argument("--checkpoint-dir", default=str(FLARE / "checkpoints/pi0_libero_pt"))
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--n-inits", type=int, default=10,
                   help="inits per prompt; total infers = 17 * n_inits")
    p.add_argument("--n-perms", type=int, default=200,
                   help="prompt-level permutations for the null (0 = skip)")
    p.add_argument("--cv-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=7000)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print("[v2] importing openpi/libero ...", flush=True)
    from openpi.training import config as _c
    from openpi.policies import policy_config
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from patch_pi0_sde import apply_patch
    from remote_multimode_matrix import format_obs, NUM_STEPS_WAIT, LIBERO_DUMMY_ACTION

    Pi0 = apply_patch()
    cfg = _c.get_config(args.policy_config)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    model = next((getattr(policy, a) for a in ("_model", "model", "_policy", "policy")
                  if hasattr(policy, a) and isinstance(getattr(policy, a), Pi0)), None)
    if model is None:
        raise RuntimeError("Could not find pi0 PyTorch instance on policy")
    for attr, val in [("flare_eta", 0.0), ("flare_verifier_fn", None), ("flare_alpha", 0.0),
                      ("flare_obs_state", None), ("flare_eta_high", None), ("flare_eta_low", None),
                      ("flare_noise_bias_direction", None), ("flare_noise_bias_strength", 0.0)]:
        setattr(model, attr, val)
    model.eval()
    force_eager_attention(model, verbose=False)

    bench = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bench.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = bench.get_task_init_states(args.task_id)
    n_inits = min(args.n_inits, len(init_states))
    layer_re = re.compile(COMBINED_LAYER_PATTERN)

    # ---- CAPTURE: one env per init; all 17 prompts share the same observation ----
    rows = []   # (prompt_name, label, group, init_i, {layer: (D,) np})
    for init_i in range(n_inits):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(args.env_seed + init_i)
        env.reset()
        obs = env.set_init_state(init_states[init_i])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        for name, prompt, label, group in PROBE_PROMPTS:
            set_all_seeds(args.seed)
            with HiddenStateCapture(model, layer_re, verbose=(init_i == 0 and name == "A1_reorder_both")) as cap:
                _ = policy.infer(format_obs(obs, prompt))
            feats = {}
            for lname, calls in cap.items():
                last = next((c for c in reversed(calls) if c is not None), None)
                if last is None:
                    raise RuntimeError(f"no capture for {lname} ({name}, init {init_i})")
                feats[lname] = pool_locus_aware(last, locus_of(lname))
            rows.append((name, label, group, init_i, feats))
        del env
        n_pg = sum(1 for k in rows[-1][4] if locus_of(k) == "paligemma")
        n_ex = sum(1 for k in rows[-1][4] if locus_of(k) == "expert")
        print(f"  init {init_i+1}/{n_inits}: {len(PROBE_PROMPTS)} prompts captured "
              f"({n_pg} paligemma + {n_ex} expert layers)", flush=True)
    if not rows:
        raise RuntimeError("no captures collected")
    if sum(1 for k in rows[0][4] if locus_of(k) == "paligemma") == 0:
        print("[v2] WARNING: 0 PaliGemma layers hooked — backbone locus missing; "
              "check COMBINED_LAYER_PATTERN against --list-layers in probe_linear_decodability.py")

    layers = sorted(rows[0][4].keys(), key=lambda n: (locus_of(n), depth_of(n)))
    y = np.array([r[1] for r in rows])
    groups = np.array([r[0] for r in rows])              # group by PROMPT
    train_groups = np.array([r[2] for r in rows])
    prompt_names = [pp[0] for pp in PROBE_PROMPTS]
    prompt_labels = {pp[0]: pp[2] for pp in PROBE_PROMPTS}
    print(f"[v2] dataset: {len(rows)} samples ({n_inits} inits x {len(PROBE_PROMPTS)} prompts), "
          f"{len(layers)} layers, class balance {np.bincount(y).tolist()}")

    rng = np.random.default_rng(args.seed)
    results = {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    for lname in layers:
        X = np.stack([r[4][lname] for r in rows])

        # --- Analysis 1: grouped CV + prompt-level permutation null ---
        aucs = grouped_cv_auc(X, y, groups, args.cv_folds, args.seed)
        auc_mean = float(np.mean(aucs)) if aucs else float("nan")
        perm_p = None
        if args.n_perms > 0 and aucs:
            null = []
            ones = sum(prompt_labels.values())
            for _ in range(args.n_perms):
                perm_names = rng.permutation(prompt_names)
                lab = {n: (1 if i < ones else 0) for i, n in enumerate(perm_names)}
                y_perm = np.array([lab[g] for g in groups])
                pa = grouped_cv_auc(X, y_perm, groups, args.cv_folds, args.seed)
                null.append(np.mean(pa) if pa else 0.5)
            perm_p = float((1 + sum(1 for v in null if v >= auc_mean)) / (1 + len(null)))

        # --- Analysis 2: negation transfer ---
        tr_mask = np.isin(train_groups, ["single_word", "both_word"])
        te_mask = train_groups == "negation"
        sc = StandardScaler()
        clf = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear")
        clf.fit(sc.fit_transform(X[tr_mask]), y[tr_mask])
        pD = clf.predict_proba(sc.transform(X[te_mask]))[:, 1]
        results[lname] = {
            "locus": locus_of(lname), "depth": depth_of(lname),
            "auc_mean": auc_mean,
            "auc_std": float(np.std(aucs)) if aucs else 0.0,
            "n_folds": len(aucs), "perm_p": perm_p,
            "neg_transfer_p_tomato_mean": float(pD.mean()),
            "neg_transfer_frac_tomato": float((pD > 0.5).mean()),
        }
        d = results[lname]
        print(f"  {d['locus']:>9s} L{d['depth']:2d}: AUC={d['auc_mean']:.3f}"
              f"{'' if perm_p is None else f' (perm p={perm_p:.3f})'}"
              f" | D->tomato-only {100*d['neg_transfer_frac_tomato']:.0f}%"
              f" (mean p={d['neg_transfer_p_tomato_mean']:.2f})", flush=True)

    summary = {
        "design": "commanded-target decode; GroupKFold by prompt; prompt-level permutation null; "
                  "negation transfer (train B/C2 vs A/C1/C3/C4, test D1-3)",
        "n_inits": n_inits, "n_samples": len(rows), "cv_folds": args.cv_folds,
        "n_perms": args.n_perms, "per_layer": results,
    }
    (out / "probe_v2_results.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "probe_v2_data.npz", y=y, groups=groups,
                        train_groups=train_groups,
                        **{f"X__{n}": np.stack([r[4][n] for r in rows]) for n in layers})
    print(f"\n[v2] done -> {out}/probe_v2_results.json")
    print("  KEY: AUC>>0.5 with perm p<0.05 at held-out wordings = commanded target is "
          "linearly decodable. D->tomato-only HIGH in backbone but compliance 0-4% = "
          "information present, unused (freezing preserves what is already preserved). "
          "D->tomato-only LOW everywhere = negation never linearly resolved (upstream deficit).")


if __name__ == "__main__":
    main()
