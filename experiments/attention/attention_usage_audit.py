#!/usr/bin/env python3
"""attention_usage_audit.py — does D-v2 fine-tuning make pi0 ATTEND/READ/USE language?

The completed replacement for the eval_lora_attention.py skeleton. Runs ENTIRELY at
inference (fits a 24GB 4090), reuses the proven AttentionCapture/VelocityCapture
infrastructure, and needs NO JAX->PyTorch converter: openpi's create_trained_policy
loads the LoRA-merged weights transparently from the named config.

It answers three nested questions by comparing baseline pi0_libero vs the LoRA
variants, in a 2x2 (model x BOS-mask) factorial:

  L1 ATTEND  — per-layer attention mass on the instruction tokens (did it rise?)
  L2 READ    — which instruction tokens? named-object SELECTIVITY:
                 attn(tomato-tokens | "...tomato...") - attn(tomato-tokens | "...alphabet...")
                 baseline ~ 0 (prompt-invariant bypass); a real fix makes this > 0
  L3 USE     — velocity divergence between prompts (cos on the real-7 robot dims):
                 baseline ~ 0.999 (output ignores language); a real fix lowers it

It ALSO verifies the BOS mask actually fired (mask-on => action->BOS attention ~ 0),
the load-bearing assertion that gates any D-lite claim.

CONDITIONS (the 2x2 + CF-only), default checkpoints from the run_configs:
  baseline    pi0_libero                    ~/flare/checkpoints/pi0_libero_pt          mask=0
  C_maskoff   pi0_libero_cf_lora            .../variant_c_v2/1499                       mask=0
  Dv2_maskoff pi0_libero_cf_lora_bos_masked .../variant_d_v2/1499                       mask=0   <- THE KEY CELL
  Dv2_maskon  pi0_libero_cf_lora_bos_masked .../variant_d_v2/1499                       mask=1   <- train-matched

USAGE (on the 4090 / remote where openpi + checkpoints live):
  python3 attention_usage_audit.py --out-dir ~/flare/results/lora_attention_audit
  # add --conditions / --prompts / --n-init-states to customize; see --help.

Capture is on-GPU; the saved JSON + figures can be re-analyzed anywhere.
"""
from __future__ import annotations

import argparse, json, os, sys, time, re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

FLARE = Path.home() / "flare"
sys.path.insert(0, str(FLARE))

# Key layout for the action expert's 867 keys (verified: img 768 + lang 48 + state 1 + act 50).
N_IMG = 768
KEY_BOS = 768               # first language position under right-padding = <bos> sink
KEY_IMG_CORNER = 303        # secondary image-patch sink
LANG_START, LANG_END = 769, 816   # instruction content tokens [769, 816)
ACT_SELF_START = 816        # state + action self-attention keys [816, 867)
N_DENOISE_REAL_DIMS = 7     # the 7 real LIBERO action dims (rest are constant padding)

# Object name -> keyword stems used to find its instruction-token positions.
OBJECT_KEYWORDS = {
    "tomato_sauce": ["tomato", "sauce"],
    "alphabet_soup": ["alphabet", "soup"],
    "milk": ["milk"],
    "cream_cheese": ["cream", "cheese"],
    "butter": ["butter"],
    "orange_juice": ["orange", "juice"],
}


# --------------------------------------------------------------------------- #
# Velocity capture (patches model.denoise_step) — faithful to probe_velocity.   #
# --------------------------------------------------------------------------- #
class VelocityCapture:
    """Capture v_t at each denoise step during policy.infer()."""
    def __init__(self, model):
        self.model = model
        self.captured: List = []
        self._orig = None
        self._had = False

    def __enter__(self):
        import torch
        if not hasattr(self.model, "denoise_step"):
            raise RuntimeError(f"{type(self.model).__name__} has no denoise_step")
        self._orig = self.model.denoise_step
        self._had = "denoise_step" in self.model.__dict__
        cap = self.captured
        orig = self._orig

        def wrapped(*a, **k):
            v = orig(*a, **k)
            if torch.is_tensor(v):
                cap.append(v.detach().to("cpu").float())
            return v
        self.model.denoise_step = wrapped
        return self.captured

    def __exit__(self, *_):
        if self._had:
            self.model.denoise_step = self._orig
        elif "denoise_step" in self.model.__dict__:
            del self.model.__dict__["denoise_step"]


# --------------------------------------------------------------------------- #
# Model loading (reuse the proven mechanism_probe path).                        #
# --------------------------------------------------------------------------- #
def load_model(config_name: str, checkpoint_dir: str):
    """Return (policy, pytorch_model) or (policy, None) if backend is JAX-only."""
    from patch_pi0_sde import apply_patch
    Pi0 = apply_patch()
    from openpi.training import config as _c
    from openpi.policies import policy_config
    from mechanism_probe_attn import force_eager_attention

    cfg = _c.get_config(config_name)
    print(f"[load] config={config_name} ckpt={checkpoint_dir}", flush=True)
    policy = policy_config.create_trained_policy(cfg, str(checkpoint_dir))

    model = None
    for attr in ("_model", "model", "_policy", "policy"):
        if hasattr(policy, attr) and isinstance(getattr(policy, attr), Pi0):
            model = getattr(policy, attr)
            break
    if model is None:
        print("[load] WARNING: no PyTorch Pi0 model found (JAX backend?). "
              "Attention hooks need PyTorch — this condition will be skipped.", flush=True)
        return policy, None

    # Pure ODE, no SDE/guidance side effects, eager attention so weights exist.
    for a, v in dict(flare_eta=0.0, flare_alpha=0.0, flare_verifier_fn=None,
                     flare_obs_state=None, flare_eta_high=None, flare_eta_low=None,
                     flare_noise_bias_direction=None, flare_noise_bias_strength=0.0).items():
        if hasattr(model, a):
            setattr(model, a, v)
    model.eval()
    force_eager_attention(model, verbose=False)
    return policy, model


# --------------------------------------------------------------------------- #
# Tokenizer: instruction position -> decoded word (openpi REAL tokenizer).      #
# --------------------------------------------------------------------------- #
def tokenize_positions(prompt: str, lang_len: int = 48) -> Dict[int, str]:
    """{absolute key position -> decoded token} for instruction tokens, BOS at 768."""
    labels: Dict[int, str] = {}
    try:
        for path in ("openpi.models.tokenizer", "openpi.shared.tokenizer"):
            try:
                mod = __import__(path, fromlist=["PaligemmaTokenizer"])
                PT = getattr(mod, "PaligemmaTokenizer")
                break
            except Exception:
                PT = None
        if PT is None:
            return labels
        try:
            tk = PT(lang_len)
        except TypeError:
            tk = PT(max_len=lang_len)
        toks = tk.tokenize(prompt)
        ids = toks[0] if isinstance(toks, tuple) else toks
        inner = getattr(tk, "_tokenizer", None)
        for i, tid in enumerate(np.asarray(ids).reshape(-1)):
            pos = N_IMG + i
            dec = inner.decode([int(tid)]) if inner is not None else str(int(tid))
            labels[pos] = dec
    except Exception as e:
        print(f"[tok] WARNING: tokenizer mapping failed: {e}", flush=True)
    return labels


def object_positions(labels: Dict[int, str], obj: str) -> List[int]:
    kws = OBJECT_KEYWORDS.get(obj, [obj])
    out = []
    for pos, tok in labels.items():
        t = tok.strip().lower()
        if any(kw in t for kw in kws):
            out.append(pos)
    return out


# --------------------------------------------------------------------------- #
# Attention aggregation.                                                         #
# --------------------------------------------------------------------------- #
def expert_layer_profiles(captured: Dict[str, list]) -> Dict[int, np.ndarray]:
    """Per expert layer -> mean attention profile over (denoise steps, heads, action queries).

    Returns {layer_idx: np.array shape [867]}.  Uses ONLY the suffix (action) query rows.
    """
    import torch
    out = {}
    for name, lst in captured.items():
        m = re.search(r"gemma_expert\.model\.layers\.(\d+)\.self_attn$", name)
        if not m:
            continue
        L = int(m.group(1))
        # keep only the denoise calls (suffix queries Q=51, full key set K=867), not the prefix call
        profs = []
        for a in lst:
            if a is None or a.dim() != 4:
                continue
            B, H, Q, K = a.shape
            if Q < 51 or K < ACT_SELF_START + 1:   # gate on the suffix query length (intent), and full keys
                continue
            assert B == 1, f"expected batch 1, got {B}"
            # average over heads, then over the ACTION query rows ONLY (drop the leading state query,
            # matching visualize_attention_verified.py's [-ACT_LEN:] / the E0 reference).
            profs.append(a[0].mean(0)[-50:].mean(0).float().numpy())  # [K]
        if profs:
            out[L] = np.mean(np.stack(profs, 0), 0)
    return out


def tower_layer_profiles(captured: Dict[str, list]) -> Dict[int, np.ndarray]:
    """Per PaliGemma language-tower (BACKBONE) layer -> mean attention profile over heads & queries.

    The tower is the prefix self-attention [1,H,816,816] (768 image + 48 language keys; BOS=768,
    instruction 769..815 — same prefix layout as the expert). This measures whether the BACKBONE
    *processes* the instruction (language UNDERSTANDING), to contrast with the expert's *use* of it.
    Returns {layer_idx: np.array shape [K_prefix]} (region_masses works: action_self comes out 0).
    """
    out = {}
    for name, lst in captured.items():
        m = re.search(r"paligemma\.model\.language_model\.layers\.(\d+)\.self_attn$", name)
        if not m:
            continue
        L = int(m.group(1))
        profs = []
        for a in lst:
            if a is None or a.dim() != 4:
                continue
            B, H, Q, K = a.shape
            if K < LANG_END:        # need the full prefix (image + 48 language keys)
                continue
            assert B == 1, f"expected batch 1, got {B}"
            # average over heads, then over the LANGUAGE query rows only (the last 48 prefix tokens =
            # image..., then 48 language). Averaging over all 816 (94% image) queries dilutes the
            # backbone's language processing — restricting to language queries is the real signal.
            profs.append(a[0].mean(0)[-48:].mean(0).float().numpy())   # [K]
        if profs:
            out[L] = np.mean(np.stack(profs, 0), 0)
    return out


def region_masses(profile: np.ndarray) -> Dict[str, float]:
    img = float(profile[0:N_IMG].sum() - profile[KEY_IMG_CORNER])
    return dict(
        image=img,
        corner=float(profile[KEY_IMG_CORNER]),
        bos=float(profile[KEY_BOS]),
        instruction=float(profile[LANG_START:LANG_END].sum()),
        action_self=float(profile[ACT_SELF_START:].sum()),
    )


def instruction_token_attention(profile: np.ndarray, labels: Dict[int, str]) -> Dict[str, float]:
    """{token_label@pos -> attention} for non-pad instruction tokens."""
    out = {}
    for pos in range(LANG_START - 1, LANG_END):   # include BOS@768 too
        tok = labels.get(pos, "")
        t = tok.strip()
        if t in ("", "<pad>", "<eos>") or t.startswith("\x00"):
            continue
        out[f"{pos}:{t}"] = float(profile[pos])
    return out


# --------------------------------------------------------------------------- #
# Per-condition run.                                                            #
# --------------------------------------------------------------------------- #
def run_condition(label, config_name, checkpoint, mask_bos, prompts, env_pack,
                  seed, out_dir) -> Optional[dict]:
    import torch
    os.environ["OPENPI_MASK_BOS"] = "1" if mask_bos else "0"
    print(f"\n==== CONDITION {label}  (OPENPI_MASK_BOS={os.environ['OPENPI_MASK_BOS']}) ====",
          flush=True)
    policy, model = load_model(config_name, checkpoint)
    if model is None:
        return None

    from mechanism_probe_attn import AttentionCapture, set_all_seeds
    res = {"label": label, "config": config_name, "checkpoint": str(checkpoint),
           "mask_bos": bool(mask_bos), "prompts": {}}

    for plabel, prompt in prompts:
        obs = env_pack["format_obs"](env_pack["obs"], prompt)
        set_all_seeds(seed)
        os.environ["OPENPI_MASK_BOS"] = "1" if mask_bos else "0"   # re-assert before infer
        with AttentionCapture(model, verbose=False) as cap, VelocityCapture(model) as vels:
            _ = policy.infer(obs)
        profiles = expert_layer_profiles(cap)
        tower = tower_layer_profiles(cap)          # BACKBONE (PaliGemma) — language understanding
        labels = tokenize_positions(prompt)
        # velocity: keep the FULL per-step field [50,32] (mean over batch) so L3 can compute a
        # per-step cosine on real-7 dims (matching eval_lora_velocity.py) — NOT a horizon-mean.
        vsteps = [v.mean(0).float().numpy().tolist() for v in vels] if vels else None  # list of [50,32]
        # normalized share of instruction attention on each object's noun tokens (for L2 READ)
        instr_mass = float(profiles[17][LANG_START:LANG_END].sum()) if 17 in profiles else float("nan")
        obj_named = {}
        if 17 in profiles:
            for obj in OBJECT_KEYWORDS:
                pos = object_positions(labels, obj)
                obj_named[obj] = {"abs": float(sum(profiles[17][p] for p in pos)),
                                  "n_tokens": len(pos)}
        res["prompts"][plabel] = {
            "prompt": prompt,
            "per_layer_region": {str(L): region_masses(p) for L, p in profiles.items()},
            "L17_instruction_tokens": (
                instruction_token_attention(profiles[17], labels) if 17 in profiles else {}),
            "object_token_attention_L17": obj_named,   # {obj: {abs, n_tokens}}
            "L17_instruction_mass": instr_mass,
            "velocity_steps": vsteps,                   # [n_steps][50][32]  (None if capture failed)
            # BACKBONE (tower) instruction attention per layer — does the VLM still UNDERSTAND language?
            "backbone_per_layer_region": {str(L): region_masses(p) for L, p in tower.items()},
        }
        instr = res["prompts"][plabel]["per_layer_region"].get("17", {}).get("instruction", float("nan"))
        bos = res["prompts"][plabel]["per_layer_region"].get("17", {}).get("bos", float("nan"))
        # backbone instruction mass at the tower's last layer (max layer present)
        bb = res["prompts"][plabel]["backbone_per_layer_region"]
        bb_instr = bb[str(max(int(k) for k in bb))]["instruction"] if bb else float("nan")
        print(f"  [{plabel}] HEAD(expert) L17 instr={instr:.4f} bos={bos:.4f} | BACKBONE(tower) instr={bb_instr:.4f}",
              flush=True)

    # free GPU before next condition
    del policy, model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return res


# --------------------------------------------------------------------------- #
# Analysis + figures (pure numpy/matplotlib; re-runnable offline).              #
# --------------------------------------------------------------------------- #
def analyze(results: List[dict], out_dir: Path,
            pair=("T0_tomato_only", "T0_alphabet_only")):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    by = {r["label"]: r for r in results}
    summary = {}

    # ---- L1 ATTEND + 2x2 mask check: L17 instruction & BOS mass per condition ----
    rows = []
    for lab, r in by.items():
        p0 = r["prompts"].get(pair[0]) or next(iter(r["prompts"].values()))
        reg = p0["per_layer_region"].get("17", {})
        rows.append((lab, reg.get("instruction", np.nan), reg.get("bos", np.nan), r["mask_bos"]))
    summary["L17_instruction_and_bos"] = [
        {"condition": l, "instruction": i, "bos": b, "mask_bos": m} for (l, i, b, m) in rows]

    # mask-fired verification: mask-on conditions should show L17 BOS ≈ 0
    summary["mask_fired_check"] = {
        l: {"L17_bos": b, "mask_on": m, "fired": (m and b < 0.01)} for (l, i, b, m) in rows}

    # Figure A: per-layer instruction mass, one line per condition
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for lab, r in by.items():
        p0 = r["prompts"].get(pair[0]) or next(iter(r["prompts"].values()))
        Ls = sorted(int(L) for L in p0["per_layer_region"])
        ys = [p0["per_layer_region"][str(L)]["instruction"] * 100 for L in Ls]
        ax.plot(Ls, ys, "-o", ms=3, label=lab)
    ax.set_xlabel("action-expert layer"); ax.set_ylabel("instruction attention mass (%)")
    ax.set_title(f"L1 ATTEND — instruction attention per layer  (prompt: {pair[0]})")
    ax.legend(fontsize=8); ax.axhspan(0, 3.7, color="red", alpha=0.05)
    fig.tight_layout(); fig.savefig(out_dir / "fig_attend_per_layer.png", dpi=150); plt.close(fig)

    # ---- BACKBONE vs HEAD: does the VLM UNDERSTAND language while the head fails to USE it? ----
    # Headline of the gradient-flow / induction story (base vs finetuned vs knowledge-insulated):
    #   backbone (tower) instruction mass = understanding ; head (expert) = usage.
    bvh = {}
    for lab, r in by.items():
        p0 = r["prompts"].get(pair[0]) or next(iter(r["prompts"].values()))
        head_reg = p0.get("per_layer_region", {})
        bb_reg = p0.get("backbone_per_layer_region", {})
        # use each tower/expert's deepest layer (max idx present)
        head_max = str(max((int(k) for k in head_reg), default=17))
        bb_max = str(max((int(k) for k in bb_reg), default=0)) if bb_reg else None
        bvh[lab] = {
            "head_instruction_mass": head_reg.get(head_max, {}).get("instruction", float("nan")),
            "backbone_instruction_mass": (bb_reg.get(bb_max, {}).get("instruction", float("nan"))
                                          if bb_max is not None else float("nan")),
            "head_bos": head_reg.get(head_max, {}).get("bos", float("nan")),
            "backbone_bos": (bb_reg.get(bb_max, {}).get("bos", float("nan")) if bb_max else float("nan")),
        }
    summary["backbone_vs_head"] = bvh
    if bvh:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        labs = list(bvh); x = np.arange(len(labs)); w = 0.38
        ax.bar(x - w / 2, [bvh[l]["backbone_instruction_mass"] * 100 for l in labs], w,
               color="#1f77b4", label="BACKBONE (PaliGemma tower) — understanding")
        ax.bar(x + w / 2, [bvh[l]["head_instruction_mass"] * 100 for l in labs], w,
               color="#d62728", label="HEAD (action expert) — usage")
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("instruction attention mass (%), deepest layer")
        ax.set_title("Backbone vs Head — where does fine-tuning corrupt language?\n"
                     "(backbone high + head low = understands but doesn't use → head-side bypass)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / "fig_backbone_vs_head.png", dpi=150); plt.close(fig)

    # Figure B: 2x2 mask check — L17 instruction & BOS
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labs = list(by); x = np.arange(len(labs))
    instr = [next(i for (l, i, b, m) in rows if l == lab) * 100 for lab in labs]
    bos = [next(b for (l, i, b, m) in rows if l == lab) * 100 for lab in labs]
    ax.bar(x - 0.2, instr, 0.38, color="#d62728", label="L17 instruction mass")
    ax.bar(x + 0.2, bos, 0.38, color="#9467bd", label="L17 BOS-sink mass")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("% of L17 attention"); ax.legend(fontsize=8)
    ax.set_title("2x2 — did weights learn to read language, and did the BOS mask fire?\n"
                 "(mask-on conditions should show BOS≈0; ATTEND-fix => instruction rises mask-OFF)")
    fig.tight_layout(); fig.savefig(out_dir / "fig_2x2_maskcheck.png", dpi=150); plt.close(fig)

    # ---- L2 READ: attention ON the named object's noun tokens, per (model, prompt) ----
    # Well-defined (the named token always exists in its OWN prompt); the contrast that matters is
    # ACROSS MODELS (baseline vs C vs D) for the same prompt — a real fix attends MORE to the noun.
    # named share = attn on the named object's tokens / total instruction attention.
    named = {pl: {"object": obj} for pl, obj in [(pair[0], "tomato_sauce"), (pair[1], "alphabet_soup")]}
    read = {}
    for lab, r in by.items():
        read[lab] = {}
        for pl, meta in named.items():
            p = r["prompts"].get(pl)
            if not p:
                continue
            ot = p.get("object_token_attention_L17", {}).get(meta["object"], {})
            imass = p.get("L17_instruction_mass", float("nan"))
            absn = ot.get("abs", float("nan")); ntok = ot.get("n_tokens", 0)
            read[lab][pl] = {"named_object": meta["object"], "abs_attn": absn, "n_tokens": ntok,
                             "instr_mass": imass,
                             "share_of_instruction": (absn / imass) if imass and imass > 0 else float("nan"),
                             "valid": ntok > 0}
    summary["L2_named_object_read"] = read
    pls = [pair[0], pair[1]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labs2 = list(read); x = np.arange(len(labs2)); wdt = 0.38
    for i, pl in enumerate(pls):
        vals = [read[l].get(pl, {}).get("abs_attn", np.nan) * 100 for l in labs2]
        ax.bar(x + (i - 0.5) * wdt, vals, wdt, label=f"{pl} (names {named[pl]['object']})")
    ax.set_xticks(x); ax.set_xticklabels(labs2, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("L17 attention on the NAMED object's tokens (%)")
    ax.set_title("L2 READ — attention on the named object's noun tokens, per model\n"
                 "(a real fix attends MORE than baseline; compare across models, same prompt)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig_read_named_object.png", dpi=150); plt.close(fig)

    # ---- L3 USE: per-step velocity cosine on real-7 dims (matches eval_lora_velocity) ----
    def _cos(a, b):
        a, b = a.ravel(), b.ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    use = {}
    R = N_DENOISE_REAL_DIMS
    for lab, r in by.items():
        pa, pb = r["prompts"].get(pair[0]), r["prompts"].get(pair[1])
        if not (pa and pb and pa.get("velocity_steps") and pb.get("velocity_steps")):
            continue
        cr, ce = [], []
        for sa, sb in zip(pa["velocity_steps"], pb["velocity_steps"]):
            A, B = np.asarray(sa), np.asarray(sb)            # [50, 32]
            cr.append(_cos(A[..., :R], B[..., :R]))          # full chunk, real-7
            ce.append(_cos(A[:5, :R], B[:5, :R]))            # executed chunk (first 5 steps), real-7
        use[lab] = {"cos_real7": float(np.mean(cr)), "cos_exec0_5_real7": float(np.mean(ce)),
                    "cos_per_step_real7": cr}
    summary["L3_velocity_cos_between_prompts"] = use
    if use:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        labs3 = list(use); x = np.arange(len(labs3))
        ax.bar(x, [use[l]["cos_real7"] for l in labs3], color="#1f77b4")
        ax.set_ylim(0.95, 1.001)
        ax.set_xticks(x); ax.set_xticklabels(labs3, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("per-step velocity cos between prompts (real-7)")
        ax.set_title("L3 USE — does the ACTION change with the prompt?\n"
                     "(baseline ≈ 0.999 = ignores language; lower = output became prompt-conditional)")
        fig.tight_layout(); fig.savefig(out_dir / "fig_use_velocity.png", dpi=150); plt.close(fig)

    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2)[:2500])
    return summary


# --------------------------------------------------------------------------- #
def setup_env(task_suite, task_id, env_seed, init_idx):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from remote_multimode_matrix import format_obs, NUM_STEPS_WAIT, LIBERO_DUMMY_ACTION
    bench = benchmark.get_benchmark_dict()[task_suite]()
    task = bench.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = bench.get_task_init_states(task_id)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(env_seed); env.reset()
    obs = env.set_init_state(init_states[init_idx])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return {"obs": obs, "format_obs": format_obs}


DEFAULT_CONDITIONS = [
    # label, config_name, checkpoint, mask_bos
    ("baseline", "pi0_libero", str(FLARE / "checkpoints/pi0_libero_pt"), 0),
    ("C_maskoff", "pi0_libero_cf_lora",
     str(FLARE / "openpi/checkpoints/pi0_libero_cf_lora/variant_c_v2/1499"), 0),
    ("Dv2_maskoff", "pi0_libero_cf_lora_bos_masked",
     str(FLARE / "openpi/checkpoints/pi0_libero_cf_lora_bos_masked/variant_d_v2/1499"), 0),
    ("Dv2_maskon", "pi0_libero_cf_lora_bos_masked",
     str(FLARE / "openpi/checkpoints/pi0_libero_cf_lora_bos_masked/variant_d_v2/1499"), 1),
]
DEFAULT_PROMPTS = [
    ("T0_tomato_only", "pick up the tomato sauce and put it in the basket"),
    ("T0_alphabet_only", "pick up the alphabet soup and put it in the basket"),
    ("T0_milk_heldout", "pick up the milk and put it in the basket"),   # held-out object
    ("T0_canonical", "put both the alphabet soup and the tomato sauce in the basket"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=7000)
    p.add_argument("--init-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conditions", default=None,
                   help="JSON file: [[label,config,checkpoint,mask_bos],...]; "
                        "default = baseline/C/Dv2(off)/Dv2(on)")
    p.add_argument("--preset", default=None, choices=["pt"],
                   help="Built-in conditions (avoids pasting a JSON). 'pt' = "
                        "baseline pi0_libero_pt + C_pt + Dv2_pt under config pi0_libero.")
    p.add_argument("--prompts", default=None,
                   help="JSON file: [[label,prompt],...]; default = tomato/alphabet/milk/canonical")
    p.add_argument("--analyze-only", default=None,
                   help="Path to an existing audit_raw.json to re-run analysis+figures offline")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        results = json.load(open(args.analyze_only))
        analyze(results, out_dir)
        return

    PT_CONDITIONS = [
        ("baseline", "pi0_libero", str(FLARE / "checkpoints/pi0_libero_pt"), 0),
        ("C_pt",     "pi0_libero", str(FLARE / "checkpoints/variant_c_v2_pt"), 0),
        ("Dv2_pt",   "pi0_libero", str(FLARE / "checkpoints/variant_d_v2_pt"), 0),
    ]
    if args.preset == "pt":
        conditions = PT_CONDITIONS
    elif args.conditions:
        conditions = json.load(open(args.conditions))
    else:
        conditions = DEFAULT_CONDITIONS
    prompts = json.load(open(args.prompts)) if args.prompts else DEFAULT_PROMPTS

    print("[audit] setting up env ...", flush=True)
    env_pack = setup_env(args.task_suite, args.task_id, args.env_seed, args.init_idx)

    results = []
    for (label, config_name, checkpoint, mask_bos) in conditions:
        try:
            r = run_condition(label, config_name, checkpoint, int(mask_bos),
                              prompts, env_pack, args.seed, out_dir)
            if r is not None:
                results.append(r)
                # checkpoint raw after each condition (so a late crash doesn't lose work)
                (out_dir / "audit_raw.json").write_text(json.dumps(results, indent=2))
        except Exception as e:
            print(f"[audit] CONDITION {label} FAILED: {type(e).__name__}: {e}", flush=True)

    if results:
        analyze(results, out_dir)
    print(f"\n[audit] done -> {out_dir}")


if __name__ == "__main__":
    main()
