"""adapter_behavioral_eval.py — comprehensive behavioral evaluation of a LoRA-trained pi0 checkpoint.

Runs 6 evaluation prompts × N trials on T0 scene, records first-moved object,
success rate, and step count. Saves per-trial JSON + headline metrics + bar
chart visualizations + sample rollout videos.

================================================================================
USAGE
================================================================================

python3 adapter_behavioral_eval.py \\
    --checkpoint ~/flare/openpi/checkpoints/pi0_libero_cf_lora/variant_c_natural_cf_lora_v1/2999 \\
    --config-name pi0_libero_cf_lora \\
    --variant-label variant_c_natural_cf \\
    --out-dir ~/flare/results/lora_ablation/variant_c \\
    --n-trials 15 \\
    --task-suite libero_10 --task-id 0

================================================================================
WHAT IT TESTS
================================================================================

6 evaluation prompts on T0 (LIVING_ROOM_SCENE2):

  T0_canonical:     "put both the alphabet soup and the tomato sauce in the basket"
                    (Baseline preservation: should stay alphabet-first 80%+)

  T0_reordered:     "put both the tomato sauce and the alphabet soup in the basket"
                    (Word-order test: if model parses order → tomato-first;
                     if rote → alphabet-first)

  T0_tomato_only:   "pick up the tomato sauce and put it in the basket"
                    (HEADLINE: does LoRA make this prompt produce tomato-first?
                     Baseline pi0_libero ≈ 0% tomato. Target: 50%+.)

  T0_alphabet_only: "pick up the alphabet soup and put it in the basket"
                    (Sanity: trained prompt for the default object → alphabet-first)

  T0_milk_test:     "pick up the milk and put it in the basket"
                    (Tests transfer to other CF training prompts)

  T0_irrelevant:    "stack the mugs"
                    (Sanity: irrelevant prompt should produce default behavior or chaos)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


EVAL_PROMPTS = [
    ("T0_canonical",     "put both the alphabet soup and the tomato sauce in the basket"),
    ("T0_reordered",     "put both the tomato sauce and the alphabet soup in the basket"),
    ("T0_tomato_only",   "pick up the tomato sauce and put it in the basket"),
    ("T0_alphabet_only", "pick up the alphabet soup and put it in the basket"),
    ("T0_milk_test",     "pick up the milk and put it in the basket"),
    ("T0_irrelevant",    "stack the mugs"),
]


def load_lora_policy(config_name: str, checkpoint_dir: Path):
    """Load a LoRA-trained pi0 policy via openpi's policy_config.

    For LoRA-trained models, openpi handles the LoRA-merged weight loading
    transparently — the Pi0Config in the named config specifies gemma_*_lora
    variants, and the checkpoint dir contains the merged params.
    """
    print(f"[load] config={config_name!r}, checkpoint={checkpoint_dir}", flush=True)

    from openpi.training import config as _c
    from openpi.policies import policy_config

    cfg = _c.get_config(config_name)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {checkpoint_dir}")

    policy = policy_config.create_trained_policy(cfg, str(checkpoint_dir))
    print(f"[load] policy loaded successfully", flush=True)
    return policy


def setup_env(task_suite: str, task_id: int, video_dir: Optional[Path] = None):
    """Create a LIBERO env for the given task."""
    print(f"[env] setting up suite={task_suite} task_id={task_id}", flush=True)
    from libero.libero import benchmark, get_libero_path

    sys.path.insert(0, str(Path.home() / "flare"))
    from patch_pi0_sde import apply_patch  # noqa
    apply_patch()  # ensure SDE patch is installed for any flare_* attrs

    import remote_multimode_matrix as mmm  # noqa

    bench = benchmark.get_benchmark_dict()[task_suite]()
    task = bench.get_task(task_id)
    bddl = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    init_states = bench.get_task_init_states(task_id)
    return {
        "task": task,
        "task_id": task_id,
        "bddl": bddl,
        "init_states": init_states,
        "n_inits": len(init_states),
        "mmm": mmm,
    }


def run_trial(policy, model, env_pack, init_state, env_seed,
               prompt, replan_steps=5, base_seed=42,
               save_video_path: Optional[Path] = None):
    """Run a single rollout with the given prompt; return outcome dict."""
    mmm = env_pack["mmm"]
    cond_ode = {"name": "ode_single", "eta": 0.0, "K": 1,
                "guidance": False, "alpha": 0.0}

    # Zero out flare_* attrs so the patched sample_actions runs ODE-only
    if model is not None:
        model.flare_eta = 0.0
        model.flare_verifier_fn = None
        model.flare_alpha = 0.0
        model.flare_obs_state = None
        model.flare_eta_high = None
        model.flare_eta_low = None
        model.flare_noise_bias_direction = None
        model.flare_noise_bias_strength = 0.0
        model.eval() if hasattr(model, "eval") else None

    video_mode = "agentview" if save_video_path else "off"
    rollout_result = mmm.rollout(
        policy=policy, model=model, task=env_pack["task"],
        task_id=env_pack["task_id"], bddl=env_pack["bddl"],
        init_state=init_state, env_seed=env_seed,
        base_seed=base_seed, cond=cond_ode,
        scorer_fn=None, guidance_callable=None,
        obs_state_for_guidance=None,
        replan_steps=replan_steps,
        video_mode=video_mode,
        language_override=prompt,
    )

    # mmm.rollout returns 8-tuple:
    #   (done, term, t, init_pos, final_pos, waypoints, moved_objects, frames)
    # Previously this line had `video_frames` and `_` swapped, so we were saving
    # waypoint DICTS as video frames (hence the `image must have at least 2
    # spatial dimensions` error). Fixed: frames is the LAST element.
    done, term, n_steps, init_pos, final_pos, _waypoints, moved, video_frames = rollout_result

    # Save video if requested — robust with shape inspection + GIF fallback + PNG safety net
    if save_video_path and video_frames is not None and len(video_frames) > 0:
        _save_rollout_video(save_video_path, video_frames)

    first_moved = moved[0]["name"] if moved else "<none>"
    return {
        "success": bool(done),
        "terminated": bool(term),
        "n_steps": int(n_steps),
        "first_moved": first_moved,
    }


def _save_rollout_video(save_video_path, video_frames):
    """Save a list of frames as MP4 (preferred) / GIF (fallback) / PNG (safety net).

    Diagnoses common failure modes:
      - frames are 1D arrays (env returned state vector instead of image)
      - frames are inconsistent shapes (env switched modes mid-rollout)
      - imageio mp4 encoder needs macro_block_size=1 for non-multiple-of-16 dims
    Always saves at least the first valid frame as a PNG, so we never come away
    with NOTHING for a successful rollout.
    """
    import imageio
    import numpy as np

    save_video_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Inspect + normalize frames ----
    valid = []
    skipped_shapes = {}
    for i, f in enumerate(video_frames):
        arr = np.asarray(f)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            # H × W × {3,4} — RGB(A), keep as uint8
            valid.append(arr.astype(np.uint8))
        elif arr.ndim == 2:
            # grayscale H × W — stack to RGB
            valid.append(np.stack([arr] * 3, axis=-1).astype(np.uint8))
        elif arr.ndim == 1 and arr.size in (256 * 256 * 3, 224 * 224 * 3, 128 * 128 * 3):
            # 1D but length matches an expected HW*3 → reshape and recover
            side = int((arr.size // 3) ** 0.5)
            valid.append(arr.reshape(side, side, 3).astype(np.uint8))
        else:
            skipped_shapes[str(arr.shape)] = skipped_shapes.get(str(arr.shape), 0) + 1

    if skipped_shapes:
        print(f"  [video] skipped frames with bad shapes: {skipped_shapes}")

    if not valid:
        print(f"  [video] no valid frames — nothing to save (got {len(video_frames)} input frames)")
        # Try to dump the first raw frame for diagnostic
        try:
            f0 = np.asarray(video_frames[0])
            diag_path = save_video_path.with_suffix(".diag.txt")
            diag_path.write_text(f"first frame: shape={f0.shape} dtype={f0.dtype} "
                                  f"min={f0.min()} max={f0.max()} sample={f0.flat[:10].tolist()}")
            print(f"  [video] diagnostic written: {diag_path}")
        except Exception:
            pass
        return

    # ---- 2. Always save first + last frame as PNG (safety net) ----
    png_first = save_video_path.with_suffix(".first.png")
    png_last = save_video_path.with_suffix(".last.png")
    try:
        imageio.imwrite(str(png_first), valid[0])
        imageio.imwrite(str(png_last), valid[-1])
    except Exception as e:
        print(f"  [video] PNG safety net failed: {e}")

    # ---- 3. Try MP4 (with macro_block_size=1 to handle non-16-multiple dims) ----
    try:
        imageio.mimsave(str(save_video_path), valid, fps=20, macro_block_size=1)
        return
    except Exception as e_mp4:
        print(f"  [video] mp4 save failed ({e_mp4}); trying GIF...")

    # ---- 4. Fallback to GIF (more permissive) ----
    try:
        gif_path = save_video_path.with_suffix(".gif")
        imageio.mimsave(str(gif_path), valid, fps=20)
        print(f"  [video] saved GIF: {gif_path.name}")
        return
    except Exception as e_gif:
        print(f"  [video] gif save failed too ({e_gif}); kept first/last PNGs only")


def find_model_attr(policy):
    """Locate the underlying model object on the policy (for setting flare_* attrs)."""
    from patch_pi0_sde import apply_patch
    Pi0 = apply_patch()
    for attr in ("_model", "model", "_policy", "policy"):
        if hasattr(policy, attr):
            obj = getattr(policy, attr)
            if isinstance(obj, Pi0):
                return obj
    # Not all backends expose a pytorch model; that's OK for JAX inference
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to LoRA-trained checkpoint dir (e.g. .../variant_c_v1/2999)")
    p.add_argument("--config-name", default="pi0_libero_cf_lora",
                   help="openpi config name (must match training config)")
    p.add_argument("--variant-label", required=True,
                   help="Label for this variant (e.g. 'variant_c_natural_cf')")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for results")
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--save-videos", action="store_true",
                   help="Save rollout videos (see --video-trials for how many per prompt)")
    p.add_argument("--video-trials", type=int, default=1000,
                   help="Save videos for the first N trials PER PROMPT (default: all). "
                        "Set lower (e.g. 3) to limit disk usage.")
    p.add_argument("--prompts", default=None,
                   help="Comma-separated prompt labels to run (default: all 6). "
                        "E.g. 'T0_tomato_only,T0_canonical' for a fast targeted re-run.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=7000)
    p.add_argument("--wandb-run-id", default=None,
                   help="If set, log eval metrics to this wandb run (resume='allow')")
    p.add_argument("--wandb-project", default="flare_cf_lora")
    p.add_argument("--training-step", type=int, default=None,
                   help="If set, log eval metrics at this wandb step")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    if args.save_videos:
        (out_dir / "videos").mkdir(exist_ok=True)

    # Save run config
    run_config = {
        "variant_label": args.variant_label,
        "config_name": args.config_name,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "n_trials": args.n_trials,
        "replan_steps": args.replan_steps,
        "seed": args.seed,
        "env_seed": args.env_seed,
        "eval_prompts": EVAL_PROMPTS,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print(f"\n=== EVAL: {args.variant_label} ===")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  out: {out_dir}")
    print(f"  prompts: {len(EVAL_PROMPTS)} × trials: {args.n_trials}")
    print(f"  total rollouts: {len(EVAL_PROMPTS) * args.n_trials}")
    print(f"  estimated time: ~{len(EVAL_PROMPTS) * args.n_trials * 30 / 60:.0f} min")
    print()

    # Load policy + env
    policy = load_lora_policy(args.config_name, Path(args.checkpoint).expanduser().resolve())
    model = find_model_attr(policy)
    env_pack = setup_env(args.task_suite, args.task_id)
    n_inits = min(args.n_trials, env_pack["n_inits"])
    print(f"[env] using {n_inits} init states")

    # Optionally filter to a subset of prompts (for fast targeted re-runs)
    prompts_to_run = EVAL_PROMPTS
    if args.prompts:
        wanted = {s.strip() for s in args.prompts.split(",")}
        prompts_to_run = [(l, p) for l, p in EVAL_PROMPTS if l in wanted]
        print(f"[eval] running subset of prompts: {[l for l, _ in prompts_to_run]}")
        if not prompts_to_run:
            print(f"[error] --prompts={args.prompts!r} matched none of "
                  f"{[l for l, _ in EVAL_PROMPTS]}")
            sys.exit(1)

    # Run all conditions
    all_results: Dict[str, List[dict]] = {}
    for prompt_label, prompt in prompts_to_run:
        print(f"\n[eval] === {prompt_label} ===")
        print(f"  prompt: {prompt!r}")
        trials = []
        for init_i in range(n_inits):
            init_state = env_pack["init_states"][init_i]
            env_seed = args.env_seed + init_i
            t0 = time.time()

            video_path = None
            if args.save_videos and init_i < args.video_trials:
                video_path = out_dir / "videos" / f"{prompt_label}_trial{init_i}.mp4"

            outcome = run_trial(
                policy, model, env_pack, init_state, env_seed,
                prompt=prompt, replan_steps=args.replan_steps,
                base_seed=args.seed + init_i,
                save_video_path=video_path,
            )
            elapsed = time.time() - t0
            outcome["init_idx"] = init_i
            outcome["elapsed_sec"] = elapsed
            trials.append(outcome)
            print(f"  trial {init_i+1}/{n_inits}: "
                  f"first={outcome['first_moved']!r:30s} "
                  f"succ={outcome['success']} steps={outcome['n_steps']} "
                  f"t={elapsed:.1f}s", flush=True)
        all_results[prompt_label] = trials

    # Compute summary metrics
    summary = {}
    for prompt_label, trials in all_results.items():
        n = len(trials)
        n_success = sum(t["success"] for t in trials)
        first_moved_dist = dict(Counter(t["first_moved"] for t in trials))
        summary[prompt_label] = {
            "n_trials": n,
            "n_success": n_success,
            "success_rate": n_success / max(n, 1),
            "first_moved_distribution": first_moved_dist,
            "mean_steps": float(np.mean([t["n_steps"] for t in trials])),
        }

    # Compute headline metrics
    def get_obj_count(prompt_label: str, obj_name: str) -> int:
        # Defensive: prompt may have been filtered out via --prompts
        if prompt_label not in summary:
            return 0
        return summary[prompt_label]["first_moved_distribution"].get(obj_name, 0)

    n = args.n_trials
    headline = {
        "variant_label": args.variant_label,
        # Headline: did LoRA make the tomato_only prompt produce tomato_sauce first?
        "tomato_rate_on_tomato_prompt": get_obj_count("T0_tomato_only", "tomato_sauce_1_pos") / n,
        # Baseline preservation: T0 canonical still produces alphabet-first?
        "alphabet_rate_on_canonical": get_obj_count("T0_canonical", "alphabet_soup_1_pos") / n,
        # Word-order test: does the reordered prompt produce tomato (parsing) or alphabet (rote)?
        "tomato_rate_on_reordered": get_obj_count("T0_reordered", "tomato_sauce_1_pos") / n,
        "alphabet_rate_on_reordered": get_obj_count("T0_reordered", "alphabet_soup_1_pos") / n,
        # Sanity: trained alphabet_only prompt → alphabet?
        "alphabet_rate_on_alphabet_only": get_obj_count("T0_alphabet_only", "alphabet_soup_1_pos") / n,
        # Per-prompt success rates
        "success_per_prompt": {l: summary[l]["success_rate"] for l, _ in EVAL_PROMPTS},
    }

    # Print headline
    print(f"\n=== HEADLINE METRICS ({args.variant_label}) ===")
    print(f"  Tomato-first under tomato_only prompt: {headline['tomato_rate_on_tomato_prompt']*100:.0f}%  "
          f"(baseline ≈ 0%, target ≥ 50%)")
    print(f"  Alphabet-first under canonical prompt: {headline['alphabet_rate_on_canonical']*100:.0f}%  "
          f"(baseline = 87%, target preserved)")
    print(f"  Reordered prompt distribution: "
          f"alphabet={headline['alphabet_rate_on_reordered']*100:.0f}% "
          f"tomato={headline['tomato_rate_on_reordered']*100:.0f}%")

    # Save all data
    data_out = {
        "run_config": run_config,
        "summary": summary,
        "headline": headline,
        "all_trials": all_results,
    }
    (out_dir / "behavioral_eval.json").write_text(json.dumps(data_out, indent=2, default=str))
    (out_dir / "headline_metrics.json").write_text(json.dumps(headline, indent=2))

    # Make plots
    try:
        make_plots(summary, headline, args.variant_label, out_dir / "plots")
    except Exception as e:
        print(f"\n[warn] plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Optional wandb logging — fires AFTER plots so we can attach the images too
    if args.wandb_run_id is not None:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                id=args.wandb_run_id,
                resume="allow",
            )
            # Headline scalars
            log_payload = {
                f"eval/headline/{k}": v
                for k, v in headline.items()
                if isinstance(v, (int, float))
            }
            # Per-prompt success + counts
            for prompt_label, s in summary.items():
                log_payload[f"eval/{prompt_label}/success_rate"] = s["success_rate"]
                log_payload[f"eval/{prompt_label}/n_success"]    = s["n_success"]
                log_payload[f"eval/{prompt_label}/mean_steps"]   = s["mean_steps"]
            # success_per_prompt as a nested dict gets exploded
            for pl, sr in headline.get("success_per_prompt", {}).items():
                log_payload[f"eval/success_per_prompt/{pl}"] = sr

            if args.training_step is not None:
                wandb.log(log_payload, step=args.training_step)
            else:
                wandb.log(log_payload)

            # First-moved breakdown table (per prompt × per object)
            table = wandb.Table(columns=["prompt", "object", "count", "frac"])
            for prompt_label, s in summary.items():
                n = max(s["n_trials"], 1)
                for obj, count in s["first_moved_distribution"].items():
                    table.add_data(prompt_label, obj, count, count / n)
            wandb.log({"eval/first_moved_table": table})

            # Attach the three plots as media
            for plot_name in ("first_moved_per_prompt.png",
                              "success_rate_per_prompt.png",
                              "prompt_sensitivity_matrix.png"):
                plot_path = out_dir / "plots" / plot_name
                if plot_path.exists():
                    wandb.log({f"eval/plots/{plot_name.replace('.png', '')}":
                               wandb.Image(str(plot_path))})

            print(f"\n[eval] logged metrics + plots to wandb run {args.wandb_run_id}")
        except Exception as e:
            print(f"\n[warn] wandb logging failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[done] saved results to {out_dir}")
    print(f"  - run_config.json")
    print(f"  - behavioral_eval.json")
    print(f"  - headline_metrics.json")
    print(f"  - plots/first_moved_per_prompt.png")
    print(f"  - plots/success_rate_per_prompt.png")
    print(f"  - plots/prompt_sensitivity_matrix.png")


def make_plots(summary: Dict, headline: Dict, variant_label: str, plot_dir: Path):
    """Generate visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    prompt_labels = [l for l, _ in EVAL_PROMPTS]

    # Plot 1: first-moved distribution per prompt (stacked bar)
    all_objects = sorted({
        obj for label in prompt_labels
        for obj in summary[label]["first_moved_distribution"].keys()
    })
    counts_matrix = np.zeros((len(prompt_labels), len(all_objects)))
    for i, label in enumerate(prompt_labels):
        for j, obj in enumerate(all_objects):
            counts_matrix[i, j] = summary[label]["first_moved_distribution"].get(obj, 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(prompt_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_objects)))
    for j, obj in enumerate(all_objects):
        if counts_matrix[:, j].sum() == 0:
            continue
        ax.bar(prompt_labels, counts_matrix[:, j], bottom=bottom,
               label=obj, color=colors[j])
        bottom += counts_matrix[:, j]
    ax.set_xlabel("Eval Prompt")
    ax.set_ylabel("# trials (out of {})".format(int(bottom[0])))
    ax.set_title(f"First-moved distribution per prompt — {variant_label}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "first_moved_per_prompt.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: success rate per prompt
    fig, ax = plt.subplots(figsize=(10, 5))
    success_rates = [summary[l]["success_rate"] * 100 for l in prompt_labels]
    bars = ax.bar(prompt_labels, success_rates, color="steelblue")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"Task success rate per prompt — {variant_label}")
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "success_rate_per_prompt.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: prompt-sensitivity matrix (which obj got picked under which prompt)
    # Restrict to most common objects + prompts
    common_objects = [obj for obj in all_objects if counts_matrix[:, all_objects.index(obj)].sum() >= 3]
    if not common_objects:
        common_objects = all_objects
    obj_idx = {obj: all_objects.index(obj) for obj in common_objects}
    sub_matrix = counts_matrix[:, [obj_idx[o] for o in common_objects]]
    # Normalize per row
    row_sums = sub_matrix.sum(axis=1, keepdims=True)
    normed = np.where(row_sums > 0, sub_matrix / row_sums, 0)

    fig, ax = plt.subplots(figsize=(max(8, len(common_objects)), 5))
    im = ax.imshow(normed, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(common_objects)))
    ax.set_xticklabels(common_objects, rotation=45, ha="right")
    ax.set_yticks(range(len(prompt_labels)))
    ax.set_yticklabels(prompt_labels)
    for i in range(len(prompt_labels)):
        for j in range(len(common_objects)):
            if normed[i, j] > 0.05:
                ax.text(j, i, f"{normed[i, j]:.0%}",
                        ha="center", va="center",
                        color="black" if normed[i, j] < 0.6 else "white",
                        fontsize=9)
    plt.colorbar(im, ax=ax, label="P(first-moved=object | prompt)")
    ax.set_title(f"Prompt sensitivity matrix — {variant_label}")
    fig.tight_layout()
    fig.savefig(plot_dir / "prompt_sensitivity_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  [plots] wrote 3 figures to {plot_dir}/")


if __name__ == "__main__":
    main()
