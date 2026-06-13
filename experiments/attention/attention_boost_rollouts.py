"""attention_boost_rollouts.py — force attention toward content tokens (and optionally
mask sinks at the same time).

The natural inverse of sink_suppression.py. Adds a POSITIVE bias to the
attention scores at specified key positions, forcing softmax to concentrate
attention there. Can be combined with sink suppression (negative bias) for a
double intervention.

Test hypothesis: if the language bypass is at the attention layer (model
ignores prompt because it attends to no-op sinks), then forcing attention onto
the actual content tokens (e.g., key 776 = " tomato") should restore prompt
sensitivity. If forcing attention does NOT change behavior, the bypass is
downstream of attention — confirming the late-layer weights are the locus and
LoRA-style retraining is required.

The intervention:
    attention_mask[..., boost_positions] += +boost_value     # FORCE attention here
    attention_mask[..., sink_positions]  := -inf             # BLOCK attention here

Both pre-softmax; softmax handles the redistribution naturally.

Usage:
    # Mode A: single boost-position set per condition, multiple prompts (original)
    python3 attention_boost_rollouts.py \\
        --policy-config pi0_libero \\
        --checkpoint-dir ~/flare/checkpoints/pi0_libero_pt \\
        --task-suite libero_10 --task-id 0 \\
        --target-layers 17 \\
        --boost-positions 776 777 \\
        --boost-value 10 \\
        --sink-positions 768 \\
        --prompts "put both the alphabet soup and the tomato sauce in the basket" \\
                  "put the tomato sauce in the basket" \\
        --n-trials 15 \\
        --plot \\
        --out-dir ~/flare/results/attn_boost_T0_L17_tomato_boost10

    # Mode B: multiple boost-position sets on the SAME prompt (within-prompt comparison)
    # Same prompt across conditions; only attention pattern varies. Eliminates
    # scene-OOD, prompt-string, and tokenization confounds — purely tests
    # whether attention-mass at specific language tokens drives object selection.
    python3 attention_boost_rollouts.py \\
        --policy-config pi0_libero \\
        --checkpoint-dir ~/flare/checkpoints/pi0_libero_pt \\
        --task-suite libero_10 --task-id 0 \\
        --target-layers 16 17 \\
        --prompts "put both the alphabet soup and the tomato sauce in the basket" \\
        --boost-position-sets '[[],[772,773],[776,777],[780],[769,770,771],[768]]' \\
        --boost-set-labels "baseline,boost_alphabet,boost_tomato,boost_basket,boost_function,boost_BOS" \\
        --boost-value-per-set '[0,5,5,5,5,5]' \\
        --n-trials 15 --plot \\
        --out-dir ~/flare/results/attn_boost_T0_within_prompt_L1617
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from mechanism_probe_attn import set_all_seeds, force_eager_attention


# =============================================================================
# ATTENTION BOOST + SUPPRESS HOOK (combined)
# =============================================================================

class AttentionBoostHook:
    """Forward-pre-hook on expert attention layers. Adds +boost_value to scores
    at boost_positions, and -inf at sink_positions.

    Either list can be empty:
      - boost only:  sink_positions=[],  boost_positions=[776, 777], boost_value=10
      - mask only:   sink_positions=[768], boost_positions=[]  (equivalent to SinkSuppression)
      - combined:    both set — force attention OFF sinks AND ONTO content

    Both biases are applied pre-softmax. Softmax handles the redistribution
    naturally — a boosted position with +10 bias becomes ~e^10 ≈ 22000× more
    attended than a position with 0 bias.

    Args:
        model: pi0 nn.Module
        target_layer_idxs: list of expert decoder layer indices
        sink_positions: list of key positions to mask (-inf). Default []
        boost_positions: list of key positions to amplify (+boost_value). Default []
        boost_value: positive bias added at boost_positions. Default 5.0
        layer_pattern_template: regex template with {idx} placeholder.
    """

    def __init__(self, model: nn.Module, target_layer_idxs: List[int],
                  sink_positions: Optional[List[int]] = None,
                  boost_positions: Optional[List[int]] = None,
                  boost_value: float = 5.0, verbose: bool = True,
                  layer_pattern_template: str = (
                      r"^paligemma_with_expert\.gemma_expert\.model\.layers\.{idx}\.self_attn$"
                  )):
        self.model = model
        self.target_layer_idxs = list(target_layer_idxs)
        self.sink_positions = list(sink_positions or [])
        self.boost_positions = list(boost_positions or [])
        self.boost_value = float(boost_value)
        self.verbose = verbose
        self.layer_pattern_template = layer_pattern_template
        self._handles: List = []
        self._target_modules: List[nn.Module] = []
        self._n_calls: int = 0
        self._last_modified_shape: Optional[Tuple[int, ...]] = None

    def _find_targets(self) -> List[nn.Module]:
        targets = []
        for idx in self.target_layer_idxs:
            pattern = re.compile(self.layer_pattern_template.format(idx=idx))
            matched = None
            for name, module in self.model.named_modules():
                if pattern.match(name):
                    matched = module
                    break
            if matched is None:
                sample = [n for n, _ in self.model.named_modules()
                          if "gemma_expert" in n][:5]
                raise RuntimeError(
                    f"No attention module matched {pattern.pattern}.\n"
                    f"Sample expert paths: {sample}"
                )
            targets.append(matched)
        return targets

    def _make_pre_hook(self):
        sinks = self.sink_positions
        boosts = self.boost_positions
        boost_value = self.boost_value
        outer = self

        def pre_hook(module, args, kwargs):
            outer._n_calls += 1
            if not sinks and not boosts:
                return args, kwargs

            mask = kwargs.get("attention_mask", None)
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return args, kwargs

            past_kv = kwargs.get("past_key_value", None)
            if past_kv is None:
                past_kv = next(
                    (a for a in args[1:]
                     if hasattr(a, "get_seq_length") or hasattr(a, "key_cache")),
                    None
                )
            past_len = 0
            if past_kv is not None:
                try:
                    past_len = past_kv.get_seq_length()
                except Exception:
                    try:
                        past_len = past_kv[0].shape[-2] if past_kv else 0
                    except Exception:
                        past_len = 0

            q_len = hidden_states.shape[1]
            k_len = past_len + q_len

            if mask is None:
                mask = torch.zeros(
                    (hidden_states.shape[0], 1, q_len, k_len),
                    dtype=hidden_states.dtype, device=hidden_states.device,
                )
            else:
                mask = mask.clone()
                if mask.shape[-1] < k_len:
                    pad = torch.zeros(
                        (*mask.shape[:-1], k_len - mask.shape[-1]),
                        dtype=mask.dtype, device=mask.device,
                    )
                    mask = torch.cat([mask, pad], dim=-1)

            # Apply sink suppression (-inf)
            if sinks:
                neg_inf = torch.finfo(mask.dtype).min
                for sp in sinks:
                    if 0 <= sp < mask.shape[-1]:
                        mask[..., sp] = neg_inf

            # Apply boost (+boost_value)
            if boosts:
                for bp in boosts:
                    if 0 <= bp < mask.shape[-1]:
                        mask[..., bp] = mask[..., bp] + boost_value

            outer._last_modified_shape = tuple(mask.shape)
            new_kwargs = dict(kwargs)
            new_kwargs["attention_mask"] = mask
            return args, new_kwargs

        return pre_hook

    def __enter__(self):
        self._target_modules = self._find_targets()
        for module in self._target_modules:
            handle = module.register_forward_pre_hook(
                self._make_pre_hook(), with_kwargs=True
            )
            self._handles.append(handle)
        if self.verbose:
            print(f"[AttentionBoost] hooked {len(self._target_modules)} layers "
                  f"({self.target_layer_idxs}); "
                  f"sinks={self.sink_positions}, boosts={self.boost_positions} (+{self.boost_value})")
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self.verbose:
            print(f"[AttentionBoost] removed; hook fired {self._n_calls} times")
        self._n_calls = 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-config", default="pi0_libero")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--bddl-override", default=None)
    p.add_argument("--target-layers", nargs="+", type=int, default=[17],
                   help="Expert decoder layer indices to intervene at")
    p.add_argument("--sink-positions", nargs="+", type=int, default=[],
                   help="Key positions to mask (-inf). E.g., 768 for BOS sink.")
    p.add_argument("--boost-positions", nargs="+", type=int, default=[],
                   help="Key positions to amplify (+boost_value). E.g., 776 for ' tomato'.")
    p.add_argument("--boost-value", type=float, default=5.0,
                   help="Positive bias added at boost_positions (default 5.0). "
                        "Used in Mode A or as fallback in Mode B if "
                        "--boost-value-per-set is omitted.")
    p.add_argument("--boost-position-sets", default=None,
                   help="MODE B: JSON list-of-lists of key positions, one set "
                        "per condition. Uses prompts[0] only; varies the boost "
                        "across conditions. E.g., '[[],[772,773],[776,777]]' "
                        "runs 3 conditions (no-op, alphabet-boost, tomato-boost) "
                        "on the same prompt.")
    p.add_argument("--boost-set-labels", default=None,
                   help="MODE B: comma-separated labels matching "
                        "--boost-position-sets. E.g., "
                        "'baseline,boost_alphabet,boost_tomato'. "
                        "If omitted, auto-generated from positions.")
    p.add_argument("--boost-value-per-set", default=None,
                   help="MODE B: JSON list of magnitudes, one per "
                        "--boost-position-sets entry. E.g., '[0,5,5]' for "
                        "(no-op, mild, mild). If omitted, uses --boost-value "
                        "for all sets.")
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--env-seed", type=int, default=7000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[boost] === LOADING MODEL + ENV ===", flush=True)
    from openpi.training import config as _c
    from openpi.policies import policy_config
    from libero.libero import benchmark, get_libero_path
    sys.path.insert(0, str(Path.home() / "flare"))
    from patch_pi0_sde import apply_patch  # noqa: E402
    import remote_multimode_matrix as mmm  # type: ignore  # noqa: E402

    Pi0 = apply_patch()
    cfg = _c.get_config(args.policy_config)
    ckpt_dir = args.checkpoint_dir or str(Path.home() / "flare/checkpoints/pi0_libero_pt")
    policy = policy_config.create_trained_policy(cfg, ckpt_dir)
    model = None
    for attr in ("_model", "model", "_policy", "policy"):
        if hasattr(policy, attr) and isinstance(getattr(policy, attr), Pi0):
            model = getattr(policy, attr)
            break
    if model is None:
        raise RuntimeError("Could not find PI0Pytorch on policy")

    model.flare_eta = 0.0
    model.flare_verifier_fn = None
    model.flare_alpha = 0.0
    model.flare_obs_state = None
    model.flare_eta_high = None
    model.flare_eta_low = None
    model.flare_noise_bias_direction = None
    model.flare_noise_bias_strength = 0.0
    model.eval()
    force_eager_attention(model, verbose=False)

    bench = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bench.get_task(args.task_id)
    bddl = (args.bddl_override or
            os.path.join(get_libero_path("bddl_files"),
                          task.problem_folder, task.bddl_file))
    init_states = bench.get_task_init_states(args.task_id)
    n_inits = min(args.n_trials, len(init_states))
    print(f"  task_id={args.task_id}, using {n_inits} init states")

    cond_ode = {"name": "ode_single", "eta": 0.0, "K": 1,
                "guidance": False, "alpha": 0.0}

    # Conditions: two modes
    # MODE A (--boost-positions): baseline_noop + one boost condition per prompt,
    #   all using the same boost positions and magnitude
    # MODE B (--boost-position-sets): hold prompts[0] fixed, vary the boost
    #   positions per condition (with optional per-set magnitudes)
    conditions: List[Tuple[str, List[int], List[int], str, float]] = []
    if args.boost_position_sets is not None:
        try:
            boost_sets = json.loads(args.boost_position_sets)
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"--boost-position-sets must be valid JSON list-of-lists: {e}"
            )
        if not isinstance(boost_sets, list) or not all(
            isinstance(s, list) for s in boost_sets
        ):
            raise SystemExit(
                "--boost-position-sets must be a JSON list-of-lists, "
                "e.g. '[[],[772,773],[776,777]]'"
            )
        if args.boost_set_labels:
            labels = [s.strip() for s in args.boost_set_labels.split(",")]
            if len(labels) != len(boost_sets):
                raise SystemExit(
                    f"--boost-set-labels has {len(labels)} entries; "
                    f"--boost-position-sets has {len(boost_sets)}"
                )
        else:
            labels = [
                f"set{i}_{'-'.join(map(str, s)) if s else 'empty'}"
                for i, s in enumerate(boost_sets)
            ]
        if args.boost_value_per_set is not None:
            try:
                values = [float(v) for v in json.loads(args.boost_value_per_set)]
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"--boost-value-per-set must be valid JSON list: {e}"
                )
            if len(values) != len(boost_sets):
                raise SystemExit(
                    f"--boost-value-per-set has {len(values)} entries; "
                    f"--boost-position-sets has {len(boost_sets)}"
                )
        else:
            values = [args.boost_value] * len(boost_sets)
        if len(args.prompts) > 1:
            print(f"[boost] NOTE: --boost-position-sets uses only prompts[0]; "
                  f"ignoring remaining {len(args.prompts) - 1} prompts")
        main_prompt = args.prompts[0]
        print(f"[boost] MODE B: {len(boost_sets)} boost-position sets on "
              f"prompt={main_prompt!r}")
        for label, positions, bv in zip(labels, boost_sets, values):
            conditions.append(
                (label, list(args.sink_positions), list(positions),
                 main_prompt, float(bv))
            )
    else:
        conditions.append(
            ("baseline_noop", [], [], args.prompts[0], args.boost_value)
        )
        for prompt_idx, prompt in enumerate(args.prompts):
            label = f"boost_prompt{prompt_idx}"
            conditions.append(
                (label, args.sink_positions, args.boost_positions, prompt,
                 args.boost_value)
            )

    all_results: Dict[str, List[dict]] = {}

    for label, sinks, boosts, prompt, boost_val in conditions:
        print(f"\n[boost] === CONDITION: {label} ===", flush=True)
        print(f"  sinks={sinks}, boosts={boosts} (+{boost_val}); prompt={prompt!r}")
        trials = []
        for init_i in range(n_inits):
            init_state = init_states[init_i]
            env_seed = args.env_seed + init_i
            t0 = time.time()
            with AttentionBoostHook(
                model, args.target_layers,
                sink_positions=sinks, boost_positions=boosts,
                boost_value=boost_val,
                verbose=(init_i == 0),
            ) as hook:
                set_all_seeds(args.seed + init_i)
                rollout_result = mmm.rollout(
                    policy=policy, model=model, task=task,
                    task_id=args.task_id, bddl=bddl,
                    init_state=init_state, env_seed=env_seed,
                    base_seed=args.seed, cond=cond_ode,
                    scorer_fn=None, guidance_callable=None,
                    obs_state_for_guidance=None,
                    replan_steps=args.replan_steps,
                    video_mode="off",
                    language_override=prompt,
                )
                hook_fires = hook._n_calls
            done, term, n_steps, init_pos, final_pos, _, moved, _ = rollout_result
            first = moved[0]["name"] if moved else "<none>"
            trials.append({
                "init_idx": init_i,
                "success": bool(done),
                "first_moved": first,
                "n_steps": int(n_steps),
                "hook_fires": hook_fires,
            })
            elapsed = time.time() - t0
            print(f"  trial {init_i+1}/{n_inits}: first={first!r:30s} "
                  f"succ={done} steps={n_steps} t={elapsed:.1f}s", flush=True)
        all_results[label] = trials

    # Headline summary
    print(f"\n[boost] === HEADLINE SUMMARY ===")
    print(f"  {'condition':<30s}  {'succ%':>6s}  first-moved distribution (top 3)")
    print(f"  {'-'*30}  {'-'*6}  {'-'*60}")
    summary = {}
    for label, trials in all_results.items():
        succ = sum(t["success"] for t in trials)
        n = len(trials)
        first_counts = Counter(t["first_moved"] for t in trials)
        top3 = ", ".join(f"{name}:{c}" for name, c in first_counts.most_common(3))
        print(f"  {label:<30s}  {100*succ/max(n,1):>5.0f}%  {top3}")
        summary[label] = {
            "n_trials": n, "n_success": succ,
            "success_rate": succ / max(n, 1),
            "first_moved_distribution": dict(first_counts),
        }

    out = {
        "task_id": args.task_id,
        "target_layers": args.target_layers,
        "sink_positions": args.sink_positions,
        "boost_positions": args.boost_positions,
        "boost_value": args.boost_value,
        "boost_position_sets": args.boost_position_sets,
        "boost_set_labels": args.boost_set_labels,
        "boost_value_per_set": args.boost_value_per_set,
        "prompts": list(args.prompts),
        "conditions": [
            {"label": l, "sinks": s, "boosts": b, "prompt": p,
             "boost_value": v}
            for (l, s, b, p, v) in conditions
        ],
        "summary": summary,
        "trials": all_results,
    }
    (out_dir / "attention_boost_results.json").write_text(json.dumps(out, indent=2))

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[boost] matplotlib unavailable")
        else:
            labels = list(all_results.keys())
            all_objs = sorted({t["first_moved"]
                                for trials in all_results.values()
                                for t in trials})
            data = {o: [Counter(t["first_moved"] for t in all_results[l]).get(o, 0)
                        for l in labels] for o in all_objs}
            fig, ax = plt.subplots(figsize=(12, 6))
            bottom = np.zeros(len(labels))
            for o, counts in data.items():
                ax.bar(labels, counts, bottom=bottom, label=o)
                bottom += np.array(counts)
            ax.set_xlabel("condition")
            ax.set_ylabel("# trials")
            if args.boost_position_sets is not None:
                title = (f"Within-prompt attention boost variants at "
                         f"L{args.target_layers} | T{args.task_id} | "
                         f"prompt={args.prompts[0][:60]!r}")
            else:
                title = (f"Attention boost at L{args.target_layers}; "
                         f"boost+{args.boost_value} on {args.boost_positions}, "
                         f"sink mask on {args.sink_positions} | T{args.task_id}")
            ax.set_title(title)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
            plt.xticks(rotation=20, ha="right")
            fig.tight_layout()
            png = out_dir / "attention_boost_distribution.png"
            fig.savefig(png, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[boost] wrote {png}")

    print(f"\n[boost] ✓ done. Files in {out_dir}")


if __name__ == "__main__":
    main()
