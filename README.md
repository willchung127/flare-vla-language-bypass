# What Does π0 Actually Do With Your Instruction?

Course project (CS 295, UC Irvine, Spring 2026) studying how a fine-tuned
vision-language-action model — π0 fine-tuned on LIBERO — actually uses its
language instruction: behaviorally, mechanistically, and under test-time
intervention.

**Read the report:** open [`docs/index.html`](docs/index.html) in a browser
(self-contained; figures and rollout videos live under `docs/assets/`).

## Findings in brief

- The instruction is **necessary** (deleting it drops success from 94–98% to
  0–24%) but is used like a **task label**: word order and syntax are ignored
  (scramble 0.84, keywords-only 0.92), while unfamiliar wording or format breaks
  it (noun synonyms 0.70, instruction buried in filler text 0.08).
- Prompts that require reading the sentence's content **within** a task fail
  almost completely under a strict metric: single-object requests 4%, negation
  0% — the policy replays its training trajectory instead.
- A leakage-controlled linear probe shows the named object is decodable at every
  action-expert layer (AUC 0.89–0.96) yet never acted on; negation is not
  linearly decodable at any layer of either tower.
- Four test-time interventions of increasing strength — sampler noise (SDE),
  attention redirection, hidden-state steering (placebo-controlled), and LoRA
  adapters — all fail, each with a control confirming the intervention itself
  worked. Weak interventions are absorbed; strong ones break the policy before
  behavior changes.

## Repository layout

| Path | Contents |
|---|---|
| `docs/` | The report webpage (self-contained; figures and videos under `docs/assets/`) |
| `experiments/prompt_conditions/` | Rollouts under prompt manipulations (delete / paraphrase / scramble / keywords / synonyms / filler / swap) and the base-vs-fine-tuned action-divergence test |
| `experiments/instruction_sweep/` | 20-prompt compositional sweep (reorder / single-object / order / negation / cross-task), plus strict-compliance and grounding scorers |
| `experiments/mechanism/` | Velocity-field and cross-attention divergence between prompts, re-scored on the executed action dimensions |
| `experiments/attention/` | Attention-budget measurement, sink analysis, forced-attention and attention-boost interventions |
| `experiments/probing/` | Leakage-controlled linear probe (GroupKFold by wording, permutation null, negation transfer) |
| `experiments/steering/` | Hidden-state steering with a dose sweep and a norm-matched random placebo |
| `experiments/adapters/` | LoRA ablation chain, counterfactual-data variants, merge and sanity tooling |
| `experiments/sampling_noise/` | ODE-vs-SDE sampling, η sweep, outcome-mode counting, guided-SDE comparison |
| `figures/make_figures.py` | Regenerates all seven report figures; loads result JSONs only — no hand-entered numbers |

## Pipeline (which script produced which figure)

1. `prompt_conditions/run_all.sh` → success-vs-condition results and the
   action-divergence ratios (**Fig. 1**).
2. `instruction_sweep/generate_instruction_sweep.py` + `run_instruction_sweep.py`
   → 300-episode sweep; `score_strict_compliance.py` re-scores it (**Fig. 2**).
3. `attention/sink_attention_analysis.py` → attention anatomy (**Fig. 3**);
   `forced_attention_eval.py` and `attention_boost_rollouts.py` → intervention
   rung 2.
4. `mechanism/velocity_divergence.py` + `score_velocity_executed_dims.py` →
   velocity-field similarity on the executed action dimensions (**Fig. 4**).
5. `probing/linear_probe_decoding.py` → both-tower probe (**Fig. 5**).
6. `steering/steering_dose_response.py` → dose-response + placebo (**Fig. 6**).
7. `adapters/run_adapter_ablation.sh` → LoRA variants (**Fig. 7**).
8. `sampling_noise/*` → η sweep, outcome-mode counts, guided-SDE comparison
   (intervention rung 1).

## Setup notes

- Model: the public `pi0_libero` checkpoint (and `pi05_libero` for the scale
  check) from [openpi](https://github.com/Physical-Intelligence/openpi);
  simulator: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
- All GPU work ran on a single shared RTX 4090 (24 GB) with `MUJOCO_GL=egl`.
- The scripts were developed and run co-located in a flat working directory
  inside an openpi checkout. They are grouped by experiment here for
  readability, and a few were given clearer names, so some intra-project
  imports, cross-references, and runner scripts refer to sibling modules by
  their original names. Result and checkpoint paths at the top of each script
  point at that environment and may need adjusting for yours.

## Data availability

The result files (~11 MB of JSON) behind every number in the report are kept out
of the repository to keep it source-only; they are available on request, and
`figures/make_figures.py` documents exactly which files each figure loads.

## License

MIT — see [LICENSE](LICENSE).
