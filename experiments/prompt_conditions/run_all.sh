#!/usr/bin/env bash
# Sequential driver for the language-bypass experiments (single GPU).
# Run from the openpi repo root:  bash experiments/run_all.sh
set -uo pipefail   # no -e: a failing stage should not kill later stages

cd "$(dirname "$0")"
mkdir -p results logs

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=${MUJOCO_GL:-egl}        # switch to osmesa if egl fails headless
export TOKENIZERS_PARALLELISM=false

RUN() {
  local name="$1"; shift
  echo "================================================================"
  echo "[$(date '+%F %T')] START $name"
  echo "================================================================"
  if uv run python "$@" 2>&1 | tee "logs/${name}.log"; then
    echo "[$(date '+%F %T')] DONE  $name"
  else
    echo "[$(date '+%F %T')] FAIL  $name (continuing with next stage)"
  fi
}

# ---- Stage 1: forward-pass divergence, finetuned + base (fast, highest value)
RUN exp_a action_divergence.py --suite libero_goal --frames-per-task 3 --include-base

# ---- Stage 2: rollouts on GOAL (IFR / paraphrase / swap) -- overnight chunk 1
RUN exp_c_goal rollout_prompt_conditions.py --suite libero_goal \
    --conds orig,empty,para,swap --trials 5

# ---- Stage 3: rollouts on SPATIAL (bypass quantification) -- overnight chunk 2
RUN exp_c_spatial rollout_prompt_conditions.py --suite libero_spatial \
    --conds orig,empty,scramble --trials 5

echo "================================================================"
echo "[$(date '+%F %T')] ALL STAGES FINISHED -- results in $(pwd)/results"
echo "================================================================"
