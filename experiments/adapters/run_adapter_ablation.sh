#!/bin/bash
# run_lora_ablation_chain_v3.sh — chain C v2 eval + A v2 train+eval + D v2 train+eval
#
# v3 changes over v2:
#   - Uses CORRECTED converter (state composition + image preprocessing fixed)
#   - batch=16 with automatic OOM fallback to batch=12
#   - 30%-checkpoint GATE: train to step 449, run quick 6-rollout eval, abort if FAIL
#   - WandB logging plumbed into eval scripts (gate, behavioral, velocity)
#   - Per-variant intermediate checkpoint deleted after gate passes (free disk)
#   - Auto-discovers wandb run ID from training log for downstream eval logging
#
# Variants run (in order):
#   C v2: natural counterfactual (already trained at batch=8 — JUST EVAL)
#   A v2: original-prompt-only LoRA, 50 demos       — train + gate + eval
#   D v2: BOS-masked LoRA on natural CF, 350 demos  — train + gate + eval
#         (D v2 needs pi0.py BOS mask patch + new TrainConfig pi0_libero_cf_lora_bos_masked)
#
# USAGE (always run in tmux for safety):
#   tmux new -d -s chain-v3 'bash ~/flare/run_lora_ablation_chain_v3.sh; exec bash'
#   tail -f ~/flare/results/lora_ablation/chain_v3.log
#
# ASSUMES:
#   - Variant C v2 trained at: ~/flare/openpi/checkpoints/pi0_libero_cf_lora/variant_c_natural_cf_lora_v2/2999
#   - Datasets for A and D: converted (or will be auto-converted from libero_10/libero_90 HDF5)
#   - openpi configs: pi0_libero_cf_lora, pi0_libero_t0_only_lora, pi0_libero_cf_lora_bos_masked (last is NEW for D v2)
#   - WandB logged in (`wandb login` once)
#   - ~/flare/openpi/.venv has the python env

set -u

# ============================================================================
# CONFIG
# ============================================================================

OPENPI_DIR=~/flare/openpi
LEROBOT_CACHE=~/.cache/huggingface/lerobot
ABLATION_DIR=~/flare/results/lora_ablation
LIBERO_HDF5=~/flare/openpi/third_party/libero/libero/datasets
WANDB_PROJECT=flare_cf_lora

# Training config
BATCH_PRIMARY=16          # try this first
BATCH_FALLBACK=12         # if BATCH_PRIMARY OOMs, retry with this
NUM_STEPS=1500            # at batch=16: ~24k effective samples (same as batch=8 × 3000)
GATE_STEP=449             # ~30% of NUM_STEPS; save at step GATE_STEP+1=450

# Eval config
N_EVAL_TRIALS=15          # full behavioral eval at end
GATE_N_TRIALS=3           # quick eval at gate (3 trials × 2 prompts = 6 rollouts)

# Variant C v2 (already trained at batch=8, 3000 steps)
C_V2_CKPT=$OPENPI_DIR/checkpoints/pi0_libero_cf_lora/variant_c_natural_cf_lora_v2/2999
C_V2_CONFIG=pi0_libero_cf_lora
C_V2_VARIANT_LABEL=variant_c_natural_cf_lora_v2

mkdir -p $ABLATION_DIR

CHAIN_LOG=$ABLATION_DIR/chain_v3.log
echo "" >> $CHAIN_LOG
echo "===================================================================" >> $CHAIN_LOG
echo "CHAIN v3 STARTED: $(date)" >> $CHAIN_LOG
echo "===================================================================" >> $CHAIN_LOG

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $CHAIN_LOG; }

# ============================================================================
# UTILITY: convert + symlink a single-prompt LeRobot dataset
# ============================================================================

convert_single_prompt_dataset() {
    local hdf5_path=$1
    local prompt=$2
    local repo_id=$3

    local out_path=~/flare/lerobot_data/$repo_id
    local symlink_path=$LEROBOT_CACHE/$repo_id

    if [ -d "$out_path" ] && [ -L "$symlink_path" ]; then
        log "  [skip convert] dataset $repo_id already exists at $out_path"
        return 0
    fi

    log "  [convert] $hdf5_path → $out_path"
    rm -rf $out_path 2>/dev/null
    ~/flare/openpi/.venv/bin/python ~/flare/convert_libero_hdf5_to_lerobot.py \
        --inputs "$hdf5_path" \
        --prompts "$prompt" \
        --out-dir "$out_path" \
        --repo-id "$repo_id" 2>&1 | tee -a $CHAIN_LOG | tail -10

    if [ ! -L "$symlink_path" ]; then
        mkdir -p $LEROBOT_CACHE
        ln -s "$out_path" "$symlink_path"
        log "  [symlink] $symlink_path → $out_path"
    fi
}

# ============================================================================
# UTILITY: compute norm stats for a config
# ============================================================================

compute_norm_stats() {
    local config_name=$1
    log "  [norm-stats] computing for $config_name"
    cd $OPENPI_DIR
    uv run scripts/compute_norm_stats.py --config-name $config_name 2>&1 | tee -a $CHAIN_LOG | tail -5
    cd - > /dev/null
}

# ============================================================================
# UTILITY: train with OOM-fallback
#   $1 = config_name
#   $2 = exp_name
#   $3 = num_steps
#   $4 = log_path
#
# Auto-sets OPENPI_MASK_BOS=1 if config name matches the BOS-masked variant
# (D-lite). Otherwise the env var is unset → make_attn_mask behaves as before.
# ============================================================================

train_with_fallback() {
    local config_name=$1
    local exp_name=$2
    local steps=$3
    local log_path=$4

    # D-lite: enable BOS mask env var for the bos_masked config
    local bos_env=""
    if [[ "$config_name" == "pi0_libero_cf_lora_bos_masked" ]]; then
        bos_env="OPENPI_MASK_BOS=1"
        log "  [d-lite] training with OPENPI_MASK_BOS=1 (BOS attention sink masked)"
    fi

    cd $OPENPI_DIR

    log "  [train] attempt 1: batch=$BATCH_PRIMARY steps=$steps"
    env $bos_env WANDB_PROJECT=$WANDB_PROJECT \
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run scripts/train.py \
        $config_name \
        --exp-name=$exp_name \
        --batch-size=$BATCH_PRIMARY \
        --num-train-steps=$steps \
        --overwrite \
        2>&1 | tee $log_path

    # If primary failed (non-zero exit) AND log contains OOM marker, fall back
    local exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -ne 0 ]; then
        if grep -qiE "out of memory|RESOURCE_EXHAUSTED|cudaErrorMemoryAllocation" "$log_path"; then
            log "  [OOM] batch=$BATCH_PRIMARY OOMed; retrying batch=$BATCH_FALLBACK"
            env $bos_env WANDB_PROJECT=$WANDB_PROJECT \
                    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
                    XLA_PYTHON_CLIENT_PREALLOCATE=false \
                uv run scripts/train.py \
                $config_name \
                --exp-name=$exp_name \
                --batch-size=$BATCH_FALLBACK \
                --num-train-steps=$steps \
                --overwrite \
                2>&1 | tee -a $log_path
            exit_code=${PIPESTATUS[0]}
        else
            log "  [error] training failed (non-OOM); see $log_path"
        fi
    fi

    cd - > /dev/null
    return $exit_code
}

# ============================================================================
# UTILITY: extract wandb run ID from a training log
# ============================================================================

extract_wandb_run_id() {
    local log_path=$1
    # WandB prints "wandb: Syncing run XXXX-YYYY" or "https://wandb.ai/.../runs/<id>"
    grep -oE 'runs/[a-z0-9]+' "$log_path" 2>/dev/null | head -1 | sed 's|runs/||'
}

# ============================================================================
# UTILITY: train + gate + eval for a single variant
#   $1 = variant_name (e.g. variant_a_v2)
#   $2 = config_name  (e.g. pi0_libero_t0_only_lora)
# ============================================================================

train_gate_eval_variant() {
    local variant=$1
    local config_name=$2
    local exp_name="${variant}"
    local out_dir=$ABLATION_DIR/$variant
    local train_log=$out_dir/train.log

    local ckpt_root=$OPENPI_DIR/checkpoints/$config_name/$exp_name
    local gate_step_padded=$(printf "%d" $GATE_STEP)        # e.g. 449
    local final_step=$(($NUM_STEPS - 1))                    # e.g. 1499
    local gate_ckpt=$ckpt_root/$gate_step_padded
    local final_ckpt=$ckpt_root/$final_step

    # D-lite: BOS mask env var for the bos_masked variant (used by all eval calls below).
    # Training picks this up via train_with_fallback's own check.
    local bos_env_prefix=""
    if [[ "$config_name" == "pi0_libero_cf_lora_bos_masked" ]]; then
        bos_env_prefix="env OPENPI_MASK_BOS=1"
        log "  [d-lite] all eval invocations for $variant will run with OPENPI_MASK_BOS=1"
    fi

    log ""
    log "==================================================================="
    log "VARIANT $variant (config: $config_name)"
    log "==================================================================="

    mkdir -p $out_dir

    # Fast skip: if BOTH behavioral and velocity eval JSON already exist for this
    # variant, nothing to do — saves ~30 min of redundant rollouts on chain re-runs.
    if [ -f "$out_dir/behavioral_eval.json" ] && [ -f "$out_dir/velocity_probes/velocity_eval.json" ]; then
        log "  [SKIP entire variant] $variant already has behavioral_eval.json + velocity_eval.json"
        log "                        delete those files to force re-eval"
        return 0
    fi

    # Marker file: gate eval passed for this variant.
    # Lets us skip phase-1 + gate on re-runs even after gate_ckpt was deleted
    # (e.g. chain killed during phase-3, relaunched — saves ~25 min of re-doing
    #  phase-1 train + gate eval that we already know passed).
    local gate_passed_marker=$out_dir/gate_PASSED

    # ------ PHASE 1: train to GATE_STEP+1 ------
    if [ -d "$gate_ckpt" ] || [ -d "$final_ckpt" ] || [ -f "$gate_passed_marker" ]; then
        log "  [skip phase-1] gate already cleared or final ckpt exists"
    else
        log "  [phase-1] train 0 → $(($GATE_STEP + 1)) (~20 min)"
        train_with_fallback $config_name $exp_name $(($GATE_STEP + 1)) $train_log
        if [ $? -ne 0 ]; then
            log "  [FAIL] phase-1 training failed for $variant — see $train_log"
            return 1
        fi
        if [ ! -d "$gate_ckpt" ]; then
            log "  [FAIL] phase-1 did not produce gate checkpoint $gate_ckpt"
            return 1
        fi
    fi

    # ------ PHASE 2: gate eval ------
    if [ -d "$final_ckpt" ] || [ -f "$gate_passed_marker" ]; then
        log "  [skip gate] gate already passed (marker or final ckpt exists)"
    else
        log "  [gate] running 6-rollout eval on $gate_ckpt"
        local gate_log=$out_dir/gate.log
        local wandb_run_id=$(extract_wandb_run_id $train_log)
        if [ -n "$wandb_run_id" ]; then
            log "  [gate] using wandb run id: $wandb_run_id"
        fi
        $bos_env_prefix ~/flare/openpi/.venv/bin/python ~/flare/eval_lora_gate.py \
            --checkpoint "$gate_ckpt" \
            --config-name "$config_name" \
            --variant-label "${variant}_gate" \
            --out-dir "$out_dir/gate_step${GATE_STEP}" \
            --n-trials $GATE_N_TRIALS \
            --training-step $GATE_STEP \
            ${wandb_run_id:+--wandb-run-id $wandb_run_id} \
            --wandb-project $WANDB_PROJECT \
            2>&1 | tee $gate_log

        # The gate script prints "GATE_RESULT: PASS|FAIL" as the last line
        if grep -q "GATE_RESULT: PASS" $gate_log; then
            log "  [gate] PASS — continuing to full training"
            # Write marker BEFORE deleting gate_ckpt; this is the durability point
            touch $gate_passed_marker
        else
            log "  [FAIL] gate failed for $variant — aborting; see $gate_log"
            log "  [note] gate checkpoint kept at $gate_ckpt for debugging"
            return 2
        fi

        # Free disk before phase-3
        log "  [cleanup] deleting gate checkpoint $gate_ckpt to free disk"
        rm -rf $gate_ckpt
    fi

    # ------ PHASE 3: train to NUM_STEPS ------
    if [ -d "$final_ckpt" ]; then
        log "  [skip phase-3] final checkpoint already exists"
    else
        log "  [phase-3] train 0 → $NUM_STEPS (~50 min)"
        train_with_fallback $config_name $exp_name $NUM_STEPS $train_log
        if [ $? -ne 0 ]; then
            log "  [FAIL] phase-3 training failed for $variant"
            return 1
        fi
        if [ ! -d "$final_ckpt" ]; then
            log "  [FAIL] phase-3 did not produce final checkpoint $final_ckpt"
            return 1
        fi
    fi

    # ------ PHASE 4: full behavioral eval ------
    log "  [eval-behavioral] full eval on $final_ckpt"
    local behav_log=$out_dir/eval_behavioral.log
    local wandb_run_id=$(extract_wandb_run_id $train_log)
    $bos_env_prefix ~/flare/openpi/.venv/bin/python ~/flare/eval_lora_behavioral.py \
        --checkpoint "$final_ckpt" \
        --config-name "$config_name" \
        --variant-label "$variant" \
        --out-dir "$out_dir" \
        --n-trials $N_EVAL_TRIALS \
        --training-step $final_step \
        --save-videos \
        ${wandb_run_id:+--wandb-run-id $wandb_run_id} \
        --wandb-project $WANDB_PROJECT \
        2>&1 | tee $behav_log

    # ------ PHASE 5: velocity eval ------
    log "  [eval-velocity] $variant"
    local vel_log=$out_dir/eval_velocity.log
    mkdir -p $out_dir/velocity_probes
    $bos_env_prefix ~/flare/openpi/.venv/bin/python ~/flare/eval_lora_velocity.py \
        --checkpoint "$final_ckpt" \
        --config-name "$config_name" \
        --variant-label "$variant" \
        --out-dir "$out_dir/velocity_probes" \
        --training-step $final_step \
        --baseline-velocity-dirs \
            ~/flare/results/probe_velocity_T0_LANGUAGE_FOLLOWING \
            ~/flare/results/probe_velocity_T0_NOISE_FLOOR \
            ~/flare/results/probe_velocity_T0_vs_T2prompt \
        ${wandb_run_id:+--wandb-run-id $wandb_run_id} \
        --wandb-project $WANDB_PROJECT \
        2>&1 | tee $vel_log || log "  [warn] velocity eval failed (likely JAX-only model; needs sow patch)"

    log "  [done] $variant — results in $out_dir"
    return 0
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

log ""
log "==================================================================="
log "PRE-FLIGHT CHECKS"
log "==================================================================="

# Disk space + cleanup of stale/partial checkpoints (run BEFORE space check)

# Delete partial C v2 ckpt from earlier disk-out crash (saves ~11 GB)
# Eval results (behavioral_eval.json etc.) stay in ~/flare/results/.../variant_c_natural_cf_lora_v2/
PARTIAL_C_V2_CKPT=$OPENPI_DIR/checkpoints/pi0_libero_cf_lora/variant_c_natural_cf_lora_v2
if [ -d "$PARTIAL_C_V2_CKPT" ]; then
    log "  [cleanup] removing partial C v2 ckpt at $PARTIAL_C_V2_CKPT (we'll retrain fresh as variant_c_v2)"
    rm -rf "$PARTIAL_C_V2_CKPT"
fi

# Delete failed D v2 ckpt dir (the previous attempt that crashed with the 4D-indexing bug)
PARTIAL_D_V2_CKPT=$OPENPI_DIR/checkpoints/pi0_libero_cf_lora_bos_masked/variant_d_v2
if [ -d "$PARTIAL_D_V2_CKPT" ]; then
    log "  [cleanup] removing failed D v2 ckpt at $PARTIAL_D_V2_CKPT"
    rm -rf "$PARTIAL_D_V2_CKPT"
fi

# Force D v2 to re-eval (delete the gate_PASSED marker if any leftover)
rm -f $ABLATION_DIR/variant_d_v2/gate_PASSED 2>/dev/null

# Disk space
free_gb=$(df /home --output=avail -BG | tail -1 | tr -d ' G')
log "  /home free space (after cleanup): ${free_gb} GB"
if [ "$free_gb" -lt 30 ]; then
    log "  [error] need at least 30 GB free on /home (have ${free_gb} GB). Aborting."
    exit 1
fi

# Required scripts
for f in \
    ~/flare/convert_libero_hdf5_to_lerobot.py \
    ~/flare/eval_lora_gate.py \
    ~/flare/eval_lora_behavioral.py \
    ~/flare/eval_lora_velocity.py \
    $OPENPI_DIR/scripts/train.py \
    $OPENPI_DIR/scripts/compute_norm_stats.py; do
    if [ ! -f "$f" ]; then
        log "  [error] missing required file: $f"
        exit 1
    fi
done

# WandB
if [ ! -f ~/.netrc ] || ! grep -q "wandb" ~/.netrc; then
    log "  [warn] wandb may not be logged in; metrics may fail to upload"
fi

log "  ✓ pre-flight passed"

# ============================================================================
# STEP 1: VARIANT C v2 TRAIN + EVAL (same flow as A v2)
#   Trains pi0_libero_cf_lora on the natural CF dataset, 1500 steps batch=16
#   with OOM fallback to batch=12. Gate eval at 30%. Full eval at end.
# ============================================================================

log ""
log "==================================================================="
log "STEP 1: VARIANT C v2 TRAIN + EVAL (natural CF, fresh full training)"
log "==================================================================="

if [ ! -d ~/flare/lerobot_data/living_room_scene2_cf ]; then
    log "  [SKIP C v2] natural CF dataset not found at ~/flare/lerobot_data/living_room_scene2_cf"
else
    compute_norm_stats pi0_libero_cf_lora
    train_gate_eval_variant variant_c_v2 pi0_libero_cf_lora
    C_RESULT=$?
    log "  [C v2 exit code] $C_RESULT  (0=success, 1=train fail, 2=gate fail)"
fi

# ============================================================================
# STEP 2: VARIANT D v2 (BOS attention mask + natural CF) — main intervention
# ============================================================================

log ""
log "==================================================================="
log "STEP 2: VARIANT D v2 — BOS-masked LoRA on natural CF (main intervention)"
log "==================================================================="

# D v2 uses the SAME dataset as C v2 (already converted as living_room_scene2_cf)
if [ ! -d ~/flare/lerobot_data/living_room_scene2_cf ]; then
    log "  [SKIP D v2] natural CF dataset not found at ~/flare/lerobot_data/living_room_scene2_cf"
else
    if ! grep -q "pi0_libero_cf_lora_bos_masked" $OPENPI_DIR/src/openpi/training/config.py; then
        log "  [SKIP D v2] config pi0_libero_cf_lora_bos_masked not in openpi config.py"
        log "             needs pi0.py BOS mask patch + new TrainConfig"
        log "             see INSTRUCTIONS_D_LITE_BOS_MASK.md"
    else
        compute_norm_stats pi0_libero_cf_lora_bos_masked
        train_gate_eval_variant variant_d_v2 pi0_libero_cf_lora_bos_masked
        D_RESULT=$?
        log "  [D v2 exit code] $D_RESULT  (0=success, 1=train fail, 2=gate fail)"

        # After D v2 done, optionally free C v2 ckpt to make room for A v2
        # (Comment this out if you want to keep C v2 for re-eval later)
        # log "  [cleanup] could free $C_V2_CKPT (11GB) to lower peak storage for A v2"
    fi
fi

# ============================================================================
# STEP 3: VARIANT A v2 (negative control — original prompt only)
# ============================================================================

log ""
log "==================================================================="
log "STEP 3: VARIANT A v2 — original T0 demos only (negative control)"
log "==================================================================="

convert_single_prompt_dataset \
    "$LIBERO_HDF5/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5" \
    "put both the alphabet soup and the tomato sauce in the basket" \
    "libero_t0_only"

if ! grep -q "pi0_libero_t0_only_lora" $OPENPI_DIR/src/openpi/training/config.py; then
    log "  [SKIP A v2] config pi0_libero_t0_only_lora not in openpi config.py"
    log "             add it per INSTRUCTIONS_LORA_ABLATION.md before re-running"
else
    compute_norm_stats pi0_libero_t0_only_lora
    train_gate_eval_variant variant_a_v2 pi0_libero_t0_only_lora
    A_RESULT=$?
    log "  [A v2 exit code] $A_RESULT  (0=success, 1=train fail, 2=gate fail)"
fi

# ============================================================================
# STEP 4: COMPARISON REPORT
# ============================================================================

log ""
log "==================================================================="
log "STEP 4: COMPARISON REPORT"
log "==================================================================="

~/flare/openpi/.venv/bin/python ~/flare/compare_lora_variants.py \
    --ablation-dir $ABLATION_DIR \
    --auto-discover \
    --out-dir $ABLATION_DIR/comparison_v3 2>&1 | tee -a $CHAIN_LOG

log ""
log "==================================================================="
log "CHAIN v3 COMPLETE: $(date)"
log "==================================================================="
log "Results: $ABLATION_DIR/"
ls -la $ABLATION_DIR/ | tee -a $CHAIN_LOG
log ""
log "Comparison: $ABLATION_DIR/comparison_v3/"
ls -la $ABLATION_DIR/comparison_v3/ 2>/dev/null | tee -a $CHAIN_LOG
