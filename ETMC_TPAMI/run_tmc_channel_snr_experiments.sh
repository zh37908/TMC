#!/usr/bin/env bash
set -euo pipefail

# Absolute paths preferred
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"

MODEL="$ROOT/TMC/ETMC_TPAMI/train_TMC_nyud2.py"

declare -a SNR_INPUT_METHODS=("none")
declare -a SNRS=(10 20)
declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins" "No_pretrain")

for method in "${SNR_INPUT_METHODS[@]}"; do
  for snr in "${SNRS[@]}"; do
    for pt in "${PRETRAINS[@]}"; do
      for run in 1; do
        NAME="train_TMC_channel_snr_${method}_${pt}_snr${snr}_run${run}"
        BASE_SAVEDIR="$ROOT/savepath/TMC_channel_snr/nyud"
        EXP_SAVEDIR="$BASE_SAVEDIR/${method}/snr${snr}/pretrain_${pt}/run${run}"
        echo "=== Running TMC_channel_snr | method=${method} | pretrain=${pt} | snr=${snr} | run=${run} ==="
        # Skip if already exists to avoid overwriting
        if [ -d "$EXP_SAVEDIR" ] && { [ -f "$EXP_SAVEDIR/model_best.pt" ] || [ -f "$EXP_SAVEDIR/checkpoint.pt" ]; }; then
          echo "[SKIP] Exists: $EXP_SAVEDIR"
          continue
        fi
        "$PYTHON" "$MODEL" \
          --snr_input_method "$method" \
          --channel_snr "$snr" \
          --pretrain "$pt" \
          --savedir "$EXP_SAVEDIR" \
          --name "$NAME" \
          --seed "$run" \
          --model tmc_channel_snr \
          --lr 1e-1
      done
    done
  done
done


# Dynamic SNR experiments
SNR_MIN=0
SNR_MAX=20
for method in "${SNR_INPUT_METHODS[@]}"; do
  for pt in "${PRETRAINS[@]}"; do
    for run in 1; do
      NAME="train_TMC_channel_snr_${method}_${pt}_dynamic_${SNR_MIN}_${SNR_MAX}_run${run}"
      BASE_SAVEDIR="$ROOT/savepath/TMC_channel_snr/nyud"
      EXP_SAVEDIR="$BASE_SAVEDIR/${method}/dynamic_${SNR_MIN}_${SNR_MAX}/pretrain_${pt}/run${run}"
      echo "=== Running TMC_channel_snr (DYNAMIC) | method=${method} | pretrain=${pt} | range=[${SNR_MIN}, ${SNR_MAX}] | run=${run} ==="
      if [ -d "$EXP_SAVEDIR" ] && { [ -f "$EXP_SAVEDIR/model_best.pt" ] || [ -f "$EXP_SAVEDIR/checkpoint.pt" ]; }; then
        echo "[SKIP] Exists: $EXP_SAVEDIR"
        continue
      fi
      "$PYTHON" "$MODEL" \
        --snr_input_method "$method" \
        --use_dynamic_snr \
        --snr_min "$SNR_MIN" \
        --snr_max "$SNR_MAX" \
        --pretrain "$pt" \
        --savedir "$EXP_SAVEDIR" \
        --name "$NAME" \
        --seed "$run" \
        --model tmc_channel_snr \
        --lr 1e-1
    done
  done
done
# Baseline TMC_channel experiments (no SNR embedding), static SNRs
declare -a BASE_SNRS=(10 20)
for snr in "${BASE_SNRS[@]}"; do
  for pt in "${PRETRAINS[@]}"; do
    for run in 1; do
      NAME="train_TMC_channel_baseline_${pt}_snr${snr}_run${run}"
      BASE_SAVEDIR="$ROOT/savepath/TMC_channel/nyud"
      EXP_SAVEDIR="$BASE_SAVEDIR/static_snr${snr}/pretrain_${pt}/run${run}"
      echo "=== Running TMC_channel BASELINE | pretrain=${pt} | snr=${snr} | run=${run} ==="
      if [ -d "$EXP_SAVEDIR" ] && { [ -f "$EXP_SAVEDIR/model_best.pt" ] || [ -f "$EXP_SAVEDIR/checkpoint.pt" ]; }; then
        echo "[SKIP] Exists: $EXP_SAVEDIR"
        continue
      fi
      "$PYTHON" "$MODEL" \
        --channel_snr "$snr" \
        --pretrain "$pt" \
        --savedir "$EXP_SAVEDIR" \
        --name "$NAME" \
        --seed "$run" \
        --model tmc_channel \
        --lr 1e-1
    done
  done
done

# Baseline TMC_channel experiments (dynamic SNR range)
BASE_SNR_MIN=0
BASE_SNR_MAX=20
for pt in "${PRETRAINS[@]}"; do
  for run in 1; do
    NAME="train_TMC_channel_baseline_${pt}_dynamic_${BASE_SNR_MIN}_${BASE_SNR_MAX}_run${run}"
    BASE_SAVEDIR="$ROOT/savepath/TMC_channel/nyud"
    EXP_SAVEDIR="$BASE_SAVEDIR/dynamic_${BASE_SNR_MIN}_${BASE_SNR_MAX}/pretrain_${pt}/run${run}"
    echo "=== Running TMC_channel BASELINE (DYNAMIC) | pretrain=${pt} | range=[${BASE_SNR_MIN}, ${BASE_SNR_MAX}] | run=${run} ==="
    if [ -d "$EXP_SAVEDIR" ] && { [ -f "$EXP_SAVEDIR/model_best.pt" ] || [ -f "$EXP_SAVEDIR/checkpoint.pt" ]; }; then
      echo "[SKIP] Exists: $EXP_SAVEDIR"
      continue
    fi
    "$PYTHON" "$MODEL" \
      --use_dynamic_snr \
      --snr_min "$BASE_SNR_MIN" \
      --snr_max "$BASE_SNR_MAX" \
      --pretrain "$pt" \
      --savedir "$EXP_SAVEDIR" \
      --name "$NAME" \
      --seed "$run" \
      --model tmc_channel \
      --lr 1e-1
  done
done


