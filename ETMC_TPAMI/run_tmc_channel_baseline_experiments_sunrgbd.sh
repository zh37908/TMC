#!/usr/bin/env bash
set -euo pipefail

# Absolute paths preferred
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"

MODEL="$ROOT/TMC/ETMC_TPAMI/train_TMC_sunrgbd.py"

declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins")
declare -a RUNS=(1 2)

# Static SNR baseline training on SUNRGBD (no SNR embedding inside the model)
declare -a BASE_SNRS=(10 20)
LR="1e-1"

for snr in "${BASE_SNRS[@]}"; do
  for pt in "${PRETRAINS[@]}"; do
    for run in "${RUNS[@]}"; do
      NAME="train_TMC_channel_baseline_${pt}_snr${snr}_lr${LR}_run${run}"
      BASE_SAVEDIR="$ROOT/savepath/TMC_channel/sunrgbd/static_snr${snr}/pretrain_${pt}/lr_${LR}"
      EXP_SAVEDIR="$BASE_SAVEDIR/run${run}"
      echo "=== Running SUNRGBD TMC_channel BASELINE | pretrain=${pt} | snr=${snr} | lr=${LR} | run=${run} ==="
      if [ -d "$EXP_SAVEDIR" ] && { [ -f "$EXP_SAVEDIR/model_best.pt" ] || [ -f "$EXP_SAVEDIR/checkpoint.pt" ]; }; then
        echo "[SKIP] Exists: $EXP_SAVEDIR"
        continue
      fi
      mkdir -p "$EXP_SAVEDIR"
      "$PYTHON" "$MODEL" \
        --channel_snr "$snr" \
        --pretrain "$pt" \
        --lr "$LR" \
        --batch_sz 16 \
        --savedir "$EXP_SAVEDIR" \
        --name "$NAME" \
        --seed "$run" \
        --model tmc_channel
    done
  done
done

echo "All SUNRGBD TMC_channel baseline runs completed."


