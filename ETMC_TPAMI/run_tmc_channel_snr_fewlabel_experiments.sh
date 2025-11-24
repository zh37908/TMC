#!/usr/bin/env bash
set -euo pipefail

# Absolute paths preferred
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"

MODEL="$ROOT/TMC/ETMC_TPAMI/train_TMC_nyud2.py"

METHOD="none"
declare -a SNRS=(10 20)
LR="1e-2"
declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins" "No_pretrain")

for SNR in "${SNRS[@]}"; do
  for pt in "${PRETRAINS[@]}"; do
    for run in 1 2; do
      NAME="train_TMC_channel_snr_fewlabel_${METHOD}_${pt}_snr${SNR}_lr${LR}_run${run}"
      BASE_SAVEDIR="$ROOT/savepath/TMC_channel_snr/nyud/${METHOD}/few-label/snr${SNR}/pretrain_${pt}/lr_${LR}"
      EXP_SAVEDIR="$BASE_SAVEDIR/run${run}"
      echo "=== Running FEW-LABEL TMC_channel_snr | method=${METHOD} | pretrain=${pt} | snr=${SNR} | lr=${LR} | run=${run} ==="
      if [ -d "$EXP_SAVEDIR" ] && { [ -f "$EXP_SAVEDIR/model_best.pt" ] || [ -f "$EXP_SAVEDIR/checkpoint.pt" ]; }; then
        echo "[SKIP] Exists: $EXP_SAVEDIR"
        continue
      fi
      "$PYTHON" "$MODEL" \
        --snr_input_method "$METHOD" \
        --channel_snr "$SNR" \
        --pretrain "$pt" \
        --lr "$LR" \
        --batch_sz 16 \
        --savedir "$EXP_SAVEDIR" \
        --name "$NAME" \
        --seed "$run" \
        --model tmc_channel_snr \
        --label_fraction 0.5
    done
  done
done


