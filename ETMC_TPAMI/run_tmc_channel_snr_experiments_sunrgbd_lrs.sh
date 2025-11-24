#!/usr/bin/env bash
set -euo pipefail

# Absolute paths preferred
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"

MODEL="$ROOT/TMC/ETMC_TPAMI/train_TMC_sunrgbd.py"

declare -a SNR_INPUT_METHODS=("none")
declare -a SNRS=(10)
# 如需只跑某些预训练，可在此调整
declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins" "No_pretrain")
declare -a LRS=("1e-1" "5e-2" "1e-2")

# Static SNR experiments (tmc_channel_snr)
for method in "${SNR_INPUT_METHODS[@]}"; do
  for snr in "${SNRS[@]}"; do
    for pt in "${PRETRAINS[@]}"; do
      for lr in "${LRS[@]}"; do
        for run in 1 2 ; do
          NAME="train_TMC_channel_snr_${method}_${pt}_snr${snr}_lr${lr}_run${run}"
          BASE_SAVEDIR="$ROOT/savepath/TMC_channel_snr/sunrgbd"
          EXP_SAVEDIR="$BASE_SAVEDIR/${method}/snr${snr}/pretrain_${pt}/lr_${lr}/run${run}"
          echo "=== Running TMC_channel_snr | method=${method} | pretrain=${pt} | snr=${snr} | lr=${lr} | run=${run} ==="
          if [ -d "$EXP_SAVEDIR" ] && { [ -f "$EXP_SAVEDIR/model_best.pt" ] || [ -f "$EXP_SAVEDIR/checkpoint.pt" ]; }; then
            echo "[SKIP] Exists: $EXP_SAVEDIR"
            continue
          fi
          "$PYTHON" "$MODEL" \
            --snr_input_method "$method" \
            --channel_snr "$snr" \
            --pretrain "$pt" \
            --lr "$lr" \
            --savedir "$EXP_SAVEDIR" \
            --name "$NAME" \
            --seed "$run" \
            --model tmc_channel_snr
        done
      done
    done
  done
done


