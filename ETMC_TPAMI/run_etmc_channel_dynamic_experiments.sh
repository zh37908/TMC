#!/usr/bin/env bash
set -euo pipefail

# ETMC Channel Dynamic Experiments
# Based on TMC dynamic experiments but for ETMC_channel models

# Absolute paths preferred
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"
TRAINER_NYUD="$ROOT/TMC/ETMC_TPAMI/train_ETMC_channel_nyud2.py"
TRAINER_SUNRGBD="$ROOT/TMC/ETMC_TPAMI/train_ETMC_channel_sunrgbd.py"

declare -a DATASETS=("nyud" "sunrgbd")

# SNR 区间（可用环境变量覆盖）
SNR_MIN=${SNR_MIN:-0}
SNR_MAX=${SNR_MAX:-20}
RANGE_TAG="range${SNR_MIN}-${SNR_MAX}"
PRETRAIN=${PRETRAIN:-"DeCUR"}  # Default pretrain method, can be overridden by environment variable

echo "=== Starting ETMC Channel Dynamic Experiments ==="
echo "Python: $PYTHON"
echo "Root: $ROOT"
echo "Datasets: ${DATASETS[*]}"
echo "SNR Range: [${SNR_MIN}, ${SNR_MAX}] dB"
echo "Range Tag: $RANGE_TAG"
echo "Pretrain: $PRETRAIN"
echo ""

for dataset in "${DATASETS[@]}"; do
  for run in 1; do
    NAME="train_ETMC_channel_${dataset}_${PRETRAIN}_${RANGE_TAG}_dynamic_run${run}"
    BASE_SAVEDIR="$ROOT/savepath/ETMC_channel_dynamic/${dataset}"
    EXP_SAVEDIR="$BASE_SAVEDIR/${PRETRAIN}/${RANGE_TAG}/dynamic/run${run}"
    
    # Select trainer and data path based on dataset
    if [[ "$dataset" == "nyud" ]]; then
      TRAINER="$TRAINER_NYUD"
      DATA_PATH="/home/hzhaobi/Multired/nyud2"
      N_CLASSES=10
    else
      TRAINER="$TRAINER_SUNRGBD"
      DATA_PATH="/home/hzhaobi/Multired/sunrgbd"
      N_CLASSES=19
    fi
    
    echo "=== Running ETMC_channel | dataset=${dataset} | pretrain=${PRETRAIN} | dynamic SNR [${SNR_MIN}, ${SNR_MAX}] dB | run=${run} ==="
    echo "Trainer: $TRAINER"
    echo "Data path: $DATA_PATH"
    echo "Save dir: $EXP_SAVEDIR"
    echo ""
    
    "$PYTHON" "$TRAINER" \
      --channel_mode dynamic \
      --snr_min "$SNR_MIN" \
      --snr_max "$SNR_MAX" \
      --pretrain "$PRETRAIN" \
      --data_path "$DATA_PATH" \
      --n_classes "$N_CLASSES" \
      --savedir "$EXP_SAVEDIR" \
      --name "$NAME" \
      --seed "$run"
    
    echo "=== Completed ETMC_channel | dataset=${dataset} | pretrain=${PRETRAIN} | dynamic SNR [${SNR_MIN}, ${SNR_MAX}] dB | run=${run} ==="
    echo ""
  done
done

echo "=== All ETMC Channel Dynamic Experiments Completed ==="
