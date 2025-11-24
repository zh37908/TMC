#!/usr/bin/env bash
set -euo pipefail

# ETMC Channel Pretrain Experiments
# Run ETMC experiments with different pretraining methods

# Absolute paths preferred
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"
TRAINER_NYUD="$ROOT/TMC/ETMC_TPAMI/train_ETMC_channel_nyud2.py"
TRAINER_SUNRGBD="$ROOT/TMC/ETMC_TPAMI/train_ETMC_channel_sunrgbd.py"

declare -a DATASETS=("nyud")
declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins" "No_pretrain")
declare -a SNRS=(0 10 20)

echo "=== Starting ETMC Channel Pretrain Experiments ==="
echo "Python: $PYTHON"
echo "Root: $ROOT"
echo "Datasets: ${DATASETS[*]}"
echo "Pretrains: ${PRETRAINS[*]}"
echo "SNRs: ${SNRS[*]}"
echo ""

for dataset in "${DATASETS[@]}"; do
  for pretrain in "${PRETRAINS[@]}"; do
    for snr in "${SNRS[@]}"; do
      for run in 1; do
        NAME="train_ETMC_channel_${dataset}_${pretrain}_static_snr${snr}_run${run}"
        BASE_SAVEDIR="$ROOT/savepath/ETMC_channel/${dataset}"
        EXP_SAVEDIR="$BASE_SAVEDIR/${pretrain}/static/snr${snr}/run${run}"
        
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
        
        echo "=== Running ETMC_channel | dataset=${dataset} | pretrain=${pretrain} | static SNR=${snr} dB | run=${run} ==="
        echo "Trainer: $TRAINER"
        echo "Data path: $DATA_PATH"
        echo "Save dir: $EXP_SAVEDIR"
        echo ""
        
        "$PYTHON" "$TRAINER" \
          --channel_mode static \
          --channel_snr "$snr" \
          --pretrain "$pretrain" \
          --data_path "$DATA_PATH" \
          --n_classes "$N_CLASSES" \
          --savedir "$EXP_SAVEDIR" \
          --name "$NAME" \
          --seed "$run"
        
        echo "=== Completed ETMC_channel | dataset=${dataset} | pretrain=${pretrain} | static SNR=${snr} dB | run=${run} ==="
        echo ""
      done
    done
  done
done

echo "=== All ETMC Channel Pretrain Experiments Completed ==="
