#!/usr/bin/env bash
set -euo pipefail

# ETMC Training Examples
# This script shows how to use the ETMC channel training scripts

PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"

echo "=== ETMC Channel Training Examples ==="
echo ""

# Example 1: Train ETMC_channel on NYUD2 with static SNR=20dB and DeCUR pretrain
echo "Example 1: ETMC_channel NYUD2 Static SNR=20dB with DeCUR pretrain"
echo "Command:"
echo "$PYTHON $ROOT/TMC/ETMC_TPAMI/train_ETMC_channel_nyud2.py \\"
echo "  --channel_mode static \\"
echo "  --channel_snr 20.0 \\"
echo "  --pretrain DeCUR \\"
echo "  --data_path /home/hzhaobi/Multired/nyud2 \\"
echo "  --n_classes 10 \\"
echo "  --savedir /home/hzhaobi/Multired/savepath/ETMC_channel/nyud/DeCUR/static/snr20/run1 \\"
echo "  --name train_ETMC_channel_nyud_DeCUR_static_snr20_run1 \\"
echo "  --seed 1"
echo ""

# Example 2: Train ETMC_channel on NYUD2 with dynamic SNR [0,20]dB and SimCLR pretrain
echo "Example 2: ETMC_channel NYUD2 Dynamic SNR [0,20]dB with SimCLR pretrain"
echo "Command:"
echo "$PYTHON $ROOT/TMC/ETMC_TPAMI/train_ETMC_channel_nyud2.py \\"
echo "  --channel_mode dynamic \\"
echo "  --snr_min 0.0 \\"
echo "  --snr_max 20.0 \\"
echo "  --pretrain SimCLR \\"
echo "  --data_path /home/hzhaobi/Multired/nyud2 \\"
echo "  --n_classes 10 \\"
echo "  --savedir /home/hzhaobi/Multired/savepath/ETMC_channel_dynamic/nyud/SimCLR/range0-20/dynamic/run1 \\"
echo "  --name train_ETMC_channel_nyud_SimCLR_range0-20_dynamic_run1 \\"
echo "  --seed 1"
echo ""

# Example 3: Train ETMC_channel on SUN RGB-D with static SNR=15dB and No pretrain
echo "Example 3: ETMC_channel SUN RGB-D Static SNR=15dB with No pretrain"
echo "Command:"
echo "$PYTHON $ROOT/TMC/ETMC_TPAMI/train_ETMC_channel_sunrgbd.py \\"
echo "  --channel_mode static \\"
echo "  --channel_snr 15.0 \\"
echo "  --pretrain No_pretrain \\"
echo "  --data_path /home/hzhaobi/Multired/sunrgbd \\"
echo "  --n_classes 19 \\"
echo "  --savedir /home/hzhaobi/Multired/savepath/ETMC_channel/sunrgbd/No_pretrain/static/snr15/run1 \\"
echo "  --name train_ETMC_channel_sunrgbd_No_pretrain_static_snr15_run1 \\"
echo "  --seed 1"
echo ""

# Example 4: Run batch static experiments with DeCUR pretrain
echo "Example 4: Batch Static Experiments with DeCUR pretrain"
echo "Command:"
echo "PRETRAIN=DeCUR bash $ROOT/TMC/ETMC_TPAMI/run_etmc_channel_static_experiments.sh"
echo ""

# Example 5: Run batch static experiments with BarlowTwins pretrain
echo "Example 5: Batch Static Experiments with BarlowTwins pretrain"
echo "Command:"
echo "PRETRAIN=BarlowTwins bash $ROOT/TMC/ETMC_TPAMI/run_etmc_channel_static_experiments.sh"
echo ""

# Example 6: Run batch dynamic experiments with custom SNR range and pretrain
echo "Example 6: Batch Dynamic Experiments (SNR [5,15]dB) with SimCLR pretrain"
echo "Command:"
echo "SNR_MIN=5 SNR_MAX=15 PRETRAIN=SimCLR bash $ROOT/TMC/ETMC_TPAMI/run_etmc_channel_dynamic_experiments.sh"
echo ""

# Example 7: Run comprehensive pretrain experiments
echo "Example 7: Comprehensive Pretrain Experiments (all pretrain methods)"
echo "Command:"
echo "bash $ROOT/TMC/ETMC_TPAMI/run_etmc_channel_pretrain_experiments.sh"
echo ""

echo "=== Key Parameters ==="
echo "--channel_mode: 'static' or 'dynamic'"
echo "--channel_snr: Fixed SNR value for static mode (e.g., 20.0)"
echo "--snr_min/--snr_max: SNR range for dynamic mode (e.g., 0.0 to 20.0)"
echo "--pretrain: Pretraining method ('DeCUR', 'SimCLR', 'BarlowTwins', 'No_pretrain')"
echo "--freeze_encoder: Whether to freeze encoder parameters (0 or 1)"
echo "--data_path: Path to dataset directory"
echo "--n_classes: Number of classes (10 for NYUD2, 19 for SUN RGB-D)"
echo "--savedir: Directory to save model checkpoints"
echo "--name: Experiment name"
echo "--seed: Random seed for reproducibility"
echo ""

echo "=== Dataset Paths ==="
echo "NYUD2: /home/hzhaobi/Multired/nyud2"
echo "SUN RGB-D: /home/hzhaobi/Multired/sunrgbd"
echo ""

echo "=== Output Structure ==="
echo "Each experiment will create:"
echo "- checkpoint.pt: Latest model checkpoint"
echo "- model_best.pt: Best model checkpoint"
echo "- args.pt: Training arguments"
echo "- logfile.log: Training log"
