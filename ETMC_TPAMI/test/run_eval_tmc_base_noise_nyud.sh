#!/usr/bin/env bash
set -euo pipefail

# Config
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"
TESTER="$ROOT/TMC/ETMC_TPAMI/test/test_TMC_base_channel_nyud.py"
SAVEDIR="/home/hzhaobi/Multired/savepath/TMC_base/nyud/ReleasedVersion"
DATAPATH="/home/hzhaobi/Multired/nyud2"

VIEWS=(rgb depth)
POWERS=(1 10 100)

OUT_CSV="$ROOT/TMC/ETMC_TPAMI/test/results_tmc_base_noise_nyud.csv"
echo "view,noise_power,acc,savedir" > "$OUT_CSV"

for VIEW in "${VIEWS[@]}"; do
  for POW in "${POWERS[@]}"; do
    echo "--- ${VIEW}, power=${POW} ---"
    RAW=$("$PYTHON" "$TESTER" \
      --savedir "$SAVEDIR" \
      --data_path "$DATAPATH" \
      --split test \
      --batch_sz 32 --n_workers 8 \
      --noisy_view "$VIEW" \
      --noise_power "$POW" \
      --noise_stage dataloader | cat)
    ACC=$(echo "$RAW" | grep -E "^acc " | awk '{print $NF}')
    echo "${VIEW},${POW},${ACC},${SAVEDIR}" >> "$OUT_CSV"
  done
done

echo "Saved results to: $OUT_CSV"


