#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"
TESTER_BASE="$ROOT/TMC/ETMC_TPAMI/test/test_TMC_channel_baseline_eval.py"
DATAPATH="$ROOT/nyud2"

PRETRAIN="DeCUR"
RUNS=(1 2)
STATIC_SNRS=(0 10 20)
DYN_MIN=0
DYN_MAX=20
NOISE_POWERS=(1 10 100 1000)

OUT_DIR="$ROOT/TMC/ETMC_TPAMI/plots/uncert_density/${PRETRAIN}"
mkdir -p "$OUT_DIR"

# Static SNR density
for run in "${RUNS[@]}"; do
  SAVEDIR="$ROOT/savepath/TMC_channel/nyud/static_snr10/pretrain_${PRETRAIN}/run${run}/train_TMC_channel_baseline_${PRETRAIN}_snr10_run${run}"
  if [ ! -d "$SAVEDIR" ]; then
    echo "[WARN] Savedir not found: $SAVEDIR" >&2
    continue
  fi
  for SNR in "${STATIC_SNRS[@]}"; do
    OUTPNG="$OUT_DIR/static_snr${SNR}_run${run}.png"
    "$PYTHON" "$TESTER_BASE" \
      --savedir "$SAVEDIR" \
      --data_path "$DATAPATH" \
      --split test \
      --batch_sz 32 --n_workers 8 \
      --snr "$SNR" \
      --save_uncert_density \
      --uncert_density_path "$OUTPNG" | cat
  done

done

# Dynamic SNR density
for run in "${RUNS[@]}"; do
  SAVEDIR_DYN="$ROOT/savepath/TMC_channel/nyud/dynamic_${DYN_MIN}_${DYN_MAX}/pretrain_${PRETRAIN}/run${run}/train_TMC_channel_baseline_${PRETRAIN}_dynamic_${DYN_MIN}_${DYN_MAX}_run${run}"
  if [ -d "$SAVEDIR_DYN" ]; then
    OUTPNG="$OUT_DIR/dynamic_${DYN_MIN}_${DYN_MAX}_run${run}.png"
    "$PYTHON" "$TESTER_BASE" \
      --savedir "$SAVEDIR_DYN" \
      --data_path "$DATAPATH" \
      --split test \
      --batch_sz 32 --n_workers 8 \
      --use_dynamic_snr \
      --snr_min "$DYN_MIN" \
      --snr_max "$DYN_MAX" \
      --save_uncert_density \
      --uncert_density_path "$OUTPNG" | cat
  fi
done

# Noisy input tests at fixed 10 dB, single-view noise
for run in "${RUNS[@]}"; do
  SAVEDIR_NOISE="$ROOT/savepath/TMC_channel/nyud/static_snr10/pretrain_${PRETRAIN}/run${run}/train_TMC_channel_baseline_${PRETRAIN}_snr10_run${run}"
  if [ -d "$SAVEDIR_NOISE" ]; then
    for POW in "${NOISE_POWERS[@]}"; do
      for VIEW in rgb depth; do
        OUTPNG="$OUT_DIR/noisy_${VIEW}_pow${POW}_run${run}.png"
        "$PYTHON" "$TESTER_BASE" \
          --savedir "$SAVEDIR_NOISE" \
          --data_path "$DATAPATH" \
          --split test \
          --batch_sz 32 --n_workers 8 \
          --snr 10 \
          --noisy_view "$VIEW" \
          --noise_power "$POW" \
          --save_uncert_density \
          --uncert_density_path "$OUTPNG" | cat
      done
    done
  fi
done

echo "Saved uncertainty density figures under: $OUT_DIR"
