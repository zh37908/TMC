#!/usr/bin/env bash
set -euo pipefail

# Config
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"
TESTER="$ROOT/TMC/ETMC_TPAMI/test/test_TMC_channel_snr_retransmit.py"
DATAPATH="$ROOT/nyud2"

# Target savedir (single run)
SAVEDIR="$ROOT/savepath/TMC/nyud/original/Adam/pretrain_resnet18/ReleasedVersion"

# Evaluation SNRs
STATIC_SNRS=(0 10 20)
DYN_MIN=0
DYN_MAX=20
# Noise powers to test (variance) for source input noise
NOISE_POWERS=(1 10 100 1000)

# Output CSV
OUT_CSV="$ROOT/TMC/ETMC_TPAMI/test/results_original_pretrain_resnet18.csv"
echo "pretrain,run,method,train_group,snr_mode,snr,acc_pre,acc_post,savedir" > "$OUT_CSV"

echo "=== Testing pretrain=resnet18 run=1 group=original ==="

# Static evaluations for 0,10,20 dB
for SNR in "${STATIC_SNRS[@]}"; do
  echo "--- static SNR=${SNR} dB ---"
  RAW=$("$PYTHON" "$TESTER" \
    --savedir "$SAVEDIR" \
    --data_path "$DATAPATH" \
    --split test \
    --batch_sz 32 --n_workers 8 \
    --ignore_channel \
    --retransmit \
    --reject_score u_evi \
    --rt_trigger high \
    --target_coverage 0.9 \
    --rt_max_trials 3 \
    --snr "$SNR" \
    --rt_mode ds \
    --report_selective | cat)
  ACC_PRE=$(echo "$RAW" | grep -E "^acc \(pre-rt\):" | awk '{print $3}')
  ACC_POST=$(echo "$RAW" | grep -E "^acc \(post-rt\):" | awk '{print $3}')
  echo "resnet18,1,original,pretrain_resnet18,static,${SNR},${ACC_PRE},${ACC_POST},${SAVEDIR}" >> "$OUT_CSV"
done

# Dynamic evaluation for [DYN_MIN, DYN_MAX]
echo "--- dynamic SNR=[${DYN_MIN},${DYN_MAX}] dB ---"
RAW=$("$PYTHON" "$TESTER" \
  --savedir "$SAVEDIR" \
  --data_path "$DATAPATH" \
  --split test \
  --batch_sz 32 --n_workers 8 \
  --ignore_channel \
  --retransmit \
  --reject_score u_evi \
  --rt_trigger high \
  --target_coverage 0.9 \
  --rt_max_trials 3 \
  --use_dynamic_snr \
  --snr_min "$DYN_MIN" \
  --snr_max "$DYN_MAX" \
  --rt_mode ds \
  --report_selective | cat)
ACC_PRE=$(echo "$RAW" | grep -E "^acc \(pre-rt\):" | awk '{print $3}')
ACC_POST=$(echo "$RAW" | grep -E "^acc \(post-rt\):" | awk '{print $3}')
echo "resnet18,1,original,pretrain_resnet18,dynamic,${DYN_MIN}-${DYN_MAX},${ACC_PRE},${ACC_POST},${SAVEDIR}" >> "$OUT_CSV"

# Noisy input tests (channel fixed at 10 dB), noise added to one view at a time
for POW in "${NOISE_POWERS[@]}"; do
  for VIEW in rgb depth; do
    echo "--- noisy ${VIEW}, pow=${POW} ---"
    RAW=$("$PYTHON" "$TESTER" \
      --savedir "$SAVEDIR" \
      --data_path "$DATAPATH" \
      --split test \
      --batch_sz 32 --n_workers 8 \
      --ignore_channel \
      --retransmit \
      --reject_score u_evi \
      --rt_trigger high \
      --target_coverage 0.9 \
      --rt_max_trials 3 \
      --snr 10 \
      --rt_mode ds \
      --noisy_view "$VIEW" \
      --noise_power "$POW" \
      --report_selective | cat)
    ACC_PRE=$(echo "$RAW" | grep -E "^acc \(pre-rt\):" | awk '{print $3}')
    ACC_POST=$(echo "$RAW" | grep -E "^acc \(post-rt\):" | awk '{print $3}')
    echo "resnet18,1,original,pretrain_resnet18,noisy_${VIEW},${POW},${ACC_PRE},${ACC_POST},${SAVEDIR}" >> "$OUT_CSV"
  done
done

echo "Saved results to: $OUT_CSV"


