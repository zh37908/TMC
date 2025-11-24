#!/usr/bin/env bash
set -euo pipefail

# Config
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"
TESTER_SNR="$ROOT/TMC/ETMC_TPAMI/test/test_TMC_channel_snr_retransmit.py"
TESTER_BASE="$ROOT/TMC/ETMC_TPAMI/test/test_TMC_channel_baseline_eval.py"
DATAPATH="$ROOT/nyud2"

# declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins" "No_pretrain")
declare -a PRETRAINS=("DeCUR")
declare -a RUNS=(1 2 3 4 5)
declare -a VIEWS=("rgb" "depth")
declare -a NOISE_POWERS=(1 10 100 1000)

OUT_CSV="$ROOT/TMC/ETMC_TPAMI/test/results_tmc_noisy.csv"
echo "model,pretrain,run,view,noise_power,acc_pre,acc_post,acc,ckpt_sha256,savedir" > "$OUT_CSV"

# Uncertainty model (TMC_channel_snr), saved under none/snr10
METHOD="none"
TRAIN_SNR=10
for pt in "${PRETRAINS[@]}"; do
  for run in "${RUNS[@]}"; do
    SAVEDIR="$ROOT/savepath/TMC_channel_snr/nyud/${METHOD}/snr${TRAIN_SNR}/pretrain_${pt}/run${run}/train_TMC_channel_snr_${METHOD}_${pt}_snr${TRAIN_SNR}_run${run}"
    if [ ! -d "$SAVEDIR" ]; then
      continue
    fi
    CKPT=""
    if [ -f "$SAVEDIR/model_best.pt" ]; then
      CKPT="$SAVEDIR/model_best.pt"
    elif [ -f "$SAVEDIR/checkpoint.pt" ]; then
      CKPT="$SAVEDIR/checkpoint.pt"
    else
      CAND=$(ls -1 "$SAVEDIR"/*.pt 2>/dev/null | head -n 1 || true)
      if [ -n "${CAND:-}" ]; then
        CKPT="$CAND"
      fi
    fi
    if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
      CKPT_SHA=$(sha256sum "$CKPT" | awk '{print $1}')
    else
      CKPT_SHA="NA"
    fi
    for view in "${VIEWS[@]}"; do
      for pow in "${NOISE_POWERS[@]}"; do
        RAW=$("$PYTHON" "$TESTER_SNR" \
          --savedir "$SAVEDIR" \
          --data_path "$DATAPATH" \
          --split test \
          --batch_sz 32 --n_workers 8 \
          --retransmit \
          --reject_score u_evi \
          --rt_trigger high \
          --target_coverage 0.9 \
          --rt_max_trials 3 \
          --snr "$TRAIN_SNR" \
          --rt_mode ds \
          --noisy_view "$view" \
          --noise_power "$pow" \
          --report_selective | cat)
        ACC_PRE=$(echo "$RAW" | grep -E "^acc \(pre-rt\):" | awk '{print $3}')
        ACC_POST=$(echo "$RAW" | grep -E "^acc \(post-rt\):" | awk '{print $3}')
        echo "tmc_channel_snr,${pt},${run},${view},${pow},${ACC_PRE},${ACC_POST},,${CKPT_SHA},${SAVEDIR}" >> "$OUT_CSV"
      done
    done
  done
done

# Baseline model (TMC_channel), saved under static_snr10
for pt in "${PRETRAINS[@]}"; do
  for run in "${RUNS[@]}"; do
    SAVEDIR="$ROOT/savepath/TMC_channel/nyud/static_snr${TRAIN_SNR}/pretrain_${pt}/run${run}/train_TMC_channel_baseline_${pt}_snr${TRAIN_SNR}_run${run}"
    if [ ! -d "$SAVEDIR" ]; then
      continue
    fi
    CKPT=""
    if [ -f "$SAVEDIR/model_best.pt" ]; then
      CKPT="$SAVEDIR/model_best.pt"
    elif [ -f "$SAVEDIR/checkpoint.pt" ]; then
      CKPT="$SAVEDIR/checkpoint.pt"
    else
      CAND=$(ls -1 "$SAVEDIR"/*.pt 2>/dev/null | head -n 1 || true)
      if [ -n "${CAND:-}" ]; then
        CKPT="$CAND"
      fi
    fi
    if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
      CKPT_SHA=$(sha256sum "$CKPT" | awk '{print $1}')
    else
      CKPT_SHA="NA"
    fi
    for view in "${VIEWS[@]}"; do
      for pow in "${NOISE_POWERS[@]}"; do
        RAW=$("$PYTHON" "$TESTER_BASE" \
          --savedir "$SAVEDIR" \
          --data_path "$DATAPATH" \
          --split test \
          --batch_sz 32 --n_workers 8 \
          --snr "$TRAIN_SNR" \
          --noisy_view "$view" \
          --noise_power "$pow" | cat)
        ACC=$(echo "$RAW" | grep -E "^acc:" | awk '{print $2}')
        echo "tmc_channel,${pt},${run},${view},${pow},,,${ACC},${CKPT_SHA},${SAVEDIR}" >> "$OUT_CSV"
      done
    done
  done
done

echo "Saved noisy evaluation results to: $OUT_CSV"


