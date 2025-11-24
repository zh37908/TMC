#!/usr/bin/env bash
set -euo pipefail

# Config
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"
TESTER="$ROOT/TMC/ETMC_TPAMI/test/test_TMC_channel_baseline_eval.py"
DATAPATH="$ROOT/nyud2"

declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins" "No_pretrain")
declare -a RUNS=(1 2)
STATIC_SNRS=(0 10 20)
DYN_MIN=0
DYN_MAX=20
NOISE_POWERS=(1 10 100 1000)

OUT_CSV="$ROOT/TMC/ETMC_TPAMI/test/results_tmc_channel_baseline_pretrains.csv"
echo "pretrain,run,snr_mode,snr,acc,depth_acc,rgb_acc,ckpt_sha256,savedir" > "$OUT_CSV"

# Static SNR results from baseline training savedirs
for pt in "${PRETRAINS[@]}"; do
  for run in "${RUNS[@]}"; do
    SAVEDIR_BASE="$ROOT/savepath/TMC_channel/nyud"

    # static evals
    for SNR in "${STATIC_SNRS[@]}"; do
      SAVEDIR="$SAVEDIR_BASE/static_snr10/pretrain_${pt}/run${run}/train_TMC_channel_baseline_${pt}_snr10_run${run}"
      if [ ! -d "$SAVEDIR" ]; then
        echo "[WARN] Savedir not found: $SAVEDIR" >&2
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
      RAW=$("$PYTHON" "$TESTER" \
        --savedir "$SAVEDIR" \
        --data_path "$DATAPATH" \
        --split test \
        --batch_sz 32 --n_workers 8 \
        --snr "$SNR" | cat)
      ACC=$(echo "$RAW" | grep -E "^acc:" | awk '{print $2}')
      DEP=$(echo "$RAW" | grep -E "^depth_acc:" | awk '{print $2}')
      RGB=$(echo "$RAW" | grep -E "^rgb_acc:" | awk '{print $2}')
      echo "${pt},${run},static,${SNR},${ACC},${DEP},${RGB},${CKPT_SHA},${SAVEDIR}" >> "$OUT_CSV"
    done

    # dynamic eval
    SAVEDIR_DYN="$SAVEDIR_BASE/dynamic_${DYN_MIN}_${DYN_MAX}/pretrain_${pt}/run${run}/train_TMC_channel_baseline_${pt}_dynamic_${DYN_MIN}_${DYN_MAX}_run${run}"
    if [ -d "$SAVEDIR_DYN" ]; then
      CKPT=""
      if [ -f "$SAVEDIR_DYN/model_best.pt" ]; then
        CKPT="$SAVEDIR_DYN/model_best.pt"
      elif [ -f "$SAVEDIR_DYN/checkpoint.pt" ]; then
        CKPT="$SAVEDIR_DYN/checkpoint.pt"
      else
        CAND=$(ls -1 "$SAVEDIR_DYN"/*.pt 2>/dev/null | head -n 1 || true)
        if [ -n "${CAND:-}" ]; then
          CKPT="$CAND"
        fi
      fi
      if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
        CKPT_SHA=$(sha256sum "$CKPT" | awk '{print $1}')
      else
        CKPT_SHA="NA"
      fi
      RAW=$("$PYTHON" "$TESTER" \
        --savedir "$SAVEDIR_DYN" \
        --data_path "$DATAPATH" \
        --split test \
        --batch_sz 32 --n_workers 8 \
        --use_dynamic_snr \
        --snr_min "$DYN_MIN" \
        --snr_max "$DYN_MAX" | cat)
      ACC=$(echo "$RAW" | grep -E "^acc:" | awk '{print $2}')
      DEP=$(echo "$RAW" | grep -E "^depth_acc:" | awk '{print $2}')
      RGB=$(echo "$RAW" | grep -E "^rgb_acc:" | awk '{print $2}')
      echo "${pt},${run},dynamic,${DYN_MIN}-${DYN_MAX},${ACC},${DEP},${RGB},${CKPT_SHA},${SAVEDIR_DYN}" >> "$OUT_CSV"
    fi

    # Noisy input tests (channel fixed at 10 dB), single view noise
    SAVEDIR_NOISE="$SAVEDIR_BASE/static_snr10/pretrain_${pt}/run${run}/train_TMC_channel_baseline_${pt}_snr10_run${run}"
    if [ -d "$SAVEDIR_NOISE" ]; then
      for POW in "${NOISE_POWERS[@]}"; do
        for VIEW in rgb depth; do
          echo "--- baseline noisy ${VIEW}, pow=${POW} ---"
          CKPT=""
          if [ -f "$SAVEDIR_NOISE/model_best.pt" ]; then
            CKPT="$SAVEDIR_NOISE/model_best.pt"
          elif [ -f "$SAVEDIR_NOISE/checkpoint.pt" ]; then
            CKPT="$SAVEDIR_NOISE/checkpoint.pt"
          else
            CAND=$(ls -1 "$SAVEDIR_NOISE"/*.pt 2>/dev/null | head -n 1 || true)
            if [ -n "${CAND:-}" ]; then
              CKPT="$CAND"
            fi
          fi
          if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
            CKPT_SHA=$(sha256sum "$CKPT" | awk '{print $1}')
          else
            CKPT_SHA="NA"
          fi
          RAW=$("$PYTHON" "$TESTER" \
            --savedir "$SAVEDIR_NOISE" \
            --data_path "$DATAPATH" \
            --split test \
            --batch_sz 32 --n_workers 8 \
            --snr 10 \
            --noisy_view "$VIEW" \
            --noise_power "$POW" | cat)
          ACC=$(echo "$RAW" | grep -E "^acc:" | awk '{print $2}')
          DEP=$(echo "$RAW" | grep -E "^depth_acc:" | awk '{print $2}')
          RGB=$(echo "$RAW" | grep -E "^rgb_acc:" | awk '{print $2}')
          echo "${pt},${run},noisy_${VIEW},${POW},${ACC},${DEP},${RGB},${CKPT_SHA},${SAVEDIR_NOISE}" >> "$OUT_CSV"
        done
      done
    fi
  done
done

echo "Saved results to: $OUT_CSV"


