#!/usr/bin/env bash
set -euo pipefail

# Config
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"
TESTER="$ROOT/TMC/ETMC_TPAMI/test/test_TMC_retransmit_sunrgbd.py"
SAVEDIR="/home/hzhaobi/Multired/savepath/TMC_channel_snr/sunrgbd/none/snr10/pretrain_DeCUR/lr_1e-1/run1/train_TMC_channel_snr_none_DeCUR_snr10_lr1e-1_run1"
DATAPATH="/home/hzhaobi/Multired/conc_data"

VIEWS=(rgb depth)
POWERS=(1 10 100)

OUT_CSV="$ROOT/TMC/ETMC_TPAMI/test/results_tmc_channel_snr_noise_DeCUR_sunrgbd.csv"
echo "view,noise_power,channel_mode,channel_snr,trigger,acc_pre,acc_post,rt_ratio,savedir" > "$OUT_CSV"

# Also evaluate clean source (no noise): view=none, power=0
ALL_VIEWS=("none" "${VIEWS[@]}")
ALL_POWERS_BY_VIEW=("0")
for V in "${VIEWS[@]}"; do
  ALL_POWERS_BY_VIEW+=("1 10 100")
done

# Channel SNR configurations: fixed 0,10,20 dB and dynamic [0,20]
declare -a CH_MODES=("fixed" "fixed" "fixed" "dynamic")
declare -a CH_SNR=("0" "10" "20" "0-20")
declare -a TRIGGERS=("low" "high")

IDX=0
for VIEW in "${ALL_VIEWS[@]}"; do
  # decide powers list per view
  if [ "$VIEW" = "none" ]; then
    POW_LIST=("0")
  else
    POW_LIST=("1" "10" "100")
  fi
  for POW in "${POW_LIST[@]}"; do
    for I in "${!CH_MODES[@]}"; do
      MODE="${CH_MODES[$I]}"
      SNR="${CH_SNR[$I]}"
      for TRIG in "${TRIGGERS[@]}"; do
        echo "--- view=${VIEW}, power=${POW}, ch_mode=${MODE}, ch_snr=${SNR}, trigger=${TRIG} ---"
        if [ "$MODE" = "fixed" ]; then
          RAW=$("$PYTHON" "$TESTER" \
            --savedir "$SAVEDIR" \
            --data_path "$DATAPATH" \
            --split test \
            --batch_sz 32 --n_workers 8 \
            --noisy_view "$VIEW" \
            --noise_power "$POW" \
            --noise_stage dataloader \
            --channel_snr_mode fixed --channel_snr_db "$SNR" \
            --retransmit --target_coverage 0.8 \
            --reject_score entropy --rt_trigger "$TRIG" | cat)
          CHMODE="fixed"
          CHSNR="$SNR"
        else
          RAW=$("$PYTHON" "$TESTER" \
            --savedir "$SAVEDIR" \
            --data_path "$DATAPATH" \
            --split test \
            --batch_sz 32 --n_workers 8 \
            --noisy_view "$VIEW" \
            --noise_power "$POW" \
            --noise_stage dataloader \
            --channel_snr_mode dynamic --snr_min 0 --snr_max 20 \
            --retransmit --target_coverage 0.8 \
            --reject_score entropy --rt_trigger "$TRIG" | cat)
          CHMODE="dynamic"
          CHSNR="[0,20]"
        fi
        ACC_PRE=$(echo "$RAW" | grep -F "acc (pre-rt):" | awk '{print $NF}')
        ACC_POST=$(echo "$RAW" | grep -F "acc (post-rt):" | awk '{print $NF}')
        RT_RATIO=$(echo "$RAW" | grep -F "retransmit_ratio:" | awk '{print $NF}')
        echo "${VIEW},${POW},${CHMODE},${CHSNR},${TRIG},${ACC_PRE},${ACC_POST},${RT_RATIO},${SAVEDIR}" >> "$OUT_CSV"
      done
    done
  done
done

echo "Saved results to: $OUT_CSV"


