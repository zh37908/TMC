SAVEDIR=/home/hzhaobi/Multired/savepath/TMC/nyud/original/Adam/pretrain_resnet18/ReleasedVersion


ROOT="/home/hzhaobi/Multired"
DATAPATH="$ROOT/nyud2"

NOISES=(1 10 100)

extract_acc() {
  # 从脚本输出中抽取 "acc (post-rt):" 的数值；若没有启用 retransmit，pre/post 一致
  # 使用 printf 返回值，避免额外换行和空格
  local out="$1"
  local acc
  acc=$(echo "$out" | grep -E "^acc \(post-rt\):" | sed -E 's/.*acc \(post-rt\):[[:space:]]*([0-9.]+)/\1/' | tail -n1)
  if [ -z "$acc" ]; then
    # 兜底：尝试读取 pre-rt
    acc=$(echo "$out" | grep -E "^acc \(pre-rt\):" | sed -E 's/.*acc \(pre-rt\):[[:space:]]*([0-9.]+)/\1/' | tail -n1)
  fi
  printf "%s" "$acc"
}

run_and_collect() {
  local view="$1"
  local -n acc_arr_ref=$2  # 通过名称引用数组
  acc_arr_ref=()
  for p in "${NOISES[@]}"; do
    echo "[RUN] view=$view, noise_power=$p"
    out=$(python /home/hzhaobi/Multired/TMC/ETMC_TPAMI/test/test_TMC_channel_snr_retransmit.py \
      --savedir "$SAVEDIR" --data_path "$DATAPATH" --split test \
      --noisy_view "$view" --noise_power "$p")
    acc=$(extract_acc "$out")
    if [ -z "$acc" ]; then
      echo "[WARN] 未能从输出中解析 acc，标记为 NaN。" >&2
      acc="NaN"
    fi
    acc_arr_ref+=("$acc")
  done
}

# 分别收集 RGB / Depth 的 acc(post-rt)
declare -a ACC_RGB
declare -a ACC_DEPTH

run_and_collect rgb ACC_RGB
run_and_collect depth ACC_DEPTH

# 打印为一张表：行=视角，列=噪声强度
printf "\nSummary (accuracy: acc post-rt)\n"
printf "%-8s" "View"
for p in "${NOISES[@]}"; do
  printf " %10s" "$p"
done
printf "\n"

printf "%-8s" "rgb"
for acc in "${ACC_RGB[@]}"; do
  printf " %10s" "$acc"
done
printf "\n"

printf "%-8s" "depth"
for acc in "${ACC_DEPTH[@]}"; do
  printf " %10s" "$acc"
done
printf "\n"