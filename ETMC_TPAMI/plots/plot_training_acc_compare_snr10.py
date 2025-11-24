#!/usr/bin/env python3
import os
import re
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


# 仅考虑 TMC_channel_snr 两种情况：
# 1) NYUD: 遍历所有 snr*/dynamic_* 子目录，统计 lr=1e-2
# 2) SUNRGBD: 仅 snr10/ 子目录，统计 lr=1e-1
SNR_BASE_ROOT_NYUD = "/home/hzhaobi/Multired/savepath/TMC_channel_snr/nyud/none"
SNR_ROOT_SUNRGBD_SNR10 = "/home/hzhaobi/Multired/savepath/TMC_channel_snr/sunrgbd/none/snr10"
PRETRAINS = ["DeCUR", "SimCLR", "BarlowTwins", "No_pretrain"]
RUNS = [1, 2, 3, 4, 5]
ONLY_LR_NYUD = "1e-2"
ONLY_LR_SUNRGBD = "1e-1"


def find_logfile(root: str, pretrain: str, run: int) -> str:
    d = os.path.join(root, f"pretrain_{pretrain}", f"run{run}")
    if not os.path.isdir(d):
        return ""
    # try to find subdir and logfile.log
    try:
        for sub in os.listdir(d):
            subp = os.path.join(d, sub)
            if os.path.isdir(subp):
                fp = os.path.join(subp, "logfile.log")
                if os.path.isfile(fp):
                    return fp
    except Exception:
        pass
    # fallback: direct logfile.log
    fp = os.path.join(d, "logfile.log")
    return fp if os.path.isfile(fp) else ""


def find_logfile_with_lr(root: str, pretrain: str, lr_str: str, run: int) -> str:
    d = os.path.join(root, f"pretrain_{pretrain}", f"lr_{lr_str}", f"run{run}")
    if not os.path.isdir(d):
        return ""
    try:
        for sub in os.listdir(d):
            subp = os.path.join(d, sub)
            if os.path.isdir(subp):
                fp = os.path.join(subp, "logfile.log")
                if os.path.isfile(fp):
                    return fp
    except Exception:
        pass
    fp = os.path.join(d, "logfile.log")
    return fp if os.path.isfile(fp) else ""


VAL_LINE = re.compile(r"(val|Test):\s*Loss:\s*([0-9.]+)\s*\|\s*depth_acc:\s*([0-9.]+),\s*rgb_acc:\s*([0-9.]+),\s*depth rgb acc:\s*([0-9.]+)")


def parse_fused_acc_series(logfile: str) -> List[float]:
    series: List[float] = []
    try:
        with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = VAL_LINE.search(line)
                if m:
                    fused = float(m.group(5))
                    series.append(fused)
    except Exception:
        return []
    return series


def aggregate_mean(series_list: List[List[float]]) -> List[float]:
    if not series_list:
        return []
    # clip to min common length to fairly average aligned epochs
    min_len = min(len(s) for s in series_list if s)
    if min_len == 0:
        return []
    mean: List[float] = []
    for i in range(min_len):
        vals = [s[i] for s in series_list if len(s) > i]
        mean.append(sum(vals) / float(len(vals)))
    return mean


def collect_means(root: str) -> Dict[str, List[float]]:
    pretrain_to_mean: Dict[str, List[float]] = {}
    for pt in PRETRAINS:
        runs: List[List[float]] = []
        for r in RUNS:
            lf = find_logfile(root, pt, r)
            if not lf:
                continue
            ser = parse_fused_acc_series(lf)
            if ser:
                runs.append(ser)
        pretrain_to_mean[pt] = aggregate_mean(runs)
    return pretrain_to_mean


def collect_means_with_lr(root: str, lr_str: str) -> Dict[str, List[float]]:
    pretrain_to_mean: Dict[str, List[float]] = {}
    for pt in PRETRAINS:
        runs: List[List[float]] = []
        for r in RUNS:
            lf = find_logfile_with_lr(root, pt, lr_str, r)
            if not lf:
                continue
            ser = parse_fused_acc_series(lf)
            if ser:
                runs.append(ser)
        pretrain_to_mean[pt] = aggregate_mean(runs)
    return pretrain_to_mean


def list_snr_dirs(root: str) -> List[str]:
    try:
        subs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    except Exception:
        return []
    return [d for d in subs if d.startswith("snr") or d.startswith("dynamic_")]


def collect_means_across_snr_for_lr(base_root: str, lr_str: str, snr_name_filter: Optional[str] = None) -> Dict[str, List[float]]:
    pretrain_to_runs: Dict[str, List[List[float]]] = {pt: [] for pt in PRETRAINS}
    snr_dirs = list_snr_dirs(base_root)
    for snr_dir in snr_dirs:
        if snr_name_filter is not None and snr_dir != snr_name_filter:
            continue
        root = os.path.join(base_root, snr_dir)
        for pt in PRETRAINS:
            for r in RUNS:
                lf = find_logfile_with_lr(root, pt, lr_str, r)
                if not lf:
                    continue
                ser = parse_fused_acc_series(lf)
                if ser:
                    pretrain_to_runs[pt].append(ser)
    # 对每个 pretrain，把不同 SNR 目录收集到的所有 run 序列一起取均值
    return {pt: aggregate_mean(runs) for pt, runs in pretrain_to_runs.items()}


# Function to find the epoch when accuracy reaches a certain threshold

def find_epoch_for_accuracy(series: List[float], threshold: float) -> int:
    for i, acc in enumerate(series):
        if acc >= threshold:
            return i + 1  # Epochs are 1-based
    return -1  # Return -1 if threshold is never reached

# Add table creation logic

def create_summary_table(pretrain_to_mean: Dict[str, List[float]], condition: str) -> pd.DataFrame:
    thresholds = [40, 45, 50]
    # 动态构建列，避免 KeyError
    data: Dict[str, List] = {'Pretrain': [], 'Condition': []}
    for t in thresholds:
        data[f'{t}% Epoch'] = []
    data['Best Acc'] = []
    for pretrain, series in pretrain_to_mean.items():
        data['Pretrain'].append(pretrain)
        data['Condition'].append(condition)
        for threshold in thresholds:
            epoch = find_epoch_for_accuracy(series, threshold / 100.0)
            data[f'{threshold}% Epoch'].append(epoch)
        data['Best Acc'].append(max(series) if series else 0)
    return pd.DataFrame(data)


def main():
    # NYUD: 所有 SNR 子目录（dynamic_*/snr*）下 lr=1e-2
    nyud_lr2_means = collect_means_across_snr_for_lr(SNR_BASE_ROOT_NYUD, ONLY_LR_NYUD, snr_name_filter=None)
    # SUNRGBD: 仅 snr10 目录下 lr=1e-1
    sunrgbd_lr1e1_means = collect_means_with_lr(SNR_ROOT_SUNRGBD_SNR10, ONLY_LR_SUNRGBD)

    # plot 2x2 subplots by pretrain
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, pt in enumerate(PRETRAINS):
        ax = axes[idx]
        a = nyud_lr2_means.get(pt, [])
        b = sunrgbd_lr1e1_means.get(pt, [])
        if a:
            ax.plot(range(1, len(a) + 1), a, linestyle="-", color="#d62728", label="NYUD | lr=1e-2 (all SNRs)")
        if b:
            ax.plot(range(1, len(b) + 1), b, linestyle="--", color="#1f77b4", label="SUNRGBD | snr10 | lr=1e-1")
        ax.set_title(pt)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fusion Acc")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("TMC_channel_snr | NYUD lr=1e-2 (all SNRs) vs SUNRGBD snr10 lr=1e-1 | Fusion Acc")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = "/home/hzhaobi/Multired/TMC/ETMC_TPAMI/plots/TMC_channel_snr_nyud_lr1e-2_vs_sunrgbd_snr10_lr1e-1_fusion_acc.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to: {out_path}")

    table_nyud = create_summary_table(nyud_lr2_means, 'NYUD_lr_1e-2_allSNRs')
    table_sunrgbd = create_summary_table(sunrgbd_lr1e1_means, 'SUNRGBD_snr10_lr_1e-1')
    summary_table = pd.concat([table_nyud, table_sunrgbd], ignore_index=True)
    summary_path = "/home/hzhaobi/Multired/TMC/ETMC_TPAMI/plots/TMC_channel_snr_nyud_lr1e-2_vs_sunrgbd_snr10_lr1e-1_summary.csv"
    summary_table.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()


