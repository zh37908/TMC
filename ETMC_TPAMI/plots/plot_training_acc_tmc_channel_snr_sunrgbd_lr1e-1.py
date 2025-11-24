#!/usr/bin/env python3
import os
import re
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch


BASE_DIR = "/home/hzhaobi/Multired/savepath/TMC_channel_snr/sunrgbd/none"
SNR_DIRS = ["snr10", "snr20"]
PRETRAINS = ["DeCUR", "SimCLR", "BarlowTwins", "No_pretrain"]
FEW_LABEL_BASE_DIR = "/home/hzhaobi/Multired/savepath/TMC_channel_snr/sunrgbd/none/few-label"

# 全局绘图风格（与 NYUD 脚本保持一致）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5
plt.rcParams['axes.linewidth'] = 0.45


def find_logfiles_for_pretrain(root: str, pretrain: str) -> List[str]:
    base = os.path.join(root, f"pretrain_{pretrain}", "lr_1e-1")
    if not os.path.isdir(base):
        return []
    logfiles: List[str] = []
    # 优先收集 run* 目录下的 logfile.log
    try:
        subs = sorted(os.listdir(base))
    except Exception:
        subs = []
    for name in subs:
        subp = os.path.join(base, name)
        if os.path.isdir(subp) and name.lower().startswith("run"):
            # 常见两种结构：runX/logfile.log 或 runX/<time>/logfile.log
            fp1 = os.path.join(subp, "logfile.log")
            if os.path.isfile(fp1):
                logfiles.append(fp1)
                continue
            try:
                for sub2 in os.listdir(subp):
                    subp2 = os.path.join(subp, sub2)
                    if os.path.isdir(subp2):
                        fp2 = os.path.join(subp2, "logfile.log")
                        if os.path.isfile(fp2):
                            logfiles.append(fp2)
            except Exception:
                pass
    # 如果没有 run*，尝试 lr 目录自身
    if not logfiles:
        fp = os.path.join(base, "logfile.log")
        if os.path.isfile(fp):
            logfiles.append(fp)
    return logfiles


# 不再通过 args.pt 过滤学习率，直接在路径 pretrain_<PT>/lr_1e-1 下收集日志


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
    # 方案A：按可用数据逐 epoch 求均值（不按最短序列截断）
    series_list = [s for s in series_list if s]
    if not series_list:
        return []
    max_len = max(len(s) for s in series_list)
    mean: List[float] = []
    for i in range(max_len):
        vals = [s[i] for s in series_list if len(s) > i]
        if not vals:
            break
        mean.append(sum(vals) / float(len(vals)))
    return mean


def env_desc_from_snr(snr_dir: str) -> str:
    m = re.match(r"snr(\d+)", snr_dir)
    if m:
        return f"{int(m.group(1))} dB, AWGN"
    return snr_dir


def plot_one_for_dir(snr_root: str, snr_dir: str, colors: Dict[str, str], out_prefix: str = "") -> None:
    if not os.path.isdir(snr_root):
        print(f"[WARN] Missing: {snr_root}")
        return

    pt_to_mean: Dict[str, List[float]] = {}
    for pt in PRETRAINS:
        series_list: List[List[float]] = []
        logfiles = find_logfiles_for_pretrain(snr_root, pt)
        for lf in logfiles:
            ser = parse_fused_acc_series(lf)
            if ser:
                series_list.append(ser)
        pt_to_mean[pt] = aggregate_mean(series_list)

    plt.figure(figsize=(3.52, 2.5))  # 近似单列比例
    for pt in PRETRAINS:
        m = pt_to_mean.get(pt, [])
        if m:
            plt.plot(range(1, len(m) + 1), m, label=pt, linewidth=1.2, color=colors.get(pt))

    ax = plt.gca()
    ax.set_title(f"Test Acc. vs Epoch ({env_desc_from_snr(snr_dir)})")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Test Acc.")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.3)
    ax.set_xlim(0, 500)
    ax.set_xticks(np.arange(0, 501, 100))
    ax.set_ylim(0.0, 0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
    ax.tick_params(direction="in", length=2.5, width=0.45, top=True, right=True)

    snr_val = snr_dir.replace("snr", "")
    out_path = f"/home/hzhaobi/Multired/TMC/ETMC_TPAMI/plots/sunrgbd_TMC_channel_snr_{out_prefix}snr{snr_val}_lr1e-1_fusion_acc.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {out_path}")


def main():
    colors = {
        "DeCUR": "#2ca02c",       # green（与 NYUD 保持一致）
        "SimCLR": "#1f77b4",      # blue
        "BarlowTwins": "#d62728", # red
        "No_pretrain": "#ff7f0e", # orange
    }

    # 标准标注
    for snr_dir in SNR_DIRS:
        snr_root = os.path.join(BASE_DIR, snr_dir)
        plot_one_for_dir(snr_root, snr_dir, colors, out_prefix="")

    # few-label 标注
    for snr_dir in SNR_DIRS:
        snr_root = os.path.join(FEW_LABEL_BASE_DIR, snr_dir)
        plot_one_for_dir(snr_root, snr_dir, colors, out_prefix="fewlabel_")


if __name__ == "__main__":
    main()


