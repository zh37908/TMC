#!/usr/bin/env python3
import os
import re
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = "/home/hzhaobi/Multired/savepath/TMC_channel_snr/nyud/none/few-label"
SNR_DIRS = ["snr10", "snr20"]
LR_STR = "1e-2"
PRETRAINS = ["DeCUR", "SimCLR", "BarlowTwins", "No_pretrain"]
SMOOTH_WINDOW = 5  # 曲线平滑窗口（奇数更佳，如 5/7）


def find_logfiles_for_pretrain(root: str, pretrain: str, lr_str: str) -> List[str]:
    base = os.path.join(root, f"pretrain_{pretrain}", f"lr_{lr_str}")
    if not os.path.isdir(base):
        return []
    logfiles: List[str] = []
    try:
        subs = sorted(os.listdir(base))
    except Exception:
        subs = []
    for name in subs:
        subp = os.path.join(base, name)
        if os.path.isdir(subp) and name.lower().startswith("run"):
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
    if not logfiles:
        fp = os.path.join(base, "logfile.log")
        if os.path.isfile(fp):
            logfiles.append(fp)
    return logfiles


VAL_LINE = re.compile(r"(val|Test):\s*Loss:\s*([0-9.]+)\s*\|\s*depth_acc:\s*([0-9.]+),\s*rgb_acc:\s*([0-9.]+),\s*depth rgb acc:\s*([0-9.]+)")

# 全局绘图风格（与主图保持一致，偏 IEEE 单列图）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5
plt.rcParams['axes.linewidth'] = 0.45


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
    # 与 sunrgbd 脚本一致：不按最短序列截断，逐 epoch 对可用序列求均值
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


def smooth_series(values: List[float], window: int = SMOOTH_WINDOW) -> List[float]:
    if window is None or window <= 1 or len(values) <= 2:
        return values
    # 使用居中滑动平均，边界自动缩小窗口
    half = window // 2
    smoothed: List[float] = []
    n = len(values)
    for i in range(n):
        s = 0.0
        c = 0
        start = max(0, i - half)
        end = min(n - 1, i + half)
        for j in range(start, end + 1):
            s += values[j]
            c += 1
        smoothed.append(s / float(c))
    return smoothed


def format_env_desc(snr_dir: str) -> str:
    # 将 "snr10" → "10 dB, AWGN"
    m = re.match(r"snr(\d+)", snr_dir)
    if m:
        return f"{int(m.group(1))} dB, AWGN"
    return snr_dir


def main():
    colors = {
        "DeCUR": "#2ca02c",       # green
        "SimCLR": "#1f77b4",      # blue
        "BarlowTwins": "#d62728", # red
        "No_pretrain": "#ff7f0e", # orange
    }

    for snr_dir in SNR_DIRS:
        snr_root = os.path.join(BASE_DIR, snr_dir)
        if not os.path.isdir(snr_root):
            print(f"[WARN] Missing: {snr_root}")
            continue

        pt_to_mean: Dict[str, List[float]] = {}
        for pt in PRETRAINS:
            series_list: List[List[float]] = []
            for lf in find_logfiles_for_pretrain(snr_root, pt, LR_STR):
                ser = parse_fused_acc_series(lf)
                if ser:
                    series_list.append(ser)
            pt_to_mean[pt] = aggregate_mean(series_list)

        plt.figure(figsize=(3.52, 2.5))  # 近似单列图比例
        plotted_any = False
        for pt in PRETRAINS:
            m = pt_to_mean.get(pt, [])
            if m:
                plotted_any = True
                ms = smooth_series(m, SMOOTH_WINDOW)
                plt.plot(range(1, len(ms) + 1), ms, label=pt, linewidth=1.2, color=colors.get(pt))

        env_desc = format_env_desc(snr_dir)
        ax = plt.gca()
        ax.set_title(f"Test Acc. vs Epoch ({env_desc})")
        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("Test Acc.")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.3)
        ax.set_xlim(0, 500)
        ax.set_xticks(np.arange(0, 501, 100))
        ax.set_ylim(0.0, 0.60)
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
        ax.tick_params(direction="in", length=2.5, width=0.45, top=True, right=True)
        # 不显示 legend，以免遮挡
        snr_val = snr_dir.replace("snr", "")
        out_path = f"/home/hzhaobi/Multired/TMC/ETMC_TPAMI/plots/nyud_TMC_channel_snr_fewlabel_snr{snr_val}_lr{LR_STR}_fusion_acc.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


