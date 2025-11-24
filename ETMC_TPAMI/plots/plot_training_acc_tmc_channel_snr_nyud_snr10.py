#!/usr/bin/env python3
import os
import re
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = "/home/hzhaobi/Multired/savepath/TMC_channel_snr/nyud/none"
PRETRAINS: List[str] = ["DeCUR", "SimCLR", "BarlowTwins", "No_pretrain"]
RUNS = [1, 2, 3, 4, 5]
SMOOTH_WINDOW = 5

# 仅绘制该学习率
ONLY_LR = "1e-2"


def find_logfile_for_lr(root: str, pretrain: str, lr_str: str, run: int) -> str:
    pre_dir = os.path.join(root, f"pretrain_{pretrain}")
    lr_dir = os.path.join(pre_dir, f"lr_{lr_str}")
    run_dir = os.path.join(lr_dir, f"run{run}")
    if not os.path.isdir(run_dir):
        return ""
    try:
        for sub in os.listdir(run_dir):
            subp = os.path.join(run_dir, sub)
            if os.path.isdir(subp):
                fp = os.path.join(subp, "logfile.log")
                if os.path.isfile(fp):
                    return fp
    except Exception:
        pass
    fp = os.path.join(run_dir, "logfile.log")
    return fp if os.path.isfile(fp) else ""


VAL_LINE = re.compile(r"(val|Test):\s*Loss:\s*([0-9.]+)\s*\|\s*depth_acc:\s*([0-9.]+),\s*rgb_acc:\s*([0-9.]+),\s*depth rgb acc:\s*([0-9.]+)")

# 全局绘图风格（尽量贴近 IEEE 单列图规范）
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
    # 与 few-label 绘图保持一致：不按最短序列截断，逐 epoch 对可用序列求均值
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
    half = window // 2
    out: List[float] = []
    n = len(values)
    for i in range(n):
        s = 0.0
        c = 0
        a = max(0, i - half)
        b = min(n - 1, i + half)
        for j in range(a, b + 1):
            s += values[j]
            c += 1
        out.append(s / float(c))
    return out


def discover_lrs(snr_root: str, pretrains: List[str]) -> List[str]:
    lr_set: set[str] = set()
    for pt in pretrains:
        pre_dir = os.path.join(snr_root, f"pretrain_{pt}")
        if not os.path.isdir(pre_dir):
            continue
        try:
            for d in os.listdir(pre_dir):
                if d.startswith("lr_") and os.path.isdir(os.path.join(pre_dir, d)):
                    lr_set.add(d.replace("lr_", "", 1))
        except Exception:
            continue
    return sorted(lr_set)


def format_env_desc(snr_name: str) -> str:
    # 将目录名（如 snr10 / dynamic_0_20）映射为标题中的信道环境描述
    if snr_name.startswith("snr"):
        m = re.match(r"snr(\d+)", snr_name)
        if m:
            return f"{int(m.group(1))} dB, AWGN"
    if snr_name.startswith("dynamic_"):
        rng = snr_name.replace("dynamic_", "", 1).replace("_", "–")
        return f"{rng} dB, AWGN (Dynamic)"
    return snr_name


def plot_one_figure(snr_name: str, snr_root: str, lr_str: str, pretrains: List[str]) -> Tuple[str, int]:
    pt_to_mean: Dict[str, List[float]] = {}
    for pt in pretrains:
        runs: List[List[float]] = []
        for r in RUNS:
            lf = find_logfile_for_lr(snr_root, pt, lr_str, r)
            if not lf:
                continue
            ser = parse_fused_acc_series(lf)
            if ser:
                runs.append(ser)
        pt_to_mean[pt] = aggregate_mean(runs)

    colors = {
        "DeCUR": "#2ca02c",       # green
        "SimCLR": "#1f77b4",      # blue
        "BarlowTwins": "#d62728", # red
        "No_pretrain": "#ff7f0e", # orange
    }

    has_any = any(pt_to_mean.get(pt) for pt in pretrains)
    if not has_any:
        return "", 0

    plt.figure(figsize=(3.52, 2.5))  # 近似示例图比例（~1.41）
    plotted = 0
    for pt in pretrains:
        m = pt_to_mean.get(pt, [])
        if m:
            ms = smooth_series(m, SMOOTH_WINDOW)
            plt.plot(range(1, len(ms) + 1), ms, label=pt, linewidth=1.2, color=colors.get(pt))
            plotted += 1

    ax = plt.gca()
    env_desc = format_env_desc(snr_name)
    ax.set_title(f"Test Acc. vs Epoch ({env_desc})")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Test Acc.")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.3)
    ax.set_xlim(0, 500)
    ax.set_xticks(np.arange(0, 501, 100))
    ax.set_ylim(0.0, 0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
    ax.tick_params(direction="in", length=2.5, width=0.45, top=True, right=True)
    # 不使用 legend，避免遮挡
    out_path = f"/home/hzhaobi/Multired/TMC/ETMC_TPAMI/plots/nyud_TMC_channel_snr_{snr_name}_lr{lr_str}_fusion_acc.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    return out_path, plotted


def main():
    try:
        snr_dirs = [d for d in os.listdir(BASE_DIR)
                    if os.path.isdir(os.path.join(BASE_DIR, d)) and (d.startswith("snr") or d.startswith("dynamic_"))]
    except Exception:
        snr_dirs = []
    snr_dirs.sort()
    if not snr_dirs:
        print(f"No snr*/dynamic_* directories under {BASE_DIR}")
        return

    for snr_name in snr_dirs:
        snr_root = os.path.join(BASE_DIR, snr_name)
        lrs = discover_lrs(snr_root, PRETRAINS)
        if not lrs:
            print(f"[WARN] No lr_* found in {snr_root}")
            continue
        # 仅绘制 lr=1e-2
        if ONLY_LR not in lrs:
            print(f"[SKIP] No lr_{ONLY_LR} found in {snr_root}")
            continue
        for lr_str in [ONLY_LR]:
            out_path, plotted = plot_one_figure(snr_name, snr_root, lr_str, PRETRAINS)
            if plotted == len(PRETRAINS):
                print(f"Saved: {out_path} (4 curves)")
            elif plotted > 0:
                print(f"Saved: {out_path} ({plotted}/4 curves)")
            else:
                print(f"[SKIP] No curves for {snr_name} lr={lr_str}")


if __name__ == "__main__":
    main()