import os
import re
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


NYUD_ROOTS = [
    # dynamic SNR
    "/home/hzhaobi/Multired/savepath/TMC_channel_dynamic/nyud",
    "/home/hzhaobi/Multired/savepath/TMCBase_channel_dynamic/nyud",
    # static SNR
    "/home/hzhaobi/Multired/savepath/TMC_channel/nyud",
    "/home/hzhaobi/Multired/savepath/TMCBase_channel/nyud",
]

# 支持的预训练标签（从路径中提取）
KNOWN_PRETRAINS = {"DeCUR", "SimCLR", "BarlowTwins", "No_pretrain"}


def find_logfiles(root: str) -> List[str]:
    logfiles: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "logfile.log":
                logfiles.append(os.path.join(dirpath, fn))
    return logfiles


def parse_meta_from_path(path: str) -> Dict[str, str]:
    parts = path.split(os.sep)
    meta: Dict[str, str] = {
        "model_root": "unknown",
        "pretrain": "unknown",
        "run": "run?",
        "spec": None,  # rangeX-Y 或 snrZ
    }
    # 模型根类别推断
    if "TMC_channel_dynamic" in path:
        meta["model_root"] = "TMC_channel_dynamic"
    elif "TMCBase_channel_dynamic" in path:
        meta["model_root"] = "TMCBase_channel_dynamic"
    elif "savepath" in parts and "TMC_channel" in parts:
        meta["model_root"] = "TMC_channel"
    elif "savepath" in parts and "TMCBase_channel" in parts:
        meta["model_root"] = "TMCBase_channel"
    # 预训练标签
    for p in parts:
        if p in KNOWN_PRETRAINS:
            meta["pretrain"] = p
            break
    # run 号
    for p in parts:
        if re.fullmatch(r"run\d+", p):
            meta["run"] = p
            break
    # 区间（如 range0-20）或静态 SNR（如 snr10）
    for p in parts:
        if p.startswith("range"):
            meta["spec"] = p
            break
    if meta["spec"] is None:
        for p in parts:
            if re.fullmatch(r"snr\d+", p):
                meta["spec"] = p
                break
    return meta


EPOCH_LINE_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*\|\s*val:\s*Loss:\s*([0-9.]+)\s*\|\s*depth_acc:\s*([0-9.]+),\s*rgb_acc:\s*([0-9.]+),\s*depth rgb acc:\s*([0-9.]+)"
)


def parse_log(logfile: str) -> Dict[str, List[Tuple[int, float, float, float]]]:
    epochs: List[int] = []
    depth_accs: List[float] = []
    rgb_accs: List[float] = []
    fusion_accs: List[float] = []

    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EPOCH_LINE_RE.search(line)
            if m:
                ep = int(m.group(1))
                # m.group(2) 是总 epoch，未使用
                # loss = float(m.group(3))
                depth_acc = float(m.group(4))
                rgb_acc = float(m.group(5))
                fusion_acc = float(m.group(6))
                epochs.append(ep)
                depth_accs.append(depth_acc)
                rgb_accs.append(rgb_acc)
                fusion_accs.append(fusion_acc)

    return {
        "epoch": epochs,
        "depth_acc": depth_accs,
        "rgb_acc": rgb_accs,
        "fusion_acc": fusion_accs,
    }


def group_key(meta: Dict[str, str]) -> Tuple[str, str, Optional[str]]:
    return meta["model_root"], meta["pretrain"], meta["spec"]


def aggregate_runs(curves: List[Dict[str, List[float]]]) -> Dict[str, np.ndarray]:
    # 过滤掉没有 epoch 数据的条目
    valid = [c for c in curves if c.get("epoch")]
    if not valid:
        return {"epoch": np.array([]), "mean": np.array([]), "std": np.array([])}

    # 对齐长度（不同 run 可能早停或长度不同），取最短长度
    min_len = min(len(c["epoch"]) for c in valid)
    if min_len == 0:
        return {"epoch": np.array([]), "mean": np.array([]), "std": np.array([])}

    fused = np.stack([np.array(c["fusion_acc"])[:min_len] for c in valid], axis=0)
    epochs = np.array(valid[0]["epoch"][:min_len])
    return {
        "epoch": epochs,
        "mean": fused.mean(axis=0),
        "std": fused.std(axis=0),
    }


def plot_groups(groups: Dict[Tuple[str, str, Optional[str]], List[Dict[str, List[float]]]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 先按 model_root 分组，分别出图
    model_to_pretrain: Dict[str, Dict[Tuple[str, Optional[str]], List[Dict[str, List[float]]]]] = {}
    for (model_root, pretrain, spec), runs in groups.items():
        model_to_pretrain.setdefault(model_root, {}).setdefault((pretrain, spec), []).extend(runs)

    for model_root, sub in model_to_pretrain.items():
        # 汇总图（包含该 model_root 下所有曲线）
        plt.figure(figsize=(10, 6))
        handles = []
        labels = []
        for (pretrain, spec), runs in sorted(sub.items()):
            agg = aggregate_runs(runs)
            if agg["epoch"].size == 0:
                continue
            epochs = agg["epoch"]
            mean = agg["mean"]
            std = agg["std"]
            label = pretrain if spec is None else f"{pretrain} ({spec})"
            line, = plt.plot(epochs, mean, linewidth=2)
            plt.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=line.get_color(), label="_nolegend_")
            handles.append(line)
            labels.append(label)

        plt.title(f"NYUD | {model_root} | Fusion accuracy vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Fusion accuracy (depth_rgb_acc)")
        if handles:
            plt.legend(handles, labels)
        plt.grid(True, linestyle="--", alpha=0.4)
        outfile = os.path.join(out_dir, f"nyud_{model_root}_fusion_acc.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=200)
        plt.close()
        print(f"Saved: {outfile}")

        # 针对静态 SNR，将 snr10 和 snr20 分开各画一张（仅适用于 static models）
        if model_root in {"TMC_channel", "TMCBase_channel"}:
            for target_spec in ("snr10", "snr20"):
                # 仅包含该 spec 的曲线，图例用预训练标签
                plt.figure(figsize=(10, 6))
                handles = []
                labels = []
                for (pretrain, spec), runs in sorted(sub.items()):
                    if spec != target_spec:
                        continue
                    agg = aggregate_runs(runs)
                    if agg["epoch"].size == 0:
                        continue
                    epochs = agg["epoch"]
                    mean = agg["mean"]
                    std = agg["std"]
                    label = pretrain
                    line, = plt.plot(epochs, mean, linewidth=2)
                    plt.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=line.get_color(), label="_nolegend_")
                    handles.append(line)
                    labels.append(label)

                if handles:
                    plt.title(f"NYUD | {model_root} | Fusion acc vs. Epoch | {target_spec}")
                    plt.xlabel("Epoch")
                    plt.ylabel("Fusion accuracy (depth_rgb_acc)")
                    plt.legend(handles, labels)
                    plt.grid(True, linestyle="--", alpha=0.4)
                    outfile = os.path.join(out_dir, f"nyud_{model_root}_{target_spec}_fusion_acc.png")
                    plt.tight_layout()
                    plt.savefig(outfile, dpi=200)
                    print(f"Saved: {outfile}")
                plt.close()


def main() -> None:
    # 输出目录
    out_dir = "/home/hzhaobi/Multired/plots"
    groups: Dict[Tuple[str, str, Optional[str]], List[Dict[str, List[float]]]] = {}

    for root in NYUD_ROOTS:
        logs = find_logfiles(root)
        for lf in logs:
            meta = parse_meta_from_path(lf)
            curves = parse_log(lf)
            key = group_key(meta)
            groups.setdefault(key, []).append(curves)

    plot_groups(groups, out_dir)


if __name__ == "__main__":
    main()


