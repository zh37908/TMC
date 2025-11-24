#!/usr/bin/env python3
import csv
import os
from collections import defaultdict


ROOT = "/home/hzhaobi/Multired/TMC/ETMC_TPAMI/test"
SNR_MODEL_CSV = os.path.join(ROOT, "results_tmc_channel_snr_pretrains.csv")
BASELINE_CSV = os.path.join(ROOT, "results_tmc_channel_baseline_pretrains.csv")
OUT_CSV = os.path.join(ROOT, "results_tmc_compare.csv")


def to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def read_snr_model(path):
    """Read new-model results (with acc_pre/acc_post) and group by (pretrain, snr_mode, snr)."""
    groups = defaultdict(lambda: {"acc_pre": [], "acc_post": []})
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["pretrain"], row["snr_mode"], row["snr"], row.get("method", ""))
            ap = to_float(row.get("acc_pre"))
            apo = to_float(row.get("acc_post"))
            if ap is not None:
                groups[key]["acc_pre"].append(ap)
            if apo is not None:
                groups[key]["acc_post"].append(apo)
    # reduce to mean
    out = {}
    for k, v in groups.items():
        acc_pre = sum(v["acc_pre"]) / len(v["acc_pre"]) if v["acc_pre"] else None
        acc_post = sum(v["acc_post"]) / len(v["acc_post"]) if v["acc_post"] else None
        out[k] = {"acc_pre": acc_pre, "acc_post": acc_post}
    return out


def read_baseline(path):
    """Read baseline results (acc, depth_acc, rgb_acc) and group by (pretrain, snr_mode, snr)."""
    groups = defaultdict(lambda: {"acc": [], "depth_acc": [], "rgb_acc": []})
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["pretrain"], row["snr_mode"], row["snr"])
            a = to_float(row.get("acc"))
            da = to_float(row.get("depth_acc"))
            ra = to_float(row.get("rgb_acc"))
            if a is not None:
                groups[key]["acc"].append(a)
            if da is not None:
                groups[key]["depth_acc"].append(da)
            if ra is not None:
                groups[key]["rgb_acc"].append(ra)
    out = {}
    for k, v in groups.items():
        acc = sum(v["acc"]) / len(v["acc"]) if v["acc"] else None
        depth_acc = sum(v["depth_acc"]) / len(v["depth_acc"]) if v["depth_acc"] else None
        rgb_acc = sum(v["rgb_acc"]) / len(v["rgb_acc"]) if v["rgb_acc"] else None
        out[k] = {"acc": acc, "depth_acc": depth_acc, "rgb_acc": rgb_acc}
    return out


def main():
    snr_model = read_snr_model(SNR_MODEL_CSV)
    baseline = read_baseline(BASELINE_CSV)

    # unify keys by (pretrain, snr_mode, snr); method from snr_model retained for reference
    keys = set()
    for (pre, mode, snr, method) in snr_model.keys():
        keys.add((pre, mode, snr))
    for (pre, mode, snr) in baseline.keys():
        keys.add((pre, mode, snr))

    header = [
        "pretrain",
        "snr_mode",
        "snr",
        "baseline_acc",
        "baseline_depth_acc",
        "baseline_rgb_acc",
        "snr_acc_pre",
        "snr_acc_post",
        "delta_post_vs_baseline",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for pre, mode, snr in sorted(keys):
            base = baseline.get((pre, mode, snr), {})
            # find any method entry; method is not essential for aggregation
            snr_entry = None
            for (k_pre, k_mode, k_snr, _method), v in snr_model.items():
                if k_pre == pre and k_mode == mode and k_snr == snr:
                    snr_entry = v
                    break

            b_acc = base.get("acc")
            b_d = base.get("depth_acc")
            b_r = base.get("rgb_acc")
            a_pre = snr_entry.get("acc_pre") if snr_entry else None
            a_post = snr_entry.get("acc_post") if snr_entry else None
            delta = (a_post - b_acc) if (a_post is not None and b_acc is not None) else None

            row = [
                pre,
                mode,
                snr,
                f"{b_acc:.4f}" if b_acc is not None else "",
                f"{b_d:.4f}" if b_d is not None else "",
                f"{b_r:.4f}" if b_r is not None else "",
                f"{a_pre:.4f}" if a_pre is not None else "",
                f"{a_post:.4f}" if a_post is not None else "",
                f"{delta:.4f}" if delta is not None else "",
            ]
            w.writerow(row)

    print(f"Wrote aggregated comparison to: {OUT_CSV}")


if __name__ == "__main__":
    main()


