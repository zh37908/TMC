#!/usr/bin/env python3
import csv
import os
from collections import defaultdict


ROOT = "/home/hzhaobi/Multired/TMC/ETMC_TPAMI/test"
IN_CSV = os.path.join(ROOT, "results_tmc_noisy.csv")
OUT_CSV = os.path.join(ROOT, "results_tmc_noisy_agg.csv")


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    groups = defaultdict(lambda: {"acc": [], "acc_pre": [], "acc_post": []})
    with open(IN_CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (
                row.get("model", ""),
                row.get("pretrain", ""),
                row.get("view", ""),
                row.get("noise_power", ""),
            )
            ap = to_float(row.get("acc_pre"))
            apo = to_float(row.get("acc_post"))
            a = to_float(row.get("acc"))
            if ap is not None:
                groups[key]["acc_pre"].append(ap)
            if apo is not None:
                groups[key]["acc_post"].append(apo)
            if a is not None:
                groups[key]["acc"].append(a)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pretrain", "view", "noise_power", "mean_acc_pre", "mean_acc_post", "mean_acc"])
        for (model, pretrain, view, powv), vals in sorted(groups.items()):
            def mean_or_blank(lst):
                return "" if not lst else f"{sum(lst)/len(lst):.4f}"
            w.writerow([
                model,
                pretrain,
                view,
                powv,
                mean_or_blank(vals["acc_pre"]),
                mean_or_blank(vals["acc_post"]),
                mean_or_blank(vals["acc"]),
            ])

    print(f"Wrote aggregated noisy results to: {OUT_CSV}")


if __name__ == "__main__":
    main()


