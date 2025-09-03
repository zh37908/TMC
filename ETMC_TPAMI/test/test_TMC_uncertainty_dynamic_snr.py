import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.aligned_conc_dataset import AlignedConcDataset
from models.TMC import TMC_base_channel, TMC_channel
from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate uncertainty of fused output under dynamic SNR")
    parser.add_argument("--savedir", type=str, required=True, help="Directory containing checkpoint.pt/model_best.pt and args.pt")
    parser.add_argument("--model_type", type=str, choices=["TMC_channel", "TMC_base_channel"], required=True,
                        help="Model variant to evaluate")
    parser.add_argument("--data_path", type=str, required=True, help="Root folder of NYUDv2 dataset (with train/test subfolders)")
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_fig", action="store_true", help="Save uncertainty histograms")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save figures and arrays")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins")
    parser.add_argument("--save_numpy", action="store_true", help="Save raw uncertainty arrays to .npy")
    return parser.parse_args()


def load_args(savedir):
    args_path = os.path.join(savedir, "args.pt")
    if os.path.exists(args_path):
        print(f"Loading args from {args_path}")
        return torch.load(args_path)

    class Dummy:
        pass

    print("args.pt not found. Using default hyper-parameters.")
    return Dummy()


def build_dataloader(args, data_path, batch_sz, n_workers):
    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]

    val_transforms = [
        transforms.Resize((args.FINE_SIZE if hasattr(args, "FINE_SIZE") else 224,
                           args.FINE_SIZE if hasattr(args, "FINE_SIZE") else 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    test_loader = torch.utils.data.DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(data_path, "test"), transform=transforms.Compose(val_transforms)),
        batch_size=batch_sz,
        shuffle=False,
        num_workers=n_workers,
    )
    return test_loader


@torch.no_grad()
def evaluate_uncertainty(model, loader, model_type, n_classes, snr_min, snr_max, device):
    model.eval()

    # 保存融合输出不确定度与正确性标记
    fused_uncertainties = []
    correctness_flags = []

    for batch in tqdm(loader, total=len(loader)):
        rgb, depth, tgt = batch["A"].to(device), batch["B"].to(device), batch["label"].to(device)

        # 动态 SNR 采样（与现有测试脚本一致，整批共享一次随机 SNR）
        if hasattr(model, "args"):
            model.args.channel_snr = float(np.random.uniform(snr_min, snr_max))

        if model_type == "TMC_channel":
            # evidential 模型返回的是 alpha
            _, _, fused_alpha = model(rgb, depth)
            # 不确定度：u = C / sum(alpha)
            S = torch.sum(fused_alpha, dim=1)  # [B]
            u = float(n_classes) / S
            fused_uncertainty = u
            fused_pred = torch.argmax(fused_alpha, dim=1)
        else:
            # 基线模型返回的是 logits
            _, _, fused_logits = model(rgb, depth)
            probs = F.softmax(fused_logits, dim=1)
            # 使用熵作为不确定度：H(p) = -sum p log p
            fused_uncertainty = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
            fused_pred = torch.argmax(fused_logits, dim=1)

        fused_uncertainties.append(fused_uncertainty.detach().cpu())
        correctness_flags.append((fused_pred == tgt).detach().cpu())

    fused_uncertainties = torch.cat(fused_uncertainties, dim=0).numpy()
    correctness_flags = torch.cat(correctness_flags, dim=0).numpy().astype(bool)

    # 统计
    overall_mean_uncertainty = float(np.mean(fused_uncertainties))
    if correctness_flags.any():
        correct_mean_uncertainty = float(np.mean(fused_uncertainties[correctness_flags]))
    else:
        correct_mean_uncertainty = float("nan")
    if (~correctness_flags).any():
        wrong_mean_uncertainty = float(np.mean(fused_uncertainties[~correctness_flags]))
    else:
        wrong_mean_uncertainty = float("nan")

    acc = float(np.mean(correctness_flags.astype(np.float32)))

    return {
        "acc": acc,
        "overall_uncertainty": overall_mean_uncertainty,
        "correct_uncertainty": correct_mean_uncertainty,
        "wrong_uncertainty": wrong_mean_uncertainty,
        "fused_uncertainties": fused_uncertainties,
        "correctness_flags": correctness_flags,
    }


def main():
    cli_args = parse_args()
    set_seed(1)

    savedir = cli_args.savedir.rstrip("/")
    args = load_args(savedir)

    # dataset 默认字段
    for attr, val in [("LOAD_SIZE", 256), ("FINE_SIZE", 224)]:
        if not hasattr(args, attr):
            setattr(args, attr, val)

    # 评测时覆盖 batch 与 workers
    args.batch_sz = cli_args.batch_sz
    args.n_workers = cli_args.n_workers

    # dataloader
    test_loader = build_dataloader(args, cli_args.data_path, cli_args.batch_sz, cli_args.n_workers)

    # 模型
    if cli_args.model_type == "TMC_channel":
        model = TMC_channel(args).to(cli_args.device)
    else:
        model = TMC_base_channel(args).to(cli_args.device)

    # 加载 checkpoint
    ckpt_path_best = os.path.join(savedir, "model_best.pt")
    ckpt_path = ckpt_path_best if os.path.exists(ckpt_path_best) else os.path.join(savedir, "checkpoint.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=cli_args.device)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)

    # 评测不确定度
    metrics = evaluate_uncertainty(
        model,
        test_loader,
        cli_args.model_type,
        getattr(args, "n_classes", 10),
        cli_args.snr_min,
        cli_args.snr_max,
        cli_args.device,
    )

    print("Uncertainty evaluation under dynamic SNR [{:.1f}, {:.1f}] dB".format(cli_args.snr_min, cli_args.snr_max))
    print("acc: {:.4f}".format(metrics["acc"]))
    print("overall_uncertainty: {:.6f}".format(metrics["overall_uncertainty"]))
    print("correct_uncertainty: {:.6f}".format(metrics["correct_uncertainty"]))
    print("wrong_uncertainty: {:.6f}".format(metrics["wrong_uncertainty"]))

    # Optional: draw and save distributions for correct vs wrong predictions
    if cli_args.save_fig:
        out_dir = cli_args.out_dir or os.path.join(savedir, "figs")
        os.makedirs(out_dir, exist_ok=True)
        u = metrics["fused_uncertainties"]
        m = metrics["correctness_flags"]
        u_correct = u[m]
        u_wrong = u[~m]

        # Overlay histogram
        plt.figure(figsize=(7, 5))
        if u_correct.size > 0:
            plt.hist(u_correct, bins=cli_args.bins, alpha=0.6, label="correct", density=True)
        if u_wrong.size > 0:
            plt.hist(u_wrong, bins=cli_args.bins, alpha=0.6, label="wrong", density=True)
        plt.xlabel("uncertainty")
        plt.ylabel("density")
        plt.title(f"Fused uncertainty distribution (acc={metrics['acc']:.3f})")
        plt.legend()
        fig_path_overlay = os.path.join(out_dir, "uncertainty_hist_overlay.png")
        plt.tight_layout()
        plt.savefig(fig_path_overlay)
        plt.close()

        # Separate subplots
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        if u_correct.size > 0:
            plt.hist(u_correct, bins=cli_args.bins, color="#2ca02c", alpha=0.8, density=True)
        plt.title("Correct")
        plt.xlabel("uncertainty")
        plt.ylabel("density")
        plt.subplot(1, 2, 2)
        if u_wrong.size > 0:
            plt.hist(u_wrong, bins=cli_args.bins, color="#d62728", alpha=0.8, density=True)
        plt.title("Wrong")
        plt.xlabel("uncertainty")
        plt.ylabel("density")
        fig_path_split = os.path.join(out_dir, "uncertainty_hist_split.png")
        plt.tight_layout()
        plt.savefig(fig_path_split)
        plt.close()

        print(f"Saved figures to: {fig_path_overlay} and {fig_path_split}")

    if cli_args.save_numpy:
        out_dir = cli_args.out_dir or os.path.join(savedir, "figs")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "fused_uncertainties.npy"), metrics["fused_uncertainties"])
        np.save(os.path.join(out_dir, "correctness_flags.npy"), metrics["correctness_flags"].astype(np.bool_))
        print(f"Saved arrays to: {out_dir}")


if __name__ == "__main__":
    main()


