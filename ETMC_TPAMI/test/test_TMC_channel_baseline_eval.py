import argparse
import os
from typing import Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import sys
_ROOT = '/home/hzhaobi/Multired/TMC/ETMC_TPAMI'
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from data.aligned_conc_dataset import AlignedConcDataset
from models.TMC import TMC_channel
from utils.utils import set_seed

import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline TMC_channel (no retransmission)")
    parser.add_argument("--savedir", type=str, required=True, help="Directory containing checkpoint.pt/model_best.pt and args.pt")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["test", "val", "valid", "train"], default="test")
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # SNR evaluation config
    parser.add_argument("--snr", type=float, default=None, help="Fixed SNR for evaluation; if None and not dynamic, infer from args.pt or path")
    parser.add_argument("--use_dynamic_snr", action="store_true", help="Enable dynamic SNR per batch (depth/rgb independently sampled in model)")
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--ckpt_file", type=str, default=None, help="Override checkpoint filename inside savedir (e.g., model_best.pt)")
    # Noise injection config
    parser.add_argument("--noisy_view", type=str, choices=["none", "rgb", "depth"], default="none",
                        help="Inject Gaussian noise to specified view for testing")
    parser.add_argument("--noise_power", type=float, default=0.0,
                        help="Gaussian noise power (variance). If 0, no noise is added.")
    parser.add_argument("--save_uncert_density", action="store_true", help="If set, plot/save density of final fused uncertainty")
    parser.add_argument("--uncert_density_path", type=str, default=None, help="Output path for uncertainty density figure (PNG)")
    return parser.parse_args()


def load_run_args(savedir: str) -> argparse.Namespace:
    args_path = os.path.join(savedir, "args.pt")
    if os.path.exists(args_path):
        return torch.load(args_path)
    class Dummy:
        pass
    return Dummy()


def build_dataloader(args: argparse.Namespace, data_path: str, batch_sz: int, n_workers: int, split: str):
    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]

    val_transforms = [
        transforms.Resize((getattr(args, "FINE_SIZE", 224), getattr(args, "FINE_SIZE", 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    # pick split dir
    candidates = [split]
    for alt in ["test", "val", "valid", "train"]:
        if alt not in candidates:
            candidates.append(alt)
    picked_dir = None
    for s in candidates:
        d = os.path.join(data_path, s)
        if os.path.isdir(d):
            picked_dir = d
            break
    if picked_dir is None:
        raise FileNotFoundError(f"No dataset split directory found under {data_path}")

    return torch.utils.data.DataLoader(
        AlignedConcDataset(args, data_dir=picked_dir, transform=transforms.Compose(val_transforms)),
        batch_size=batch_sz,
        shuffle=False,
        num_workers=n_workers,
    )


def infer_fixed_snr(cli_args: argparse.Namespace, run_args: argparse.Namespace) -> Optional[float]:
    if cli_args.snr is not None:
        return float(cli_args.snr)
    if hasattr(run_args, "channel_snr"):
        try:
            return float(run_args.channel_snr)
        except Exception:
            pass
    # best-effort parse from path
    import re
    m = re.search(r"snr(\d+(?:\.\d+)?)", cli_args.savedir, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, num_classes: int, device: str, use_dynamic_snr: bool, fixed_snr: Optional[float], noisy_view: str, noise_power: float) -> dict:
    model.eval()
    correct_depth, correct_rgb, correct_fused, total = 0, 0, 0, 0
    fused_uncert = []
    for batch in tqdm(loader, total=len(loader)):
        rgb, depth, tgt = batch["A"].to(device), batch["B"].to(device), batch["label"].to(device)
        if noise_power > 0.0 and noisy_view != "none":
            std_pix = float(noise_power) ** 0.5
            # per-channel std used in Normalize; adding noise before Normalize is equivalent to adding (noise/std) after Normalize
            std_vec = torch.tensor([0.1474, 0.1950, 0.1646], dtype=rgb.dtype, device=rgb.device).view(1, -1, 1, 1)
            noise_scale = std_pix / std_vec
            if noisy_view == "rgb":
                rgb = rgb + torch.randn_like(rgb) * noise_scale
            elif noisy_view == "depth":
                depth = depth + torch.randn_like(depth) * noise_scale
        if use_dynamic_snr:
            snr_in = None
        else:
            fs = float(fixed_snr) if fixed_snr is not None else float(getattr(model.args, "channel_snr", 20.0))
            snr_in = torch.full((rgb.shape[0],), fs, dtype=torch.float32, device=rgb.device)

        depth_a, rgb_a, fused_a = model(rgb, depth, snr_in)
        # evidential alphas -> use argmax for accuracy
        pd = torch.argmax(depth_a, dim=1)
        pr = torch.argmax(rgb_a, dim=1)
        pf = torch.argmax(fused_a, dim=1)

        correct_depth += int((pd == tgt).sum().item())
        correct_rgb += int((pr == tgt).sum().item())
        correct_fused += int((pf == tgt).sum().item())
        total += int(tgt.shape[0])

        u_fused = float(num_classes) / torch.sum(fused_a, dim=1)
        fused_uncert.append(u_fused.detach().cpu())

    fused_uncert_all = torch.cat(fused_uncert, dim=0).numpy() if len(fused_uncert) > 0 else np.array([])

    return {
        "depth_acc": float(correct_depth) / float(total) if total > 0 else 0.0,
        "rgb_acc": float(correct_rgb) / float(total) if total > 0 else 0.0,
        "acc": float(correct_fused) / float(total) if total > 0 else 0.0,
        "uncert_post_all": fused_uncert_all,
    }


def main():
    cli_args = parse_args()
    set_seed(1)

    savedir = cli_args.savedir.rstrip("/")
    run_args = load_run_args(savedir)

    # defaults required by dataset/model
    defaults = {
        "LOAD_SIZE": 256,
        "FINE_SIZE": 224,
        "img_embed_pool_type": "avg",
        "num_image_embeds": 1,
        "img_hidden_sz": 512,
        "hidden": [512],
        "dropout": 0.1,
        "n_classes": 10,
        "channel_hidden": [512],
        "channel_size": 256,
    }
    for k, v in defaults.items():
        if not hasattr(run_args, k):
            setattr(run_args, k, v)

    run_args.batch_sz = cli_args.batch_sz
    run_args.n_workers = cli_args.n_workers

    test_loader = build_dataloader(run_args, cli_args.data_path, cli_args.batch_sz, cli_args.n_workers, cli_args.split)

    # build model
    model = TMC_channel(run_args).to(cli_args.device)

    # load checkpoint
    ckpt_path: Optional[str] = None
    if cli_args.ckpt_file is not None:
        candidate = os.path.join(savedir, cli_args.ckpt_file)
        if os.path.exists(candidate):
            ckpt_path = candidate
    if ckpt_path is None:
        ckpt_path_best = os.path.join(savedir, "model_best.pt")
        ckpt_path = ckpt_path_best if os.path.exists(ckpt_path_best) else os.path.join(savedir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            try:
                files = os.listdir(savedir)
            except Exception:
                files = []
            pt_files = [f for f in files if f.endswith(".pt") or f.endswith(".pth")]
            pick = pt_files[0] if len(pt_files) > 0 else None
            if pick is not None:
                ckpt_path = os.path.join(savedir, pick)
    assert ckpt_path is not None and os.path.exists(ckpt_path), f"Checkpoint not found under savedir: {savedir}"
    checkpoint = torch.load(ckpt_path, map_location=cli_args.device)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)

    # resolve SNR mode
    fixed_snr = None
    if not cli_args.use_dynamic_snr:
        fixed_snr = infer_fixed_snr(cli_args, run_args)
        if fixed_snr is not None and hasattr(model, "args"):
            model.args.channel_snr = float(fixed_snr)
    else:
        if hasattr(model, "args"):
            model.args.snr_min = float(cli_args.snr_min)
            model.args.snr_max = float(cli_args.snr_max)

    metrics = evaluate(
        model,
        test_loader,
        getattr(run_args, "n_classes", 10),
        cli_args.device,
        bool(cli_args.use_dynamic_snr),
        fixed_snr,
        str(getattr(cli_args, "noisy_view", "none")),
        float(getattr(cli_args, "noise_power", 0.0)),
    )

    if fixed_snr is not None:
        print(f"Eval at fixed SNR: {fixed_snr:.1f} dB")
    else:
        print(f"Eval under dynamic SNR range: [{float(cli_args.snr_min):.1f}, {float(cli_args.snr_max):.1f}] dB")

    print(f"depth_acc: {metrics['depth_acc']:.4f}")
    print(f"rgb_acc:   {metrics['rgb_acc']:.4f}")
    print(f"acc:       {metrics['acc']:.4f}")

    # Optionally plot/save fused uncertainty density
    if getattr(cli_args, "save_uncert_density", False) and metrics.get("uncert_post_all") is not None and metrics["uncert_post_all"].size > 0:
        import matplotlib.pyplot as plt
        u_all = metrics["uncert_post_all"]
        fig = plt.figure(figsize=(6, 4))
        try:
            sns.kdeplot(u_all, fill=True, color="#1f77b4", alpha=0.6)
        except Exception:
            plt.hist(u_all, bins=60, density=True, color="#1f77b4", alpha=0.6)
        plt.xlabel("Fused uncertainty")
        plt.ylabel("Density")
        plt.title("Uncertainty density (baseline)")
        plt.grid(True, alpha=0.3)
        out_path = cli_args.uncert_density_path or os.path.join(savedir, "uncertainty_density.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved uncertainty density figure to: {out_path}")


if __name__ == "__main__":
    main()


