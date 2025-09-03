import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torchvision.transforms as transforms

"""
将脚本移动到子目录后，增加动态 sys.path 注入，确保可以从
`TMC/ETMC_TPAMI` 目录正确导入 `models/` 与 `data/` 模块。
"""
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from data.aligned_conc_dataset import AlignedConcDataset
from models.TMC import TMC_base_channel, TMC_channel, ce_loss
from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate static TMC/TMC_base models under dynamic SNR")
    parser.add_argument("--savedir", type=str, required=True, help="Directory containing checkpoint.pt/model_best.pt and args.pt")
    parser.add_argument("--model_type", type=str, choices=["TMC_channel", "TMC_base_channel"], required=True,
                        help="Model variant to evaluate")
    parser.add_argument("--data_path", type=str, required=True, help="Root folder of NYUDv2 dataset (with train/test subfolders)")
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snr_fixed", type=float, default=None, help="If set, evaluate with a fixed static SNR (overrides snr_min/max)")
    return parser.parse_args()


def load_args(savedir):
    """Load training args if available, otherwise create a minimal dummy object."""
    args_path = os.path.join(savedir, "args.pt")
    if os.path.exists(args_path):
        print(f"Loading args from {args_path}")
        return torch.load(args_path)

    class Dummy:  # fallback minimal args
        pass
    print("args.pt not found. Using default hyper-parameters.")
    return Dummy()


def build_dataloaders(args, data_path, batch_sz, n_workers):
    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]

    train_transforms = [
        transforms.Resize((args.LOAD_SIZE if hasattr(args, "LOAD_SIZE") else 256, args.LOAD_SIZE if hasattr(args, "LOAD_SIZE") else 256)),
        transforms.RandomCrop((args.FINE_SIZE if hasattr(args, "FINE_SIZE") else 224, args.FINE_SIZE if hasattr(args, "FINE_SIZE") else 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    val_transforms = [
        transforms.Resize((args.FINE_SIZE if hasattr(args, "FINE_SIZE") else 224, args.FINE_SIZE if hasattr(args, "FINE_SIZE") else 224)),
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


def evaluate(model, loader, model_type, n_classes, snr_min, snr_max, device, snr_fixed=None):
    model.eval()
    ce_criterion = ce_loss
    ce_criterion_eval = nn.CrossEntropyLoss()

    losses, depth_preds, rgb_preds, depthrgb_preds, tgts = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            rgb, depth, tgt = batch["A"].to(device), batch["B"].to(device), batch["label"].to(device)
            # set channel snr: fixed if provided, else dynamic sampling
            if hasattr(model, "args"):
                if snr_fixed is not None:
                    model.args.channel_snr = float(snr_fixed)
                else:
                    model.args.channel_snr = float(np.random.uniform(snr_min, snr_max))

            if model_type == "TMC_channel":
                depth_alpha, rgb_alpha, depth_rgb_alpha = model(rgb, depth)
                loss = ce_criterion(tgt, depth_alpha, n_classes, np.inf, getattr(model.args, "annealing_epoch", 10)) + \
                       ce_criterion(tgt, rgb_alpha, n_classes, np.inf, getattr(model.args, "annealing_epoch", 10)) + \
                       ce_criterion(tgt, depth_rgb_alpha, n_classes, np.inf, getattr(model.args, "annealing_epoch", 10))
                depth_logit = depth_alpha
                rgb_logit = rgb_alpha
                depthrgb_logit = depth_rgb_alpha
            else:  # TMC_base_channel
                depth_logit, rgb_logit, depthrgb_logit = model(rgb, depth)
                loss = ce_criterion_eval(depth_logit, tgt) + ce_criterion_eval(rgb_logit, tgt) + ce_criterion_eval(depthrgb_logit, tgt)

            losses.append(loss.item())
            depth_preds.append(depth_logit.argmax(dim=1).cpu().numpy())
            rgb_preds.append(rgb_logit.argmax(dim=1).cpu().numpy())
            depthrgb_preds.append(depthrgb_logit.argmax(dim=1).cpu().numpy())
            tgts.append(tgt.cpu().numpy())

    tgts = [l for sl in tgts for l in sl]
    depth_preds = [l for sl in depth_preds for l in sl]
    rgb_preds = [l for sl in rgb_preds for l in sl]
    depthrgb_preds = [l for sl in depthrgb_preds for l in sl]

    metrics = {
        "loss": np.mean(losses),
        "depth_acc": accuracy_score(tgts, depth_preds),
        "rgb_acc": accuracy_score(tgts, rgb_preds),
        "depthrgb_acc": accuracy_score(tgts, depthrgb_preds),
    }
    return metrics


def main():
    cli_args = parse_args()
    set_seed(1)

    savedir = cli_args.savedir.rstrip("/")
    args = load_args(savedir)

    # Override dataset related fields if missing
    for attr, val in [("LOAD_SIZE", 256), ("FINE_SIZE", 224)]:
        if not hasattr(args, attr):
            setattr(args, attr, val)

    # batch size / num_workers override for evaluation
    args.batch_sz = cli_args.batch_sz
    args.n_workers = cli_args.n_workers

    # Build dataloader
    test_loader = build_dataloaders(args, cli_args.data_path, cli_args.batch_sz, cli_args.n_workers)

    # Prepare model
    if cli_args.model_type == "TMC_channel":
        model = TMC_channel(args).to(cli_args.device)
    else:
        model = TMC_base_channel(args).to(cli_args.device)

    # Load checkpoint (prefer model_best.pt else checkpoint.pt)
    ckpt_path_best = os.path.join(savedir, "model_best.pt")
    ckpt_path = ckpt_path_best if os.path.exists(ckpt_path_best) else os.path.join(savedir, "checkpoint.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=cli_args.device)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)

    # Evaluate
    metrics = evaluate(model, test_loader, cli_args.model_type, getattr(args, "n_classes", 10),
                       cli_args.snr_min, cli_args.snr_max, cli_args.device, cli_args.snr_fixed)

    if cli_args.snr_fixed is not None:
        print("Evaluation under static SNR {:.1f} dB".format(cli_args.snr_fixed))
    else:
        print("Evaluation under dynamic SNR [{:.1f}, {:.1f}] dB".format(cli_args.snr_min, cli_args.snr_max))
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
