import argparse
import os
import re
from typing import Optional

import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from PIL import Image

import sys
_ROOT = '/home/hzhaobi/Multired/TMC/ETMC_TPAMI'
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from data.aligned_conc_dataset import AlignedConcDataset
from models.TMC import TMC, TMC_channel_snr
from utils.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TMC (no retransmit, optional source noise)")
    parser.add_argument("--savedir", type=str, required=True, help="Directory containing checkpoint.pt/model_best.pt and args.pt")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["test", "val", "valid", "train"], default="test")
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_file", type=str, default=None, help="Override checkpoint filename inside savedir (e.g., model_best.pt)")

    # Source noise injection (on inputs before model)
    parser.add_argument("--noisy_view", type=str, choices=["none", "rgb", "depth"], default="none",
                        help="Inject Gaussian noise to specified view for testing")
    parser.add_argument("--noise_power", type=float, default=0.0,
                        help="Gaussian noise power (variance). If 0, no noise is added.")
    parser.add_argument("--noise_stage", type=str, choices=["normalized", "pixel", "dataloader"], default="normalized",
                        help="Where to inject noise: normalized (current tensor space) or pixel (denorm -> add -> renorm)")
    return parser.parse_args()


def load_run_args(savedir: str) -> argparse.Namespace:
    args_path = os.path.join(savedir, "args.pt")
    if os.path.exists(args_path):
        return torch.load(args_path)

    class Dummy:
        pass

    return Dummy()


def build_dataloader(args: argparse.Namespace, data_path: str, batch_sz: int, n_workers: int, split: str,
                     noisy_view: str = "none", noise_power: float = 0.0, noise_stage: str = "normalized"):
    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]

    # default pipeline (normalized)
    tf_resize = transforms.Resize((getattr(args, "FINE_SIZE", 224), getattr(args, "FINE_SIZE", 224)))
    tf_totensor = transforms.ToTensor()
    tf_normalize = transforms.Normalize(mean=mean, std=std)
    val_transforms = [tf_resize, tf_totensor, tf_normalize]

    candidates = [split]
    for alt in ["test", "val", "valid", "train"]:
        if alt not in candidates:
            candidates.append(alt)
    picked_dir = None
    tried = []
    for s in candidates:
        d = os.path.join(data_path, s)
        tried.append(d)
        if os.path.isdir(d):
            picked_dir = d
            break
    if picked_dir is None:
        raise FileNotFoundError(f"No dataset split directory found. Tried: {tried}")

    # Build dataset depending on noise_stage
    if str(noise_stage) == "dataloader" and float(noise_power) > 0.0 and str(noisy_view) != "none":
        class NoisyAlignedConcDataset(AlignedConcDataset):
            def __init__(self, cfg, data_dir, view: str, power: float, mean_vec, std_vec):
                super().__init__(cfg, data_dir=data_dir, transform=None, labeled=True)
                self.view = view
                self.power = float(power)
                self.mean = torch.tensor(mean_vec, dtype=torch.float32).view(3, 1, 1)
                self.std = torch.tensor(std_vec, dtype=torch.float32).view(3, 1, 1)

            def __getitem__(self, index):
                if self.labeled:
                    img_path, label = self.imgs[index]
                else:
                    img_path = self.imgs[index]
                    label = None
                img_name = os.path.basename(img_path)
                AB_conc = Image.open(img_path).convert('RGB')
                w, h = AB_conc.size
                w2 = int(w / 2)
                if w2 > self.cfg.FINE_SIZE:
                    A = AB_conc.crop((0, 0, w2, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
                    B = AB_conc.crop((w2, 0, w, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
                else:
                    A = AB_conc.crop((0, 0, w2, h))
                    B = AB_conc.crop((w2, 0, w, h))
                # ToTensor -> [0,1]
                A = tf_totensor(A)
                B = tf_totensor(B)
                # Add pixel noise on selected view
                if self.power > 0.0:
                    std_pix = self.power ** 0.5
                    if self.view == 'rgb':
                        A = torch.clamp(A + torch.randn_like(A) * std_pix, 0.0, 1.0)
                    elif self.view == 'depth':
                        B = torch.clamp(B + torch.randn_like(B) * std_pix, 0.0, 1.0)
                # Normalize
                A = (A - self.mean) / self.std
                B = (B - self.mean) / self.std
                out = {'A': A, 'B': B, 'img_name': img_name}
                if label is not None:
                    out['label'] = label
                return out

        ds = NoisyAlignedConcDataset(args, picked_dir, str(noisy_view), float(noise_power), mean, std)
    else:
        # standard transform path (includes Normalize). Noise (if any) will be injected later in evaluate()
        ds = AlignedConcDataset(args, data_dir=picked_dir, transform=transforms.Compose(val_transforms))

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_sz,
        shuffle=False,
        num_workers=n_workers,
    )


@torch.no_grad()
def evaluate_tmc(
    model: torch.nn.Module,
    loader,
    device: str,
    noisy_view: str,
    noise_power: float,
    num_classes: int,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(loader, total=len(loader)):
        rgb, depth, tgt = batch["A"].to(device), batch["B"].to(device), batch["label"].to(device)

        # optional input noise injection (skip if done in dataloader)
        stage = getattr(evaluate_tmc, "_noise_stage", "normalized")
        if stage != "dataloader" and float(noise_power) > 0.0 and noisy_view != "none":
            std_pix = float(noise_power) ** 0.5
            mean_vec = torch.tensor([0.4951, 0.3601, 0.4587], dtype=rgb.dtype, device=rgb.device).view(1, -1, 1, 1)
            std_vec = torch.tensor([0.1474, 0.1950, 0.1646], dtype=rgb.dtype, device=rgb.device).view(1, -1, 1, 1)
            if stage == "pixel":
                if noisy_view == "rgb":
                    rgb_pix = rgb * std_vec + mean_vec
                    rgb_pix = torch.clamp(rgb_pix + torch.randn_like(rgb_pix) * std_pix, 0.0, 1.0)
                    rgb = (rgb_pix - mean_vec) / std_vec
                elif noisy_view == "depth":
                    depth_pix = depth * std_vec + mean_vec
                    depth_pix = torch.clamp(depth_pix + torch.randn_like(depth_pix) * std_pix, 0.0, 1.0)
                    depth = (depth_pix - mean_vec) / std_vec
            else:
                noise_scale = std_pix / std_vec
                if noisy_view == "rgb":
                    rgb = rgb + torch.randn_like(rgb) * noise_scale
                elif noisy_view == "depth":
                    depth = depth + torch.randn_like(depth) * noise_scale

        # forward: support both TMC (rgb, depth) and TMC_channel_snr (rgb, depth, snr)
        try:
            depth_a, rgb_a, fused_a = model(rgb, depth)
        except TypeError:
            # TMC_channel_snr: pass a very high SNR to effectively disable channel noise
            snr_in = torch.full((rgb.shape[0],), 60.0, dtype=torch.float32, device=rgb.device)
            depth_a, rgb_a, fused_a = model(rgb, depth, snr_in)
        pred = torch.argmax(fused_a, dim=1)
        correct += int((pred == tgt).sum().item())
        total += int(tgt.numel())

    return float(correct) / float(total) if total > 0 else 0.0


def main():
    cli_args = parse_args()
    set_seed(1)

    savedir = cli_args.savedir.rstrip("/")
    run_args = load_run_args(savedir)

    # minimal defaults
    defaults = {
        "LOAD_SIZE": 256,
        "FINE_SIZE": 224,
        "img_embed_pool_type": "avg",
        "num_image_embeds": 1,
        "img_hidden_sz": 512,
        "hidden": [512],
        "dropout": 0.1,
        "n_classes": 10,
    }
    for k, v in defaults.items():
        if not hasattr(run_args, k):
            setattr(run_args, k, v)

    run_args.batch_sz = cli_args.batch_sz
    run_args.n_workers = cli_args.n_workers

    test_loader = build_dataloader(
        run_args, cli_args.data_path, cli_args.batch_sz, cli_args.n_workers, cli_args.split,
        noisy_view=str(getattr(cli_args, "noisy_view", "none")),
        noise_power=float(getattr(cli_args, "noise_power", 0.0)),
        noise_stage=str(getattr(cli_args, "noise_stage", "normalized")),
    )

    # load checkpoint (before model init for auto-arch)
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
            preferred = [f for f in pt_files if "best" in f.lower() or "checkpoint" in f.lower()]
            pick = preferred[0] if len(preferred) > 0 else (pt_files[0] if len(pt_files) > 0 else None)
            if pick is not None:
                ckpt_path = os.path.join(savedir, pick)
    assert ckpt_path is not None and os.path.exists(ckpt_path), f"Checkpoint not found under savedir. Tried default names and scan: {savedir}"
    checkpoint = torch.load(ckpt_path, map_location=cli_args.device)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Ensure channel config is valid (default to awgn); we will pass high SNR to null noise
    if not hasattr(run_args, "channel_type"):
        setattr(run_args, "channel_type", "awgn")
    if not hasattr(run_args, "snr_input_method"):
        setattr(run_args, "snr_input_method", "none")

    # Auto-detect architecture: if channel encoders exist in ckpt, use TMC_channel_snr
    state_keys = list(state.keys())
    is_channel_arch = any(k.startswith("depthchannel_enc.") or k.startswith("rgbchannel_enc.") for k in state_keys)
    if is_channel_arch:
        model = TMC_channel_snr(run_args).to(cli_args.device)
    else:
        model = TMC(run_args).to(cli_args.device)

    model.load_state_dict(state, strict=False)

    # pass noise stage into evaluate function via attribute (simple plumb without refactor)
    setattr(evaluate_tmc, "_noise_stage", str(getattr(cli_args, "noise_stage", "normalized")))

    acc = evaluate_tmc(
        model,
        test_loader,
        cli_args.device,
        str(getattr(cli_args, "noisy_view", "none")),
        float(getattr(cli_args, "noise_power", 0.0)),
        int(getattr(run_args, "n_classes", 10)),
    )

    if float(getattr(cli_args, "noise_power", 0.0)) > 0.0 and str(getattr(cli_args, "noisy_view", "none")) != "none":
        print(f"acc (with source noise: view={cli_args.noisy_view}, power={cli_args.noise_power}): {acc:.4f}")
    else:
        print(f"acc (clean): {acc:.4f}")


if __name__ == "__main__":
    main()


