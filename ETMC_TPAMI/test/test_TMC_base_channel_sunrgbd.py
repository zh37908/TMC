import argparse
import os
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
from models.TMC import TMC_base_channel
from utils.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TMC_base_channel on SUNRGBD (optional source noise)")
    p.add_argument("--savedir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--split", type=str, choices=["test", "val", "valid", "train"], default="test")
    p.add_argument("--batch_sz", type=int, default=32)
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_file", type=str, default=None)

    # source noise
    p.add_argument("--noisy_view", type=str, choices=["none", "rgb", "depth"], default="none")
    p.add_argument("--noise_power", type=float, default=0.0)
    p.add_argument("--noise_stage", type=str, choices=["normalized", "pixel", "dataloader"], default="normalized")

    # channel snr for eval (TMC_base_channel uses args.channel_snr in forward)
    p.add_argument("--channel_snr_db", type=float, default=None, help="If set, override run_args.channel_snr for eval")
    return p.parse_args()


def load_run_args(savedir: str) -> argparse.Namespace:
    args_path = os.path.join(savedir, "args.pt")
    if os.path.exists(args_path):
        return torch.load(args_path)
    class Dummy: pass
    return Dummy()


def build_dataloader(args: argparse.Namespace, data_path: str, batch_sz: int, n_workers: int, split: str,
                     noisy_view: str = "none", noise_power: float = 0.0, noise_stage: str = "normalized"):
    # SUNRGBD stats
    mean = [0.6983, 0.3918, 0.4474]
    std = [0.1648, 0.1359, 0.1644]

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

    if str(noise_stage) == "dataloader" and float(noise_power) > 0.0 and str(noisy_view) != "none":
        class NoisyAlignedConcDataset(AlignedConcDataset):
            def __init__(self, cfg, data_dir, view: str, power: float, mean_vec, std_vec):
                super().__init__(cfg, data_dir=data_dir, transform=None, labeled=True)
                self.view = view
                self.power = float(power)
                self.mean = torch.tensor(mean_vec, dtype=torch.float32).view(3, 1, 1)
                self.std = torch.tensor(std_vec, dtype=torch.float32).view(3, 1, 1)
            def __getitem__(self, index):
                img_path, label = self.imgs[index]
                img_name = os.path.basename(img_path)
                AB = Image.open(img_path).convert('RGB')
                w, h = AB.size; w2 = int(w/2)
                if w2 > self.cfg.FINE_SIZE:
                    A = AB.crop((0,0,w2,h)).resize((self.cfg.LOAD_SIZE,self.cfg.LOAD_SIZE), Image.BICUBIC)
                    B = AB.crop((w2,0,w,h)).resize((self.cfg.LOAD_SIZE,self.cfg.LOAD_SIZE), Image.BICUBIC)
                else:
                    A = AB.crop((0,0,w2,h)); B = AB.crop((w2,0,w,h))
                A = tf_totensor(A); B = tf_totensor(B)
                if self.power > 0.0:
                    std_pix = self.power ** 0.5
                    if self.view == 'rgb':
                        A = torch.clamp(A + torch.randn_like(A)*std_pix, 0.0, 1.0)
                    elif self.view == 'depth':
                        B = torch.clamp(B + torch.randn_like(B)*std_pix, 0.0, 1.0)
                A = (A - self.mean)/self.std; B = (B - self.mean)/self.std
                return { 'A': A, 'B': B, 'img_name': img_name, 'label': label }
        ds = NoisyAlignedConcDataset(args, picked_dir, str(noisy_view), float(noise_power), mean, std)
    else:
        ds = AlignedConcDataset(args, data_dir=picked_dir, transform=transforms.Compose(val_transforms))

    return torch.utils.data.DataLoader(ds, batch_size=batch_sz, shuffle=False, num_workers=n_workers)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: str,
    noisy_view: str,
    noise_power: float,
) -> float:
    model.eval()
    correct = 0; total = 0
    # optional per-batch pixel/normalized injection
    for batch in tqdm(loader, total=len(loader)):
        rgb, depth, tgt = batch['A'].to(device), batch['B'].to(device), batch['label'].to(device)
        depth_logits, rgb_logits, depth_rgb_logits = model(rgb, depth)
        pred = torch.argmax(depth_rgb_logits, dim=1)
        correct += int((pred == tgt).sum().item()); total += int(tgt.numel())
    return float(correct)/float(total) if total>0 else 0.0


def main():
    cli = parse_args()
    set_seed(1)

    savedir = cli.savedir.rstrip('/')
    run_args = load_run_args(savedir)

    # SUNRGBD defaults
    defaults = {
        'LOAD_SIZE': 256, 'FINE_SIZE': 224,
        'img_embed_pool_type': 'avg', 'num_image_embeds': 1,
        'img_hidden_sz': 512, 'hidden': [512], 'dropout': 0.1,
        'n_classes': 19,
        # channel path params used by TMC_base_channel
        'channel_hidden': [512], 'channel_size': 256,
        'channel_type': 'awgn', 'channel_snr': 20.0,
    }
    for k,v in defaults.items():
        if not hasattr(run_args, k): setattr(run_args, k, v)

    # allow overriding eval SNR
    if cli.channel_snr_db is not None:
        run_args.channel_snr = float(cli.channel_snr_db)

    run_args.batch_sz = cli.batch_sz; run_args.n_workers = cli.n_workers

    loader = build_dataloader(
        run_args, cli.data_path, cli.batch_sz, cli.n_workers, cli.split,
        noisy_view=str(getattr(cli, 'noisy_view', 'none')),
        noise_power=float(getattr(cli, 'noise_power', 0.0)),
        noise_stage=str(getattr(cli, 'noise_stage', 'normalized')),
    )

    # build and load
    model = TMC_base_channel(run_args).to(cli.device)
    # pick ckpt
    ckpt_path: Optional[str] = None
    if cli.ckpt_file:
        candidate = os.path.join(savedir, cli.ckpt_file)
        if os.path.exists(candidate): ckpt_path = candidate
    if ckpt_path is None:
        best = os.path.join(savedir, 'model_best.pt'); last = os.path.join(savedir, 'checkpoint.pt')
        ckpt_path = best if os.path.exists(best) else last
        if not os.path.exists(ckpt_path):
            files = []
            try: files = os.listdir(savedir)
            except Exception: pass
            pt = [f for f in files if f.endswith('.pt') or f.endswith('.pth')]
            pref = [f for f in pt if 'best' in f.lower() or 'checkpoint' in f.lower()]
            pick = pref[0] if len(pref)>0 else (pt[0] if len(pt)>0 else None)
            if pick is not None: ckpt_path = os.path.join(savedir, pick)
    assert ckpt_path and os.path.exists(ckpt_path), f"Checkpoint not found under {savedir}"
    ckpt = torch.load(ckpt_path, map_location=cli.device)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    acc = evaluate(model, loader, cli.device, str(cli.noisy_view), float(cli.noise_power))
    if float(cli.noise_power) > 0.0 and str(cli.noisy_view) != 'none':
        print(f"acc (with source noise: view={cli.noisy_view}, power={cli.noise_power}): {acc:.4f}")
    else:
        print(f"acc (clean): {acc:.4f}")


if __name__ == '__main__':
    main()


