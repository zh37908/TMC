import argparse
import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import sys
_ROOT = '/home/hzhaobi/Multired/TMC/ETMC_TPAMI'
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from data.aligned_conc_dataset import AlignedConcDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize clean vs. source-noised images for RGB/Depth")
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--split', type=str, default='test', choices=['test', 'val', 'valid', 'train'])
    p.add_argument('--num_samples', type=int, default=6)
    p.add_argument('--out_dir', type=str, default='./visualizations/source_noise')
    p.add_argument('--views', type=str, default='rgb,depth', help='comma separated: rgb,depth')
    p.add_argument('--noise_powers', type=str, default='1,10,100', help='comma separated powers (variance)')
    return p.parse_args()


def build_loader(data_path: str, split: str, batch_sz: int = 32, n_workers: int = 4):
    class Dummy:
        pass
    args = Dummy()
    setattr(args, 'FINE_SIZE', 224)
    setattr(args, 'LOAD_SIZE', 256)
    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]
    val_tf = [
        transforms.Resize((getattr(args, 'FINE_SIZE', 224), getattr(args, 'FINE_SIZE', 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    # pick split directory with fallbacks
    for s in [split, 'test', 'val', 'valid', 'train']:
        d = os.path.join(data_path, s)
        if os.path.isdir(d):
            split_dir = d
            break
    else:
        raise FileNotFoundError(f'No dataset split directory found under {data_path}')

    ds = AlignedConcDataset(args, data_dir=split_dir, transform=transforms.Compose(val_tf))
    return torch.utils.data.DataLoader(ds, batch_size=batch_sz, shuffle=False, num_workers=n_workers)


def denormalize(img: torch.Tensor) -> np.ndarray:
    # img: (C,H,W), normalized with mean/std below
    mean = torch.tensor([0.4951, 0.3601, 0.4587], dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std = torch.tensor([0.1474, 0.1950, 0.1646], dtype=img.dtype, device=img.device).view(-1, 1, 1)
    x = img * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    x_np = x.detach().cpu().permute(1, 2, 0).numpy()
    return x_np


def add_noise(x: torch.Tensor, view: str, power: float) -> torch.Tensor:
    if power <= 0.0:
        return x
    std_pix = float(power) ** 0.5
    std_vec = torch.tensor([0.1474, 0.1950, 0.1646], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    noise_scale = std_pix / std_vec
    if view == 'rgb' or view == 'depth':
        return x + torch.randn_like(x) * noise_scale
    return x


def make_grid_figure(clean_list: List[torch.Tensor], noisy_list: List[torch.Tensor], title: str, out_path: str):
    n = len(clean_list)
    cols = 2
    rows = n
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    if rows == 1:
        axes = np.array([axes])
    for i in range(n):
        clean_img = denormalize(clean_list[i])
        noisy_img = denormalize(noisy_list[i])
        axes[i, 0].imshow(clean_img)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('clean')
        axes[i, 1].imshow(noisy_img)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('noisy')
    fig.suptitle(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    views = [v.strip() for v in args.views.split(',') if v.strip() in ('rgb', 'depth')]
    noise_powers = [float(x.strip()) for x in args.noise_powers.split(',') if x.strip()]

    loader = build_loader(args.data_path, args.split)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collected_rgb: List[torch.Tensor] = []
    collected_depth: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch['A']
            depth = batch['B']
            for i in range(rgb.shape[0]):
                if len(collected_rgb) < args.num_samples:
                    collected_rgb.append(rgb[i])
                if len(collected_depth) < args.num_samples:
                    collected_depth.append(depth[i])
            if len(collected_rgb) >= args.num_samples and len(collected_depth) >= args.num_samples:
                break

    # stack and move to device for noise ops
    rgb_stack = torch.stack(collected_rgb, dim=0).to(device)
    depth_stack = torch.stack(collected_depth, dim=0).to(device)

    for view in views:
        for power in noise_powers:
            if view == 'rgb':
                clean = rgb_stack
                noisy = add_noise(rgb_stack, 'rgb', power)
                clean_list = [clean[i].cpu() for i in range(clean.shape[0])]
                noisy_list = [noisy[i].cpu() for i in range(noisy.shape[0])]
            else:
                clean = depth_stack
                noisy = add_noise(depth_stack, 'depth', power)
                clean_list = [clean[i].cpu() for i in range(clean.shape[0])]
                noisy_list = [noisy[i].cpu() for i in range(noisy.shape[0])]

            out_path = os.path.join(args.out_dir, view, f"{view}_noise_power_{int(power) if power.is_integer() else power}.png")
            title = f"{view.upper()} | noise power={power}"
            make_grid_figure(clean_list, noisy_list, title, out_path)
            print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()


