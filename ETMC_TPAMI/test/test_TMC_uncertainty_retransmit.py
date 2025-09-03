import argparse
import os
import re
from typing import Dict, Optional
import sys
sys.path.append('/home/hzhaobi/Multired/TMC/ETMC_TPAMI')

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.aligned_conc_dataset import AlignedConcDataset
from models.TMC import TMC_channel, TMC_base_channel
from utils.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retransmission strategy based on fused uncertainty threshold")
    parser.add_argument("--savedir", type=str, required=True, help="Directory containing checkpoint.pt/model_best.pt and args.pt")
    parser.add_argument("--model_type", type=str, choices=["TMC_channel", "TMC_base_channel"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snr", type=float, default=None, help="Fixed SNR for evaluation; if None, infer from args.pt or path")

    # Retransmission config
    parser.add_argument("--retransmit", action="store_true", help="Enable retransmission (single retry per sample)")
    parser.add_argument("--rt_threshold", type=float, default=None, help="Threshold on fused uncertainty to trigger retransmission")
    parser.add_argument("--rt_trigger", type=str, choices=["low", "high"], default="low", help="Trigger when fused uncertainty is below(low)/above(high) threshold")
    parser.add_argument("--rt_pick_view", type=str, choices=["min", "max"], default="max", help="Pick view with min/max per-view uncertainty for retransmission")
    parser.add_argument("--rt_snr", type=float, default=None, help="SNR(dB) used only for retransmission forward; default: reuse --snr/inferred")

    # Fusion robustness options (evidential)
    parser.add_argument("--ds_discount", action="store_true", help="Apply reliability discount before DS fusion on evidential outputs")
    parser.add_argument("--choose_better", action="store_true", default=True, help="For triggered samples, keep the result (pre vs retransmit) with lower uncertainty")
    return parser.parse_args()


def load_run_args(savedir: str) -> argparse.Namespace:
    args_path = os.path.join(savedir, "args.pt")
    if os.path.exists(args_path):
        return torch.load(args_path)

    class Dummy:
        pass

    return Dummy()


def build_dataloader(args: argparse.Namespace, data_path: str, batch_sz: int, n_workers: int):
    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]

    val_transforms = [
        transforms.Resize((getattr(args, "FINE_SIZE", 224), getattr(args, "FINE_SIZE", 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return torch.utils.data.DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(data_path, "test"), transform=transforms.Compose(val_transforms)),
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
    m = re.search(r"snr(\d+(?:\.\d+)?)", cli_args.savedir, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def ds_combin_two(alpha1: torch.Tensor, alpha2: torch.Tensor, num_classes: int) -> torch.Tensor:
    S1 = torch.sum(alpha1, dim=1, keepdim=True)
    S2 = torch.sum(alpha2, dim=1, keepdim=True)
    E1 = alpha1 - 1.0
    E2 = alpha2 - 1.0
    b1 = E1 / S1.expand_as(E1)
    b2 = E2 / S2.expand_as(E2)
    u1 = num_classes / S1
    u2 = num_classes / S2

    bb = torch.bmm(b1.view(-1, num_classes, 1), b2.view(-1, 1, num_classes))
    bb_sum = torch.sum(bb, dim=(1, 2))
    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    K = bb_sum - bb_diag

    bu = b1 * u2.expand_as(b1)
    ub = b2 * u1.expand_as(b2)

    b_a = (b1 * b2 + bu + ub) / (1.0 - K).view(-1, 1).expand_as(b1)
    u_a = (u1 * u2) / (1.0 - K).view(-1, 1).expand_as(u1)
    S_a = num_classes / u_a
    e_a = b_a * S_a.expand_as(b_a)
    return e_a + 1.0


def apply_discount(alpha: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Reliability discount: alpha' = 1 + gamma * (alpha - 1).
    gamma shape can be (B,1) or (B,) and will broadcast to alpha's shape.
    """
    return 1.0 + (alpha - 1.0) * gamma


@torch.no_grad()
def evaluate_with_retransmission(
    model: torch.nn.Module,
    loader,
    model_type: str,
    num_classes: int,
    device: str,
    rt_cfg: Dict,
) -> Dict:
    model.eval()

    all_uncert_pre, all_uncert_post = [], []
    all_correct_pre, all_correct_post = [], []
    num_rt, num_total = 0, 0

    for batch in tqdm(loader, total=len(loader)):
        rgb, depth, tgt = batch["A"].to(device), batch["B"].to(device), batch["label"].to(device)

        if model_type == "TMC_channel":
            depth_a, rgb_a, fused_a = model(rgb, depth)
            u_fused = float(num_classes) / torch.sum(fused_a, dim=1)
            pred_fused = torch.argmax(fused_a, dim=1)

            # record pre-RT
            all_uncert_pre.append(u_fused.detach().cpu())
            all_correct_pre.append((pred_fused == tgt).detach().cpu())

            # setup RT mask
            do_rt_mask = None
            if rt_cfg["enabled"] and rt_cfg["threshold"] is not None:
                tau = rt_cfg["threshold"]
                do_rt_mask = (u_fused < tau) if rt_cfg["trigger"] == "low" else (u_fused > tau)

            # default post equals pre
            fused_post = fused_a
            if do_rt_mask is not None and torch.any(do_rt_mask):
                # one more forward pass for retransmission (optionally with different SNR)
                orig_snr = None
                if hasattr(model, "args") and rt_cfg.get("rt_snr") is not None:
                    orig_snr = getattr(model.args, "channel_snr", None)
                    model.args.channel_snr = float(rt_cfg["rt_snr"])
                depth_a2, rgb_a2, _ = model(rgb, depth)
                if hasattr(model, "args") and rt_cfg.get("rt_snr") is not None and orig_snr is not None:
                    model.args.channel_snr = orig_snr
                u_depth = float(num_classes) / torch.sum(depth_a, dim=1)
                u_rgb = float(num_classes) / torch.sum(rgb_a, dim=1)

                fused_list = []
                for i in range(fused_a.shape[0]):
                    num_total += 1
                    if bool(do_rt_mask[i].item()):
                        num_rt += 1
                        pick_rgb = (u_rgb[i] <= u_depth[i]) if rt_cfg["pick_view"] == "min" else (u_rgb[i] >= u_depth[i])
                        if bool(pick_rgb):
                            a_left, a_right = depth_a[i:i+1], rgb_a2[i:i+1]
                            if rt_cfg.get("ds_discount"):
                                u_l = float(num_classes) / torch.sum(a_left, dim=1, keepdim=True)
                                u_r = float(num_classes) / torch.sum(a_right, dim=1, keepdim=True)
                                g_l = 1.0 / (1.0 + u_l)
                                g_r = 1.0 / (1.0 + u_r)
                                a_left = apply_discount(a_left, g_l)
                                a_right = apply_discount(a_right, g_r)
                            fused_i = ds_combin_two(a_left, a_right, num_classes)
                        else:
                            a_left, a_right = depth_a2[i:i+1], rgb_a[i:i+1]
                            if rt_cfg.get("ds_discount"):
                                u_l = float(num_classes) / torch.sum(a_left, dim=1, keepdim=True)
                                u_r = float(num_classes) / torch.sum(a_right, dim=1, keepdim=True)
                                g_l = 1.0 / (1.0 + u_l)
                                g_r = 1.0 / (1.0 + u_r)
                                a_left = apply_discount(a_left, g_l)
                                a_right = apply_discount(a_right, g_r)
                            fused_i = ds_combin_two(a_left, a_right, num_classes)
                        if rt_cfg.get("choose_better", True):
                            u_i_pre = float(num_classes) / torch.sum(fused_a[i:i+1], dim=1)
                            u_i_post = float(num_classes) / torch.sum(fused_i, dim=1)
                            if (u_i_post > u_i_pre).item():
                                fused_i = fused_a[i:i+1]
                        fused_list.append(fused_i)
                    else:
                        fused_list.append(fused_a[i:i+1])
                fused_post = torch.cat(fused_list, dim=0)
            else:
                num_total += fused_a.shape[0]

            pred_post = torch.argmax(fused_post, dim=1)
            u_post = float(num_classes) / torch.sum(fused_post, dim=1)
            all_uncert_post.append(u_post.detach().cpu())
            all_correct_post.append((pred_post == tgt).detach().cpu())

        else:  # TMC_base_channel
            depth_l, rgb_l, fused_l = model(rgb, depth)
            probs = F.softmax(fused_l, dim=1)
            u_fused = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
            pred_fused = torch.argmax(fused_l, dim=1)

            # record pre-RT
            all_uncert_pre.append(u_fused.detach().cpu())
            all_correct_pre.append((pred_fused == tgt).detach().cpu())

            do_rt_mask = None
            if rt_cfg["enabled"] and rt_cfg["threshold"] is not None:
                tau = rt_cfg["threshold"]
                do_rt_mask = (u_fused < tau) if rt_cfg["trigger"] == "low" else (u_fused > tau)

            fused_post = fused_l
            if do_rt_mask is not None and torch.any(do_rt_mask):
                # approximation: retransmit fused head directly (optionally with different SNR)
                orig_snr = None
                if hasattr(model, "args") and rt_cfg.get("rt_snr") is not None:
                    orig_snr = getattr(model.args, "channel_snr", None)
                    model.args.channel_snr = float(rt_cfg["rt_snr"])
                _, _, fused_l2 = model(rgb, depth)
                if hasattr(model, "args") and rt_cfg.get("rt_snr") is not None and orig_snr is not None:
                    model.args.channel_snr = orig_snr
                fused_list = []
                for i in range(fused_l.shape[0]):
                    num_total += 1
                    if bool(do_rt_mask[i].item()):
                        num_rt += 1
                        if rt_cfg.get("choose_better", True):
                            p_pre = F.softmax(fused_l[i:i+1], dim=1)
                            p_post = F.softmax(fused_l2[i:i+1], dim=1)
                            u_pre = -torch.sum(p_pre * torch.log(torch.clamp(p_pre, min=1e-12)), dim=1)
                            u_post = -torch.sum(p_post * torch.log(torch.clamp(p_post, min=1e-12)), dim=1)
                            fused_list.append(fused_l[i:i+1] if (u_post > u_pre).item() else fused_l2[i:i+1])
                        else:
                            fused_list.append(fused_l2[i:i+1])
                    else:
                        fused_list.append(fused_l[i:i+1])
                fused_post = torch.cat(fused_list, dim=0)
            else:
                num_total += fused_l.shape[0]

            probs_post = F.softmax(fused_post, dim=1)
            u_post = -torch.sum(probs_post * torch.log(torch.clamp(probs_post, min=1e-12)), dim=1)
            pred_post = torch.argmax(fused_post, dim=1)
            all_uncert_post.append(u_post.detach().cpu())
            all_correct_post.append((pred_post == tgt).detach().cpu())

    u0 = torch.cat(all_uncert_pre, dim=0).numpy()
    u1 = torch.cat(all_uncert_post, dim=0).numpy()
    m0 = torch.cat(all_correct_pre, dim=0).numpy().astype(bool)
    m1 = torch.cat(all_correct_post, dim=0).numpy().astype(bool)

    metrics = {
        "acc_pre": float(np.mean(m0.astype(np.float32))),
        "acc_post": float(np.mean(m1.astype(np.float32))),
        "uncert_pre": float(np.mean(u0)),
        "uncert_post": float(np.mean(u1)),
        "retransmit_ratio": (float(num_rt) / float(num_total)) if num_total > 0 else 0.0,
    }
    return metrics


def main():
    cli_args = parse_args()
    set_seed(1)

    savedir = cli_args.savedir.rstrip("/")
    run_args = load_run_args(savedir)

    # fill minimal defaults to build dataset
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

    test_loader = build_dataloader(run_args, cli_args.data_path, cli_args.batch_sz, cli_args.n_workers)

    # build model
    if cli_args.model_type == "TMC_channel":
        model = TMC_channel(run_args).to(cli_args.device)
    else:
        model = TMC_base_channel(run_args).to(cli_args.device)

    # load checkpoint
    ckpt_path_best = os.path.join(savedir, "model_best.pt")
    ckpt_path = ckpt_path_best if os.path.exists(ckpt_path_best) else os.path.join(savedir, "checkpoint.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location=cli_args.device)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)

    # set fixed SNR if provided/inferred
    fixed_snr = infer_fixed_snr(cli_args, run_args)
    if fixed_snr is not None and hasattr(model, "args"):
        model.args.channel_snr = float(fixed_snr)

    rt_cfg = {
        "enabled": bool(cli_args.retransmit and cli_args.rt_threshold is not None),
        "threshold": cli_args.rt_threshold,
        "trigger": cli_args.rt_trigger,
        "pick_view": cli_args.rt_pick_view,
        "rt_snr": cli_args.rt_snr,
        "ds_discount": bool(cli_args.ds_discount),
        "choose_better": bool(cli_args.choose_better),
    }

    metrics = evaluate_with_retransmission(
        model,
        test_loader,
        cli_args.model_type,
        getattr(run_args, "n_classes", 10),
        cli_args.device,
        rt_cfg,
    )

    if fixed_snr is not None:
        print(f"Eval at fixed SNR: {fixed_snr:.1f} dB")
    print(f"acc (pre-rt):  {metrics['acc_pre']:.4f}")
    print(f"acc (post-rt): {metrics['acc_post']:.4f}")
    print(f"uncertainty (pre-rt):  {metrics['uncert_pre']:.6f}")
    print(f"uncertainty (post-rt): {metrics['uncert_post']:.6f}")
    print(f"retransmit_ratio: {metrics['retransmit_ratio']:.4f}")


if __name__ == "__main__":
    main()


