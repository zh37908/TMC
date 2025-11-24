import argparse
import os
import re
from typing import Dict, Optional

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
from models.TMC import TMC_channel_snr
from utils.utils import set_seed

import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retransmission for TMC_channel_snr (fixed SNR)")
    parser.add_argument("--savedir", type=str, required=True, help="Directory containing checkpoint.pt/model_best.pt and args.pt")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["test", "val", "valid", "train"], default="test")
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snr", type=float, default=None, help="Fixed SNR for evaluation; if None, infer from args.pt or path")
    parser.add_argument("--use_dynamic_snr", action="store_true", help="Enable dynamic SNR per batch (depth/rgb independently sampled in model)")
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--ckpt_file", type=str, default=None, help="Override checkpoint filename inside savedir (e.g., model_best.pt)")

    # Retransmission config (evidential-only)
    parser.add_argument("--retransmit", action="store_true")
    parser.add_argument("--rt_threshold", type=float, default=None)
    parser.add_argument("--rt_trigger", type=str, choices=["low", "high"], default="low")
    parser.add_argument("--reject_score", type=str, choices=["u_evi", "entropy", "margin", "maxprob"], default="u_evi")
    parser.add_argument("--target_coverage", type=float, default=0.95)
    parser.add_argument("--ds_discount", action="store_true")
    parser.add_argument("--choose_better", action="store_true", default=True)
    parser.add_argument("--rt_max_trials", type=int, default=3)
    parser.add_argument("--rt_discard_unqualified", action="store_true")
    parser.add_argument("--report_selective", action="store_true")
    parser.add_argument("--rt_mode", type=str, choices=["ds", "avgfeat"], default="ds",
                        help="Retransmit combine mode: ds=evidential DS combine; avgfeat=average channel features then classify")
    # Noise injection config
    parser.add_argument("--noisy_view", type=str, choices=["none", "rgb", "depth"], default="none",
                        help="Inject Gaussian noise to specified view for testing")
    parser.add_argument("--noise_power", type=float, default=0.0,
                        help="Gaussian noise power (variance). If 0, no noise is added.")
    # Dump uncertainties
    parser.add_argument("--dump_uncert_path", type=str, default=None,
                        help="If set, save per-sample uncertainties to NPZ at this path (contains u_pre, u_post, correctness masks)")
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

    # pick split directory, fallback if not found
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
    m = re.search(r"snr(\d+(?:\.\d+)?)", cli_args.savedir, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


@torch.no_grad()
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
    return 1.0 + (alpha - 1.0) * gamma


@torch.no_grad()
def compute_reject_score_evi(alpha: torch.Tensor, num_classes: int, mode: str) -> torch.Tensor:
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / torch.clamp(S, min=1e-12)
    if mode == "u_evi":
        return float(num_classes) / torch.sum(alpha, dim=1)
    if mode == "entropy":
        return -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
    if mode == "margin":
        top2 = torch.topk(probs, k=2, dim=1).values
        return top2[:, 0] - top2[:, 1]
    if mode == "maxprob":
        return torch.max(probs, dim=1).values
    return float(num_classes) / torch.sum(alpha, dim=1)


@torch.no_grad()
def scan_reject_scores(
    model: torch.nn.Module,
    loader,
    num_classes: int,
    device: str,
    score_mode: str,
    use_dynamic_snr: bool,
    fixed_snr: Optional[float],
    noisy_view: str,
    noise_power: float,
) -> Dict[str, np.ndarray]:
    model.eval()
    scores_depth, scores_rgb = [], []
    for batch in tqdm(loader, total=len(loader)):
        rgb, depth = batch["A"].to(device), batch["B"].to(device)
        # optional input noise injection (before channel)
        if float(noise_power) > 0.0 and noisy_view != "none":
            std_pix = float(noise_power) ** 0.5
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
        depth_a, rgb_a, _ = model(rgb, depth, snr_in)
        s_depth = compute_reject_score_evi(depth_a, num_classes, score_mode)
        s_rgb = compute_reject_score_evi(rgb_a, num_classes, score_mode)
        scores_depth.append(s_depth.detach().cpu())
        scores_rgb.append(s_rgb.detach().cpu())
    return {
        "depth": torch.cat(scores_depth, dim=0).numpy(),
        "rgb": torch.cat(scores_rgb, dim=0).numpy(),
    }


@torch.no_grad()
def evaluate_with_retransmission(
    model: torch.nn.Module,
    loader,
    num_classes: int,
    device: str,
    rt_cfg: Dict,
    use_dynamic_snr: bool,
    fixed_snr: Optional[float],
) -> Dict:
    model.eval()

    all_uncert_pre, all_uncert_post = [], []
    all_correct_pre, all_correct_post = [], []
    all_do_rt_masks = []
    num_rt, num_total = 0, 0

    for batch in tqdm(loader, total=len(loader)):
        rgb, depth, tgt = batch["A"].to(device), batch["B"].to(device), batch["label"].to(device)
        # optional input noise injection (before channel)
        if float(rt_cfg.get("noise_power", 0.0)) > 0.0 and rt_cfg.get("noisy_view", "none") != "none":
            std = float(rt_cfg["noise_power"]) ** 0.5
            if rt_cfg["noisy_view"] == "rgb":
                rgb = rgb + torch.randn_like(rgb) * std
            elif rt_cfg["noisy_view"] == "depth":
                depth = depth + torch.randn_like(depth) * std

        if use_dynamic_snr:
            snr_in = None
        else:
            fs = float(fixed_snr) if fixed_snr is not None else float(getattr(model.args, "channel_snr", 20.0))
            snr_in = torch.full((rgb.shape[0],), fs, dtype=torch.float32, device=rgb.device)
        depth_a, rgb_a, fused_a = model(rgb, depth, snr_in)
        u_fused = float(num_classes) / torch.sum(fused_a, dim=1)
        pred_fused = torch.argmax(fused_a, dim=1)

        all_uncert_pre.append(u_fused.detach().cpu())
        all_correct_pre.append((pred_fused == tgt).detach().cpu())

        # per-view masks
        do_rt_depth_mask, do_rt_rgb_mask = None, None
        do_rt_any = None
        if rt_cfg["enabled"] and (rt_cfg.get("threshold_depth") is not None or rt_cfg.get("threshold_rgb") is not None):
            tau_depth = rt_cfg.get("threshold_depth", rt_cfg.get("threshold"))
            tau_rgb = rt_cfg.get("threshold_rgb", rt_cfg.get("threshold"))
            score_mode = rt_cfg.get("reject_score", "u_evi")
            score_depth = compute_reject_score_evi(depth_a, num_classes, score_mode)
            score_rgb = compute_reject_score_evi(rgb_a, num_classes, score_mode)
            if rt_cfg["trigger"] == "low":
                do_rt_depth_mask = (score_depth < tau_depth) if tau_depth is not None else None
                do_rt_rgb_mask = (score_rgb < tau_rgb) if tau_rgb is not None else None
            else:
                do_rt_depth_mask = (score_depth > tau_depth) if tau_depth is not None else None
                do_rt_rgb_mask = (score_rgb > tau_rgb) if tau_rgb is not None else None
            if do_rt_depth_mask is None and do_rt_rgb_mask is None:
                do_rt_any = None
            elif do_rt_depth_mask is None:
                do_rt_any = do_rt_rgb_mask
            elif do_rt_rgb_mask is None:
                do_rt_any = do_rt_depth_mask
            else:
                do_rt_any = do_rt_depth_mask | do_rt_rgb_mask

        fused_post = fused_a
        if do_rt_any is not None and torch.any(do_rt_any):
            max_trials = int(rt_cfg.get("rt_max_trials", 1))
            discard_fail = bool(rt_cfg.get("rt_discard_unqualified", False))
            rt_mode = str(rt_cfg.get("rt_mode", "ds"))

            # Precompute pre-channel features for avgfeat mode (once per batch)
            if rt_mode == "avgfeat":
                depth_feat_pre = model.depthenc(depth)
                depth_feat_pre = torch.flatten(depth_feat_pre, start_dim=1)
                depth_feat_pre = model._forward_mlp(depth_feat_pre, model.depthchannel_enc)
                rgb_feat_pre = model.rgbenc(rgb)
                rgb_feat_pre = torch.flatten(rgb_feat_pre, start_dim=1)
                rgb_feat_pre = model._forward_mlp(rgb_feat_pre, model.rgbchannel_enc)

            fused_list = []
            for i in range(fused_a.shape[0]):
                num_total += 2
                if do_rt_depth_mask is not None:
                    num_rt += int(bool(do_rt_depth_mask[i].item()))
                if do_rt_rgb_mask is not None:
                    num_rt += int(bool(do_rt_rgb_mask[i].item()))

                def still_triggers(score_tensor: torch.Tensor, tau: float, trigger: str) -> bool:
                    if tau is None:
                        return False
                    return bool((score_tensor < tau).item()) if trigger == "low" else bool((score_tensor > tau).item())

                score_mode = rt_cfg.get("reject_score", "u_evi")

                if rt_mode == "avgfeat":
                    # prepare SNR scalars per sample
                    if use_dynamic_snr:
                        d_scalar = torch.empty((), device=device).uniform_(float(getattr(model.args, 'snr_min', 0.0)), float(getattr(model.args, 'snr_max', 20.0))).item()
                        r_scalar = torch.empty((), device=device).uniform_(float(getattr(model.args, 'snr_min', 0.0)), float(getattr(model.args, 'snr_max', 20.0))).item()
                    else:
                        sfix = float(fixed_snr) if fixed_snr is not None else float(getattr(model.args, 'channel_snr', 20.0))
                        d_scalar = sfix
                        r_scalar = sfix

                    # start with original alphas
                    a_left_cur = depth_a[i:i+1]
                    a_right_cur = rgb_a[i:i+1]

                    # depth avg-feature retransmit
                    if do_rt_depth_mask is not None and bool(do_rt_depth_mask[i].item()):
                        tau_d = rt_cfg.get("threshold_depth", rt_cfg.get("threshold"))
                        trials = 0
                        avg_d = None
                        while trials < max_trials:
                            ch = model.channel(depth_feat_pre[i:i+1], float(d_scalar))
                            avg_d = ch if avg_d is None else (avg_d * trials + ch) / float(trials + 1)
                            trials += 1
                            # classify with SNR embedding policy
                            if getattr(model, 'snr_input_method', 'none') == 'mlp':
                                snr_inp = torch.full((1, 1), float(d_scalar), dtype=torch.float32, device=device)
                                snr_emb = model.snr_embed(snr_inp)
                                fused_feat = model.fuse_depth(torch.cat([avg_d, snr_emb], dim=-1))
                            elif getattr(model, 'snr_input_method', 'none') in ('concat', 'add'):
                                snr_inp = torch.full((1, 1), float(d_scalar), dtype=torch.float32, device=device)
                                snr_emb = model.snr_embed(snr_inp)
                                fused_feat = model._fuse_with_snr(avg_d, snr_emb, model.snr_input_method)
                            else:
                                fused_feat = avg_d
                            logits = model._forward_mlp(fused_feat, model.clf_depth)
                            a_left_cur = F.softplus(logits) + 1.0
                            s_d = compute_reject_score_evi(a_left_cur, num_classes, score_mode)
                            if not still_triggers(s_d, tau_d, rt_cfg["trigger"]):
                                break
                        if discard_fail:
                            s_d_final = compute_reject_score_evi(a_left_cur, num_classes, score_mode)
                            if still_triggers(s_d_final, tau_d, rt_cfg["trigger"]):
                                a_left_cur = torch.ones_like(a_left_cur)

                    # rgb avg-feature retransmit
                    if do_rt_rgb_mask is not None and bool(do_rt_rgb_mask[i].item()):
                        tau_r = rt_cfg.get("threshold_rgb", rt_cfg.get("threshold"))
                        trials = 0
                        avg_r = None
                        while trials < max_trials:
                            ch = model.channel(rgb_feat_pre[i:i+1], float(r_scalar))
                            avg_r = ch if avg_r is None else (avg_r * trials + ch) / float(trials + 1)
                            trials += 1
                            if getattr(model, 'snr_input_method', 'none') == 'mlp':
                                snr_inp = torch.full((1, 1), float(r_scalar), dtype=torch.float32, device=device)
                                snr_emb = model.snr_embed(snr_inp)
                                fused_feat = model.fuse_rgb(torch.cat([avg_r, snr_emb], dim=-1))
                            elif getattr(model, 'snr_input_method', 'none') in ('concat', 'add'):
                                snr_inp = torch.full((1, 1), float(r_scalar), dtype=torch.float32, device=device)
                                snr_emb = model.snr_embed(snr_inp)
                                fused_feat = model._fuse_with_snr(avg_r, snr_emb, model.snr_input_method)
                            else:
                                fused_feat = avg_r
                            logits = model._forward_mlp(fused_feat, model.clf_rgb)
                            a_right_cur = F.softplus(logits) + 1.0
                            s_r = compute_reject_score_evi(a_right_cur, num_classes, score_mode)
                            if not still_triggers(s_r, tau_r, rt_cfg["trigger"]):
                                break
                        if discard_fail:
                            s_r_final = compute_reject_score_evi(a_right_cur, num_classes, score_mode)
                            if still_triggers(s_r_final, tau_r, rt_cfg["trigger"]):
                                a_right_cur = torch.ones_like(a_right_cur)

                else:
                    # original DS-based alpha retransmit
                    a_left_cur = depth_a[i:i+1]
                    a_right_cur = rgb_a[i:i+1]
                    if do_rt_depth_mask is not None and bool(do_rt_depth_mask[i].item()):
                        tau_d = rt_cfg.get("threshold_depth", rt_cfg.get("threshold"))
                        trials = 0
                        while trials < max_trials:
                            a_left_cur = ds_combin_two(a_left_cur, depth_a[i:i+1], num_classes)
                            trials += 1
                            s_d = compute_reject_score_evi(a_left_cur, num_classes, score_mode)
                            if not still_triggers(s_d, tau_d, rt_cfg["trigger"]):
                                break
                        if discard_fail:
                            s_d_final = compute_reject_score_evi(a_left_cur, num_classes, score_mode)
                            if still_triggers(s_d_final, tau_d, rt_cfg["trigger"]):
                                a_left_cur = torch.ones_like(a_left_cur)
                    if do_rt_rgb_mask is not None and bool(do_rt_rgb_mask[i].item()):
                        tau_r = rt_cfg.get("threshold_rgb", rt_cfg.get("threshold"))
                        trials = 0
                        while trials < max_trials:
                            a_right_cur = ds_combin_two(a_right_cur, rgb_a[i:i+1], num_classes)
                            trials += 1
                            s_r = compute_reject_score_evi(a_right_cur, num_classes, score_mode)
                            if not still_triggers(s_r, tau_r, rt_cfg["trigger"]):
                                break
                        if discard_fail:
                            s_r_final = compute_reject_score_evi(a_right_cur, num_classes, score_mode)
                            if still_triggers(s_r_final, tau_r, rt_cfg["trigger"]):
                                a_right_cur = torch.ones_like(a_right_cur)

                # optional reliability discount
                if rt_cfg.get("ds_discount"):
                    u_l = float(num_classes) / torch.sum(a_left_cur, dim=1, keepdim=True)
                    u_r = float(num_classes) / torch.sum(a_right_cur, dim=1, keepdim=True)
                    g_l = 1.0 / (1.0 + u_l)
                    g_r = 1.0 / (1.0 + u_r)
                    a_left_use = apply_discount(a_left_cur, g_l)
                    a_right_use = apply_discount(a_right_cur, g_r)
                else:
                    a_left_use = a_left_cur
                    a_right_use = a_right_cur

                fused_i = ds_combin_two(a_left_use, a_right_use, num_classes)

                if rt_cfg.get("choose_better", True):
                    u_i_pre = float(num_classes) / torch.sum(fused_a[i:i+1], dim=1)
                    u_i_post = float(num_classes) / torch.sum(fused_i, dim=1)
                    if (u_i_post > u_i_pre).item():
                        fused_i = fused_a[i:i+1]
                fused_list.append(fused_i)
            fused_post = torch.cat(fused_list, dim=0)
        else:
            num_total += 2 * fused_a.shape[0]

        pred_post = torch.argmax(fused_post, dim=1)
        u_post = float(num_classes) / torch.sum(fused_post, dim=1)
        all_uncert_post.append(u_post.detach().cpu())
        all_correct_post.append((pred_post == tgt).detach().cpu())
        if do_rt_any is None:
            all_do_rt_masks.append(torch.zeros_like(u_fused, dtype=torch.bool).cpu())
        else:
            all_do_rt_masks.append(do_rt_any.detach().cpu())

    u0 = torch.cat(all_uncert_pre, dim=0).numpy()
    u1 = torch.cat(all_uncert_post, dim=0).numpy()
    m0 = torch.cat(all_correct_pre, dim=0).numpy().astype(bool)
    m1 = torch.cat(all_correct_post, dim=0).numpy().astype(bool)

    do_rt = torch.cat(all_do_rt_masks, dim=0).numpy().astype(bool)
    coverage = float(1.0 - np.mean(do_rt.astype(np.float32))) if do_rt.size > 0 else 1.0
    sel_risk = None
    if np.any(~do_rt):
        sel_risk = float(1.0 - np.mean(m0[~do_rt].astype(np.float32)))

    metrics = {
        "acc_pre": float(np.mean(m0.astype(np.float32))),
        "acc_post": float(np.mean(m1.astype(np.float32))),
        "uncert_pre": float(np.mean(u0)),
        "uncert_post": float(np.mean(u1)),
        "retransmit_ratio": (float(num_rt) / float(num_total)) if num_total > 0 else 0.0,
        "coverage": coverage,
        "selective_risk": sel_risk,
        "uncert_post_all": u1,
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

    test_loader = build_dataloader(run_args, cli_args.data_path, cli_args.batch_sz, cli_args.n_workers, cli_args.split)

    # build model
    model = TMC_channel_snr(run_args).to(cli_args.device)

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
            preferred = [f for f in pt_files if "best" in f.lower() or "checkpoint" in f.lower()]
            pick = preferred[0] if len(preferred) > 0 else (pt_files[0] if len(pt_files) > 0 else None)
            if pick is not None:
                ckpt_path = os.path.join(savedir, pick)
    assert ckpt_path is not None and os.path.exists(ckpt_path), f"Checkpoint not found under savedir. Tried default names and scan: {savedir}"
    checkpoint = torch.load(ckpt_path, map_location=cli_args.device)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)

    # resolve SNR mode: fixed or dynamic
    fixed_snr = None
    if not cli_args.use_dynamic_snr:
        fixed_snr = infer_fixed_snr(cli_args, run_args)
        if fixed_snr is not None and hasattr(model, "args"):
            model.args.channel_snr = float(fixed_snr)
    else:
        # pass dynamic range to model args (used when snr=None in forward)
        if hasattr(model, "args"):
            model.args.snr_min = float(cli_args.snr_min)
            model.args.snr_max = float(cli_args.snr_max)

    # per-view auto threshold
    auto_tau = {"depth": None, "rgb": None}
    if cli_args.target_coverage is not None and cli_args.rt_threshold is None:
        scores = scan_reject_scores(
            model,
            test_loader,
            getattr(run_args, "n_classes", 10),
            cli_args.device,
            cli_args.reject_score,
            bool(cli_args.use_dynamic_snr),
            fixed_snr,
            str(getattr(cli_args, "noisy_view", "none")),
            float(getattr(cli_args, "noise_power", 0.0)),
        )
        cov = float(cli_args.target_coverage)
        cov = min(max(cov, 0.0), 1.0)
        q = cov if cli_args.rt_trigger == "high" else (1.0 - cov)
        auto_tau = {
            "depth": float(np.quantile(scores["depth"], q)) if scores["depth"].size > 0 else None,
            "rgb": float(np.quantile(scores["rgb"], q)) if scores["rgb"].size > 0 else None,
        }

    rt_cfg = {
        "enabled": bool(cli_args.retransmit and (cli_args.rt_threshold is not None or auto_tau["depth"] is not None or auto_tau["rgb"] is not None)),
        "threshold": (cli_args.rt_threshold if cli_args.rt_threshold is not None else None),
        "threshold_depth": (cli_args.rt_threshold if cli_args.rt_threshold is not None else auto_tau["depth"]),
        "threshold_rgb": (cli_args.rt_threshold if cli_args.rt_threshold is not None else auto_tau["rgb"]),
        "trigger": cli_args.rt_trigger,
        "reject_score": cli_args.reject_score,
        "ds_discount": bool(cli_args.ds_discount),
        "choose_better": bool(cli_args.choose_better),
        "rt_max_trials": int(cli_args.rt_max_trials),
        "rt_discard_unqualified": bool(cli_args.rt_discard_unqualified),
        "rt_mode": str(getattr(cli_args, "rt_mode", "ds")),
        # pass noise settings into eval flow
        "noisy_view": str(getattr(cli_args, "noisy_view", "none")),
        "noise_power": float(getattr(cli_args, "noise_power", 0.0)),
    }

    metrics = evaluate_with_retransmission(
        model,
        test_loader,
        getattr(run_args, "n_classes", 10),
        cli_args.device,
        rt_cfg,
        bool(cli_args.use_dynamic_snr),
        fixed_snr,
    )

    # Optionally dump per-sample uncertainties (pre and post)
    if getattr(cli_args, "dump_uncert_path", None):
        # We don't have arrays returned, so re-scan quickly to collect uncertainties only
        # To avoid heavy recomputation, mirror core logic to capture u_pre/u_post
        model.eval()
        u_pre_all, u_post_all = [], []
        correct_pre_all, correct_post_all = [], []
        do_rt_all = []
        for batch in test_loader:
            rgb, depth, tgt = batch["A"].to(cli_args.device), batch["B"].to(cli_args.device), batch["label"].to(cli_args.device)
            if float(rt_cfg.get("noise_power", 0.0)) > 0.0 and rt_cfg.get("noisy_view", "none") != "none":
                std = float(rt_cfg["noise_power"]) ** 0.5
                if rt_cfg["noisy_view"] == "rgb":
                    rgb = rgb + torch.randn_like(rgb) * std
                elif rt_cfg["noisy_view"] == "depth":
                    depth = depth + torch.randn_like(depth) * std
            if bool(cli_args.use_dynamic_snr):
                snr_in = None
            else:
                fs = float(fixed_snr) if fixed_snr is not None else float(getattr(model.args, "channel_snr", 20.0))
                snr_in = torch.full((rgb.shape[0],), fs, dtype=torch.float32, device=rgb.device)
            depth_a, rgb_a, fused_a = model(rgb, depth, snr_in)
            u_fused = float(getattr(run_args, "n_classes", 10)) / torch.sum(fused_a, dim=1)
            pred_fused = torch.argmax(fused_a, dim=1)
            # per-view masks
            do_rt_depth_mask, do_rt_rgb_mask = None, None
            do_rt_any = None
            tau_depth = rt_cfg.get("threshold_depth", rt_cfg.get("threshold"))
            tau_rgb = rt_cfg.get("threshold_rgb", rt_cfg.get("threshold"))
            score_mode = rt_cfg.get("reject_score", "u_evi")
            if rt_cfg["enabled"] and (tau_depth is not None or tau_rgb is not None):
                score_depth = compute_reject_score_evi(depth_a, getattr(run_args, "n_classes", 10), score_mode)
                score_rgb = compute_reject_score_evi(rgb_a, getattr(run_args, "n_classes", 10), score_mode)
                if rt_cfg["trigger"] == "low":
                    do_rt_depth_mask = (score_depth < tau_depth) if tau_depth is not None else None
                    do_rt_rgb_mask = (score_rgb < tau_rgb) if tau_rgb is not None else None
                else:
                    do_rt_depth_mask = (score_depth > tau_depth) if tau_depth is not None else None
                    do_rt_rgb_mask = (score_rgb > tau_rgb) if tau_rgb is not None else None
                if do_rt_depth_mask is None and do_rt_rgb_mask is None:
                    do_rt_any = None
                elif do_rt_depth_mask is None:
                    do_rt_any = do_rt_rgb_mask
                elif do_rt_rgb_mask is None:
                    do_rt_any = do_rt_depth_mask
                else:
                    do_rt_any = do_rt_depth_mask | do_rt_rgb_mask

            # record pre
            u_pre_all.append(u_fused.detach().cpu())
            correct_pre_all.append((pred_fused == tgt).detach().cpu())
            do_rt_all.append(torch.zeros_like(u_fused, dtype=torch.bool).cpu() if do_rt_any is None else do_rt_any.detach().cpu())

            # post (reuse logic from evaluate_with_retransmission but simplified: DS loop only)
            fused_post = fused_a
            if do_rt_any is not None and torch.any(do_rt_any):
                fused_list = []
                num_classes = int(getattr(run_args, "n_classes", 10))
                for i in range(fused_a.shape[0]):
                    a_left_cur = depth_a[i:i+1]
                    a_right_cur = rgb_a[i:i+1]
                    if do_rt_depth_mask is not None and bool(do_rt_depth_mask[i].item()):
                        trials = 0
                        tau_d = tau_depth
                        while trials < int(rt_cfg.get("rt_max_trials", 1)):
                            a_left_cur = ds_combin_two(a_left_cur, depth_a[i:i+1], num_classes)
                            trials += 1
                            s_d = compute_reject_score_evi(a_left_cur, num_classes, score_mode)
                            if not ((s_d < tau_d) if rt_cfg["trigger"] == "low" else (s_d > tau_d)):
                                break
                    if do_rt_rgb_mask is not None and bool(do_rt_rgb_mask[i].item()):
                        trials = 0
                        tau_r = tau_rgb
                        while trials < int(rt_cfg.get("rt_max_trials", 1)):
                            a_right_cur = ds_combin_two(a_right_cur, rgb_a[i:i+1], num_classes)
                            trials += 1
                            s_r = compute_reject_score_evi(a_right_cur, num_classes, score_mode)
                            if not ((s_r < tau_r) if rt_cfg["trigger"] == "low" else (s_r > tau_r)):
                                break
                    fused_i = ds_combin_two(a_left_cur, a_right_cur, num_classes)
                    if rt_cfg.get("choose_better", True):
                        u_i_pre = float(num_classes) / torch.sum(fused_a[i:i+1], dim=1)
                        u_i_post = float(num_classes) / torch.sum(fused_i, dim=1)
                        if (u_i_post > u_i_pre).item():
                            fused_i = fused_a[i:i+1]
                    fused_list.append(fused_i)
                fused_post = torch.cat(fused_list, dim=0)
            u_post = float(getattr(run_args, "n_classes", 10)) / torch.sum(fused_post, dim=1)
            pred_post = torch.argmax(fused_post, dim=1)
            u_post_all.append(u_post.detach().cpu())
            correct_post_all.append((pred_post == tgt).detach().cpu())

        u_pre_np = torch.cat(u_pre_all, dim=0).numpy()
        u_post_np = torch.cat(u_post_all, dim=0).numpy()
        corr_pre_np = torch.cat(correct_pre_all, dim=0).numpy().astype(bool)
        corr_post_np = torch.cat(correct_post_all, dim=0).numpy().astype(bool)
        do_rt_np = torch.cat(do_rt_all, dim=0).numpy().astype(bool)
        os.makedirs(os.path.dirname(cli_args.dump_uncert_path), exist_ok=True)
        np.savez(cli_args.dump_uncert_path, u_pre=u_pre_np, u_post=u_post_np, correct_pre=corr_pre_np, correct_post=corr_post_np, do_rt=do_rt_np)

    # Save density of final fused uncertainty
    if cli_args.save_uncert_density and metrics.get("uncert_post_all") is not None:
        import matplotlib.pyplot as plt
        u_all = metrics["uncert_post_all"]
        fig = plt.figure(figsize=(6, 4))
        try:
            sns.kdeplot(u_all, fill=True, color="#1f77b4", alpha=0.6)
        except Exception:
            # Fallback to histogram-based density
            plt.hist(u_all, bins=60, density=True, color="#1f77b4", alpha=0.6)
        plt.xlabel("Fused uncertainty")
        plt.ylabel("Density")
        plt.title("Uncertainty density (post retransmission)")
        plt.grid(True, alpha=0.3)
        out_path = cli_args.uncert_density_path or os.path.join(savedir, "uncertainty_density.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved uncertainty density figure to: {out_path}")

    if fixed_snr is not None:
        print(f"Eval at fixed SNR: {fixed_snr:.1f} dB")
    else:
        print(f"Eval under dynamic SNR range: [{float(cli_args.snr_min):.1f}, {float(cli_args.snr_max):.1f}] dB")
    if cli_args.target_coverage is not None and cli_args.rt_threshold is None and (rt_cfg.get("threshold_depth") is not None or rt_cfg.get("threshold_rgb") is not None):
        td = rt_cfg.get("threshold_depth")
        tr = rt_cfg.get("threshold_rgb")
        if td is not None:
            print(f"Auto-selected threshold depth (score={cli_args.reject_score}, trigger={cli_args.rt_trigger}) -> tau_d={td:.6f}")
        if tr is not None:
            print(f"Auto-selected threshold rgb   (score={cli_args.reject_score}, trigger={cli_args.rt_trigger}) -> tau_r={tr:.6f}")
    print(f"acc (pre-rt):  {metrics['acc_pre']:.4f}")
    print(f"acc (post-rt): {metrics['acc_post']:.4f}")
    print(f"uncertainty (pre-rt):  {metrics['uncert_pre']:.6f}")
    print(f"uncertainty (post-rt): {metrics['uncert_post']:.6f}")
    print(f"retransmit_ratio: {metrics['retransmit_ratio']:.4f}")
    if cli_args.report_selective:
        print(f"coverage (as non-retransmit fraction): {metrics['coverage']:.4f}")
        if metrics["selective_risk"] is not None:
            print(f"selective_risk (on accepted, pre-rt): {metrics['selective_risk']:.4f}")


if __name__ == "__main__":
    main()


