import argparse
import os
from typing import Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description="Evaluate TMC on SUNRGBD with optional retransmission and source noise")
    parser.add_argument("--savedir", type=str, required=True, help="Directory containing checkpoint.pt/model_best.pt and args.pt")
    parser.add_argument("--data_path", type=str, required=True, help="Root directory of SUNRGBD (expects split subfolders)")
    parser.add_argument("--split", type=str, choices=["test", "val", "valid", "train"], default="test")
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_file", type=str, default=None, help="Override checkpoint filename inside savedir (e.g., model_best.pt)")

    # Source noise injection
    parser.add_argument("--noisy_view", type=str, choices=["none", "rgb", "depth"], default="none",
                        help="Inject Gaussian noise to specified view for testing")
    parser.add_argument("--noise_power", type=float, default=0.0,
                        help="Gaussian noise power (variance). If 0, no noise is added.")
    parser.add_argument("--noise_stage", type=str, choices=["normalized", "pixel", "dataloader"], default="normalized",
                        help="Where to inject noise: normalized (tensor space), pixel (denorm->add->renorm), or dataloader (inject during load)")
    # Channel SNR control at eval for TMC_channel_snr
    parser.add_argument("--channel_snr_mode", type=str, choices=["auto", "fixed", "dynamic"], default="auto",
                        help="auto: use channel_snr from checkpoint args if available; fixed: use --channel_snr_db; dynamic: sample per-batch SNR in [snr_min, snr_max]")
    parser.add_argument("--channel_snr_db", type=float, default=10.0,
                        help="SNR (dB) used at eval when --channel_snr_mode=fixed")
    parser.add_argument("--snr_min", type=float, default=0.0, help="Min SNR (dB) when --channel_snr_mode=dynamic")
    parser.add_argument("--snr_max", type=float, default=20.0, help="Max SNR (dB) when --channel_snr_mode=dynamic")
    # Retransmission config (DS-based)
    parser.add_argument("--retransmit", action="store_true", help="Enable retransmission per-view based on uncertainty score")
    parser.add_argument("--rt_threshold", type=float, default=None, help="Global threshold for retransmission trigger")
    parser.add_argument("--rt_trigger", type=str, choices=["low", "high"], default="low",
                        help="Trigger when score is low or high relative to threshold")
    parser.add_argument("--reject_score", type=str, choices=["u_evi", "entropy", "margin", "maxprob"], default="u_evi",
                        help="Score function for retransmission decision")
    parser.add_argument("--target_coverage", type=float, default=0.9,
                        help="If set and no --rt_threshold, auto-choose per-view thresholds to reach this non-retransmit coverage")
    parser.add_argument("--ds_discount", action="store_true", help="Apply reliability discount before DS fuse")
    parser.add_argument("--choose_better", action="store_true", default=True, help="Keep original fused if retransmit worsens uncertainty")
    parser.add_argument("--rt_max_trials", type=int, default=3, help="Max DS combination trials per triggered view")
    parser.add_argument("--rt_discard_unqualified", action="store_true",
                        help="If after max trials still triggered, set that view to vacuous evidence")
    parser.add_argument("--report_selective", action="store_true", help="Report coverage/selective-risk statistics")
    parser.add_argument("--rt_mode", type=str, choices=["ds", "avgfeat"], default="ds",
                        help="Retransmit combine mode: ds=evidential DS combine; avgfeat=average channel features then classify")
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
    # SUNRGBD normalization stats (follow train_TMC_sunrgbd.py)
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
                A = tf_totensor(A)
                B = tf_totensor(B)
                if self.power > 0.0:
                    std_pix = self.power ** 0.5
                    if self.view == 'rgb':
                        A = torch.clamp(A + torch.randn_like(A) * std_pix, 0.0, 1.0)
                    elif self.view == 'depth':
                        B = torch.clamp(B + torch.randn_like(B) * std_pix, 0.0, 1.0)
                A = (A - self.mean) / self.std
                B = (B - self.mean) / self.std
                out = {'A': A, 'B': B, 'img_name': img_name}
                if label is not None:
                    out['label'] = label
                return out

        ds = NoisyAlignedConcDataset(args, picked_dir, str(noisy_view), float(noise_power), mean, std)
    else:
        ds = AlignedConcDataset(args, data_dir=picked_dir, transform=transforms.Compose(val_transforms))

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_sz,
        shuffle=False,
        num_workers=n_workers,
    )


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
    is_channel_arch: bool,
    eval_snr_db: float,
    is_dynamic: bool = False,
) -> dict:
    model.eval()
    scores_depth, scores_rgb = [], []
    # noise stage shared with evaluate
    stage = getattr(evaluate_tmc, "_noise_stage", "normalized")
    for batch in tqdm(loader, total=len(loader)):
        rgb, depth = batch["A"].to(device), batch["B"].to(device)
        # optional source noise (not needed if injected at dataloader)
        # Reuse the same stats as evaluate_tmc
        std_pix = 0.0
        mean_vec = torch.tensor([0.6983, 0.3918, 0.4474], dtype=rgb.dtype, device=rgb.device).view(1, -1, 1, 1)
        std_vec = torch.tensor([0.1648, 0.1359, 0.1644], dtype=rgb.dtype, device=rgb.device).view(1, -1, 1, 1)
        # here we do not inject additional noise; dataset already contains noise for dataloader stage
        if is_channel_arch:
            if is_dynamic:
                snr_in = None
            else:
                snr_in = torch.full((rgb.shape[0],), float(eval_snr_db), dtype=torch.float32, device=rgb.device)
            depth_a, rgb_a, _ = model(rgb, depth, snr_in)
        else:
            depth_a, rgb_a, _ = model(rgb, depth)
        s_depth = compute_reject_score_evi(depth_a, num_classes, score_mode)
        s_rgb = compute_reject_score_evi(rgb_a, num_classes, score_mode)
        scores_depth.append(s_depth.detach().cpu())
        scores_rgb.append(s_rgb.detach().cpu())
    return {
        "depth": torch.cat(scores_depth, dim=0).numpy(),
        "rgb": torch.cat(scores_rgb, dim=0).numpy(),
    }


@torch.no_grad()
def evaluate_tmc(
    model: torch.nn.Module,
    loader,
    device: str,
    noisy_view: str,
    noise_power: float,
    num_classes: int,
    is_channel_arch: bool,
    eval_snr_db: Optional[float],
    is_dynamic: bool = False,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(loader, total=len(loader)):
        rgb, depth, tgt = batch["A"].to(device), batch["B"].to(device), batch["label"].to(device)

        stage = getattr(evaluate_tmc, "_noise_stage", "normalized")
        if stage != "dataloader" and float(noise_power) > 0.0 and noisy_view != "none":
            std_pix = float(noise_power) ** 0.5
            mean_vec = torch.tensor([0.6983, 0.3918, 0.4474], dtype=rgb.dtype, device=rgb.device).view(1, -1, 1, 1)
            std_vec = torch.tensor([0.1648, 0.1359, 0.1644], dtype=rgb.dtype, device=rgb.device).view(1, -1, 1, 1)
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

        if is_channel_arch:
            # pass chosen evaluation SNR (dB) or dynamic None
            if is_dynamic:
                snr_in = None
            else:
                snr_db = float(eval_snr_db) if eval_snr_db is not None else 10.0
                snr_in = torch.full((rgb.shape[0],), snr_db, dtype=torch.float32, device=rgb.device)
            depth_a, rgb_a, fused_a = model(rgb, depth, snr_in)
        else:
            depth_a, rgb_a, fused_a = model(rgb, depth)
        pred = torch.argmax(fused_a, dim=1)
        correct += int((pred == tgt).sum().item())
        total += int(tgt.numel())

    return float(correct) / float(total) if total > 0 else 0.0


@torch.no_grad()
def evaluate_with_retransmission(
    model: torch.nn.Module,
    loader,
    num_classes: int,
    device: str,
    rt_cfg: dict,
    is_channel_arch: bool,
    eval_snr_db: Optional[float],
    is_dynamic: bool = False,
) -> dict:
    model.eval()
    all_uncert_pre, all_uncert_post = [], []
    all_correct_pre, all_correct_post = [], []
    all_do_rt_masks = []
    num_rt, num_total = 0, 0

    for batch in tqdm(loader, total=len(loader)):
        rgb, depth, tgt = batch["A"].to(device), batch["B"].to(device), batch["label"].to(device)

        # Source noise: handled by dataset if stage=dataloader; for other stages evaluate_tmc already injects,
        # but here we keep inputs as provided by loader to be consistent with threshold scan.

        if is_channel_arch:
            if is_dynamic:
                snr_in = None
            else:
                snr_db = float(eval_snr_db) if eval_snr_db is not None else 10.0
                snr_in = torch.full((rgb.shape[0],), snr_db, dtype=torch.float32, device=rgb.device)
            depth_a, rgb_a, fused_a = model(rgb, depth, snr_in)
        else:
            depth_a, rgb_a, fused_a = model(rgb, depth)

        u_fused = float(num_classes) / torch.sum(fused_a, dim=1)
        pred_fused = torch.argmax(fused_a, dim=1)
        all_uncert_pre.append(u_fused.detach().cpu())
        all_correct_pre.append((pred_fused == tgt).detach().cpu())

        # Determine retransmission masks
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
            fused_list = []
            rt_mode = str(rt_cfg.get("rt_mode", "ds"))
            # Precompute pre-channel features for avgfeat mode (only for channel arch)
            if rt_mode == "avgfeat" and is_channel_arch:
                depth_feat_pre = model.depthenc(depth)
                depth_feat_pre = torch.flatten(depth_feat_pre, start_dim=1)
                depth_feat_pre = model._forward_mlp(depth_feat_pre, model.depthchannel_enc)
                rgb_feat_pre = model.rgbenc(rgb)
                rgb_feat_pre = torch.flatten(rgb_feat_pre, start_dim=1)
                rgb_feat_pre = model._forward_mlp(rgb_feat_pre, model.rgbchannel_enc)
            for i in range(fused_a.shape[0]):
                num_total += 2
                if do_rt_depth_mask is not None:
                    num_rt += int(bool(do_rt_depth_mask[i].item()))
                if do_rt_rgb_mask is not None:
                    num_rt += int(bool(do_rt_rgb_mask[i].item()))
                score_mode = rt_cfg.get("reject_score", "u_evi")
                a_left_cur = depth_a[i:i+1]
                a_right_cur = rgb_a[i:i+1]
                if rt_mode == "avgfeat" and is_channel_arch:
                    # evaluation snr (dynamic: per-trial sample in [snr_min, snr_max])
                    snr_min = float(getattr(model.args, "snr_min", 0.0))
                    snr_max = float(getattr(model.args, "snr_max", 20.0))
                    snr_scalar = float(eval_snr_db) if (eval_snr_db is not None and not is_dynamic) else None
                    # depth avg-feature retransmit
                    if do_rt_depth_mask is not None and bool(do_rt_depth_mask[i].item()):
                        tau_d = rt_cfg.get("threshold_depth", rt_cfg.get("threshold"))
                        trials = 0
                        avg_d = None
                        while trials < max_trials:
                            cur_snr = float(torch.empty((), device=device).uniform_(snr_min, snr_max).item()) if (is_dynamic or snr_scalar is None) else float(snr_scalar)
                            ch = model.channel(depth_feat_pre[i:i+1], cur_snr)
                            avg_d = ch if avg_d is None else (avg_d * trials + ch) / float(trials + 1)
                            trials += 1
                            # classify with SNR embedding policy
                            if getattr(model, 'snr_input_method', 'none') == 'mlp':
                                snr_inp = torch.full((1, 1), cur_snr, dtype=torch.float32, device=device)
                                snr_emb = model.snr_embed(snr_inp)
                                fused_feat = model.fuse_depth(torch.cat([avg_d, snr_emb], dim=-1))
                            elif getattr(model, 'snr_input_method', 'none') in ('concat', 'add'):
                                snr_inp = torch.full((1, 1), cur_snr, dtype=torch.float32, device=device)
                                snr_emb = model.snr_embed(snr_inp)
                                fused_feat = model._fuse_with_snr(avg_d, snr_emb, model.snr_input_method)
                            else:
                                fused_feat = avg_d
                            logits = model._forward_mlp(fused_feat, model.clf_depth)
                            a_left_cur = F.softplus(logits) + 1.0
                            s_d = compute_reject_score_evi(a_left_cur, num_classes, score_mode)
                            if not ((s_d < tau_d) if rt_cfg["trigger"] == "low" else (s_d > tau_d)):
                                break
                        if discard_fail:
                            s_d_final = compute_reject_score_evi(a_left_cur, num_classes, score_mode)
                            if ((s_d_final < tau_d) if rt_cfg["trigger"] == "low" else (s_d_final > tau_d)):
                                a_left_cur = torch.ones_like(a_left_cur)
                    # rgb avg-feature retransmit
                    if do_rt_rgb_mask is not None and bool(do_rt_rgb_mask[i].item()):
                        tau_r = rt_cfg.get("threshold_rgb", rt_cfg.get("threshold"))
                        trials = 0
                        avg_r = None
                        while trials < max_trials:
                            cur_snr = float(torch.empty((), device=device).uniform_(snr_min, snr_max).item()) if (is_dynamic or snr_scalar is None) else float(snr_scalar)
                            ch = model.channel(rgb_feat_pre[i:i+1], cur_snr)
                            avg_r = ch if avg_r is None else (avg_r * trials + ch) / float(trials + 1)
                            trials += 1
                            if getattr(model, 'snr_input_method', 'none') == 'mlp':
                                snr_inp = torch.full((1, 1), cur_snr, dtype=torch.float32, device=device)
                                snr_emb = model.snr_embed(snr_inp)
                                fused_feat = model.fuse_rgb(torch.cat([avg_r, snr_emb], dim=-1))
                            elif getattr(model, 'snr_input_method', 'none') in ('concat', 'add'):
                                snr_inp = torch.full((1, 1), cur_snr, dtype=torch.float32, device=device)
                                snr_emb = model.snr_embed(snr_inp)
                                fused_feat = model._fuse_with_snr(avg_r, snr_emb, model.snr_input_method)
                            else:
                                fused_feat = avg_r
                            logits = model._forward_mlp(fused_feat, model.clf_rgb)
                            a_right_cur = F.softplus(logits) + 1.0
                            s_r = compute_reject_score_evi(a_right_cur, num_classes, score_mode)
                            if not ((s_r < tau_r) if rt_cfg["trigger"] == "low" else (s_r > tau_r)):
                                break
                        if discard_fail:
                            s_r_final = compute_reject_score_evi(a_right_cur, num_classes, score_mode)
                            if ((s_r_final < tau_r) if rt_cfg["trigger"] == "low" else (s_r_final > tau_r)):
                                a_right_cur = torch.ones_like(a_right_cur)
                else:
                    # DS-based retransmit (original)
                    if do_rt_depth_mask is not None and bool(do_rt_depth_mask[i].item()):
                        tau_d = rt_cfg.get("threshold_depth", rt_cfg.get("threshold"))
                        trials = 0
                        while trials < max_trials:
                            a_left_cur = ds_combin_two(a_left_cur, depth_a[i:i+1], num_classes)
                            trials += 1
                            s_d = compute_reject_score_evi(a_left_cur, num_classes, score_mode)
                            if not ((s_d < tau_d) if rt_cfg["trigger"] == "low" else (s_d > tau_d)):
                                break
                        if discard_fail:
                            s_d_final = compute_reject_score_evi(a_left_cur, num_classes, score_mode)
                            if ((s_d_final < tau_d) if rt_cfg["trigger"] == "low" else (s_d_final > tau_d)):
                                a_left_cur = torch.ones_like(a_left_cur)
                    if do_rt_rgb_mask is not None and bool(do_rt_rgb_mask[i].item()):
                        tau_r = rt_cfg.get("threshold_rgb", rt_cfg.get("threshold"))
                        trials = 0
                        while trials < max_trials:
                            a_right_cur = ds_combin_two(a_right_cur, rgb_a[i:i+1], num_classes)
                            trials += 1
                            s_r = compute_reject_score_evi(a_right_cur, num_classes, score_mode)
                            if not ((s_r < tau_r) if rt_cfg["trigger"] == "low" else (s_r > tau_r)):
                                break
                        if discard_fail:
                            s_r_final = compute_reject_score_evi(a_right_cur, num_classes, score_mode)
                            if ((s_r_final < tau_r) if rt_cfg["trigger"] == "low" else (s_r_final > tau_r)):
                                a_right_cur = torch.ones_like(a_right_cur)

                # Optional discount then fuse
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
    }
    return metrics


def main():
    cli_args = parse_args()
    set_seed(1)

    savedir = cli_args.savedir.rstrip("/")
    run_args = load_run_args(savedir)

    # SUNRGBD defaults
    defaults = {
        "LOAD_SIZE": 256,
        "FINE_SIZE": 224,
        "img_embed_pool_type": "avg",
        "num_image_embeds": 1,
        "img_hidden_sz": 512,
        "hidden": [512],
        "dropout": 0.1,
        "n_classes": 19,
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

    if not hasattr(run_args, "channel_type"):
        setattr(run_args, "channel_type", "awgn")
    if not hasattr(run_args, "snr_input_method"):
        setattr(run_args, "snr_input_method", "none")

    state_keys = list(state.keys())
    is_channel_arch = any(k.startswith("depthchannel_enc.") or k.startswith("rgbchannel_enc.") for k in state_keys)
    if is_channel_arch:
        model = TMC_channel_snr(run_args).to(cli_args.device)
    else:
        model = TMC(run_args).to(cli_args.device)

    model.load_state_dict(state, strict=False)

    setattr(evaluate_tmc, "_noise_stage", str(getattr(cli_args, "noise_stage", "normalized")))

    # decide evaluation SNR
    if is_channel_arch:
        mode = str(getattr(cli_args, "channel_snr_mode", "auto"))
        if mode == "fixed":
            eval_snr_db = float(getattr(cli_args, "channel_snr_db", 10.0))
            is_dynamic = False
        elif mode == "dynamic":
            eval_snr_db = None
            is_dynamic = True
            # Pass dynamic range to model args (used when snr_in=None)
            setattr(run_args, "snr_min", float(getattr(cli_args, "snr_min", 0.0)))
            setattr(run_args, "snr_max", float(getattr(cli_args, "snr_max", 20.0)))
            if hasattr(model, "args"):
                model.args.snr_min = float(getattr(cli_args, "snr_min", 0.0))
                model.args.snr_max = float(getattr(cli_args, "snr_max", 20.0))
        else:
            eval_snr_db = float(getattr(run_args, "channel_snr", 10.0))
            is_dynamic = False
    else:
        eval_snr_db = None
        is_dynamic = False

    # Auto-select per-view thresholds for target coverage if requested
    auto_tau = {"depth": None, "rgb": None}
    if cli_args.target_coverage is not None and cli_args.rt_threshold is None:
        scores = scan_reject_scores(
            model,
            test_loader,
            int(getattr(run_args, "n_classes", 19)),
            cli_args.device,
            cli_args.reject_score,
            bool(is_channel_arch),
            float(eval_snr_db) if (eval_snr_db is not None) else float(getattr(cli_args, "channel_snr_db", 10.0)),
            bool(is_dynamic),
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
    }

    metrics = evaluate_with_retransmission(
        model,
        test_loader,
        int(getattr(run_args, "n_classes", 19)),
        cli_args.device,
        rt_cfg,
        bool(is_channel_arch),
        eval_snr_db,
        bool(is_dynamic),
    )

    if cli_args.target_coverage is not None and cli_args.rt_threshold is None and (rt_cfg.get("threshold_depth") is not None or rt_cfg.get("threshold_rgb") is not None):
        td = rt_cfg.get("threshold_depth")
        tr = rt_cfg.get("threshold_rgb")
        if td is not None:
            print(f"Auto-selected threshold depth (score={cli_args.reject_score}, trigger={cli_args.rt_trigger}) -> tau_d={td:.6f}")
        if tr is not None:
            print(f"Auto-selected threshold rgb   (score={cli_args.reject_score}, trigger={cli_args.rt_trigger}) -> tau_r={tr:.6f}")
    print(f"acc (pre-rt):  {metrics['acc_pre']:.4f}")
    print(f"acc (post-rt): {metrics['acc_post']:.4f}")
    print(f"retransmit_ratio: {metrics['retransmit_ratio']:.4f}")
    if cli_args.report_selective:
        print(f"coverage (as non-retransmit fraction): {metrics['coverage']:.4f}")
        if metrics["selective_risk"] is not None:
            print(f"selective_risk (on accepted, pre-rt): {metrics['selective_risk']:.4f}")


if __name__ == "__main__":
    main()


