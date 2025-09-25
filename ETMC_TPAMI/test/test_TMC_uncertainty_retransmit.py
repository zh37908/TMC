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
    parser.add_argument("--split", type=str, choices=["test", "val", "valid", "train"], default="test", help="Dataset split subfolder to use under data_path")
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snr", type=float, default=None, help="Fixed SNR for evaluation; if None, infer from args.pt or path")
    parser.add_argument("--ckpt_file", type=str, default=None, help="Override checkpoint filename inside savedir (e.g., model_best.pt)")

    # Retransmission config
    parser.add_argument("--retransmit", action="store_true", help="Enable retransmission (single retry per sample)")
    parser.add_argument("--rt_threshold", type=float, default=None, help="Threshold on fused uncertainty to trigger retransmission")
    parser.add_argument("--rt_trigger", type=str, choices=["low", "high"], default="low", help="Trigger when fused uncertainty is below(low)/above(high) threshold")
    parser.add_argument("--rt_pick_view", type=str, choices=["min", "max"], default="max", help="Pick view with min/max per-view uncertainty for retransmission")
    parser.add_argument("--rt_snr", type=float, default=None, help="SNR(dB) used only for retransmission forward; default: reuse --snr/inferred")
    parser.add_argument("--rt_max_trials", type=int, default=3, help="Max per-view retransmission DS-combine trials without extra forward (evidential)")
    parser.add_argument("--rt_discard_unqualified", action="store_true", help="Discard a view if it still fails threshold after max trials (evidential)")

    # Fusion robustness options (evidential)
    parser.add_argument("--ds_discount", action="store_true", help="Apply reliability discount before DS fusion on evidential outputs")
    parser.add_argument("--choose_better", action="store_true", default=True, help="For triggered samples, keep the result (pre vs retransmit) with lower uncertainty")

    # Rejection/Selective prediction options used to drive retransmission
    parser.add_argument(
        "--reject_score",
        type=str,
        choices=["u_evi", "entropy", "margin", "maxprob"],
        default="u_evi",
        help="Rejection score used to trigger retransmission: evidential uncertainty, entropy, probability margin, or max probability",
    )
    parser.add_argument(
        "--target_coverage",
        type=float,
        default=None,
        help="If set and --rt_threshold is None, auto-select threshold to achieve target coverage (accepted fraction).",
    )
    parser.add_argument(
        "--report_selective",
        action="store_true",
        help="Report coverage and selective risk (risk on accepted/non-retransmitted samples).",
    )
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
def compute_reject_score(
    fused_out: torch.Tensor,
    model_type: str,
    num_classes: int,
    mode: str,
) -> torch.Tensor:
    """Compute rejection score from fused outputs.
    For evidential (alpha) outputs: use alpha to derive probabilities.
    For non-evidential (logits) outputs: use softmax probabilities.
    """
    if model_type == "TMC_channel":
        # fused_out is evidential alpha
        alpha = fused_out
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
        # fallback
        return float(num_classes) / torch.sum(alpha, dim=1)
    else:
        # fused_out are logits
        probs = F.softmax(fused_out, dim=1)
        if mode in ("u_evi", "entropy"):
            return -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
        if mode == "margin":
            top2 = torch.topk(probs, k=2, dim=1).values
            return top2[:, 0] - top2[:, 1]
        if mode == "maxprob":
            return torch.max(probs, dim=1).values
        return -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)


@torch.no_grad()
def scan_reject_scores(
    model: torch.nn.Module,
    loader,
    model_type: str,
    num_classes: int,
    device: str,
    score_mode: str,
) -> Dict[str, np.ndarray]:
    """单次遍历：为每个视角分别计算拒绝分数，用于按视角阈值的自动选择。

    返回字典：{"depth": np.ndarray, "rgb": np.ndarray}
    """
    model.eval()
    scores_depth, scores_rgb = [], []
    for batch in tqdm(loader, total=len(loader)):
        rgb, depth = batch["A"].to(device), batch["B"].to(device)
        if model_type == "TMC_channel":
            depth_a, rgb_a, _ = model(rgb, depth)
            s_depth = compute_reject_score(depth_a, model_type, num_classes, score_mode)
            s_rgb = compute_reject_score(rgb_a, model_type, num_classes, score_mode)
        else:
            depth_l, rgb_l, _ = model(rgb, depth)
            s_depth = compute_reject_score(depth_l, model_type, num_classes, score_mode)
            s_rgb = compute_reject_score(rgb_l, model_type, num_classes, score_mode)
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
    model_type: str,
    num_classes: int,
    device: str,
    rt_cfg: Dict,
) -> Dict:
    model.eval()

    all_uncert_pre, all_uncert_post = [], []
    all_correct_pre, all_correct_post = [], []
    all_do_rt_masks = []
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

            # setup per-view RT masks
            do_rt_depth_mask, do_rt_rgb_mask = None, None
            if rt_cfg["enabled"] and (rt_cfg.get("threshold_depth") is not None or rt_cfg.get("threshold_rgb") is not None):
                # per-view thresholds with fallback to a shared threshold if provided
                tau_depth = rt_cfg.get("threshold_depth", rt_cfg.get("threshold"))
                tau_rgb = rt_cfg.get("threshold_rgb", rt_cfg.get("threshold"))
                score_mode = rt_cfg.get("reject_score", "u_evi")
                # per-view scores based on evidential outputs
                score_depth = compute_reject_score(depth_a, model_type, num_classes, score_mode)
                score_rgb = compute_reject_score(rgb_a, model_type, num_classes, score_mode)
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
            else:
                do_rt_any = None

            # default post equals pre
            fused_post = fused_a
            if do_rt_any is not None and torch.any(do_rt_any):
                # evidential per-view multi-trial retransmission without extra forward
                max_trials = int(rt_cfg.get("rt_max_trials", 1))
                discard_fail = bool(rt_cfg.get("rt_discard_unqualified", False))

                fused_list = []
                for i in range(fused_a.shape[0]):
                    # count per view (always two per sample)
                    num_total += 2
                    if do_rt_depth_mask is not None:
                        num_rt += int(bool(do_rt_depth_mask[i].item()))
                    if do_rt_rgb_mask is not None:
                        num_rt += int(bool(do_rt_rgb_mask[i].item()))

                    # start with original per-view evidences
                    a_left_cur = depth_a[i:i+1]
                    a_right_cur = rgb_a[i:i+1]

                    # helper to decide trigger given a score tensor of shape [1]
                    def still_triggers(score_tensor: torch.Tensor, tau: float, trigger: str) -> bool:
                        if tau is None:
                            return False
                        if trigger == "low":
                            return bool((score_tensor < tau).item())
                        else:
                            return bool((score_tensor > tau).item())

                    score_mode = rt_cfg.get("reject_score", "u_evi")

                    # depth view trials (combine THEN validate)
                    if do_rt_depth_mask is not None and bool(do_rt_depth_mask[i].item()):
                        tau_d = rt_cfg.get("threshold_depth", rt_cfg.get("threshold"))
                        trials = 0
                        while trials < max_trials:
                            # first combine with original evidence to emulate a new transmission
                            a_left_cur = ds_combin_two(a_left_cur, depth_a[i:i+1], num_classes)
                            trials += 1
                            # then validate against threshold
                            s_d = compute_reject_score(a_left_cur, model_type, num_classes, score_mode)
                            if not still_triggers(s_d, tau_d, rt_cfg["trigger"]):
                                break
                        # after trials, if still not pass and need discard -> vacuous evidence (all ones)
                        if discard_fail:
                            s_d_final = compute_reject_score(a_left_cur, model_type, num_classes, score_mode)
                            if still_triggers(s_d_final, tau_d, rt_cfg["trigger"]):
                                a_left_cur = torch.ones_like(a_left_cur)

                    # rgb view trials (combine THEN validate)
                    if do_rt_rgb_mask is not None and bool(do_rt_rgb_mask[i].item()):
                        tau_r = rt_cfg.get("threshold_rgb", rt_cfg.get("threshold"))
                        trials = 0
                        while trials < max_trials:
                            a_right_cur = ds_combin_two(a_right_cur, rgb_a[i:i+1], num_classes)
                            trials += 1
                            s_r = compute_reject_score(a_right_cur, model_type, num_classes, score_mode)
                            if not still_triggers(s_r, tau_r, rt_cfg["trigger"]):
                                break
                        if discard_fail:
                            s_r_final = compute_reject_score(a_right_cur, model_type, num_classes, score_mode)
                            if still_triggers(s_r_final, tau_r, rt_cfg["trigger"]):
                                a_right_cur = torch.ones_like(a_right_cur)

                    # prepare evidences for fusion: apply optional discount
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
                # no retransmission triggered; still count per-view total
                num_total += 2 * fused_a.shape[0]

            pred_post = torch.argmax(fused_post, dim=1)
            u_post = float(num_classes) / torch.sum(fused_post, dim=1)
            all_uncert_post.append(u_post.detach().cpu())
            all_correct_post.append((pred_post == tgt).detach().cpu())
            if do_rt_any is None:
                all_do_rt_masks.append(torch.zeros_like(u_fused, dtype=torch.bool).cpu())
            else:
                all_do_rt_masks.append(do_rt_any.detach().cpu())

        else:  # TMC_base_channel
            depth_l, rgb_l, fused_l = model(rgb, depth)
            probs = F.softmax(fused_l, dim=1)
            u_fused = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
            pred_fused = torch.argmax(fused_l, dim=1)

            # record pre-RT
            all_uncert_pre.append(u_fused.detach().cpu())
            all_correct_pre.append((pred_fused == tgt).detach().cpu())

            # per-view RT masks (logits)
            do_rt_depth_mask, do_rt_rgb_mask = None, None
            do_rt_any = None
            if rt_cfg["enabled"] and (rt_cfg.get("threshold_depth") is not None or rt_cfg.get("threshold_rgb") is not None):
                tau_depth = rt_cfg.get("threshold_depth", rt_cfg.get("threshold"))
                tau_rgb = rt_cfg.get("threshold_rgb", rt_cfg.get("threshold"))
                score_mode = rt_cfg.get("reject_score", "entropy")
                score_depth = compute_reject_score(depth_l, model_type, num_classes, score_mode)
                score_rgb = compute_reject_score(rgb_l, model_type, num_classes, score_mode)
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

            fused_post = fused_l
            if do_rt_any is not None and torch.any(do_rt_any):
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
                    # count per view
                    num_total += 2
                    if do_rt_depth_mask is not None:
                        num_rt += int(bool(do_rt_depth_mask[i].item()))
                    if do_rt_rgb_mask is not None:
                        num_rt += int(bool(do_rt_rgb_mask[i].item()))
                    if bool(do_rt_any[i].item()):
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
                # no retransmission triggered; still count per-view total
                num_total += 2 * fused_l.shape[0]

            probs_post = F.softmax(fused_post, dim=1)
            u_post = -torch.sum(probs_post * torch.log(torch.clamp(probs_post, min=1e-12)), dim=1)
            pred_post = torch.argmax(fused_post, dim=1)
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
    # selective risk on accepted (non-RT) using pre-RT correctness
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
    if cli_args.model_type == "TMC_channel":
        model = TMC_channel(run_args).to(cli_args.device)
    else:
        model = TMC_base_channel(run_args).to(cli_args.device)

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
            # fallback: search any *.pt or *.pth under savedir
            try:
                files = os.listdir(savedir)
            except Exception:
                files = []
            pt_files = [f for f in files if f.endswith(".pt") or f.endswith(".pth")]
            # prefer names containing 'best' or 'checkpoint'
            preferred = [f for f in pt_files if "best" in f.lower() or "checkpoint" in f.lower()]
            pick = preferred[0] if len(preferred) > 0 else (pt_files[0] if len(pt_files) > 0 else None)
            if pick is not None:
                ckpt_path = os.path.join(savedir, pick)
    assert ckpt_path is not None and os.path.exists(ckpt_path), f"Checkpoint not found under savedir. Tried default names and scan: {savedir}"
    checkpoint = torch.load(ckpt_path, map_location=cli_args.device)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)

    # set fixed SNR if provided/inferred
    fixed_snr = infer_fixed_snr(cli_args, run_args)
    if fixed_snr is not None and hasattr(model, "args"):
        model.args.channel_snr = float(fixed_snr)

    # If target coverage is provided but threshold is not, scan to set threshold
    auto_tau = {"depth": None, "rgb": None}
    if cli_args.target_coverage is not None and cli_args.rt_threshold is None:
        scores = scan_reject_scores(
            model,
            test_loader,
            cli_args.model_type,
            getattr(run_args, "n_classes", 10),
            cli_args.device,
            cli_args.reject_score,
        )
        cov = float(cli_args.target_coverage)
        cov = min(max(cov, 0.0), 1.0)
        if cli_args.rt_trigger == "high":
            q = cov
        else:
            q = 1.0 - cov
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
        "pick_view": cli_args.rt_pick_view,
        "rt_snr": cli_args.rt_snr,
        "ds_discount": bool(cli_args.ds_discount),
        "choose_better": bool(cli_args.choose_better),
        "reject_score": cli_args.reject_score,
        "rt_max_trials": int(cli_args.rt_max_trials),
        "rt_discard_unqualified": bool(cli_args.rt_discard_unqualified),
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


