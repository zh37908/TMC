import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score

from data.aligned_conc_dataset import AlignedConcDataset
from utils.utils import *
from utils.logger import create_logger
from models.TMC import (
    TMC_base_channel,
    TMC_channel,
    TMC_base_channel_dynamic,
    TMC_channel_dynamic,
    ce_loss,
)
from models.pretrain_models import DeCUR, SimCLR, BarlowTwins


def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # dataset and io
    parser.add_argument("--data_path", type=str, default="/home/hzhaobi/Multired/nyud2")
    parser.add_argument("--savedir", type=str, default="./savepath/TMC_any/nyud/")
    parser.add_argument("--name", type=str, default="train_TMC_any")

    # model selection
    parser.add_argument("--model_kind", type=str, choices=["base", "evidential"], default="evidential",
                        help="Choose base (softmax+CE) or evidential (Dirichlet+ce_loss)")
    parser.add_argument("--channel_mode", type=str, choices=["static", "dynamic"], default="static",
                        help="Choose channel mode: static uses --channel_snr; dynamic samples per-forward in [snr_min, snr_max]")

    # channel params
    parser.add_argument("--channel_snr", type=float, default=20.0, help="SNR(dB) for static channel")
    parser.add_argument("--snr_min", type=float, default=0.0, help="Min SNR(dB) for dynamic channel")
    parser.add_argument("--snr_max", type=float, default=20.0, help="Max SNR(dB) for dynamic channel")

    # training
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam",
                        help="Choose optimizer: adam or sgd")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--nesterov", action="store_true", help="Use Nesterov momentum for SGD")
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)

    # image encoder and classifier heads
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"]) 
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[64, 64])
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--channel_hidden", nargs="*", type=int, default=[512])
    parser.add_argument("--channel_size", type=int, default=32)

    # evidential training misc
    parser.add_argument("--annealing_epoch", type=int, default=10)

    # pretrain / backbone
    parser.add_argument("--pretrain", type=str, default='DeCUR', choices=['DeCUR','SimCLR','BarlowTwins','No_pretrain'])
    parser.add_argument("--freeze_encoder", type=int, default=0)
    parser.add_argument("--backbone", type=str, default='resnet18')
    parser.add_argument("--rda", type=bool, default=False)
    parser.add_argument('--projector', default='8192-8192-8192', type=str, metavar='MLP', help='projector MLP')
    return parser


def get_optimizer(model, args):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # SGD
    return optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=bool(args.nesterov), weight_decay=1e-5
    )


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def build_dataloaders(args):
    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]
    train_transforms = [
        transforms.Resize((args.LOAD_SIZE, args.LOAD_SIZE)),
        transforms.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    val_transforms = [
        transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    train_loader = torch.utils.data.DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'train'), transform=transforms.Compose(train_transforms)),
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'test'), transform=transforms.Compose(val_transforms)),
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
    )
    return train_loader, test_loader


def build_model(args):
    # choose class by kind and channel mode
    if args.model_kind == "base":
        model_cls = TMC_base_channel_dynamic if args.channel_mode == "dynamic" else TMC_base_channel
    else:
        model_cls = TMC_channel_dynamic if args.channel_mode == "dynamic" else TMC_channel
    model = model_cls(args).cuda()

    # load pretrain trunk if requested
    if args.pretrain != 'No_pretrain':
        if args.pretrain == 'DeCUR':
            model_pt = DeCUR(args)
            checkpoint = torch.load('/home/hzhaobi/model_DeCUR_nopretrain.pth')
        elif args.pretrain == 'SimCLR':
            model_pt = SimCLR(args)
            checkpoint = torch.load('/home/hzhaobi/model_SimCLR.pth')
        elif args.pretrain == 'BarlowTwins':
            model_pt = BarlowTwins(args)
            checkpoint = torch.load('/home/hzhaobi/model_BarlowTwins_nopretrain.pth')
        else:
            model_pt = None
            checkpoint = None
        if model_pt is not None:
            model_pt.load_state_dict(checkpoint)
            backbone1_trunk = nn.Sequential(*list(model_pt.backbone_1.children())[:-1])
            backbone2_trunk = nn.Sequential(*list(model_pt.backbone_2.children())[:-1])
            model.rgbenc.model.load_state_dict(backbone1_trunk.state_dict(), strict=True)
            model.depthenc.model.load_state_dict(backbone2_trunk.state_dict(), strict=True)

    if args.freeze_encoder:
        for p in model.rgbenc.model.parameters():
            p.requires_grad = False
        for p in model.depthenc.model.parameters():
            p.requires_grad = False

    return model


def model_forward(i_epoch, model, args, batch):
    rgb, depth, tgt = batch['A'], batch['B'], batch['label']
    rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()

    if args.model_kind == "base":
        depth_logits, rgb_logits, depth_rgb_logits = model(rgb, depth)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(depth_logits, tgt) + criterion(rgb_logits, tgt) + criterion(depth_rgb_logits, tgt)
        depth_pred = depth_logits.argmax(dim=1).detach().cpu().numpy()
        rgb_pred = rgb_logits.argmax(dim=1).detach().cpu().numpy()
        depth_rgb_pred = depth_rgb_logits.argmax(dim=1).detach().cpu().numpy()
    else:
        depth_alpha, rgb_alpha, depth_rgb_alpha = model(rgb, depth)
        loss = ce_loss(tgt, depth_alpha, args.n_classes, i_epoch, args.annealing_epoch) \
             + ce_loss(tgt, rgb_alpha, args.n_classes, i_epoch, args.annealing_epoch) \
             + ce_loss(tgt, depth_rgb_alpha, args.n_classes, i_epoch, args.annealing_epoch)
        depth_pred = depth_alpha.argmax(dim=1).detach().cpu().numpy()
        rgb_pred = rgb_alpha.argmax(dim=1).detach().cpu().numpy()
        depth_rgb_pred = depth_rgb_alpha.argmax(dim=1).detach().cpu().numpy()

    return loss, depth_pred, rgb_pred, depth_rgb_pred, tgt


@torch.no_grad()
def eval_epoch(i_epoch, data, model, args):
    model.eval()
    losses, depth_preds, rgb_preds, depthrgb_preds, tgts = [], [], [], [], []
    for batch in data:
        loss, depth_pred, rgb_pred, depth_rgb_pred, tgt = model_forward(i_epoch, model, args, batch)
        losses.append(loss.item())
        depth_preds.append(depth_pred)
        rgb_preds.append(rgb_pred)
        depthrgb_preds.append(depth_rgb_pred)
        tgts.append(tgt.cpu().numpy())

    metrics = {"loss": np.mean(losses)}
    tgts = [l for sl in tgts for l in sl]
    depth_preds = [l for sl in depth_preds for l in sl]
    rgb_preds = [l for sl in rgb_preds for l in sl]
    depthrgb_preds = [l for sl in depthrgb_preds for l in sl]
    metrics["depth_acc"] = accuracy_score(tgts, depth_preds)
    metrics["rgb_acc"] = accuracy_score(tgts, rgb_preds)
    metrics["depthrgb_acc"] = accuracy_score(tgts, depthrgb_preds)
    return metrics


def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, test_loader = build_dataloaders(args)
    model = build_model(args)

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    logger = create_logger(f"{args.savedir}/logfile.log", args)

    # persist args for test scripts
    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    # auto-resume if checkpoint exists and --resume passed
    if getattr(args, "resume", False) and os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    for i_epoch in range(start_epoch, args.max_epochs):
        model.train()
        optimizer.zero_grad()
        train_losses = []
        for batch in tqdm(train_loader, total=len(train_loader)):
            # static channel: ensure snr fixed; dynamic handled inside model
            if args.channel_mode == "static":
                model.args.channel_snr = float(args.channel_snr)

            loss, *_ = model_forward(i_epoch, model, args, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_losses.append(loss.item())
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        metrics = eval_epoch(i_epoch, test_loader, model, args)
        logger.info(f"Epoch {i_epoch + 1}/{args.max_epochs} | Train Loss: {np.mean(train_losses):.4f}")
        log_metrics("val", metrics, logger)
        logger.info(
            f"Epoch {i_epoch + 1}/{args.max_epochs} | val: Loss: {metrics['loss']:.5f} | depth_acc: {metrics['depth_acc']:.5f}, "
            f"rgb_acc: {metrics['rgb_acc']:.5f}, depth rgb acc: {metrics['depthrgb_acc']:.5f}"
        )

        tuning_metric = metrics["depthrgb_acc"]
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Continue training without early stopping.")

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    test_metrics = eval_epoch(np.inf, test_loader, model, args)
    logger.info(
        f"Test: Loss: {test_metrics['loss']:.5f} | depth_acc: {test_metrics['depth_acc']:.5f}, "
        f"rgb_acc: {test_metrics['rgb_acc']:.5f}, depth rgb acc: {test_metrics['depthrgb_acc']:.5f}"
    )
    log_metrics("Test", test_metrics, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Unified trainer for TMC (base/evidential) with static/dynamic channel")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()


