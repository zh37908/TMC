import argparse
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models.TMC import TMC_channel_snr as TMC, ce_loss
import torchvision.transforms as transforms
from data.aligned_conc_dataset import AlignedConcDataset
from utils.utils import *
from utils.logger import create_logger
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="/home/hzhaobi/Multired/nyud2")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="ReleasedVersion")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/TMC/nyud/original/Adam/pretrain_resnet18/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    # SNR-related args for TMC_channel_snr (static SNR training)
    parser.add_argument("--channel_snr", type=float, default=10.0, help="Static SNR (dB) during training/eval")
    parser.add_argument("--snr_input_method", type=str, default="none", choices=["none", "mlp", "concat", "add"], help="How SNR is injected into model; 'none' disables SNR embedding")
    # Channel module config required by TMC_channel_snr
    parser.add_argument("--channel_type", type=str, default="awgn", choices=["none", "awgn", "rayleigh"], help="Channel type for feature perturbation")
    parser.add_argument("--channel_hidden", nargs="*", type=int, default=[512], help="Hidden sizes for channel encoders")
    parser.add_argument("--channel_size", type=int, default=256, help="Projected channel feature size")
    # Optional SNR embed config/range (used if启用动态或非none融合)
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--snr_embed_dim", type=int, default=64)


def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_forward(i_epoch, model, args, ce_loss, batch):
    rgb, depth, tgt = batch['A'], batch['B'], batch['label']

    rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
    # static 10 dB (or args.channel_snr) vector for the whole batch
    snr_in = torch.full((rgb.shape[0],), float(args.channel_snr), dtype=torch.float32, device=rgb.device)
    depth_alpha, rgb_alpha, depth_rgb_alpha = model(rgb, depth, snr_in)

    loss = ce_loss(tgt, depth_alpha, args.n_classes, i_epoch, args.annealing_epoch) + \
           ce_loss(tgt, rgb_alpha, args.n_classes, i_epoch, args.annealing_epoch) + \
           ce_loss(tgt, depth_rgb_alpha, args.n_classes, i_epoch, args.annealing_epoch)
    return loss, depth_alpha, rgb_alpha, depth_rgb_alpha, tgt


def model_eval(i_epoch, data, model, args, criterion):
    model.eval()
    with torch.no_grad():
        losses, depth_preds, rgb_preds, depthrgb_preds, tgts = [], [], [], [], []
        fused_uncerts = []
        for batch in data:
            loss, depth_alpha, rgb_alpha, depth_rgb_alpha, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())

            depth_pred = depth_alpha.argmax(dim=1).cpu().detach().numpy()
            rgb_pred = rgb_alpha.argmax(dim=1).cpu().detach().numpy()
            depth_rgb_pred = depth_rgb_alpha.argmax(dim=1).cpu().detach().numpy()

            depth_preds.append(depth_pred)
            rgb_preds.append(rgb_pred)
            depthrgb_preds.append(depth_rgb_pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

            # collect fused uncertainty: K / sum(alpha)
            K = int(getattr(args, 'n_classes', 10))
            u = float(K) / torch.sum(depth_rgb_alpha, dim=1)
            fused_uncerts.append(u.cpu())

    metrics = {"loss": np.mean(losses)}

    tgts = [l for sl in tgts for l in sl]
    depth_preds = [l for sl in depth_preds for l in sl]
    rgb_preds = [l for sl in rgb_preds for l in sl]
    depthrgb_preds = [l for sl in depthrgb_preds for l in sl]
    metrics["depth_acc"] = accuracy_score(tgts, depth_preds)
    metrics["rgb_acc"] = accuracy_score(tgts, rgb_preds)
    metrics["depthrgb_acc"] = accuracy_score(tgts, depthrgb_preds)
    metrics["fused_uncert_all"] = torch.cat(fused_uncerts, dim=0).numpy() if len(fused_uncerts) > 0 else np.array([])
    return metrics


def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]
    train_transforms = list()
    train_transforms.append(transforms.Resize((args.LOAD_SIZE, args.LOAD_SIZE)))
    train_transforms.append(transforms.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize(mean=mean, std=std))
    val_transforms = list()
    val_transforms.append(transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.Normalize(mean=mean, std=std))

    train_loader = DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'train'), transform=transforms.Compose(train_transforms)),
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers)
    test_loader = DataLoader(
            AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'test'), transform=transforms.Compose(val_transforms)),
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers)
    model = TMC(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    ckpt_path = os.path.join(args.savedir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        try:
            # Load weights only; do NOT resume optimizer/scheduler to avoid optimizer-type mismatch
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            logger.info("Loaded model weights from checkpoint (strict=False); optimizer/scheduler not resumed.")
        except Exception as e:
            # filter by matching keys and shapes
            model_sd = model.state_dict()
            src_sd = checkpoint.get("state_dict", {})
            filtered = {}
            for k, v in src_sd.items():
                if k in model_sd and hasattr(v, 'shape') and hasattr(model_sd[k], 'shape') and tuple(v.shape) == tuple(model_sd[k].shape):
                    filtered[k] = v
            if len(filtered) > 0:
                model_sd.update(filtered)
                model.load_state_dict(model_sd)
                logger.warning(f"Partially loaded {len(filtered)} compatible tensors from checkpoint; optimizer/scheduler not resumed. Reason: {e}")
            else:
                logger.warning(f"Checkpoint incompatible with current model; starting fresh. Reason: {e}")

    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()
        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, depth_out, rgb_out, depthrgb, tgt = model_forward(i_epoch, model, args, ce_loss, batch)
            if args.gradient_accumulation_steps > 1:
                 loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics = model_eval(
            np.inf, test_loader, model, args, ce_loss
        )
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("val", metrics, logger)
        logger.info(
            "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}, depth rgb acc: {:.5f}".format(
                "val", metrics["loss"], metrics["depth_acc"], metrics["rgb_acc"], metrics["depthrgb_acc"]
            )
        )
        tuning_metric = metrics["depthrgb_acc"]

        # every 50 epochs: save train/test uncertainty density snapshots
        if ((i_epoch + 1) % 10) == 0:
            try:
                # eval on train loader (model in eval mode inside model_eval)
                metrics_train_snap = model_eval(np.inf, train_loader, model, args, ce_loss)
                u_train = metrics_train_snap.get("fused_uncert_all", np.array([]))
                u_test = metrics.get("fused_uncert_all", np.array([]))
                plots_dir = os.path.join(args.savedir, "plots")
                _save_uncert_density(u_train, os.path.join(plots_dir, f"uncert_epoch_{i_epoch+1:04d}_train.png"), f"Uncertainty density (train @epoch {i_epoch+1})")
                _save_uncert_density(u_test, os.path.join(plots_dir, f"uncert_epoch_{i_epoch+1:04d}_test.png"), f"Uncertainty density (test @epoch {i_epoch+1})")
                logger.info(f"Saved uncertainty density snapshots for epoch {i_epoch+1} -> {plots_dir}")
            except Exception as e:
                logger.info(f"Failed saving per-epoch uncertainty density (epoch {i_epoch+1}): {e}")

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
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    test_metrics = model_eval(
        np.inf, test_loader, model, args, ce_loss
    )
    logger.info(
        "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}, depth rgb acc: {:.5f}".format(
            "Test", test_metrics["loss"], test_metrics["depth_acc"], test_metrics["rgb_acc"],
            test_metrics["depthrgb_acc"]
        )
    )
    log_metrics(f"Test", test_metrics, logger)

    # plot and save fused uncertainty density for best checkpoint
    try:
        uncert = test_metrics.get("fused_uncert_all", np.array([]))
        if uncert.size > 0:
            out_dir = os.path.join(args.savedir, "plots")
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, "uncertainty_density_best.png")
            plt.figure(figsize=(6, 4))
            try:
                sns.kdeplot(uncert, fill=True, color="#1f77b4", alpha=0.6)
            except Exception:
                plt.hist(uncert, bins=60, density=True, color="#1f77b4", alpha=0.6)
            plt.xlabel("Fused uncertainty")
            plt.ylabel("Density")
            plt.title("Uncertainty density (best checkpoint)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_png, dpi=200)
            plt.close()
            logger.info(f"Saved uncertainty density figure: {out_png}")
            # print brief stats
            logger.info(
                "Uncertainty stats (best): mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
                    float(np.mean(uncert)), float(np.std(uncert)), float(np.min(uncert)), float(np.max(uncert))
                )
            )
    except Exception as e:
        logger.info(f"Failed to save/print uncertainty density: {e}")


def _save_uncert_density(uncert_arr, out_path: str, title: str):
    if uncert_arr is None or (hasattr(uncert_arr, 'size') and uncert_arr.size == 0):
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    try:
        sns.kdeplot(uncert_arr, fill=True, color="#1f77b4", alpha=0.6)
    except Exception:
        plt.hist(uncert_arr, bins=60, density=True, color="#1f77b4", alpha=0.6)
    plt.xlabel("Fused uncertainty")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()