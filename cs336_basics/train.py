"""
Training script for cs336 language model.

Example usage:
    python -m cs336_basics.train \
        --train-path data/TinyStoriesV2-GPT4-train.txt.npy \
        --val-path data/TinyStoriesV2-GPT4-valid.txt.npy \
        --checkpoint-dir checkpoints/ \
        --vocab-size 10000 \
        --d-model 512 \
        --num-heads 8 \
        --num-layers 6 \
        --d-ff 2048 \
        --context-length 256 \
        --batch-size 32 \
        --max-iters 10000 \
        --lr 3e-4
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.transformer import Transformer, cross_entropy
from cs336_basics.optimizer import AdamW, lr_cosine_schedule, gradient_clipping
from cs336_basics.training_utils import get_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer language model")

    # Data
    p.add_argument("--train-path", type=str, required=True,
                   help="Path to the training data file (numpy .npy or raw binary)")
    p.add_argument("--val-path", type=str, required=True,
                   help="Path to the validation data file (numpy .npy or raw binary)")
    p.add_argument("--dtype", type=str, default="uint16",
                   choices=["uint16", "uint32", "int32", "int64"],
                   help="numpy dtype for memmap (ignored for .npy files)")

    # Model hyperparameters
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--theta", type=float, default=10000.0,
                   help="RoPE theta base")

    # Optimizer hyperparameters
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5,
                   help="Minimum LR at the end of cosine decay")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm (0 to disable)")

    # LR schedule
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--lr-decay-iters", type=int, default=None,
                   help="Total iters for cosine decay (defaults to --max-iters)")

    # Training loop
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=10000)
    p.add_argument("--eval-interval", type=int, default=500,
                   help="Evaluate on validation set every N iters")
    p.add_argument("--eval-iters", type=int, default=20,
                   help="Number of batches to average for val loss estimate")
    p.add_argument("--log-interval", type=int, default=50,
                   help="Log training loss every N iters")
    p.add_argument("--checkpoint-interval", type=int, default=1000,
                   help="Save a checkpoint every N iters (0 to disable)")

    # Checkpointing
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from")

    # Device
    p.add_argument("--device", type=str, default=None,
                   help="Device to use (default: cuda if available, else cpu)")

    # Weights & Biases
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="cs336-lm")
    p.add_argument("--wandb-run-name", type=str, default=None)

    return p.parse_args()


def load_dataset(path: str, dtype: str) -> np.ndarray:
    """Load a dataset with np.memmap for memory efficiency, or np.load for .npy files."""
    path = Path(path)
    if path.suffix == ".npy":
        # Small enough to load fully, or use memory-mapped .npy
        return np.load(path, mmap_mode="r")
    else:
        # Raw binary file — use memmap so it's never fully read into RAM
        return np.memmap(path, dtype=dtype, mode="r")


@torch.no_grad()
def estimate_val_loss(
    model: Transformer,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        # logits: (B, T, V) — flatten to (B*T, V) for cross_entropy
        B, T, V = logits.shape
        loss = cross_entropy(logits.view(B * T, V), y.view(B * T))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(
    path: str,
    model: Transformer,
    optimizer: AdamW,
    iteration: int,
    val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )
    print(f"  [checkpoint] saved to {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: AdamW,
    device: str,
) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    iteration = ckpt["iteration"]
    print(f"  [checkpoint] resumed from {path} at iter {iteration}, val_loss={ckpt['val_loss']:.4f}")
    return iteration


def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------ device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # ------------------------------------------------------------------- data
    print("Loading datasets …")
    train_data = load_dataset(args.train_path, args.dtype)
    val_data = load_dataset(args.val_path, args.dtype)
    print(f"  train tokens: {len(train_data):,}  |  val tokens: {len(val_data):,}")

    # ------------------------------------------------------------------ model
    model = Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        theta=args.theta,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {num_params:,}")

    # --------------------------------------------------------------- optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters is not None else args.max_iters

    # -------------------------------------------------------- optional resume
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer, device)

    # ----------------------------------------------------------- optional W&B
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            wandb.watch(model, log_freq=args.log_interval)
        except ImportError:
            print("Warning: wandb not installed; disabling W&B logging.")
            args.wandb = False

    # --------------------------------------------------------------- training
    model.train()
    best_val_loss = float("inf")
    t0 = time.time()

    for iteration in range(start_iter, args.max_iters):
        # ---- LR schedule
        lr = lr_cosine_schedule(
            iteration=iteration,
            alpha_min=args.min_lr,
            alpha_max=args.lr,
            warmup_iters=args.warmup_iters,
            total_iters=lr_decay_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ---- forward + backward
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad()

        logits = model(x)
        B, T, V = logits.shape
        loss = cross_entropy(logits.view(B * T, V), y.view(B * T))
        loss.backward()

        if args.grad_clip > 0:
            gradient_clipping(list(model.parameters()), args.grad_clip)

        optimizer.step()

        # ---- periodic logging
        if (iteration + 1) % args.log_interval == 0:
            dt = time.time() - t0
            tokens_per_sec = args.log_interval * args.batch_size * args.context_length / dt
            print(
                f"iter {iteration + 1:>7d} | loss {loss.item():.4f} | "
                f"lr {lr:.2e} | {tokens_per_sec:,.0f} tok/s"
            )
            if args.wandb:
                wandb.log({"train/loss": loss.item(), "lr": lr, "iter": iteration + 1})
            t0 = time.time()

        # ---- periodic validation
        if (iteration + 1) % args.eval_interval == 0:
            val_loss = estimate_val_loss(
                model, val_data, args.batch_size, args.context_length, device, args.eval_iters
            )
            print(f"  [val] iter {iteration + 1:>7d} | val_loss {val_loss:.4f}")
            if args.wandb:
                wandb.log({"val/loss": val_loss, "iter": iteration + 1})

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    os.path.join(args.checkpoint_dir, "best.pt"),
                    model, optimizer, iteration + 1, val_loss,
                )

        # ---- periodic checkpoint
        if args.checkpoint_interval > 0 and (iteration + 1) % args.checkpoint_interval == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"iter_{iteration + 1:07d}.pt"),
                model, optimizer, iteration + 1, loss.item(),
            )

    # ---- final checkpoint
    save_checkpoint(
        os.path.join(args.checkpoint_dir, "final.pt"),
        model, optimizer, args.max_iters, loss.item(),
    )

    if args.wandb:
        wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    train(parse_args())
