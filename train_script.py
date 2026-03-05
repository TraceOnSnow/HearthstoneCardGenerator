"""train_script.py – Entry-point for training the HearthstoneCardGenerator model.

Usage::

    python train_script.py --data_dir /path/to/data --epochs 50

The script:
1. Builds the CLIP encoder and GNN model.
2. Loads train / val / test data via :class:`~utils.data_loader.HearthstoneDataLoader`.
3. Runs the training loop with periodic validation.
4. Saves the best checkpoint to ``--output_dir``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.clip_encoder import CLIPEncoder
from models.gnn_model import GNNModel
from utils.data_loader import HearthstoneDataLoader


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Hearthstone Card Generator (GNN + CLIP)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing card images and metadata JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory where model checkpoints are saved.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Samples per mini-batch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=256,
        help="Hidden layer width for the GNN.",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=128,
        help="Output embedding dimensionality for the GNN.",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=256,
        help="Projection dimensionality for the CLIP encoder.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model identifier.",
    )
    parser.add_argument(
        "--unfreeze_clip",
        action="store_true",
        default=False,
        help="Unfreeze CLIP encoder weights during training (default: frozen).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader worker processes."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (e.g. 'cuda', 'cpu'). Auto-detected if not set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(args: argparse.Namespace, device: torch.device) -> tuple[CLIPEncoder, GNNModel]:
    clip_encoder = CLIPEncoder(
        model_name=args.clip_model,
        projection_dim=args.projection_dim,
        freeze_clip=not args.unfreeze_clip,
    ).to(device)

    # GNN input dim = projection_dim (from CLIP) + stat features + one-hot features
    # Stat features: cost, attack, health (up to 3)
    # One-hot features: 6 card types + 12 classes = 18
    gnn_in_channels = args.projection_dim + 3 + 18

    gnn_model = GNNModel(
        in_channels=gnn_in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
    ).to(device)

    return clip_encoder, gnn_model


def run_epoch(
    clip_encoder: CLIPEncoder,
    gnn_model: GNNModel,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
) -> float:
    """Run a single epoch and return the average loss."""
    clip_encoder.train(train)
    gnn_model.train(train)

    total_loss = 0.0
    num_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
            pixel_values = batch.get("pixel_values")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("label")

            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
                image_emb, text_emb = clip_encoder(pixel_values, input_ids, attention_mask)
                # Fuse embeddings by averaging
                node_features = (image_emb + text_emb) / 2
            else:
                _, node_features = clip_encoder(None, input_ids, attention_mask)

            # For a simple supervised setting, use the projected features directly.
            # A real graph structure would be built from card relationships here.
            if labels is not None:
                labels = labels.to(device)
                # Use node_features directly when no graph edges are available.
                logits = gnn_model.fc(node_features)
                loss = criterion(logits, labels)

                if train and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data_loader_factory = HearthstoneDataLoader(
        data_dir=args.data_dir,
        clip_model_name=args.clip_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loaders = data_loader_factory.get_all_loaders()

    # Models
    clip_encoder, gnn_model = build_models(args, device)

    # Optimizer (only update non-frozen parameters)
    params = list(filter(lambda p: p.requires_grad, clip_encoder.parameters()))
    params += list(gnn_model.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            clip_encoder, gnn_model, loaders["train"], optimizer, criterion, device, train=True
        )
        val_loss = run_epoch(
            clip_encoder, gnn_model, loaders["val"], None, criterion, device, train=False
        )
        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "clip_encoder": clip_encoder.state_dict(),
                "gnn_model": gnn_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            save_path = output_dir / "best_model.pt"
            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved best checkpoint to {save_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
