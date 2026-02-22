"""
Training Script for ResNet18 Plant Disease Classifier.

Trains a ResNet18 model on the PlantVillage dataset (or compatible structure)
for 16-class plant disease classification (15 diseases + 1 healthy).

Features:
- Data augmentation (flip, rotation, brightness/contrast jitter)
- Adam optimizer with ReduceLROnPlateau scheduler
- CrossEntropyLoss
- Early stopping with configurable patience
- Saves best model (.pth weights)
- Exports TorchScript model (.pt) for edge inference
- Classification report and confusion matrix

Usage:
    python train.py --data_dir /path/to/plantvillage --epochs 30 --batch_size 32

Dataset Structure Expected:
    data_dir/
    ├── Apple___Apple_scab/
    │   ├── image001.jpg
    │   └── ...
    ├── Corn_(maize)___Common_rust/
    │   └── ...
    ├── Healthy/
    │   └── ...
    └── ... (16 class folders)
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.config import (
    DISEASE_CLASSES,
    NUM_CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def get_train_transforms() -> transforms.Compose:
    """Training data augmentation and normalization transforms.

    Augmentations applied:
    - Random horizontal flip (p=0.5)
    - Random rotation (±15°)
    - Random brightness and contrast adjustment
    - Random resized crop to 224×224
    - ImageNet normalization

    Returns:
        Composed torchvision transforms for training.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Validation/test data transforms (no augmentation).

    Returns:
        Composed torchvision transforms for validation/testing.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(data_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Load dataset and split into train/val/test sets.

    Args:
        data_dir: Path to the root dataset directory with class subfolders.
        train_ratio: Fraction of data for training (default 0.8).
        val_ratio: Fraction of data for validation (default 0.1).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names, class_to_idx).
    """
    # Load full dataset to get class info
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    class_to_idx = full_dataset.class_to_idx

    logger.info(f"Dataset loaded from: {data_dir}")
    logger.info(f"Total images: {len(full_dataset)}")
    logger.info(f"Classes found ({len(class_names)}): {class_names}")

    # Verify class count matches expected
    if len(class_names) != NUM_CLASSES:
        logger.warning(
            f"Expected {NUM_CLASSES} classes, found {len(class_names)}. "
            "Proceeding with found classes."
        )

    # Split dataset
    total = len(full_dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"Split: train={train_size}, val={val_size}, test={test_size}")

    # Apply transforms (need to override the transform for each subset)
    train_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=get_train_transforms())
    val_dataset_full = datasets.ImageFolder(root=data_dir, transform=get_val_transforms())

    # Re-split with transforms applied
    train_dataset_aug = torch.utils.data.Subset(train_dataset.dataset, train_dataset.indices)
    val_dataset_aug = torch.utils.data.Subset(val_dataset_full, val_dataset.indices)
    test_dataset_aug = torch.utils.data.Subset(val_dataset_full, test_dataset.indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=args.batch_size if 'args' in dir() else 32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset_aug,
        batch_size=args.batch_size if 'args' in dir() else 32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset_aug,
        batch_size=args.batch_size if 'args' in dir() else 32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names, class_to_idx


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Build ResNet18 with pretrained ImageNet weights and modified FC layer.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze early layers initially.

    Returns:
        Modified ResNet18 model.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all layers except layer4 and fc (if requested)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen (except layer4 and fc).")

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    logger.info(f"FC layer replaced: {in_features} → {num_classes}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> tuple:
    """Train for one epoch.

    Args:
        model: The CNN model.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        epoch: Current epoch number.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """Validate the model.

    Args:
        model: The CNN model.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    class_names: list,
) -> dict:
    """Evaluate the model on the test set and produce a classification report.

    Args:
        model: Trained CNN model.
        test_loader: Test data loader.
        device: Compute device.
        class_names: List of class names.

    Returns:
        Dictionary with per-class and overall metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute per-class metrics
    from collections import defaultdict
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})

    for pred, label in zip(all_preds, all_labels):
        label_name = class_names[label] if label < len(class_names) else str(label)
        pred_name = class_names[pred] if pred < len(class_names) else str(pred)

        class_metrics[label_name]["total"] += 1
        if pred == label:
            class_metrics[label_name]["tp"] += 1
        else:
            class_metrics[label_name]["fn"] += 1
            class_metrics[pred_name]["fp"] += 1

    # Print classification report
    overall_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    overall_total = len(all_labels)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

    report = {
        "overall_accuracy": overall_accuracy,
        "total_samples": overall_total,
        "per_class": {},
    }

    print("\n" + "=" * 75)
    print("CLASSIFICATION REPORT")
    print("=" * 75)
    print(f"{'Class':<50} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 75)

    for name in class_names:
        m = class_metrics[name]
        precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0
        recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report["per_class"][name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": m["total"],
        }
        print(f"{name:<50} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

    print("-" * 75)
    print(f"{'Overall Accuracy':<50} {'':<10} {'':<10} {overall_accuracy:>10.4f}")
    print(f"Total samples: {overall_total}")
    print("=" * 75)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_torchscript(model: nn.Module, save_path: str, device: str) -> None:
    """Export the trained model as TorchScript for edge inference.

    Args:
        model: Trained PyTorch model.
        save_path: Path to save the TorchScript model.
        device: Device the model is on.
    """
    model.eval()
    model_cpu = model.to("cpu")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    try:
        traced_model = torch.jit.trace(model_cpu, dummy_input)
        traced_model.save(save_path)
        logger.info(f"TorchScript model exported to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to export TorchScript: {e}")

    # Move model back to original device
    model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    """Main training function.

    Orchestrates the full training pipeline: data loading, model building,
    training with early stopping, evaluation, and model export.

    Args:
        args: Command-line arguments.
    """
    # ── Device setup ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # ── Ensure output directory exists ────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────
    logger.info(f"Loading dataset from: {args.data_dir}")
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=get_train_transforms())
    class_names = full_dataset.classes
    num_classes = len(class_names)

    logger.info(f"Found {len(full_dataset)} images in {num_classes} classes")

    # Split the dataset
    total = len(full_dataset)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    test_size = total - train_size - val_size

    train_indices, val_indices, test_indices = random_split(
        range(total),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create datasets with appropriate transforms
    train_dataset = datasets.ImageFolder(root=args.data_dir, transform=get_train_transforms())
    val_dataset = datasets.ImageFolder(root=args.data_dir, transform=get_val_transforms())
    test_dataset = datasets.ImageFolder(root=args.data_dir, transform=get_val_transforms())

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    logger.info(f"Data split: train={train_size}, val={val_size}, test={test_size}")

    # ── Build model ───────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes, freeze_backbone=args.freeze_backbone)
    model = model.to(device)
    logger.info("Model built and moved to device.")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True,
    )

    # ── Training loop with early stopping ─────────────────────────────────
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Early stopping patience: {args.patience}")
    print()

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Unfreeze all layers after warmup epochs
        if epoch == args.unfreeze_epoch and args.freeze_backbone:
            logger.info(f"Unfreezing all layers at epoch {epoch}.")
            for param in model.parameters():
                param.requires_grad = True
            # Update optimizer to include all parameters with lower LR
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate * 0.1, weight_decay=1e-4)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"  ✓ New best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience={args.patience})")
                break

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s ({total_time / 60:.1f} min)")

    # ── Load best model and save ──────────────────────────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model weights.")

    # Save PyTorch weights
    weights_path = str(MODELS_DIR / "plant_disease_model.pth")
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Model weights saved to: {weights_path}")

    # Export TorchScript
    torchscript_path = str(MODELS_DIR / "plant_disease_model_scripted.pt")
    export_torchscript(model, torchscript_path, device)

    # Save class mapping
    class_mapping_path = str(MODELS_DIR / "class_mapping.json")
    mapping = {
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "idx_to_class": {str(idx): name for idx, name in enumerate(class_names)},
        "num_classes": num_classes,
    }
    with open(class_mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Class mapping saved to: {class_mapping_path}")

    # Save training history
    history_path = str(MODELS_DIR / "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")

    # ── Evaluate on test set ──────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    report = evaluate_test(model, test_loader, device, class_names)

    # Save report
    report_path = str(MODELS_DIR / "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved to: {report_path}")

    print("\n✓ Training complete!")
    print(f"  Model weights: {weights_path}")
    print(f"  TorchScript:   {torchscript_path}")
    print(f"  Class mapping: {class_mapping_path}")
    print(f"  Test accuracy: {report['overall_accuracy']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ResNet18 plant disease classifier on PlantVillage dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to dataset root directory with class subfolders.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Initial learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (number of epochs without improvement).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", default=True,
        help="Freeze backbone layers initially (unfreeze at --unfreeze_epoch).",
    )
    parser.add_argument(
        "--no_freeze", action="store_true", default=False,
        help="Train all layers from the start.",
    )
    parser.add_argument(
        "--unfreeze_epoch", type=int, default=5,
        help="Epoch at which to unfreeze all layers (if backbone is frozen).",
    )
    args = parser.parse_args()

    if args.no_freeze:
        args.freeze_backbone = False

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
