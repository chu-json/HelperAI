#!/usr/bin/env python3
"""Train a ResNet-18 classifier on CT series mosaic PNGs.

The model predicts the metadata-derived body region label (`head` or `chest`)
from the mosaic images created by build_midrc_mosaic_dataset.py.

Inputs:
    outputs/mosaic_manifest.csv
        Manifest with SeriesInstanceUID, body_region, and mosaic_path columns.
        body_region is mapped as: head -> 0, chest -> 1.
    data/mosaics/*.png
        Mosaic PNGs referenced by the manifest's mosaic_path column.

Outputs under outputs/resnet18_head_chest/:
    config.json
        Training configuration for the run.
    label_mapping.json
        Numeric class mapping used by the model.
    splits.csv
        Train/validation/test assignment for each series.
    history.csv
        Per-epoch train loss and validation metrics.
    best_resnet18.pt
        Weights from the epoch with best validation balanced accuracy.
    final_resnet18.pt
        Weights after the final training epoch.
    test_metrics.json
        Test accuracy, balanced accuracy, classification report, and confusion
        matrix.
    test_predictions.csv
        Per-series test predictions.

Usage:
    python scripts/train_head_chest_resnet18.py --epochs 10
    python scripts/train_head_chest_resnet18.py --epochs 1 --no-pretrained
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logger = logging.getLogger("train_head_chest_resnet18")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "mosaic_manifest.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "resnet18_head_chest"
LABEL_TO_INDEX = {"head": 0, "chest": 1}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    manifest: str
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    val_fraction: float
    test_fraction: float
    pretrained: bool
    image_size: int
    num_workers: int
    device: str


class MosaicDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(row["resolved_mosaic_path"]).convert("RGB")
        label = torch.tensor(int(row["label_index"]), dtype=torch.long)
        return self.transform(image), label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_manifest_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"mosaic_path", "body_region", "SeriesInstanceUID"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")

    frame = manifest.copy()
    frame["body_region"] = frame["body_region"].astype(str).str.lower()
    frame = frame[frame["body_region"].isin(LABEL_TO_INDEX)].copy()
    frame["label_index"] = frame["body_region"].map(LABEL_TO_INDEX)
    frame["resolved_mosaic_path"] = frame["mosaic_path"].map(
        lambda value: str(resolve_manifest_path(str(value), REPO_ROOT))
    )
    missing_files = [
        path for path in frame["resolved_mosaic_path"].map(Path) if not path.exists()
    ]
    if missing_files:
        sample = ", ".join(str(path) for path in missing_files[:5])
        raise FileNotFoundError(
            f"{len(missing_files)} mosaic file(s) are missing; first missing: {sample}"
        )
    if frame["label_index"].nunique() != 2:
        raise ValueError("manifest must contain both head and chest labels")
    return frame.reset_index(drop=True)


def split_manifest(
    frame: pd.DataFrame,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be between 0 and 1")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    train_val, test = train_test_split(
        frame,
        test_size=test_fraction,
        random_state=seed,
        stratify=frame["label_index"],
    )
    val_size_of_train_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=val_size_of_train_val,
        random_state=seed,
        stratify=train_val["label_index"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(
        drop=True
    )


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_TO_INDEX))
    return model


def make_loader(
    frame: pd.DataFrame,
    transform: transforms.Compose,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = MosaicDataset(frame, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    labels = [0, 1]
    return {
        "loss": total_loss / max(len(y_true), 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[INDEX_TO_LABEL[index] for index in labels],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_examples += labels.size(0)
    return total_loss / max(total_examples, 1)


def write_splits(
    output_dir: Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    split_frame = pd.concat(
        [
            train.assign(split="train"),
            val.assign(split="val"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    columns = [
        "split",
        "SeriesInstanceUID",
        "body_region",
        "label_index",
        "mosaic_path",
        "resolved_mosaic_path",
    ]
    split_frame[columns].to_csv(output_dir / "splits.csv", index=False)


def write_predictions(
    output_dir: Path,
    test: pd.DataFrame,
    final_metrics: dict[str, Any],
) -> None:
    predictions = test.copy()
    predictions["true_label"] = final_metrics["y_true"]
    predictions["predicted_label"] = final_metrics["y_pred"]
    predictions["true_label_name"] = predictions["true_label"].map(INDEX_TO_LABEL)
    predictions["predicted_label_name"] = predictions["predicted_label"].map(
        INDEX_TO_LABEL
    )
    predictions[
        [
            "SeriesInstanceUID",
            "body_region",
            "true_label_name",
            "predicted_label_name",
            "mosaic_path",
        ]
    ].to_csv(output_dir / "test_predictions.csv", index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, mps, etc.",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Use ImageNet-pretrained ResNet-18 weights.",
    )
    pretrained_group.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Initialize ResNet-18 from scratch.",
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_manifest(args.manifest)
    train, val, test = split_manifest(
        frame,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    train_transform, eval_transform = build_transforms(args.image_size)
    train_loader = make_loader(
        train,
        train_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = make_loader(
        val,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        test,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = build_model(args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    config = TrainConfig(
        manifest=str(args.manifest),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        pretrained=args.pretrained,
        image_size=args.image_size,
        num_workers=args.num_workers,
        device=str(device),
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "label_mapping.json").write_text(
        json.dumps(LABEL_TO_INDEX, indent=2) + "\n",
        encoding="utf-8",
    )
    write_splits(args.output_dir, train, val, test)

    history = []
    best_val_balanced_accuracy = -1.0
    best_model_path = args.output_dir / "best_resnet18.pt"
    logger.info(
        "Training on %s with %d train / %d val / %d test examples",
        device,
        len(train),
        len(val),
        len(test),
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_metrics = evaluate(model, val_loader, device=device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        history.append(epoch_row)
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_acc=%.3f val_bal_acc=%.3f",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["balanced_accuracy"],
        )
        if val_metrics["balanced_accuracy"] > best_val_balanced_accuracy:
            best_val_balanced_accuracy = val_metrics["balanced_accuracy"]
            torch.save(model.state_dict(), best_model_path)

    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    final_model_path = args.output_dir / "final_resnet18.pt"
    torch.save(model.state_dict(), final_model_path)

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device=device)
    metrics_for_json = {
        key: value
        for key, value in test_metrics.items()
        if key not in {"y_true", "y_pred"}
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(metrics_for_json, indent=2) + "\n",
        encoding="utf-8",
    )
    write_predictions(args.output_dir, test, test_metrics)

    print()
    print("=" * 72)
    print(f"  train examples       : {len(train)}")
    print(f"  val examples         : {len(val)}")
    print(f"  test examples        : {len(test)}")
    print(f"  device               : {device}")
    print(f"  pretrained           : {args.pretrained}")
    print(f"  test accuracy        : {test_metrics['accuracy']:.3f}")
    print(f"  test balanced acc    : {test_metrics['balanced_accuracy']:.3f}")
    print(f"  output dir           : {args.output_dir}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
