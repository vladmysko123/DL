#!/usr/bin/env python
# cv_eval_from_checkpoint_all.py
#
# This script DOES NOT retrain models.
# It:
#   1) Finds all saved "best" models in models/ folder.
#   2) For each checkpoint: loads the model and evaluates it on K stratified folds
#      of the training data (k-fold evaluation of a fixed model).
#   3) Prints a summary table with mean accuracy per model.
#   4) Prints a summary of training hyperparameters used earlier.
#
# Important: This is NOT true k-fold cross-validation in the training sense,
# because the model weights are fixed and not re-trained per fold.

from pathlib import Path
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# -------------------------------------------------
#  Config
# -------------------------------------------------
DATA_DIR = Path("lnu-butterflies")
TRAIN_CSV = DATA_DIR / "train.csv"
CLASSES_TXT = DATA_DIR / "classes.txt"
MODELS_DIR = Path("models")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------------------------------------
#  Data / transforms
# -------------------------------------------------
train_csv = pd.read_csv(TRAIN_CSV)
classes = [line.strip() for line in CLASSES_TXT.read_text().splitlines() if line.strip()]
num_classes = len(classes)
print(f"Loaded {len(train_csv)} train samples, {num_classes} classes")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ButterflyDataset(Dataset):
    """
    Uses train.csv structure:
        id, path, label
    where:
        - path is relative to DATA_DIR
        - label is class name (e.g. "MONARCH")
    """
    def __init__(self, df: pd.DataFrame, root_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_str = row["label"]
        label_idx = classes.index(label_str)
        return img, label_idx


def make_loader(df: pd.DataFrame, batch_size: int = 32):
    ds = ButterflyDataset(df, DATA_DIR, transform=val_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


# -------------------------------------------------
#  Load model from checkpoint (no training)
# -------------------------------------------------
def build_arch(arch: str, num_classes: int):
    arch = arch.lower()

    if arch == "alexnet":
        try:
            weights = models.AlexNet_Weights.IMAGENET1K_V1
            model = models.alexnet(weights=weights)
        except Exception:
            model = models.alexnet(pretrained=True)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif arch == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
        except Exception:
            model = resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "convnext_tiny":
        model = timm.create_model(
            "convnext_tiny",
            pretrained=False,          # we’ll load our own weights
            num_classes=num_classes,
        )

    elif arch == "convnext_base":
        model = timm.create_model(
            "convnext_base",
            pretrained=False,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model


def infer_arch_from_filename(fname: str) -> str | None:
    """
    Very simple heuristic: detect architecture from checkpoint file name.
    """
    name = fname.lower()
    if "alexnet" in name:
        return "alexnet"
    if "resnet18" in name:
        return "resnet18"
    if "convnext_tiny" in name:
        return "convnext_tiny"
    # convnext base might be named convnext_base or just convnext_butterflies_best
    if "convnext_base" in name:
        return "convnext_base"
    if "convnext" in name and "tiny" not in name:
        return "convnext_base"
    return None


def load_trained_model(ckpt_path: Path, default_arch: str, num_classes: int):
    """
    Load a trained model from a checkpoint file.
    """
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    # Try to infer arch from checkpoint, otherwise use default
    arch = checkpoint.get("arch", checkpoint.get("model_name", default_arch))
    print(f"  checkpoint arch field: {arch} (default={default_arch})")

    model = build_arch(arch, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model, arch


# -------------------------------------------------
#  1) "K-fold cross-validation" evaluation of a fixed model
# -------------------------------------------------
def kfold_evaluate_fixed_model(
    model,
    n_splits: int = 3,
    batch_size: int = 32,
):
    """
    This evaluates a SINGLE fixed model over K stratified folds of the train data.
    There is NO retraining here. Strictly speaking this is k-fold evaluation,
    not true cross-validation training.
    """
    X = train_csv["path"].values
    y = train_csv["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_accs = []

    print(f"    === Fixed-model {n_splits}-fold evaluation ===")

    for fold, (_, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"    Fold {fold}/{n_splits}")

        # we only use the validation fold (since model is fixed)
        fold_val_df = train_csv.iloc[val_idx]
        val_loader = make_loader(fold_val_df, batch_size=batch_size)

        all_true = []
        all_pred = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_true.append(labels.cpu().numpy())
                all_pred.append(preds.cpu().numpy())

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        acc = accuracy_score(y_true, y_pred)
        fold_accs.append(acc)
        print(f"      accuracy on fold {fold}: {acc:.4f}")

    mean_acc = float(np.mean(fold_accs))
    print(f"    Fold accuracies: {[float(a) for a in fold_accs]}")
    print(f"    Mean K-fold evaluation accuracy: {mean_acc:.4f}")
    return fold_accs, mean_acc


# -------------------------------------------------
#  2) Hyperparameter "tuning" summary (no retraining)
# -------------------------------------------------
def hyperparam_summary():
    """
    This does NOT tune hyperparameters (no training).
    It just summarizes the hyperparameters that were used for different models
    based on your training scripts.

    You can show this table in the report as:
    - comparison of chosen hyperparameters
    - explanation of why convnext used smaller lr, etc.
    """
    rows = [
        # Model,      LR,      WeightDecay, BatchSize, Epochs,   Optimizer
        ["alexnet",        1e-3,   5e-4,        64,       10,      "SGD(momentum=0.9)"],
        ["resnet18",       1e-3,   5e-4,        64,       10,      "SGD(momentum=0.9)"],
        ["convnext_tiny",  1e-4,   1e-4,        16,       10,      "AdamW"],
        ["convnext_base",  1e-4,   1e-4,         4,       10,      "AdamW"],
        # YOLOv8m-cls is separate, trained via CLI
        ["yolov8m-cls",    None,   None,        16,       40,      "YOLO defaults"],
    ]
    df = pd.DataFrame(rows, columns=[
        "Model", "LR", "WeightDecay", "BatchSize", "Epochs", "Optimizer",
    ])
    print("\nHyperparameter summary (from training scripts):")
    print(df)
    return df


# -------------------------------------------------
#  Main
# -------------------------------------------------
if __name__ == "__main__":
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    ckpts = sorted(MODELS_DIR.glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No .pth files found in {MODELS_DIR}")

    print("\nFound checkpoints:")
    for c in ckpts:
        print(" -", c.name)

    eval_rows = []

    for ckpt_path in ckpts:
        fname = ckpt_path.name
        arch_guess = infer_arch_from_filename(fname)
        if arch_guess is None:
            print(f"\nSkipping {fname}: cannot infer architecture from name")
            continue

        print(f"\n================ {fname} ================")
        print(f"  inferred arch: {arch_guess}")

        model, arch_used = load_trained_model(ckpt_path, default_arch=arch_guess, num_classes=num_classes)
        fold_accs, mean_acc = kfold_evaluate_fixed_model(
            model,
            n_splits=3,
            batch_size=32,
        )

        eval_rows.append({
            "Checkpoint": fname,
            "Arch (inferred)": arch_guess,
            "Arch (ckpt)": arch_used,
            "Mean K-fold acc": mean_acc,
        })

    # Summary table for all checkpoints
    if eval_rows:
        eval_df = pd.DataFrame(eval_rows)
        print("\n\n=== Summary over all checkpoints ===")
        print(eval_df)

    # Hyperparameter info
    hp_df = hyperparam_summary()
