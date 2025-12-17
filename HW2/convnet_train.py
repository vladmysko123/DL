import os
from pathlib import Path
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from timm.optim.optim_factory import create_optimizer_v2
from timm.scheduler.cosine_lr import CosineLRScheduler

import mlflow
import mlflow.pytorch

# -------------------------------------------------
# Config
# -------------------------------------------------
DATA_DIR = Path("lnu-butterflies")
TRAIN_DIR = DATA_DIR / "train"

BATCH_SIZE = 8                 # Safe for RTX 3050 (4GB)
NUM_EPOCHS = 10
VAL_SPLIT = 0.1
RANDOM_SEED = 42

IMG_SIZE = 192                 # HUGE speedup vs 224
BASE_LR = 5e-4                 # Higher LR for head-only training
MIN_LR = 1e-6
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = OUT_DIR / "convnext_base_fast_best.pth"
best_val_acc = 0.0

# ---------------- MLflow ----------------
MLFLOW_TRACKING_URI = "file:D:/dl/hw2/mlruns"
EXPERIMENT_NAME = "convnext-base-fast"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# -------------------------------------------------
# Repro
# -------------------------------------------------
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.benchmark = True

USE_AUTOCast = DEVICE == "cuda"
AUTOCAST_KW = dict(device_type="cuda", dtype=torch.bfloat16)

# -------------------------------------------------
# Transforms (192x192)
# -------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -------------------------------------------------
# Dataset
# -------------------------------------------------
base_dataset = datasets.ImageFolder(TRAIN_DIR)
num_classes = len(base_dataset.classes)
print("Classes:", base_dataset.classes)

n_total = len(base_dataset)
n_val = int(n_total * VAL_SPLIT)
n_train = n_total - n_val

train_subset, val_subset = random_split(
    base_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = self.transform(img)
        return img, label


train_dataset = TransformDataset(train_subset, train_transform)
val_dataset = TransformDataset(val_subset, val_transform)

NUM_WORKERS = 2   # Faster on Windows laptops

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

# -------------------------------------------------
# Model (ConvNeXt-Base, FROZEN)
# -------------------------------------------------
model_name = "convnext_base"

model = timm.create_model(
    model_name,
    num_classes=num_classes,
    pretrained=True,
)

# 🔥 Freeze backbone (CRITICAL)
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

model = model.to(DEVICE, memory_format=torch.channels_last)
print("Model:", model_name, "| Backbone frozen")

# -------------------------------------------------
# Loss, Optimizer, Scheduler
# -------------------------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = create_optimizer_v2(
    filter(lambda p: p.requires_grad, model.parameters()),
    opt="adamw",
    lr=BASE_LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = CosineLRScheduler(
    optimizer,
    t_initial=NUM_EPOCHS,
    lr_min=MIN_LR,
    warmup_t=WARMUP_EPOCHS,
    warmup_lr_init=BASE_LR * 0.1,
    t_in_epochs=True,
)

# -------------------------------------------------
# Train / Eval
# -------------------------------------------------
def train_one_epoch(epoch):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    start = time.time()

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(**AUTOCAST_KW):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        loss_sum += loss.item() * images.size(0)
        correct += torch.sum(preds == labels).item()
        total += images.size(0)

    loss = loss_sum / total
    acc = correct / total
    print(f"[Train] Epoch {epoch+1}: Loss={loss:.4f} Acc={acc:.4f} Time={time.time()-start:.1f}s")
    return loss, acc


def eval_one_epoch(epoch):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    with torch.no_grad(), torch.autocast(**AUTOCAST_KW):
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss_sum += loss.item() * images.size(0)
            correct += torch.sum(preds == labels).item()
            total += images.size(0)

    loss = loss_sum / total
    acc = correct / total
    print(f"[Val]   Epoch {epoch+1}: Loss={loss:.4f} Acc={acc:.4f}")
    return loss, acc


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    with mlflow.start_run():

        mlflow.log_params({
            "model": model_name,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "base_lr": BASE_LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS,
            "optimizer": "AdamW",
            "scheduler": "CosineLR",
            "frozen_backbone": True,
            "num_classes": num_classes,
        })

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(epoch)
            val_loss, val_acc = eval_one_epoch(epoch)

            scheduler.step(epoch + 1)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_to_idx": base_dataset.class_to_idx,
                        "model_name": model_name,
                        "img_size": IMG_SIZE,
                        "frozen_backbone": True,
                    },
                    BEST_MODEL_PATH,
                )
                mlflow.log_artifact(str(BEST_MODEL_PATH))
                mlflow.log_metric("best_val_acc", best_val_acc)
                print(f"✓ New best model saved (val_acc={best_val_acc:.4f})")

        mlflow.pytorch.log_model(model, artifact_path="model")

        print("Training finished")
        print("Best val_acc:", best_val_acc)
        print("Best model saved to:", BEST_MODEL_PATH)
