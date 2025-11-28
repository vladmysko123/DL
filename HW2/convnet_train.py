import os
from pathlib import Path
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from timm.optim.optim_factory import create_optimizer_v2
from timm.scheduler.cosine_lr import CosineLRScheduler

# -----------------------------------
# Config
# -----------------------------------
DATA_DIR = Path("lnu-butterflies")
TRAIN_DIR = DATA_DIR / "train"

BATCH_SIZE = 16            # convnext_tiny is still non-trivial; adjust if OOM
NUM_EPOCHS = 10            # you can increase later if needed
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# Optimizer / Scheduler config
BASE_LR = 5e-4             # base LR (works well with warmup + cosine)
MIN_LR = 1e-6              # final LR at the end of training
WEIGHT_DECAY = 0.05        # typical strong WD for ConvNeXt / ViT-style models
WARMUP_EPOCHS = 2          # warmup for first N epochs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = OUT_DIR / "convnext_base_butterflies_best.pth"
best_val_acc = 0.0

# -----------------------------------
# Repro
# -----------------------------------
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.benchmark = True
if DEVICE == "cuda":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

USE_AUTOCast = DEVICE == "cuda"
if USE_AUTOCast:
    AUTOCAST_KW = dict(device_type="cuda", dtype=torch.bfloat16)
else:
    AUTOCAST_KW = dict(device_type="cpu", dtype=torch.float32)

# -----------------------------------
# Transforms
# -----------------------------------
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
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
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -----------------------------------
# Dataset & split
# -----------------------------------
base_dataset = datasets.ImageFolder(TRAIN_DIR)
num_classes = len(base_dataset.classes)
print("Found classes:", base_dataset.classes)
print("num_classes:", num_classes)

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

NUM_WORKERS = 4  # if Windows complains, drop to 2 or 0

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)

# -----------------------------------
# ConvNeXt-Tiny from timm (pretrained)
# -----------------------------------
model_name = "convnext_tiny"

model = timm.create_model(
    model_name,
    num_classes=num_classes,
    pretrained=True,   # uses ImageNet pretrained weights
)

print("Creating model:", model_name, "pretrained=True")
model = model.to(DEVICE)
print("Model device:", next(model.parameters()).device)

# -----------------------------------
# Loss, Optimizer, Scheduler
# -----------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Better AdamW via timm (handles proper param groups, no-decay on norm/bias)
optimizer = create_optimizer_v2(
    model,
    opt='adamw',
    lr=BASE_LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
)

# Cosine LR with warmup (epoch-based)
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=NUM_EPOCHS,         # total number of epochs
    lr_min=MIN_LR,                # final LR
    warmup_t=WARMUP_EPOCHS,       # warmup length in epochs
    warmup_lr_init=BASE_LR * 0.1, # LR at start of warmup
    t_in_epochs=True,             # we step per epoch
)


def train_one_epoch(epoch: int):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    start = time.time()

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AUTOCast:
            with torch.autocast(**AUTOCAST_KW):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()

        # optional: gradient clipping for stability on small GPUs
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    print(
        f"[Train] Epoch {epoch+1}: "
        f"Loss={epoch_loss:.4f} Acc={epoch_acc:.4f} "
        f"Time={time.time()-start:.1f}s"
    )
    return epoch_loss, epoch_acc


def eval_one_epoch(epoch: int):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if USE_AUTOCast:
                with torch.autocast(**AUTOCAST_KW):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    print(f"[Val]   Epoch {epoch+1}: Loss={epoch_loss:.4f} Acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


if __name__ == "__main__":

    for epoch in range(NUM_EPOCHS):
        train_one_epoch(epoch)
        _, val_acc = eval_one_epoch(epoch)

        # step timm cosine scheduler with current epoch index (1-based)
        scheduler.step(epoch + 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": base_dataset.class_to_idx,
                    "model_name": model_name,
                },
                BEST_MODEL_PATH,
            )
            print(f"✓ New best model saved (val_acc={best_val_acc:.4f})")

    print("Training finished. Best val_acc:", best_val_acc)
    print("Best model saved to:", BEST_MODEL_PATH)
