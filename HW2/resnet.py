import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

# ---------- Config ----------
DATA_DIR = Path("lnu-butterflies")   # CHANGE if needed
TRAIN_DIR = DATA_DIR / "train"
CLASSES_TXT = DATA_DIR / "classes.txt"

BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
VAL_SPLIT = 0.1  # 10% of train as validation
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Where to save the best model
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = OUT_DIR / "alexnet_butterflies_best.pth"
best_val_acc = 0.0

# ---------- Reproducibility ----------
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---------- Transforms ----------
# Dataset says images are 224x224, but we still do Resize/Crop for safety & augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
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

# ---------- Dataset & split ----------
full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
num_classes = len(full_dataset.classes)
print("Found classes in train folder:", full_dataset.classes)
print("num_classes:", num_classes)

n_total = len(full_dataset)
n_val = int(n_total * VAL_SPLIT)
n_train = n_total - n_val
train_dataset, val_dataset = random_split(
    full_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)

# apply val_transform to validation subset (wrap with Subset)
val_dataset = Subset(
    datasets.ImageFolder(TRAIN_DIR, transform=val_transform),
    val_dataset.indices
)

# DataLoaders — num_workers=0 so Windows multiprocessing won't complain
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

# ---------- ResNet-18 model ----------
from torchvision.models import resnet18, ResNet18_Weights

try:
    weights = ResNet18_Weights.IMAGENET1K_V1  # ImageNet pretrained
    model = resnet18(weights=weights)
except Exception:
    model = resnet18(pretrained=True)

# Replace the final fully-connected layer for our num_classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ---------- Training helpers ----------
def train_one_epoch(epoch: int):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    start_time = time.time()

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    elapsed = time.time() - start_time

    print(f"[Train] Epoch {epoch+1}: "
          f"Loss={epoch_loss:.4f} Acc={epoch_acc:.4f} Time={elapsed:.1f}s")

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


# ---------- Train loop ----------
if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = eval_one_epoch(epoch)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": full_dataset.class_to_idx,
            }, BEST_MODEL_PATH)
            print(f"✓ New best model saved (val_acc={best_val_acc:.4f})")

    print("Training finished. Best val_acc:", best_val_acc)
    print("Best model saved to:", BEST_MODEL_PATH)
