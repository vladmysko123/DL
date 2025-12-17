import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

import mlflow
import mlflow.pytorch

# ---------- Config ----------
DATA_DIR = Path("lnu-butterflies")
TRAIN_DIR = DATA_DIR / "train"

BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
VAL_SPLIT = 0.1
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Output
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = OUT_DIR / "resnet18_butterflies_best.pth"
best_val_acc = 0.0

# ---------- MLflow ----------
MLFLOW_TRACKING_URI = "file:D:/dl/hw2/mlruns"   # CHANGE if needed
EXPERIMENT_NAME = "resnet18-butterflies"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ---------- Reproducibility ----------
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---------- Transforms ----------
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

# ---------- Dataset ----------
full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
num_classes = len(full_dataset.classes)
print("Classes:", full_dataset.classes)

n_total = len(full_dataset)
n_val = int(n_total * VAL_SPLIT)
n_train = n_total - n_val

train_dataset, val_split = random_split(
    full_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)

val_dataset = Subset(
    datasets.ImageFolder(TRAIN_DIR, transform=val_transform),
    val_split.indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ---------- Model ----------
try:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
except Exception:
    model = resnet18(pretrained=True)

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
def train_one_epoch(epoch):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0
    start = time.time()

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += images.size(0)

    loss = running_loss / total
    acc = running_corrects / total
    print(f"[Train] Epoch {epoch+1}: Loss={loss:.4f} Acc={acc:.4f} Time={time.time()-start:.1f}s")
    return loss, acc


def eval_one_epoch(epoch):
    model.eval()
    running_loss, running_corrects, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += images.size(0)

    loss = running_loss / total
    acc = running_corrects / total
    print(f"[Val]   Epoch {epoch+1}: Loss={loss:.4f} Acc={acc:.4f}")
    return loss, acc


# ---------- Train ----------
if __name__ == "__main__":
    with mlflow.start_run():

        mlflow.log_params({
            "model": "resnet18",
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LR,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "val_split": VAL_SPLIT,
            "num_classes": num_classes,
            "optimizer": "SGD",
            "scheduler": "StepLR",
        })

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(epoch)
            val_loss, val_acc = eval_one_epoch(epoch)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": full_dataset.class_to_idx,
                }, BEST_MODEL_PATH)

                mlflow.log_artifact(str(BEST_MODEL_PATH))
                mlflow.log_metric("best_val_acc", best_val_acc)
                print(f"✓ New best model saved (val_acc={best_val_acc:.4f})")

        mlflow.pytorch.log_model(model, artifact_path="model")

        print("Training finished")
        print("Best val_acc:", best_val_acc)
        print("Best model saved to:", BEST_MODEL_PATH)
