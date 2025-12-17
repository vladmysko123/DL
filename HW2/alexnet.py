import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import AlexNet_Weights

from training_utils import DataConfig, prepare_dataloaders, set_seed, train_model

# ---------------- Config ----------------
DATA_DIR = Path("lnu-butterflies") / "train"
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = OUT_DIR / "alexnet_butterflies_best.pth"

BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
VAL_SPLIT = 0.1
RANDOM_SEED = 42
NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "butterflies-classification"


def build_model(num_classes: int) -> nn.Module:
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=weights)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def main() -> None:
    set_seed(RANDOM_SEED)

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    data_cfg = DataConfig(
        data_dir=DATA_DIR,
        val_split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    train_loader, val_loader, class_to_idx, classes = prepare_dataloaders(
        data_cfg, train_transform, val_transform, seed=RANDOM_SEED
    )
    num_classes = len(classes)
    print("Classes:", classes)

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="alexnet"):
        mlflow.log_params(
            {
                "model": "alexnet",
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "learning_rate": LR,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "val_split": VAL_SPLIT,
                "num_classes": num_classes,
                "optimizer": "SGD",
                "scheduler": "StepLR",
                "pretrained": True,
            }
        )

        best_val_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            best_model_path=BEST_MODEL_PATH,
            class_to_idx=class_to_idx,
            model_name="alexnet",
        )

        mlflow.pytorch.log_model(model, artifact_path="model")
        print("Training finished")
        print("Best val_acc:", best_val_acc)
        print("Model saved to:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
