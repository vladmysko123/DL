from pathlib import Path

import mlflow
import mlflow.pytorch
import timm
import torch
import torch.nn as nn
from timm.optim.optim_factory import create_optimizer_v2
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchvision import transforms

from training_utils import DataConfig, prepare_dataloaders, set_seed, train_model

DATA_DIR = Path("lnu-butterflies") / "train"
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = OUT_DIR / "convnext_base_fast_best.pth"

BATCH_SIZE = 8
NUM_EPOCHS = 10
VAL_SPLIT = 0.1
RANDOM_SEED = 42
IMG_SIZE = 192
BASE_LR = 5e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 1
NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
print("Using device:", DEVICE)

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "butterflies-classification"


def build_model(num_classes: int) -> nn.Module:
    model = timm.create_model("convnext_base", num_classes=num_classes, pretrained=True)
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    return model.to(DEVICE, memory_format=torch.channels_last)


def main() -> None:
    set_seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
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
            transforms.CenterCrop(IMG_SIZE),
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
        persistent_workers=True,
    )
    train_loader, val_loader, class_to_idx, classes = prepare_dataloaders(
        data_cfg, train_transform, val_transform, seed=RANDOM_SEED
    )
    num_classes = len(classes)
    print("Classes:", classes)

    model = build_model(num_classes)

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

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="convnext_base"):
        mlflow.log_params(
            {
                "model": "convnext_base",
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
            model_name="convnext_base",
            use_amp=USE_AMP,
            amp_dtype=torch.bfloat16,
            channels_last=True,
            extra_state={"img_size": IMG_SIZE, "frozen_backbone": True},
        )

        mlflow.pytorch.log_model(model, artifact_path="model")
        print("Training finished")
        print("Best val_acc:", best_val_acc)
        print("Best model saved to:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
