import contextlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import mlflow
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DataConfig:
    data_dir: Path
    val_split: float
    batch_size: int
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False


def prepare_dataloaders(
    data_cfg: DataConfig,
    train_transform,
    val_transform,
    seed: int,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Iterable[str]]:
    """Create train/val loaders with distinct transforms and deterministic split."""
    base_dataset = datasets.ImageFolder(data_cfg.data_dir, transform=train_transform)
    n_total = len(base_dataset)
    n_val = int(n_total * data_cfg.val_split)
    n_train = n_total - n_val

    train_subset, val_split = random_split(
        base_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # Rebuild val dataset to apply the validation transform
    val_dataset = Subset(
        datasets.ImageFolder(data_cfg.data_dir, transform=val_transform),
        val_split.indices,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle_train,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.persistent_workers,
    )

    return train_loader, val_loader, base_dataset.class_to_idx, base_dataset.classes


def _format_batch(images, device: str, channels_last: bool):
    images = images.to(device, non_blocking=True)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    device: str,
    best_model_path: Path,
    class_to_idx: Dict[str, int],
    model_name: str,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
    extra_state: Optional[Dict] = None,
) -> float:
    """Generic training loop with MLflow logging and checkpointing."""
    best_val_acc = 0.0
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
    )

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        start = time.time()

        for images, labels in train_loader:
            images = _format_batch(images, device, channels_last)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss_sum += loss.item() * images.size(0)
            train_correct += torch.sum(preds == labels).item()
            train_total += images.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = _format_batch(images, device, channels_last)
                labels = labels.to(device, non_blocking=True)

                with autocast_ctx:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss_sum += loss.item() * images.size(0)
                val_correct += torch.sum(preds == labels).item()
                val_total += images.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        if scheduler is not None:
            if hasattr(scheduler, "t_initial"):
                scheduler.step(epoch + 1)
            else:
                scheduler.step()

        print(
            f"[{model_name}] Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"time={time.time()-start:.1f}s"
        )

        if mlflow.active_run():
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
            state = {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "model_name": model_name,
            }
            if extra_state:
                state.update(extra_state)
            torch.save(state, best_model_path)
            if mlflow.active_run():
                mlflow.log_artifact(str(best_model_path))
                mlflow.log_metric("best_val_acc", best_val_acc)
            print(f"New best model saved to {best_model_path} (val_acc={best_val_acc:.4f})")

    return best_val_acc
