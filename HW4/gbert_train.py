import csv
import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

from torch.cuda.amp import autocast, GradScaler

MODEL_NAME = "deepset/gbert-base"
MAX_LEN = 256
BATCH_SIZE = 8
ACCUM_STEPS = 4     
LR = 3e-5
EPOCHS = 8

TRAIN_PATH = "dataset/train.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        item = {k: torch.tensor(v) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_train(path):
    texts, labels, label_names = [], [], []

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["text"].strip()
            l = row["label"].strip()
            texts.append(t)
            labels.append(l)
            label_names.append(l)

    classes = sorted(list(set(label_names)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = [class_to_idx[c] for c in labels]

    return texts, y, classes

def train_one_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with autocast():
            outputs = model(**batch)
            loss = outputs.loss / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * ACCUM_STEPS

    return total_loss / len(loader)

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading full training data...")
    texts, y, class_names = load_train(TRAIN_PATH)

    dataset = TextDataset(texts, y, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    print("Loading GBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names)
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )

    total_steps = len(loader) * EPOCHS // ACCUM_STEPS

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = GradScaler()

    for epoch in range(EPOCHS):
        print(f"\n===== FULL TRAIN Epoch {epoch + 1}/{EPOCHS} =====")
        train_loss = train_one_epoch(model, loader, optimizer, scheduler, scaler)
        print(f"Train Loss: {train_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    SAVE_DIR = "models/gbert_full"

    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f"\nFULL MODEL SAVED → {SAVE_DIR}")
    print("Now run the submission script!")


if __name__ == "__main__":
    set_seed()
    main()
