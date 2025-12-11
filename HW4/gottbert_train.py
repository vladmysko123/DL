import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from torch.cuda.amp import autocast, GradScaler

MODEL_NAME = "uklfr/gottbert-base"
MAX_LEN = 256
BATCH_SIZE = 8
ACCUM_STEPS = 4
LR = 2e-5
EPOCHS = 10

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
    texts, labels, name_list = [], [], []

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(row["label"].strip())
            name_list.append(row["label"].strip())

    classes = sorted(list(set(name_list)))
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

    print("Loading ALL training data...")
    texts, y, class_names = load_train(TRAIN_PATH)

    train_ds = TextDataset(texts, y, tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    print("Loading GottBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names)
    ).to(DEVICE)

    model.config.label2id = {c: i for i, c in enumerate(class_names)}
    model.config.id2label = {i: c for i, c in enumerate(class_names)}

    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = len(train_loader) * EPOCHS // ACCUM_STEPS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = GradScaler()

    print("\n============ TRAINING ON FULL DATA ============\n")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
        print(f"Train Loss: {train_loss:.4f}")

    print("\nTraining complete.")

    os.makedirs("models", exist_ok=True)
    output_dir = "models/gottbert_full"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved → {output_dir}")
    print("Use this model for final Kaggle submission.")

if __name__ == "__main__":
    set_seed()
    main()
