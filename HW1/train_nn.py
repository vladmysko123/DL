import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

# === MLflow bits ===
import mlflow
import mlflow.pytorch

# (optional) if you run a tracking server/UI (e.g., mlflow server on 127.0.0.1:5000):
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("20ng-mlp-svd")  # auto-creates experiment if missing

dataset = fetch_20newsgroups(subset='all')
X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=30000,  # <-- logged below
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

svd = TruncatedSVD(n_components=200, random_state=42)  # <-- logged below
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd  = svd.transform(X_test_tfidf)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(np.unique(y_train))
in_dim = X_train_svd.shape[1]

Xtr = torch.tensor(X_train_svd, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.long)
Xte = torch.tensor(X_test_svd,  dtype=torch.float32)
yte = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=512, shuffle=False)

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(in_dim, num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

best_acc, patience, wait = 0.0, 5, 0

# === Start an MLflow run ===
with mlflow.start_run(run_name="mlp-svd-baseline"):

    # ---- log static params once ----
    mlflow.log_params({
        "tfidf_stop_words": "english",
        "tfidf_max_features": 30000,
        "tfidf_sublinear_tf": True,
        "svd_n_components": 200,
        "model_hidden_1": 512,
        "model_hidden_2": 256,
        "dropout": 0.3,
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "patience": patience,
        "device": str(device),
        "num_classes": int(num_classes),
        "input_dim": int(in_dim),
        "random_state": 42,
        "test_size": 0.2,
    })

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # evaluate
        model.eval()
        with torch.no_grad():
            logits = model(Xte.to(device))
            preds = logits.argmax(dim=1)
            acc = (preds.cpu() == yte).float().mean().item()

        print(f"Epoch {epoch+1:02d} | test acc={acc:.4f}")

        # log per-epoch metrics
        mlflow.log_metric("test_accuracy", acc, step=epoch+1)

        if acc > best_acc + 1e-4:
            best_acc, wait = acc, 0
            torch.save(model.state_dict(), "mlp_svd_best.pt")
            mlflow.log_metric("best_accuracy", best_acc, step=epoch+1)
            # keep a copy of the best weights as an artifact
            mlflow.log_artifact("mlp_svd_best.pt", artifact_path="checkpoints")
        else:
            wait += 1
            if wait >= patience:
                print("Early stop")
                break

    # Final evaluation & rich logging
    # (Use safe weights_only=True when available; fallback for older torch)
    try:
        state = torch.load("mlp_svd_best.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load("mlp_svd_best.pt", map_location=device)
    model.load_state_dict(state)

    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()

    final_acc = accuracy_score(y_test, preds)
    print(f"MLP test accuracy: {final_acc:.4f}")
    mlflow.log_metric("final_test_accuracy", final_acc)

    # classification report as a text artifact
    report_str = classification_report(y_test, preds, target_names=dataset.target_names, digits=3)
    print(report_str)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)
    mlflow.log_artifact("classification_report.txt", artifact_path="reports")

    # save & log preprocessors (log explicit files to avoid recursive/long paths)
    joblib.dump(tfidf, "tfidf_20ng.joblib")
    joblib.dump(svd,   "svd_20ng_200.joblib")
    mlflow.log_artifact("tfidf_20ng.joblib",   artifact_path="preprocessing")
    mlflow.log_artifact("svd_20ng_200.joblib", artifact_path="preprocessing")

    # log the full PyTorch model (for later mlflow models serve)
    # Note: this logs the model with its state dict; preprocessing is logged separately above.
    mlflow.pytorch.log_model(model, artifact_path="model",
                             registered_model_name=None)  # keep local; no registry needed

# Quick sanity prediction (not part of the MLflow run)
def predict_texts(texts):
    X = tfidf.transform(texts)
    X = svd.transform(X)
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(Xt)
        labels = logits.argmax(dim=1).cpu().numpy().tolist()
    return [dataset.target_names[i] for i in labels]

print(predict_texts([
    "GPU driver fails on my Mac laptop",
    "Theology debate about atheism and religion",
]))
