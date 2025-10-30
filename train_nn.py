import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report
import joblib

dataset = fetch_20newsgroups(subset='all')
X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=30000,  
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

svd = TruncatedSVD(n_components=200, random_state=42)  
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

for epoch in range(50):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        preds = logits.argmax(dim=1)
        acc = (preds.cpu() == yte).float().mean().item()

    print(f"Epoch {epoch+1:02d} | test acc={acc:.4f}")

    if acc > best_acc + 1e-4:
        best_acc, wait = acc, 0
        torch.save(model.state_dict(), "mlp_svd_best.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stop")
            break

model.load_state_dict(torch.load("mlp_svd_best.pt", map_location=device))
model.eval()
with torch.no_grad():
    logits = model(Xte.to(device))
    preds = logits.argmax(dim=1).cpu().numpy()

print(f"MLP test accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds, digits=3))

joblib.dump(tfidf, "tfidf_20ng.joblib")
joblib.dump(svd,   "svd_20ng_200.joblib")
torch.save(model.state_dict(), "mlp_svd_best.pt")

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
