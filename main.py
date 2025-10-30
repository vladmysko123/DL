import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 1) Load & split
dataset = fetch_20newsgroups(subset='all')
X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) TF-IDF (smaller for speed)
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=30000,     # ↓ from 50k
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# 3) SVD (LSA) — smaller for speed
svd = TruncatedSVD(n_components=150, random_state=42)  # ↓ from 300
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd  = svd.transform(X_test_tfidf)

# 4) DataFrames for PyCaret
cols = [f"svd_{i}" for i in range(X_train_svd.shape[1])]
train_df = pd.DataFrame(X_train_svd, columns=cols); train_df["label"] = y_train
test_df  = pd.DataFrame(X_test_svd,  columns=cols); test_df["label"]  = y_test

# 5) PyCaret — limit models + folds + time
from pycaret.classification import (
    setup, compare_models, tune_model, finalize_model, predict_model, save_model
)

clf = setup(
    data=train_df,
    target="label",
    session_id=42,
    fold=3,                 # fewer CV folds
    normalize=True,
    html=False,             # less UI overhead
    log_experiment=False,   # no MLflow logging
    verbose=True
)

# Fast, strong shortlist (skip very slow models like SVM/KNN/QDA/LDA/GPC)
# If lightgbm/xgboost aren't installed, remove them from the list.
include_models = ["lr", "ridge", "nb", "rf", "lightgbm", "xgboost"]

best = compare_models(
    include=include_models,
    n_select=1,
    turbo=False,          
    budget_time=300         
)

# Short, bounded tuning (optional)
best_tuned = tune_model(
    best,
    optimize="Accuracy",
    n_iter=20,              # small search
    choose_better=True
)

final_model = finalize_model(best_tuned)

# 6) Test evaluation
test_preds = predict_model(final_model, data=test_df)
print(test_preds.head())

# 7) Save
save_model(final_model, "best_20ng_pycaret")
