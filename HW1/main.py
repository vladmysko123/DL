# ===== optional sklearn shim if you use sklearn>=1.4 with older PyCaret =====
# from contextlib import contextmanager
# import sklearn.utils  # type: ignore
# @contextmanager
# def _noop_timer(*args, **kwargs): yield
# if not hasattr(sklearn.utils, "_print_elapsed_time"):
#     sklearn.utils._print_elapsed_time = _noop_timer  # type: ignore[attr-defined]
# =============================================================================

import os, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import mlflow

EXPERIMENT_NAME = "20NG-PyCaret"
mlflow.set_experiment(EXPERIMENT_NAME)

ART_DIR = Path("artifacts"); ART_DIR.mkdir(exist_ok=True)

def close_all_runs():
    while mlflow.active_run() is not None:
        mlflow.end_run()

close_all_runs()

dataset = fetch_20newsgroups(subset='all')
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(stop_words='english', max_features=30000, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

svd = TruncatedSVD(n_components=150, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd  = svd.transform(X_test_tfidf)

joblib.dump(tfidf, ART_DIR / "tfidf_20ng.joblib")
joblib.dump(svd,   ART_DIR / "svd_20ng_150.joblib")

cols = [f"svd_{i}" for i in range(X_train_svd.shape[1])]
train_df = pd.DataFrame(X_train_svd, columns=cols); train_df["label"] = y_train
test_df  = pd.DataFrame(X_test_svd,  columns=cols); test_df["label"]  = y_test

def _filter_available(models):
    avail = set(models)
    try:
        import lightgbm 
    except Exception:
        avail.discard("lightgbm")
    try:
        import xgboost  
    except Exception:
        avail.discard("xgboost")
    return list(avail)

include_models = _filter_available(["lr", "ridge", "nb", "rf", "lightgbm", "xgboost"])

from pycaret.classification import (
    setup, compare_models, tune_model, finalize_model, predict_model, save_model, pull
)

clf = setup(
    data=train_df,
    target="label",
    session_id=42,
    fold=3,
    html=False,
    log_experiment=True,
    experiment_name=EXPERIMENT_NAME,
    experiment_custom_tags={"dataset":"20newsgroups","features":"tfidf+svd"},
    log_plots=True,
    log_profile=False,
    log_data=False,
    verbose=True
)

best = compare_models(
    include=include_models,
    n_select=1,
    turbo=False,
    budget_time=300
)

leaderboard = pull()
lb_path = ART_DIR / "leaderboard.csv"
leaderboard.to_csv(lb_path, index=False)

best_tuned = tune_model(best, optimize="Accuracy", n_iter=20, choose_better=True)
final_model = finalize_model(best_tuned)

test_preds = predict_model(final_model, data=test_df)
pred_path = ART_DIR / "test_predictions_head.csv"
test_preds.head(50).to_csv(pred_path, index=False)

save_model(final_model, str(ART_DIR / "best_20ng_pycaret"))

close_all_runs()

with mlflow.start_run(run_name="extra_artifacts_attach"):
    mlflow.log_params({
        "vectorizer":"tfidf",
        "tfidf_stop_words":"english",
        "tfidf_max_features":30000,
        "tfidf_sublinear_tf":True,
        "svd_n_components":150,
        "test_size":0.2,
        "random_state":42
    })
    mlflow.log_artifact(str(ART_DIR / "tfidf_20ng.joblib"), artifact_path="preprocessing")
    mlflow.log_artifact(str(ART_DIR / "svd_20ng_150.joblib"), artifact_path="preprocessing")
    mlflow.log_artifact(str(lb_path), artifact_path="reports")
    mlflow.log_artifact(str(pred_path), artifact_path="reports")
    mlflow.log_artifact(str(ART_DIR / "best_20ng_pycaret.pkl"), artifact_path="model_pickles")

print("Done. In MLflow experiment:", EXPERIMENT_NAME)
print("- PyCaret runs for compare/tune/finalize (with metrics/plots)")
print("- Extra run 'extra_artifacts_attach' with preprocessing & leaderboard")
