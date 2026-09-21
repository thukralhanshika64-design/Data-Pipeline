"""
PROJECT 5 - Twin Predictors
Same workflow, two problem types:
  * CLASSIFICATION  - will this customer churn?      (discrete target)  -> precision/recall/F1/AUC
  * REGRESSION      - what will this house sell for? (continuous target) -> MAE/RMSE/R2/MAPE
plus (3) the bridge between them: regress-then-threshold vs a direct classifier.

Run:  python projects/p05_twin_predictors/main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error, median_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.db import get_dataset, log_experiment, save_table

PROJECT = "p05_twin_predictors"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


# =========================================================================== #
# CLASSIFICATION
# =========================================================================== #
def clf_metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred), roc_auc=roc_auc_score(y, proba))


def classification():
    print("\n=== CLASSIFICATION: churn ===")
    df = get_dataset("customers")
    y = df["churned"]
    X = df[["tenure_months", "monthly_charges", "total_charges", "num_support_calls",
            "contract_type", "internet_service", "payment_method"]]
    num, cat = list(X.columns[:4]), list(X.columns[4:])
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)

    models = {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1),
    }
    fitted, rows = {}, []
    for name, m in models.items():
        pipe = Pipeline([("prep", prep), ("m", m)]).fit(Xtr, ytr)
        fitted[name] = pipe
        proba = pipe.predict_proba(Xte)[:, 1]
        met = clf_metrics(yte, proba)
        rows.append({"model": name, **met})
        log_experiment(PROJECT, f"clf_{name}", name, met, dataset_name="customers")
    print(pd.DataFrame(rows).set_index("model").round(3).to_string())

    best = fitted["LogisticRegression"]
    proba = best.predict_proba(Xte)[:, 1]
    print("\nThreshold matters! LogisticRegression at different decision thresholds:")
    for thr in (0.5, 0.35, 0.25):
        m = clf_metrics(yte, proba, thr)
        print(f"  thr={thr:<4} precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, thr in zip(ax, (0.5, 0.3)):
        ConfusionMatrixDisplay.from_predictions(yte, (proba >= thr).astype(int), ax=a, colorbar=False)
        a.set_title(f"Confusion matrix, threshold={thr}")
    fig.tight_layout(); fig.savefig(OUT / "clf_confusion.png", dpi=110); plt.close(fig)


# =========================================================================== #
# REGRESSION
# =========================================================================== #
def reg_metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return dict(MAE=mean_absolute_error(y, pred), RMSE=np.sqrt(mse), MedianAE=median_absolute_error(y, pred),
                R2=r2_score(y, pred), MAPE=mean_absolute_percentage_error(y, pred))


def regression():
    print("\n=== REGRESSION: house price ===")
    df = get_dataset("houses")
    y = df["price"]
    X = df.drop(columns=["price", "house_id"])
    num = [c for c in X.columns if c != "neighborhood"]
    prep = ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), ["neighborhood"])])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)

    models = {
        "Dummy (mean)": DummyRegressor(),
        "LinearRegression": LinearRegression(),
        "LinearReg on log(price)": TransformedTargetRegressor(LinearRegression(), func=np.log, inverse_func=np.exp),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        pipe = Pipeline([("prep", prep), ("m", m)]).fit(Xtr, ytr)
        fitted[name] = pipe
        met = reg_metrics(yte, pipe.predict(Xte))
        rows.append({"model": name, **met})
        log_experiment(PROJECT, f"reg_{name}", name, met, dataset_name="houses")
    tbl = pd.DataFrame(rows).set_index("model")
    fmt = tbl.copy()
    for c in ("MAE", "RMSE", "MedianAE"):
        fmt[c] = fmt[c].map("{:,.0f}".format)
    fmt["R2"] = fmt["R2"].round(3); fmt["MAPE"] = (fmt["MAPE"] * 100).round(1).astype(str) + "%"
    print(fmt.to_string())

    pred = fitted["RandomForest"].predict(Xte)
    err = yte.to_numpy() - pred
    print(f"\nRMSE/MAE ratio = {np.sqrt(np.mean(err**2)) / np.mean(np.abs(err)):.2f}  "
          "(1.25 ~ Gaussian errors; much higher => a few HUGE errors dominate RMSE)")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(yte, pred, s=6, alpha=0.5); lim = [0, yte.max()]
    ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("actual"); ax[0].set_ylabel("predicted")
    ax[0].set_title("Predicted vs actual (RandomForest)")
    ax[1].hist(err, bins=50); ax[1].set_title("Residuals"); ax[1].set_xlabel("actual - predicted")
    fig.tight_layout(); fig.savefig(OUT / "reg_diagnostics.png", dpi=110); plt.close(fig)

    out = Xte.copy(); out["actual"] = yte.values; out["predicted"] = pred.round(0)
    save_table(out.reset_index(drop=True), "house_predictions")
    return fitted["RandomForest"], Xte, yte


# =========================================================================== #
# BRIDGE: regression <-> classification
# =========================================================================== #
def bridge():
    print("\n=== BRIDGE: 'is the house expensive (top 25%)?' ===")
    df = get_dataset("houses")
    X = df.drop(columns=["price", "house_id"])
    cutoff = df["price"].quantile(0.75)
    y_cls = (df["price"] > cutoff).astype(int)
    num = [c for c in X.columns if c != "neighborhood"]
    prep = ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), ["neighborhood"])])
    Xtr, Xte, ytr, yte, ptr, pte = train_test_split(X, y_cls, df["price"], test_size=0.25, random_state=1,
                                                    stratify=y_cls)
    reg = Pipeline([("prep", prep), ("m", RandomForestRegressor(200, min_samples_leaf=3, random_state=0,
                                                                n_jobs=-1))]).fit(Xtr, ptr)
    clf = Pipeline([("prep", prep), ("m", RandomForestClassifier(200, min_samples_leaf=3, random_state=0,
                                                                 n_jobs=-1))]).fit(Xtr, ytr)
    auc_reg = roc_auc_score(yte, reg.predict(Xte))          # predicted price used directly as a score
    auc_clf = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    print(f"  regress-then-threshold AUC = {auc_reg:.3f}   direct classifier AUC = {auc_clf:.3f}")
    print("  Regression keeps MORE information (the magnitude), so it is often competitive - but the")
    print("  right choice depends on the decision: if you only need the yes/no, optimise for it directly.")


if __name__ == "__main__":
    classification()
    regression()
    bridge()
    print(f"\nPlots saved in {OUT}")
