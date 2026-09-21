"""
PROJECT 7 - Cross-Validation Playground
Six experiments that prove WHY and WHEN each validation strategy is needed.

  A. Hold-out is noisy      : one train/test split vs K-Fold on a small dataset
  B. Stratification         : KFold vs StratifiedKFold on 1% positives
  C. Time-series leakage    : shuffled KFold vs TimeSeriesSplit
  D. Group leakage          : KFold vs GroupKFold (many rows per customer)
  E. Pipeline leakage       : feature selection OUTSIDE vs INSIDE CV on pure noise
  F. Nested CV              : the optimism of "best score from the grid search"

Run:  python projects/p07_cv_playground/main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.db import get_dataset, log_experiment

PROJECT = "p07_cv_playground"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def header(t):
    print(f"\n{'=' * 4} {t} {'=' * (72 - len(t))}")


# --------------------------------------------------------------------------- #
def exp_a_holdout_noise():
    header("A. Hold-out estimate is noisy (small dataset)")
    df = get_dataset("customers").sample(500, random_state=0)
    X = df[["tenure_months", "monthly_charges", "num_support_calls"]]
    y = df["churned"]
    model = make_pipeline(StandardScaler(), LogisticRegression())
    holdout = []
    for seed in range(100):                        # 100 different "lucky/unlucky" splits
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        holdout.append(model.fit(Xtr, ytr).score(Xte, yte))
    holdout = np.array(holdout)
    cv5 = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    cv10 = cross_val_score(model, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=0))
    print(f"  single 80/20 hold-out over 100 seeds: mean={holdout.mean():.3f}  min={holdout.min():.3f}  "
          f"max={holdout.max():.3f}  (range = {holdout.max() - holdout.min():.3f})")
    print(f"  5-fold CV  : mean={cv5.mean():.3f} +- {cv5.std():.3f}      10-fold CV: mean={cv10.mean():.3f} +- {cv10.std():.3f}")
    print("  -> With a single split you could report anywhere in that range depending on the random seed.")
    plt.figure(figsize=(6, 3.6)); plt.hist(holdout, bins=20, alpha=0.7, label="single hold-out (100 seeds)")
    plt.axvline(cv5.mean(), c="r", label="5-fold CV mean"); plt.legend(); plt.xlabel("accuracy")
    plt.title("Same model, same data: hold-out lottery"); plt.tight_layout()
    plt.savefig(OUT / "A_holdout_noise.png", dpi=110); plt.close()


def exp_b_stratification():
    header("B. Stratified vs plain K-Fold on 1% positives")
    y = np.zeros(1000, dtype=int); y[np.random.default_rng(0).choice(1000, 10, replace=False)] = 1
    X = np.zeros((1000, 1))
    for name, cv in (("KFold(5, shuffle)", KFold(5, shuffle=True, random_state=1)),
                     ("StratifiedKFold(5)", StratifiedKFold(5, shuffle=True, random_state=1))):
        pos = [int(y[te].sum()) for _, te in cv.split(X, y)]
        print(f"  {name:<20} positives per validation fold: {pos}")
    print("  -> Plain KFold can leave a fold with 0 positives (recall/AUC undefined). Always stratify classification.")


def exp_c_time_series():
    header("C. Time-series: shuffled KFold leaks the future")
    rng = np.random.default_rng(0)
    t = np.arange(600)
    y = 0.05 * t + 5 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 1, len(t))      # trend + season + noise
    lag1 = np.r_[y[0], y[:-1]]
    X = np.c_[t, np.sin(2 * np.pi * t / 50), np.cos(2 * np.pi * t / 50), lag1]
    rf = RandomForestRegressor(200, random_state=0, n_jobs=-1)
    shuffled = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
    tscv = cross_val_score(rf, X, y, cv=TimeSeriesSplit(5), scoring="r2")
    print(f"  shuffled KFold  R2 = {shuffled.mean():.3f}   <- looks fantastic")
    print(f"  TimeSeriesSplit R2 = {tscv.mean():.3f}   <- honest: the model must predict a FUTURE it hasn't seen")
    print("  Why: shuffled folds let the model interpolate between past AND future neighbours; a tree model")
    print("  also cannot extrapolate the upward trend beyond the training range.")
    log_experiment(PROJECT, "timeseries_shuffled_vs_tscv", "RandomForest",
                   {"shuffled_r2": shuffled.mean(), "timeseries_split_r2": tscv.mean()}, default_split="cv")


def exp_d_group_leakage():
    header("D. Grouped data: the same customer appears in train AND validation")
    rng = np.random.default_rng(0)
    n_cust, rows = 150, 8
    fingerprint = rng.normal(0, 1, (n_cust, 15))               # each customer has a unique signature
    label = rng.integers(0, 2, n_cust)                         # label is RANDOM per customer -> nothing generalisable
    X = np.repeat(fingerprint, rows, axis=0) + rng.normal(0, 0.3, (n_cust * rows, 15))
    y = np.repeat(label, rows)
    groups = np.repeat(np.arange(n_cust), rows)
    rf = RandomForestClassifier(150, random_state=0, n_jobs=-1)
    leaky = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean()
    honest = cross_val_score(rf, X, y, cv=GroupKFold(5), groups=groups).mean()
    print(f"  KFold accuracy      = {leaky:.3f}   <- model just memorised customers")
    print(f"  GroupKFold accuracy = {honest:.3f}   <- ~0.5 = coin-flip, the truth (labels were random!)")
    print("  Use GroupKFold whenever rows are not independent (patients, users, devices, sessions, documents).")


def exp_e_pipeline_leakage():
    header("E. Pipeline leakage: feature selection on ALL data before CV")
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(100, 2000)), rng.integers(0, 2, 100)   # PURE NOISE: true AUC = 0.5
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    Xsel = SelectKBest(f_classif, k=20).fit_transform(X, y)         # WRONG: label information leaks in
    wrong = cross_val_score(LogisticRegression(max_iter=1000), Xsel, y, cv=cv, scoring="roc_auc").mean()
    pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    right = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()          # RIGHT: selection inside each fold
    print(f"  selection BEFORE cv : AUC = {wrong:.3f}   <- 'discovered' signal in pure noise!")
    print(f"  selection INSIDE cv : AUC = {right:.3f}   <- ~0.5, as it must be")
    print("  Rule: every step that learns from data (scaling, imputing, encoding, selection, PCA, SMOTE)"
          "\n        belongs INSIDE the pipeline that is cross-validated.")
    log_experiment(PROJECT, "pipeline_leakage_demo", "SelectKBest+LogReg",
                   {"leaky_auc": wrong, "correct_auc": right}, default_split="cv")


def exp_f_nested_cv():
    header("F. Nested CV: honest performance of 'tune hyper-parameters + evaluate'")
    df = get_dataset("customers").sample(150, random_state=1)
    X = df[["tenure_months", "monthly_charges", "num_support_calls", "total_charges"]].fillna(0)
    y = df["churned"]
    grid = {"kneighborsclassifier__n_neighbors": list(range(1, 40)),
            "kneighborsclassifier__weights": ["uniform", "distance"]}
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
    outer = StratifiedKFold(5, shuffle=True, random_state=0)
    inner = StratifiedKFold(5, shuffle=True, random_state=1)
    gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc").fit(X, y)
    nested = cross_val_score(GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc"), X, y, cv=outer, scoring="roc_auc")
    print(f"  grid-search best score (non-nested): AUC = {gs.best_score_:.3f}   <- optimistic (max over 78 candidates)")
    print(f"  nested CV estimate                 : AUC = {nested.mean():.3f} +- {nested.std():.3f}   <- honest")
    print("  The selection bias is bigger with small data and large grids. Nested CV = inner loop tunes,"
          "\n  outer loop measures. Then train the final model on ALL data with the chosen hyper-parameters.")


if __name__ == "__main__":
    exp_a_holdout_noise()
    exp_b_stratification()
    exp_c_time_series()
    exp_d_group_leakage()
    exp_e_pipeline_leakage()
    exp_f_nested_cv()
