"""
PROJECT 9 - Ensemble Arena

  1. Bagging FROM SCRATCH         (bootstrap + average)            -> reduces VARIANCE
  2. Gradient Boosting FROM SCRATCH (fit trees to residuals)       -> reduces BIAS
     ...both compared against scikit-learn on house prices
  3. Classification arena: single tree, Bagging, RandomForest, ExtraTrees, AdaBoost, GradientBoosting,
     HistGradientBoosting, Voting, Stacking on churn  (5-fold CV AUC + fit time)
  4. Random-Forest extras: OOB score, feature importance
  5. Boosting: learning-rate vs number-of-trees trade-off (early stopping)

Run:  python projects/p09_ensemble_arena/main.py
"""
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, BaggingRegressor, ExtraTreesClassifier,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from common.db import get_dataset, log_experiment

PROJECT = "p09_ensemble_arena"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# From-scratch ensembles (regression, squared loss)
# --------------------------------------------------------------------------- #
class SimpleBagging:
    """Train each deep tree on a bootstrap sample; predict the average."""

    def __init__(self, n_estimators=50, max_depth=None, seed=0):
        self.n_estimators, self.max_depth, self.seed = n_estimators, max_depth, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.trees_ = []
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, n)                                   # sample WITH replacement
            self.trees_.append(DecisionTreeRegressor(max_depth=self.max_depth, random_state=i).fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees_], axis=0)


class SimpleGradientBoosting:
    """F_m(x) = F_{m-1}(x) + lr * tree_m(x), where tree_m is fitted to the residuals y - F_{m-1}(x).
    For squared loss the residual IS the negative gradient - hence 'gradient' boosting."""

    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3):
        self.n_estimators, self.lr, self.max_depth = n_estimators, learning_rate, max_depth

    def fit(self, X, y):
        self.f0_ = float(np.mean(y))
        pred = np.full(len(y), self.f0_)
        self.trees_ = []
        for i in range(self.n_estimators):
            residual = y - pred                                             # negative gradient of 0.5*(y-F)^2
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=i).fit(X, residual)
            pred += self.lr * tree.predict(X)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        return self.f0_ + self.lr * np.sum([t.predict(X) for t in self.trees_], axis=0)


def regression_part():
    print("\n[1-2] From-scratch Bagging & Gradient Boosting vs scikit-learn (house price, RMSE - lower is better)")
    df = get_dataset("houses")
    X, y = df.drop(columns=["price", "house_id"]), df["price"].to_numpy()
    pre = ColumnTransformer([("num", "passthrough", [c for c in X.columns if c != "neighborhood"]),
                             ("cat", OneHotEncoder(sparse_output=False), ["neighborhood"])])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    Xtr, Xte = pre.fit_transform(Xtr), pre.transform(Xte)
    rmse = lambda m: np.sqrt(mean_squared_error(yte, m.predict(Xte)))
    models = {
        "single deep tree": DecisionTreeRegressor(random_state=0),
        "shallow tree (depth 3)": DecisionTreeRegressor(max_depth=3, random_state=0),
        "Bagging (scratch, 100 trees)": SimpleBagging(100),
        "BaggingRegressor (sklearn)": BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, random_state=0),
        "RandomForest (sklearn)": RandomForestRegressor(200, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "GradientBoosting (scratch)": SimpleGradientBoosting(300, 0.1, 3),
        "GradientBoosting (sklearn)": GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0),
    }
    for name, m in models.items():
        t0 = time.time(); m.fit(Xtr, ytr)
        print(f"    {name:<32} RMSE = {rmse(m):>10,.0f}   ({time.time() - t0:.1f}s)")
    print("    Scratch versions land close to sklearn's -> you now know what those libraries actually do.")


# --------------------------------------------------------------------------- #
def load_churn():
    df = get_dataset("customers")
    y = df["churned"]
    X = df[["tenure_months", "monthly_charges", "total_charges", "num_support_calls",
            "contract_type", "internet_service", "payment_method"]]
    num, cat = list(X.columns[:4]), list(X.columns[4:])
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
    return X, y, pre


def classification_arena():
    print("\n[3] Classification arena on churn (5-fold CV ROC-AUC)")
    X, y, pre = load_churn()
    rf = lambda: RandomForestClassifier(150, min_samples_leaf=5, random_state=0, n_jobs=-1)
    hgb = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, random_state=0)
    lr = lambda: LogisticRegression(max_iter=1000)
    models = {
        "Decision tree (unpruned)": DecisionTreeClassifier(random_state=0),
        "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=0),
        "Bagging (100 trees)": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0, n_jobs=-1),
        "Random Forest": rf(),
        "Extra Trees": ExtraTreesClassifier(150, min_samples_leaf=5, random_state=0, n_jobs=-1),
        "AdaBoost (stumps)": AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=0),
        "HistGradientBoosting": hgb(),
        "Voting (LR+RF+HGB, soft)": VotingClassifier([("lr", lr()), ("rf", rf()), ("hgb", hgb())], voting="soft"),
        "Stacking (LR meta)": StackingClassifier([("rf", rf()), ("hgb", hgb())], final_estimator=lr(), cv=3),
        "LogisticRegression (ref.)": lr(),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for name, m in models.items():
        t0 = time.time()
        s = cross_val_score(Pipeline([("pre", pre), ("m", m)]), X, y, cv=cv, scoring="roc_auc")
        rows.append((name, s.mean(), s.std(), time.time() - t0))
        log_experiment(PROJECT, name, name, {"cv_roc_auc_mean": s.mean(), "cv_roc_auc_std": s.std()},
                       dataset_name="customers", default_split="cv")
    tbl = pd.DataFrame(rows, columns=["model", "auc_mean", "auc_std", "cv_time_s"]).set_index("model")
    print(tbl.sort_values("auc_mean", ascending=False).round(4).to_string())
    print("    Read with the std in mind: differences smaller than ~1 std between models are not meaningful.")
    print("    An unpruned tree overfits (high variance); bagging/forests fix that; boosting builds strength from")
    print("    weak learners; a plain LogisticRegression can be very competitive when signal is mostly linear.")


def rf_extras():
    print("\n[4] Random Forest extras: out-of-bag (OOB) score and feature importance")
    X, y, pre = load_churn()
    Xt = pre.fit_transform(X)
    names = pre.get_feature_names_out()
    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.25, stratify=y, random_state=0)
    rf = RandomForestClassifier(300, min_samples_leaf=5, oob_score=True, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    print(f"    OOB accuracy = {rf.oob_score_:.3f}   |   held-out test accuracy = {rf.score(Xte, yte):.3f}"
          "   (OOB ~ a free validation set: each tree is tested on the ~37% rows it never saw)")
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False).head(6)
    print("    top impurity-based importances:", {k.split('__')[-1]: round(v, 3) for k, v in imp.items()})
    print("    (impurity importance is biased toward high-cardinality/continuous features - confirm with permutation importance)")


def boosting_tradeoff():
    print("\n[5] Boosting: learning rate vs number of trees (validation AUC per boosting round)")
    X, y, pre = load_churn()
    Xt = pre.fit_transform(X)
    Xtr, Xva, ytr, yva = train_test_split(Xt, y, test_size=0.3, stratify=y, random_state=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    for lr_ in (0.3, 0.1, 0.03):
        gb = GradientBoostingClassifier(n_estimators=400, learning_rate=lr_, max_depth=3, subsample=0.8,
                                        random_state=0).fit(Xtr, ytr)
        aucs = [roc_auc_score(yva, proba[:, 1]) for proba in gb.staged_predict_proba(Xva)]
        best = int(np.argmax(aucs)) + 1
        print(f"    learning_rate={lr_:<5} best round={best:>3}  best val AUC={max(aucs):.4f}  AUC at 400 rounds={aucs[-1]:.4f}")
        ax.plot(aucs, label=f"lr={lr_}")
    ax.set_xlabel("boosting rounds"); ax.set_ylabel("validation AUC"); ax.legend()
    ax.set_title("Small learning rate needs more trees but is smoother")
    fig.tight_layout(); fig.savefig(OUT / "boosting_lr_vs_trees.png", dpi=110); plt.close(fig)
    print("    High learning rate peaks early then overfits; small learning rate + many trees + early stopping is the robust recipe.")


if __name__ == "__main__":
    regression_part()
    classification_arena()
    rf_extras()
    boosting_tradeoff()
