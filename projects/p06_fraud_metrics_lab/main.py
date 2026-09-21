"""
PROJECT 6 - Fraud Metrics Lab
Learn every evaluation metric by watching it succeed or fail on a ~1% fraud problem.

  1. Accuracy paradox          (a useless model scores 99% accuracy)
  2. Precision / Recall / F1 / ROC-AUC / PR-AUC for 3 real models
  3. ROC vs Precision-Recall   (why ROC-AUC flatters imbalanced problems)
  4. Cost-based threshold      (turn a model into a money-saving decision)
  5. Probability calibration   (class_weight='balanced' breaks probabilities)
  6. Regression metrics        (MAE vs MSE/RMSE, outliers, mean vs median)

Split is TIME-BASED (train = first 70% of the timeline, test = last 30%), like production.

Run:  python projects/p06_fraud_metrics_lab/main.py
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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.db import get_dataset, log_experiment

PROJECT = "p06_fraud_metrics"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
REVIEW_COST = 5.0            # cost of manually reviewing one flagged transaction (false positive)


def load_split():
    df = get_dataset("transactions").sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * 0.7)
    feats = ["amount", "hour", "is_foreign", "distance_from_home_km", "card_age_days",
             "txn_count_last_24h", "merchant_category"]
    return df.iloc[:cut][feats], df.iloc[:cut]["is_fraud"], df.iloc[cut:][feats], df.iloc[cut:]["is_fraud"]


def prep():
    num = ["amount", "hour", "is_foreign", "distance_from_home_km", "card_age_days", "txn_count_last_24h"]
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), ["merchant_category"])])


def metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred), roc_auc=roc_auc_score(y, proba),
                pr_auc=average_precision_score(y, proba))


def main():
    Xtr, ytr, Xte, yte = load_split()
    print(f"train={len(Xtr)} test={len(Xte)}  fraud rate train={ytr.mean():.4%} test={yte.mean():.4%} "
          f"(test frauds: {int(yte.sum())})")

    # ---- 1+2: accuracy paradox and the metric table -------------------------------------------------
    models = {
        "Dummy (never fraud)": DummyClassifier(strategy="most_frequent"),
        "LogReg (balanced)": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest (balanced)": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                          random_state=0, n_jobs=-1),
        "HistGB (balanced)": HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
    }
    fitted, probas, rows = {}, {}, []
    for name, m in models.items():
        pipe = Pipeline([("prep", prep()), ("m", m)]).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        fitted[name], probas[name] = pipe, p
        met = metrics(yte, p)
        rows.append({"model": name, **met})
        log_experiment(PROJECT, name, name, met, dataset_name="transactions")
    print("\n[1-2] Metrics at threshold 0.5 (note Dummy: accuracy ~99% but recall 0)")
    print(pd.DataFrame(rows).set_index("model").round(3).to_string())

    # ---- 3: ROC vs PR -------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name in list(models)[1:]:
        fpr, tpr, _ = roc_curve(yte, probas[name]); ax[0].plot(fpr, tpr, label=name)
        pr, rc, _ = precision_recall_curve(yte, probas[name]); ax[1].plot(rc, pr, label=name)
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_title("ROC curve"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[1].axhline(yte.mean(), c="k", ls=":", label="random"); ax[1].set_title("Precision-Recall curve")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].legend(fontsize=8)
    # Model + threshold selection must NOT touch the test set: carve a chronological validation
    # slice (last 20% of the training period), refit candidates on the first 80%, choose there.
    n_fit = int(len(Xtr) * 0.8)
    val_p = {}
    for name in list(models)[1:]:
        m = Pipeline([("prep", prep()), ("m", models[name].__class__(**models[name].get_params()))])
        m.fit(Xtr.iloc[:n_fit], ytr.iloc[:n_fit])
        val_p[name] = m.predict_proba(Xtr.iloc[n_fit:])[:, 1]
    yval, amt_val = ytr.iloc[n_fit:].to_numpy(), Xtr.iloc[n_fit:]["amount"].to_numpy()
    best = max(val_p, key=lambda n: average_precision_score(yval, val_p[n]))
    print(f"\n[3] Best model by PR-AUC on the VALIDATION slice: {best}. Notice ROC-AUC looks great for every"
          "\n    model while PR-AUC is noticeably lower - with ~1% positives even a tiny FPR means many false alarms.")

    # ---- 4: cost-based threshold (chosen on validation, reported on test) ----------------------------
    thrs = np.linspace(0.01, 0.99, 99)
    def total_cost(p, y, amt, t):
        flag = p >= t
        fn_cost = amt[(~flag) & (y == 1)].sum()             # fraud we missed = money lost
        fp_cost = REVIEW_COST * (flag & (y == 0)).sum()     # good customers we bothered
        return fn_cost + fp_cost
    val_costs = np.array([total_cost(val_p[best], yval, amt_val, t) for t in thrs])
    t_best = float(thrs[val_costs.argmin()])

    p, amt, y = probas[best], Xte["amount"].to_numpy(), yte.to_numpy()
    test_costs = np.array([total_cost(p, y, amt, t) for t in thrs])
    print(f"\n[4] Cost-based decision ({best}); FN cost = fraud amount, FP cost = ${REVIEW_COST:.0f} per review")
    print(f"    threshold chosen on validation = {t_best:.2f}")
    print(f"    TEST cost, no model (approve all)     : ${amt[y == 1].sum():>9,.0f}")
    print(f"    TEST cost, default threshold 0.50     : ${total_cost(p, y, amt, 0.5):>9,.0f}")
    print(f"    TEST cost, validation-chosen threshold: ${total_cost(p, y, amt, t_best):>9,.0f}   "
          f"(recall={recall_score(y, p >= t_best):.2f}, precision={precision_score(y, p >= t_best, zero_division=0):.2f})")
    print(f"    (oracle threshold picked ON test would give ${test_costs.min():,.0f} - that's cheating, shown only "
          "to see how close honest tuning gets)")
    ax[2].plot(thrs, test_costs, label="test cost"); ax[2].plot(thrs, val_costs, label="validation cost")
    ax[2].axvline(t_best, c="r", ls="--"); ax[2].set_title("Business cost vs threshold")
    ax[2].set_xlabel("threshold"); ax[2].set_ylabel("total cost ($)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "roc_pr_cost.png", dpi=110); plt.close(fig)
    log_experiment(PROJECT, "cost_optimal_threshold", best,
                   {"threshold": t_best, "total_cost": total_cost(p, y, amt, t_best),
                    "cost_no_model": amt[y == 1].sum()}, default_split="test")

    # ---- 5: calibration -----------------------------------------------------------------------------
    plain = Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
    p_plain = plain.predict_proba(Xte)[:, 1]
    p_bal = probas["LogReg (balanced)"]
    print(f"\n[5] Calibration: true fraud rate in test = {yte.mean():.4f}")
    print(f"    LogReg plain     mean predicted prob = {p_plain.mean():.4f}   Brier = {brier_score_loss(yte, p_plain):.5f}")
    print(f"    LogReg balanced  mean predicted prob = {p_bal.mean():.4f}   Brier = {brier_score_loss(yte, p_bal):.5f}")
    print("    class_weight='balanced' improves recall but its outputs are NOT real probabilities -")
    print(f"    ROC-AUC is essentially unchanged ({roc_auc_score(yte, p_plain):.3f} vs {roc_auc_score(yte, p_bal):.3f}) - "
          "ranking quality survives, the probability scale does not.\n    Recalibrate (CalibratedClassifierCV) if you need true probabilities.")

    # ---- 6: regression metrics ----------------------------------------------------------------------
    print("\n[6] Regression metrics: MAE vs RMSE")
    y_true = np.full(20, 100.0)
    good = y_true + np.tile([-5, 5], 10)                    # every error is 5
    bad = good.copy(); bad[0] = 300                         # ...except one error of 200
    for label, pred in (("20 errors of 5        ", good), ("19 errors of 5 + one 200", bad)):
        print(f"    {label}: MAE={mean_absolute_error(y_true, pred):6.1f}   RMSE={np.sqrt(mean_squared_error(y_true, pred)):6.1f}")
    amounts = Xtr["amount"].to_numpy()                       # skewed data: best constant guess?
    test_amt = Xte["amount"].to_numpy()
    for label, c in (("mean  ", amounts.mean()), ("median", np.median(amounts))):
        print(f"    constant prediction = {label} ({c:7.2f}):  MAE={mean_absolute_error(test_amt, np.full_like(test_amt, c)):7.2f}"
              f"   RMSE={np.sqrt(mean_squared_error(test_amt, np.full_like(test_amt, c))):7.2f}")
    print("    -> RMSE is minimised by the MEAN, MAE by the MEDIAN; RMSE punishes big misses, MAE is robust.")


if __name__ == "__main__":
    main()
