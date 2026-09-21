"""
PROJECT 10 - Production Model Picker
"Several models perform well. Which one ships?"  -> MEASURE every criterion, then decide by
business profile.

Pipeline
  1. Benchmark candidates: CV performance (mean+-std), training time, single-row latency (p50/p95),
     batch throughput, model size on disk.  Interpretability & maintenance are judgement scores (1-5).
  2. Statistical-tie check: which models are within 1 std of the best AUC?
  3. Hard GATES per business profile (e.g. p95 latency <= X ms, must be explainable) -> disqualify.
  4. Weighted SCORECARD per profile -> winner. Whether the winner changes between profiles depends on how
     large the accuracy gap is: try --dataset customers (simple model dominates) vs customers_nl (real trade-off).
  5. Train the winner on all training data, save artifact, register it in the DB model registry.

Run:
    python projects/p10_production_model_picker/main.py                       # non-linear data, profile realtime_api
    python projects/p10_production_model_picker/main.py --profile regulated_lending
    python projects/p10_production_model_picker/main.py --dataset customers   # linear-ish data: simple model dominates
Then serve it:
    uvicorn projects.p10_production_model_picker.serve:app --reload
"""
import argparse
import io
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common.db import get_dataset, log_experiment, next_model_version, register_model

PROJECT = "p10_model_picker"
MODEL_NAME = "churn_model"
MODELS_DIR = Path(__file__).parent / "models"
FEATURES_NUM = ["tenure_months", "monthly_charges", "total_charges", "num_support_calls"]
FEATURES_CAT = ["contract_type", "internet_service", "payment_method"]

# Judgement scores 1 (bad) .. 5 (great). In a real team you debate these - they are NOT measured.
CANDIDATES = {
    #  name                        factory                                                        interp maint
    "LogisticRegression":   (lambda: LogisticRegression(max_iter=1000),                                5, 5),
    "DecisionTree(depth5)": (lambda: DecisionTreeClassifier(max_depth=5, random_state=0),              4, 4),
    "RandomForest":         (lambda: RandomForestClassifier(200, min_samples_leaf=5, random_state=0, n_jobs=1), 2, 3),
    "GradientBoosting":     (lambda: GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=0), 2, 3),
    "HistGradientBoosting": (lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, random_state=0), 2, 3),
    "KNN(k=25)":            (lambda: KNeighborsClassifier(25),                                          3, 3),
    "MLP(64,32)":           (lambda: MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=0), 1, 2),
}

# Business profiles: weights (sum to 1) + hard gates. Edit these to match YOUR product.
PROFILES = {
    "realtime_api": dict(   # user-facing scoring endpoint, tight latency budget, small team
        weights=dict(performance=0.30, latency=0.30, size=0.05, train_cost=0.05, interpretability=0.10, maintenance=0.20),
        gates=dict(max_p95_ms=12.0, min_auc=0.70, min_interp=1)),   # 12 ms model budget (machine-dependent!)
    "batch_max_accuracy": dict(  # e.g. ad/recommendation ranking: 0.5% AUC = real money, latency irrelevant
        weights=dict(performance=0.85, latency=0.01, size=0.01, train_cost=0.02, interpretability=0.03, maintenance=0.08),
        gates=dict(max_p95_ms=1000.0, min_auc=0.70, min_interp=1)),
    "regulated_lending": dict(  # decisions must be explainable to customers/regulators
        weights=dict(performance=0.25, latency=0.03, size=0.02, train_cost=0.03, interpretability=0.42, maintenance=0.25),
        gates=dict(max_p95_ms=100.0, min_auc=0.70, min_interp=4)),
}


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), FEATURES_NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), FEATURES_CAT)])


def build(name):
    return Pipeline([("prep", make_preprocessor()), ("model", CANDIDATES[name][0]())])


def benchmark(Xtr, ytr, Xte, yte, n_latency=100):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for name, (_, interp, maint) in CANDIDATES.items():
        cvs = cross_val_score(build(name), Xtr, ytr, cv=cv, scoring="roc_auc")
        pipe = build(name)
        t0 = time.perf_counter(); pipe.fit(Xtr, ytr); train_s = time.perf_counter() - t0
        test_auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])

        for i in range(20):                                    # warm-up (first calls are always slower)
            pipe.predict_proba(Xte.iloc[[i]])
        singles, block_p95 = [], []                            # one request = one row (like an API call)
        for b in range(3):                                     # 3 blocks -> median of block p95s resists noise
            blk = []
            for i in range(n_latency):
                row = Xte.iloc[[20 + b * n_latency + i]]
                t0 = time.perf_counter(); pipe.predict_proba(row); blk.append((time.perf_counter() - t0) * 1000)
            singles += blk
            block_p95.append(np.percentile(blk, 95))
        batch = Xte.iloc[:1000]
        t0 = time.perf_counter(); pipe.predict_proba(batch); batch_s = time.perf_counter() - t0
        buf = io.BytesIO(); joblib.dump(pipe, buf)

        rows.append(dict(model=name, cv_auc=cvs.mean(), cv_std=cvs.std(), test_auc=test_auc, train_s=train_s,
                         p50_ms=np.percentile(singles, 50), p95_ms=float(np.median(block_p95)),
                         rows_per_s=len(batch) / batch_s, size_kb=buf.getbuffer().nbytes / 1024,
                         interp=interp, maint=maint))
    return pd.DataFrame(rows).set_index("model")


def minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return pd.Series(1.0, index=s.index) if rng == 0 else (s - s.min()) / rng


def scorecard(res: pd.DataFrame, profile: dict) -> pd.DataFrame:
    g, w = profile["gates"], profile["weights"]
    ok = (res.p95_ms <= g["max_p95_ms"]) & (res.cv_auc >= g["min_auc"]) & (res.interp >= g["min_interp"])
    s = pd.DataFrame(index=res.index)
    s["performance"] = minmax(res.cv_auc)
    # Latency is scored against the SLA budget, NOT min-max across candidates: full marks up to half the
    # budget, 0 at the budget. (Min-max stretched 2-3 ms of timing noise into a huge score gap.)
    budget = g["max_p95_ms"]
    s["latency"] = ((budget - res.p95_ms) / (0.5 * budget)).clip(0, 1)
    s["size"] = 1 - minmax(np.log10(res.size_kb))
    s["train_cost"] = 1 - minmax(np.log10(res.train_s))
    s["interpretability"] = (res.interp - 1) / 4
    s["maintenance"] = (res.maint - 1) / 4
    s["weighted_score"] = sum(s[k] * v for k, v in w.items())
    s["passes_gates"] = ok
    s.loc[~ok, "weighted_score"] = np.nan                     # disqualified
    return s.sort_values("weighted_score", ascending=False)


def sensitivity(res: pd.DataFrame, base: dict):
    """How robust is the decision to our (subjective) weights? Sweep the performance weight 0 -> 1 and
    rescale the other weights proportionally; report who wins."""
    others = {k: v for k, v in base["weights"].items() if k != "performance"}
    tot = sum(others.values())
    print("Sensitivity: winner as the weight on PERFORMANCE varies (other weights keep their ratios)")
    prev = None
    for wp in np.round(np.arange(0.0, 1.0001, 0.1), 1):
        w = {"performance": wp, **{k: (1 - wp) * v / tot for k, v in others.items()}}
        sc = scorecard(res, dict(weights=w, gates=base["gates"]))
        win = sc.index[0]
        flag = "   <-- winner changes here" if prev and win != prev else ""
        print(f"  performance weight {wp:.1f} -> {win}{flag}")
        prev = win
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="realtime_api")
    ap.add_argument("--dataset", choices=["customers_nl", "customers"], default="customers_nl",
                    help="customers_nl: non-linear churn rules (model choice is a real trade-off); "
                         "customers: mostly-linear rules (a simple model dominates)")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()

    df = get_dataset(args.dataset)
    X, y = df[FEATURES_NUM + FEATURES_CAT], df["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"Dataset: {args.dataset}. Benchmarking candidates (this takes ~1 minute)...\n")
    res = benchmark(Xtr, ytr, Xte, yte)
    pd.set_option("display.width", 200)
    print(res.round({"cv_auc": 4, "cv_std": 4, "test_auc": 4, "train_s": 2, "p50_ms": 2, "p95_ms": 2,
                     "rows_per_s": 0, "size_kb": 0}).to_string())

    best = res.cv_auc.idxmax()
    tied = res.index[res.cv_auc >= res.cv_auc[best] - res.cv_std[best]].tolist()
    print(f"\nStatistical tie: best CV AUC = {best} ({res.cv_auc[best]:.4f} +- {res.cv_std[best]:.4f}). "
          f"Models within 1 std: {tied}")
    print("-> Accuracy alone cannot separate them; that is exactly why the other criteria decide.\n")

    winners = {}
    for pname, prof in PROFILES.items():
        sc = scorecard(res, prof)
        winners[pname] = sc.index[0]
        dq = sc.index[~sc.passes_gates].tolist()
        top = sc.weighted_score.iloc[0]
        tied = [m for m in sc.index[1:] if sc.weighted_score[m] >= top - 0.03]   # within 0.03 = effectively a tie
        print(f"--- profile: {pname:<18} winner: {sc.index[0]:<22} disqualified by gates: {dq or 'none'}")
        if tied:
            print(f"    close call - within 0.03 of the winner: {tied}  (decide with judgement / a pilot, not the decimals)")
        print(sc[["performance", "latency", "interpretability", "maintenance", "weighted_score"]].head(4).round(3).to_string())
        print()
    sensitivity(res, {"weights": PROFILES["realtime_api"]["weights"], "gates": PROFILES["batch_max_accuracy"]["gates"]})
    for pname, w in winners.items():
        log_experiment(PROJECT, f"profile_{pname}", w, {"winner_cv_auc": res.cv_auc[w]}, params={"profile": pname},
                       dataset_name=args.dataset, default_split="cv")

    # ---- ship the winner for the chosen profile -------------------------------------------------
    winner = winners[args.profile]
    print(f"Chosen profile = {args.profile}  ->  shipping: {winner}")
    if args.no_register:
        return
    final = build(winner).fit(Xtr, ytr)                     # in real life: refit on ALL labelled data
    test_auc = roc_auc_score(yte, final.predict_proba(Xte)[:, 1])
    MODELS_DIR.mkdir(exist_ok=True)
    exp_id = log_experiment(PROJECT, f"final_{winner}", winner, {"roc_auc": test_auc},
                            params={"profile": args.profile}, dataset_name=args.dataset, default_split="test")
    version = next_model_version(MODEL_NAME)
    path = MODELS_DIR / f"{MODEL_NAME}_v{version}.joblib"
    joblib.dump(final, path)
    register_model(MODEL_NAME, path.relative_to(ROOT).as_posix(), exp_id, stage="production")   # archives the previous production version
    print(f"Saved {path.name} (held-out test AUC {test_auc:.4f}) and registered {MODEL_NAME} v{version} as PRODUCTION.")
    print("Serve it with:  uvicorn projects.p10_production_model_picker.serve:app --port 8000")


if __name__ == "__main__":
    main()
