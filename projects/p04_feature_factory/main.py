"""
PROJECT 4 - Feature Factory
Measure (not guess!) how much each feature-engineering technique helps on churn data.

Ablation ladder (each step ADDS to the previous one):
  L0  raw numerics + one-hot categoricals                      <- baseline
  L1  + log transform, ratios, flags, interaction, missing-indicators, ordinal contract
  L2  + date features (cyclical month, Q4 flag)
  L3  + high-cardinality `city` via TARGET ENCODING
  L3b (alternative) `city` via one-hot -> compare high-cardinality strategies

Everything lives inside sklearn Pipelines -> imputers/scalers/encoders are fitted on the
TRAIN part of each CV fold only (no leakage).

Run:  python projects/p04_feature_factory/main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler, TargetEncoder)

from common.db import get_dataset, log_experiment, save_table

PROJECT = "p04_feature_factory"
NUM0 = ["tenure_months", "monthly_charges", "total_charges", "num_support_calls"]


# --------------------------------------------------------------------------- #
# Stateless feature creation (safe to run before the split: uses only the row itself)
# --------------------------------------------------------------------------- #
def add_features(X: pd.DataFrame, level: int) -> pd.DataFrame:
    X = X.copy()
    if level >= 1:
        X["log_monthly"] = np.log(X["monthly_charges"])                       # fix right skew
        X["avg_spend"] = X["total_charges"] / X["tenure_months"]              # ratio (NaN propagates)
        X["is_new_customer"] = (X["tenure_months"] <= 6).astype(int)          # threshold flag
        X["m2m_x_support"] = (X["contract_type"] == "month-to-month") * X["num_support_calls"]  # interaction
    if level >= 2:
        m = X["signup_date"].dt.month
        X["signup_month_sin"] = np.sin(2 * np.pi * m / 12)                    # cyclical encoding
        X["signup_month_cos"] = np.cos(2 * np.pi * m / 12)
        X["signup_q4"] = m.isin([11, 12]).astype(int)                         # domain knowledge
    return X


def build_preprocessor(level: int, city: str | None):
    num = list(NUM0)
    if level >= 1:
        num += ["log_monthly", "avg_spend", "is_new_customer", "m2m_x_support"]
    if level >= 2:
        num += ["signup_month_sin", "signup_month_cos", "signup_q4"]

    # add_indicator=True adds a 0/1 "was missing" column -> missingness itself can be predictive
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=level >= 1)),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                         ("oh", OneHotEncoder(handle_unknown="ignore"))])
    parts = [("num", num_pipe, num)]

    if level >= 1:   # contract is ORDERED: month-to-month < one-year < two-year
        parts.append(("contract", OrdinalEncoder(categories=[["month-to-month", "one-year", "two-year"]]),
                      ["contract_type"]))
        parts.append(("cat", cat_pipe, ["internet_service", "payment_method"]))
    else:
        parts.append(("cat", cat_pipe, ["contract_type", "internet_service", "payment_method"]))

    if city == "target":   # high-cardinality: replace category by (smoothed, cross-fitted) mean target
        parts.append(("city", TargetEncoder(target_type="binary", random_state=0), ["city"]))
    elif city == "onehot":
        parts.append(("city", OneHotEncoder(handle_unknown="ignore"), ["city"]))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def make_pipeline(level, city, model):
    return Pipeline([("features", FunctionTransformer(add_features, kw_args={"level": level})),
                     ("prep", build_preprocessor(level, city)),
                     ("model", model)])


def main():
    df = get_dataset("customers")
    y = df["churned"]
    X = df.drop(columns=["churned", "customer_id"])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    configs = {
        "L0 baseline (raw + one-hot)": (0, None),
        "L1 + transforms/ratios/flags/ordinal": (1, None),
        "L2 + date features": (2, None),
        "L3 + city target-encoded": (2, "target"),
        "L3b city one-hot (alt)": (2, "onehot"),
    }
    models = {
        "LogReg": lambda: LogisticRegression(max_iter=2000),
        "HistGB": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=0),
    }

    results = {}                      # (config, model) -> (mean, std)
    for name, (level, city) in configs.items():
        for mname, mk in models.items():
            scores = cross_val_score(make_pipeline(level, city, mk()), X, y, cv=cv, scoring="roc_auc")
            results[(name, mname)] = (scores.mean(), scores.std())
            log_experiment(PROJECT, name, mname, {"cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()},
                           params={"level": level, "city": city}, dataset_name="customers", default_split="cv")

    base_name = next(iter(configs))
    print("5-fold CV ROC-AUC (mean +- std) and change vs the L0 baseline")
    print("  ^ / v = change larger than the fold-to-fold noise, ~ = within noise (don't trust it)\n")
    print(f"{'feature set':<40}" + "".join(f"{m:>26}" for m in models))
    for name in configs:
        row = f"{name:<40}"
        for mname in models:
            m, sd = results[(name, mname)]
            m0, sd0 = results[(base_name, mname)]
            d = m - m0
            flag = "" if name == base_name else (" ^" if d > max(sd, sd0) else " v" if d < -max(sd, sd0) else " ~")
            row += f"{m:>10.4f} +-{sd:.3f} ({d:+.3f}{flag or '  '})"
        print(row)
    print("\nHow to read this table: keep a feature idea only if it earns a '^'. Ideas that show '~' cost\n"
          "complexity for no measurable gain. Also compare the two models: the same feature can help one model\n"
          "family and not the other, so always ablate per model. Target encoding is designed for HIGH cardinality; with 30 cities\n"
          "one-hot is just as good. Experiment: get_dataset('customers', refresh=True, n_cities=500) and re-run\n"
          "(in the author's test target-encoding edged one-hot by ~0.005 AUC - small, within noise; the benefit\n"
          "is expected to grow as categories get rarer, and one-hot's column count explodes).\n")

    # ---- feature ranking + save a small 'feature store' table to the DB ---------------------------
    full = make_pipeline(2, "target", LogisticRegression(max_iter=2000)).fit(X, y)
    Z = full[:-1].transform(X)                                 # transformed matrix (numpy or DataFrame)
    names = full[:-1].named_steps["prep"].get_feature_names_out()
    Zdf = pd.DataFrame(np.asarray(Z), columns=names)
    mi = pd.Series(mutual_info_classif(Zdf, y, random_state=0), index=names).sort_values(ascending=False)
    print("\nTop-8 features by mutual information (exploration only - do NOT select features on all data before CV!):")
    print(mi.head(8).round(4).to_string())
    Zdf.insert(0, "customer_id", df["customer_id"].values)
    Zdf["churned"] = y.values
    save_table(Zdf, "customer_features")
    print(f"\nSaved engineered matrix {Zdf.shape} to DB table 'customer_features' (a mini feature store).")


if __name__ == "__main__":
    main()
