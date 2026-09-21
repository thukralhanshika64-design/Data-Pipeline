"""
PROJECT 1 - Learning Paradigms Lab
One business (a telecom company), three learning paradigms:

  supervised    -> predict WHO will churn                (labels exist)
  unsupervised  -> discover customer SEGMENTS            (no labels)
  reinforcement -> learn WHICH email campaign to send    (reward feedback, no dataset)

Run from the repo root:
    python projects/p01_learning_paradigms_lab/main.py            # all three
    python projects/p01_learning_paradigms_lab/main.py --part rl
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.db import get_dataset, log_experiment, save_table

PROJECT = "p01_learning_paradigms"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. SUPERVISED: learn f(X) -> y from labelled examples
# --------------------------------------------------------------------------- #
def supervised():
    print("\n=== SUPERVISED: churn prediction ===")
    df = get_dataset("customers")
    X = df[["tenure_months", "monthly_charges", "num_support_calls"]].copy()
    X["is_month_to_month"] = (df["contract_type"] == "month-to-month").astype(int)
    y = df["churned"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc, auc = accuracy_score(y_te, pred), roc_auc_score(y_te, proba)
    print(classification_report(y_te, pred, digits=3))
    print(f"accuracy={acc:.3f}  roc_auc={auc:.3f}  (baseline accuracy if we always say 'no churn': "
          f"{1 - y_te.mean():.3f})")

    coefs = {c: round(float(w), 3) for c, w in zip(X.columns, model[-1].coef_[0])}
    print("standardised coefficients:", coefs)
    log_experiment(PROJECT, "supervised_logreg", "LogisticRegression",
                   {"accuracy": acc, "roc_auc": auc}, dataset_name="customers")


# --------------------------------------------------------------------------- #
# 2. UNSUPERVISED: find structure with NO target column
# --------------------------------------------------------------------------- #
def unsupervised():
    print("\n=== UNSUPERVISED: customer segmentation (K-Means) ===")
    df = get_dataset("customers")
    feats = ["tenure_months", "monthly_charges", "num_support_calls"]
    Z = StandardScaler().fit_transform(df[feats])          # scaling is CRUCIAL for distance-based models

    ks, inertia, sil = range(2, 9), [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Z, km.labels_, sample_size=2000, random_state=0))
    best_k = list(ks)[int(np.argmax(sil))]
    print("silhouette by k:", {k: round(s, 3) for k, s in zip(ks, sil)}, "-> best k =", best_k)

    labels = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(Z)
    df["segment"] = labels
    profile = df.groupby("segment").agg(
        size=("customer_id", "count"),
        tenure=("tenure_months", "mean"),
        monthly=("monthly_charges", "mean"),
        support_calls=("num_support_calls", "mean"),
        churn_rate=("churned", "mean"),   # label used ONLY to interpret, not to train
    ).round(2)
    print(profile)
    save_table(df[["customer_id", "segment"]], "customer_segments")   # write results back to the DB

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
    ax[1].plot(list(ks), sil, "o-", color="tab:green"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k")
    fig.tight_layout(); fig.savefig(OUT / "unsupervised_k_selection.png", dpi=120); plt.close(fig)
    log_experiment(PROJECT, "unsupervised_kmeans", "KMeans", {"silhouette": max(sil)},
                   params={"k": best_k}, dataset_name="customers")


# --------------------------------------------------------------------------- #
# 3. REINFORCEMENT: learn by trial & error from rewards (multi-armed bandit)
# --------------------------------------------------------------------------- #
WARMUP = 5  # pulls per arm before we trust our estimates


def run_bandit(strategy, rates, steps=10000, eps=0.1, ucb_c=1.414, seed=0):
    rng = np.random.default_rng(seed)
    k = len(rates)
    counts, values = np.zeros(k), np.zeros(k)
    best = max(rates)
    regret = np.zeros(steps)
    for t in range(steps):
        if strategy == "random":
            a = int(rng.integers(k))
        elif t < k * WARMUP:
            a = t % k
        elif strategy == "greedy":                       # pure exploitation
            a = int(np.argmax(values))
        elif strategy == "epsilon_greedy":               # explore with prob eps
            a = int(rng.integers(k)) if rng.random() < eps else int(np.argmax(values))
        else:                                            # UCB: optimism in the face of uncertainty
            a = int(np.argmax(values + ucb_c * np.sqrt(np.log(t + 1) / counts)))
        reward = float(rng.random() < rates[a])          # click / no click
        counts[a] += 1
        values[a] += (reward - values[a]) / counts[a]    # incremental mean
        regret[t] = (regret[t - 1] if t else 0.0) + best - rates[a]
    return regret


def reinforcement(n_seeds=30):
    print("\n=== REINFORCEMENT: which email campaign should we send? ===")
    print(f"(each strategy is run over {n_seeds} random seeds - a single run can be lucky!)")
    true_ctr = [0.020, 0.035, 0.050, 0.030, 0.045]       # hidden from the agent
    strategies = {
        "random": dict(strategy="random"),
        "greedy": dict(strategy="greedy"),
        "epsilon_greedy(0.1)": dict(strategy="epsilon_greedy", eps=0.1),
        "ucb_textbook(c=1.41)": dict(strategy="ucb", ucb_c=1.414),
        "ucb_tuned(c=0.2)": dict(strategy="ucb", ucb_c=0.2),
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    print(f"{'strategy':<22}{'mean regret':>12}{'std':>8}{'worst':>9}")
    for name, kw in strategies.items():
        runs = np.array([run_bandit(rates=true_ctr, seed=s, **kw) for s in range(n_seeds)])
        final = runs[:, -1]
        print(f"{name:<22}{final.mean():>12.1f}{final.std():>8.1f}{final.max():>9.1f}")
        ax.plot(runs.mean(0), label=name)
        log_experiment(PROJECT, f"bandit_{name}", name, {"mean_total_regret": final.mean(),
                       "std_total_regret": final.std()}, default_split="sim")
    ax.set_xlabel("email sent"); ax.set_ylabel("cumulative regret (mean of seeds)"); ax.legend(fontsize=8)
    ax.set_title("Exploration vs exploitation")
    fig.tight_layout(); fig.savefig(OUT / "rl_regret.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "supervised", "unsupervised", "rl"], default="all")
    part = ap.parse_args().part
    if part in ("all", "supervised"): supervised()
    if part in ("all", "unsupervised"): unsupervised()
    if part in ("all", "rl"): reinforcement()
