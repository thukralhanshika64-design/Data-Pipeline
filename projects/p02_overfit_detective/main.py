"""
PROJECT 2 - Overfit Detective
Detect overfitting/underfitting with (A) validation curves, (B) learning curves,
then fix it with (C) regularisation / more data / simpler model and
(D) early stopping. (E) shows the same story with tree depth on real data.

Run:  python projects/p02_overfit_detective/main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common.db import log_experiment

PROJECT = "p02_overfit_detective"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
NOISE = 0.3                      # irreducible noise -> best achievable MSE ~ NOISE**2 = 0.09
RNG = np.random.default_rng(0)


def make_data(n, seed):
    r = np.random.default_rng(seed)
    x = r.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + r.normal(0, NOISE, n)
    return x.reshape(-1, 1), y


def poly(degree, alpha=None):
    reg = LinearRegression() if alpha is None else Ridge(alpha=alpha)
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)


def diagnose(train_err, val_err, target_err):
    """Rule-of-thumb diagnosis used in interviews."""
    if train_err > 1.5 * target_err:
        return "UNDERFITTING (high bias: even the training error is bad)"
    if val_err > 1.5 * train_err and val_err > 1.5 * target_err:
        return "OVERFITTING (high variance: big train/val gap)"
    return "GOOD FIT"


# --------------------------------------------------------------------------- #
def part_a_validation_curve():
    print("\n[A] Validation curve: model complexity (polynomial degree) vs error")
    Xtr, ytr = make_data(30, seed=1)
    Xva, yva = make_data(500, seed=2)
    degrees = range(1, 16)
    tr_err, va_err = [], []
    for d in degrees:
        m = poly(d).fit(Xtr, ytr)
        tr_err.append(mean_squared_error(ytr, m.predict(Xtr)))
        va_err.append(mean_squared_error(yva, m.predict(Xva)))
    for d, t, v in zip(degrees, tr_err, va_err):
        if d in (1, 3, 4, 8, 12, 15):
            print(f"  degree={d:>2}  train MSE={t:8.3f}  val MSE={v:12.3f}  -> {diagnose(t, v, NOISE**2)}")
    best = list(degrees)[int(np.argmin(va_err))]
    print(f"  best degree by validation error: {best}")

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    ax[0].semilogy(list(degrees), tr_err, "o-", label="train")
    ax[0].semilogy(list(degrees), va_err, "o-", label="validation")
    ax[0].axhline(NOISE**2, ls="--", c="grey", label="noise floor")
    ax[0].set_xlabel("polynomial degree"); ax[0].set_ylabel("MSE (log)"); ax[0].legend()
    ax[0].set_title("Validation curve")
    xs = np.linspace(0, 1, 300).reshape(-1, 1)
    for a, d in zip(ax[1:], (1, 4, 15)):
        m = poly(d).fit(Xtr, ytr)
        a.scatter(Xtr, ytr, s=14); a.plot(xs, np.sin(2 * np.pi * xs), "g--", label="truth")
        a.plot(xs, m.predict(xs), "r", label=f"degree {d}"); a.set_ylim(-2, 2); a.legend()
        a.set_title({1: "Underfit", 4: "Good fit", 15: "Overfit"}[d])
    fig.tight_layout(); fig.savefig(OUT / "A_validation_curve.png", dpi=110); plt.close(fig)
    log_experiment(PROJECT, "validation_curve", "poly", {"best_degree": best,
                   "val_mse": min(va_err)}, default_split="val")


def part_b_learning_curve():
    print("\n[B] Learning curves: does MORE DATA help?")
    X, y = make_data(400, seed=3)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, d in zip(ax, (1, 12)):
        sizes, tr, va = learning_curve(poly(d), X, y, cv=5, scoring="neg_mean_squared_error",
                                       train_sizes=np.linspace(0.08, 1, 8), shuffle=True, random_state=0)
        tr, va = -tr.mean(1), -va.mean(1)
        a.plot(sizes, tr, "o-", label="train"); a.plot(sizes, va, "o-", label="cv")
        a.axhline(NOISE**2, ls="--", c="grey"); a.set_yscale("log"); a.set_title(f"degree {d}")
        a.set_xlabel("training examples"); a.legend()
        print(f"  degree={d:>2}: cv-train gap {va[0] - tr[0]:8.3f} (n={sizes[0]}) -> {va[-1] - tr[-1]:.3f} "
              f"(n={sizes[-1]});  final cv MSE = {va[-1]:.3f}")
    print("  Reading: high-bias model -> curves converge HIGH (more data won't help);"
          "\n           high-variance model -> big gap that SHRINKS with more data.")
    fig.tight_layout(); fig.savefig(OUT / "B_learning_curves.png", dpi=110); plt.close(fig)


def part_c_fixes():
    print("\n[C] Fixing an overfit degree-12 model (30 training points)")
    Xtr, ytr = make_data(30, seed=1)
    Xbig, ybig = make_data(300, seed=4)
    Xte, yte = make_data(1000, seed=5)
    candidates = {
        "overfit baseline (deg 12)": (poly(12), Xtr, ytr),
        "+ Ridge alpha=1 (deg 12)": (poly(12, alpha=1.0), Xtr, ytr),
        "+ Ridge alpha=10 (deg 12)": (poly(12, alpha=10.0), Xtr, ytr),
        "simpler model (deg 4)": (poly(4), Xtr, ytr),
        "10x more data (deg 12)": (poly(12), Xbig, ybig),
    }
    print(f"  {'strategy':<28}{'train MSE':>10}{'test MSE':>10}")
    for name, (m, X, y) in candidates.items():
        m.fit(X, y)
        tr, te = mean_squared_error(y, m.predict(X)), mean_squared_error(yte, m.predict(Xte))
        print(f"  {name:<28}{tr:>10.3f}{te:>10.3f}")
        log_experiment(PROJECT, name, "poly", {"train": {"mse": tr}, "test": {"mse": te}})


def part_d_early_stopping():
    print("\n[D] Early stopping with gradient boosting")
    X, y = make_data(400, seed=6)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0)
    gb = GradientBoostingRegressor(n_estimators=600, learning_rate=0.1, max_depth=4, random_state=0)
    gb.fit(Xtr, ytr)
    tr = [mean_squared_error(ytr, p) for p in gb.staged_predict(Xtr)]
    va = [mean_squared_error(yva, p) for p in gb.staged_predict(Xva)]
    best = int(np.argmin(va)) + 1
    print(f"  train MSE keeps falling ({tr[-1]:.4f} at 600 trees) but validation is best at {best} trees "
          f"({min(va):.4f}) and worsens to {va[-1]:.4f}")
    auto = GradientBoostingRegressor(n_estimators=600, learning_rate=0.1, max_depth=4, random_state=0,
                                     validation_fraction=0.2, n_iter_no_change=10, tol=1e-4).fit(Xtr, ytr)
    print(f"  sklearn's built-in n_iter_no_change stopped automatically at {auto.n_estimators_} trees")
    plt.figure(figsize=(6, 3.8))
    plt.plot(tr, label="train"); plt.plot(va, label="validation"); plt.axvline(best, c="k", ls=":")
    plt.xlabel("boosting rounds"); plt.ylabel("MSE"); plt.legend(); plt.title("Early stopping")
    plt.tight_layout(); plt.savefig(OUT / "D_early_stopping.png", dpi=110); plt.close()


def part_e_tree_depth():
    print("\n[E] Real data: decision-tree depth on breast-cancer dataset")
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    depths = list(range(1, 16))
    tr, te = [], []
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        tr.append(t.score(Xtr, ytr)); te.append(t.score(Xte, yte))
    for d in (1, 3, 5, 15):
        print(f"  max_depth={d:>2}  train acc={tr[d-1]:.3f}  test acc={te[d-1]:.3f}")
    print("  Note: an unrestricted tree memorises training data (train acc 1.0). "
          "Fixes: max_depth, min_samples_leaf, ccp_alpha (pruning), or switch to a Random Forest.")
    plt.figure(figsize=(6, 3.8))
    plt.plot(depths, tr, "o-", label="train"); plt.plot(depths, te, "o-", label="test")
    plt.xlabel("max_depth"); plt.ylabel("accuracy"); plt.legend(); plt.title("Tree depth")
    plt.tight_layout(); plt.savefig(OUT / "E_tree_depth.png", dpi=110); plt.close()


if __name__ == "__main__":
    part_a_validation_curve()
    part_b_learning_curve()
    part_c_fixes()
    part_d_early_stopping()
    part_e_tree_depth()
    print(f"\nPlots saved in {OUT}")
