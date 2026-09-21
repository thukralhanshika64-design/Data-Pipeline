"""
PROJECT 3 - Bias-Variance Simulator
Empirically verify   E[(y - f_hat(x))^2] = Bias^2 + Variance + sigma^2

How: the true function f is KNOWN, so we can draw hundreds of independent training sets,
fit the same model on each, and look at the *distribution of predictions* at fixed x's.

    bias^2(x)   = ( mean_over_datasets f_hat(x) - f(x) )^2      -> systematic error
    variance(x) = var_over_datasets  f_hat(x)                   -> sensitivity to the dataset
    noise       = sigma^2                                       -> irreducible

Run:  python projects/p03_bias_variance_simulator/main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from common.db import log_experiment

PROJECT = "p03_bias_variance"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

SIGMA, N_TRAIN, N_SIMS = 0.3, 40, 200
X_TEST = np.linspace(0.02, 0.98, 100).reshape(-1, 1)
F_TEST = np.sin(2 * np.pi * X_TEST).ravel()


def f(x):
    return np.sin(2 * np.pi * x)


def simulate(make_model, seed=0):
    """Fit `make_model()` on N_SIMS independent training sets; return bias2, variance, emp. test MSE."""
    rng = np.random.default_rng(seed)
    preds = np.empty((N_SIMS, len(X_TEST)))
    emp = []
    for i in range(N_SIMS):
        x = rng.uniform(0, 1, N_TRAIN).reshape(-1, 1)
        y = f(x).ravel() + rng.normal(0, SIGMA, N_TRAIN)
        p = make_model().fit(x, y).predict(X_TEST)
        preds[i] = p
        y_new = F_TEST + rng.normal(0, SIGMA, len(X_TEST))     # fresh noisy test labels
        emp.append(np.mean((p - y_new) ** 2))
    bias2 = np.mean((preds.mean(0) - F_TEST) ** 2)
    var = np.mean(preds.var(0))
    return bias2, var, float(np.mean(emp))


def sweep(name, grid, factory):
    rows = []
    for g in grid:
        b2, v, emp = simulate(lambda g=g: factory(g))
        rows.append((g, b2, v, b2 + v + SIGMA**2, emp))
    print(f"\n{name}")
    print(f"  {'complexity':>10}{'bias^2':>10}{'variance':>10}{'b2+var+s2':>11}{'measured':>10}")
    for g, b2, v, tot, emp in rows:
        print(f"  {g!s:>10}{b2:>10.4f}{v:>10.4f}{tot:>11.4f}{emp:>10.4f}")
    return rows


def plot(ax, rows, title, xlabel, log_x=False):
    g = [r[0] for r in rows]
    ax.plot(g, [r[1] for r in rows], "o-", label="bias²")
    ax.plot(g, [r[2] for r in rows], "o-", label="variance")
    ax.plot(g, [r[3] for r in rows], "k--", label="bias²+var+σ²")
    ax.axhline(SIGMA**2, c="grey", ls=":", label="σ² (noise)")
    ax.set_yscale("log"); ax.set_title(title); ax.set_xlabel(xlabel); ax.legend(fontsize=8)
    if log_x: ax.set_xscale("log")


if __name__ == "__main__":
    poly = lambda d: make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), LinearRegression())
    r_poly = sweep("Polynomial regression (complexity = degree)", [1, 2, 3, 4, 6, 8, 10, 12], poly)
    r_knn = sweep("k-NN regression (complexity = 1/k: small k = complex)", [1, 2, 3, 5, 8, 12, 20, 30],
                  lambda k: KNeighborsRegressor(n_neighbors=k))
    r_tree = sweep("Decision tree (complexity = max_depth)", [1, 2, 3, 4, 5, 6, 8, 10],
                   lambda d: DecisionTreeRegressor(max_depth=d, random_state=0))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    plot(ax[0], r_poly, "Polynomial degree", "degree")
    plot(ax[1], r_knn, "k-NN (right = simpler)", "k")
    plot(ax[2], r_tree, "Decision tree depth", "max_depth")
    fig.tight_layout(); fig.savefig(OUT / "bias_variance_curves.png", dpi=110); plt.close(fig)

    # Bonus: ensembles attack VARIANCE. One deep tree vs 50 bagged deep trees.
    print("\nBonus - why bagging works (variance reduction, bias barely changes):")
    one = simulate(lambda: DecisionTreeRegressor(random_state=0))
    bag = simulate(lambda: BaggingRegressor(DecisionTreeRegressor(random_state=0), n_estimators=50, random_state=0))
    print(f"  single deep tree : bias²={one[0]:.4f} variance={one[1]:.4f}")
    print(f"  bagged 50 trees  : bias²={bag[0]:.4f} variance={bag[1]:.4f}")

    best = min(r_poly, key=lambda r: r[4])
    log_experiment(PROJECT, "poly_sweep", "poly", {"best_degree": best[0], "measured_mse": best[4],
                   "bias2": best[1], "variance": best[2]}, params={"sigma": SIGMA, "n_train": N_TRAIN})
    print(f"\nSweet spot for polynomial: degree {best[0]} (measured MSE {best[4]:.4f}; noise floor {SIGMA**2:.4f})")
    print(f"Plot saved: {OUT / 'bias_variance_curves.png'}")
