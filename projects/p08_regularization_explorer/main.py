"""
PROJECT 8 - Regularization Path Explorer

  1. Ridge (L2) FROM SCRATCH  - closed form, verified against scikit-learn
  2. Lasso (L1) FROM SCRATCH  - coordinate descent + soft-thresholding, verified against scikit-learn
  3. Coefficient paths + test error vs regularisation strength
  4. Choosing alpha with cross-validation (RidgeCV / LassoCV / ElasticNetCV) + feature recovery
  5. The correlated-features behaviour that separates L1, L2 and Elastic Net
  6. L1 vs L2 for classification (logistic regression, parameter C = 1/lambda)

Run:  python projects/p08_regularization_explorer/main.py
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import (ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression,
                                  LogisticRegression, Ridge, RidgeCV)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.db import log_experiment

PROJECT = "p08_regularization"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
N_TRAIN, P, K_TRUE, SIGMA = 100, 50, 8, 3.0


# --------------------------------------------------------------------------- #
# Data: p=50 correlated features, only 8 truly matter, noisy target
# --------------------------------------------------------------------------- #
def make_data(n, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(P)
    cov = 0.5 ** np.abs(idx[:, None] - idx[None, :])              # AR(1) correlation between features
    X = rng.multivariate_normal(np.zeros(P), cov, size=n)
    beta = np.zeros(P)
    beta[:K_TRUE] = [4, -3, 2.5, -2, 1.5, 1.5, -1, 1]
    y = X @ beta + rng.normal(0, SIGMA, n)
    return X, y, beta


# --------------------------------------------------------------------------- #
# From-scratch implementations (X standardised, y centred -> no intercept needed)
# --------------------------------------------------------------------------- #
def ridge_closed_form(X, y, alpha):
    """minimise ||y - Xw||^2 + alpha ||w||^2   ->   w = (X'X + alpha I)^-1 X'y"""
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def soft_threshold(rho, lam):
    return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)


def lasso_coordinate_descent(X, y, alpha, n_iter=2000, tol=1e-10):
    """minimise (1/2n)||y - Xw||^2 + alpha ||w||_1   (same objective as sklearn.Lasso)"""
    n, p = X.shape
    w = np.zeros(p)
    z = (X ** 2).sum(0) / n
    r = y - X @ w
    for _ in range(n_iter):
        w_old = w.copy()
        for j in range(p):
            r += X[:, j] * w[j]                       # remove feature j's contribution
            rho = X[:, j] @ r / n
            w[j] = soft_threshold(rho, alpha) / z[j]  # exact 1-D minimiser -> exact zeros appear!
            r -= X[:, j] * w[j]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


def main():
    Xtr_raw, ytr_raw, beta = make_data(N_TRAIN, seed=0)
    Xte_raw, yte_raw, _ = make_data(2000, seed=1)
    sc = StandardScaler().fit(Xtr_raw)                          # fit on TRAIN only
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)
    y_mean = ytr_raw.mean()
    ytr, yte = ytr_raw - y_mean, yte_raw - y_mean
    rmse = lambda w: float(np.sqrt(np.mean((yte - Xte @ w) ** 2)))
    print(f"n_train={N_TRAIN}, p={P}, truly relevant features={K_TRUE}, noise sigma={SIGMA} "
          f"(best possible test RMSE ~ {SIGMA})")

    # ---- 1. Ridge from scratch --------------------------------------------------------------
    w_mine = ridge_closed_form(Xtr, ytr, alpha=10.0)
    w_sk = Ridge(alpha=10.0, fit_intercept=False).fit(Xtr, ytr).coef_
    print(f"\n[1] Ridge scratch vs sklearn: max |diff| in coefficients = {np.max(np.abs(w_mine - w_sk)):.2e}")

    # ---- 2. Lasso from scratch --------------------------------------------------------------
    w_mine = lasso_coordinate_descent(Xtr, ytr, alpha=0.3)
    w_sk = Lasso(alpha=0.3, fit_intercept=False, tol=1e-10, max_iter=100000).fit(Xtr, ytr).coef_
    print(f"[2] Lasso scratch vs sklearn: max |diff| = {np.max(np.abs(w_mine - w_sk)):.2e}; "
          f"non-zero coefficients: scratch={np.sum(w_mine != 0)}, sklearn={np.sum(w_sk != 0)}")

    # ---- 3. Paths ---------------------------------------------------------------------------
    ridge_alphas = np.logspace(-2, 4, 40)
    lasso_alphas = np.logspace(-3, 0.7, 40)
    ridge_path = np.array([ridge_closed_form(Xtr, ytr, a) for a in ridge_alphas])
    lasso_path = np.array([Lasso(alpha=a, fit_intercept=False, max_iter=50000).fit(Xtr, ytr).coef_
                           for a in lasso_alphas])
    ridge_err = [rmse(w) for w in ridge_path]
    lasso_err = [rmse(w) for w in lasso_path]
    ols = LinearRegression(fit_intercept=False).fit(Xtr, ytr).coef_
    print(f"\n[3] OLS (no regularisation) test RMSE = {rmse(ols):.3f}   (overfits: p=50 vs n=100)")
    print(f"    best Ridge over grid: alpha={ridge_alphas[np.argmin(ridge_err)]:.2f} RMSE={min(ridge_err):.3f}")
    print(f"    best Lasso over grid: alpha={lasso_alphas[np.argmin(lasso_err)]:.3f} RMSE={min(lasso_err):.3f} "
          f"(non-zeros: {np.sum(lasso_path[np.argmin(lasso_err)] != 0)})")
    print("    (grid-best is picked on test for illustration only; section 4 does it properly with CV)")

    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    ax[0].semilogx(ridge_alphas, ridge_path); ax[0].set_title("Ridge path: shrinks, never exactly 0")
    ax[1].semilogx(lasso_alphas, lasso_path); ax[1].set_title("Lasso path: coefficients hit exactly 0")
    ax[2].semilogx(lasso_alphas, [(np.abs(w) > 1e-12).sum() for w in lasso_path], "o-")
    ax[2].set_title("Lasso: # non-zero coefficients")
    ax[3].semilogx(ridge_alphas, ridge_err, label="Ridge"); ax[3].semilogx(lasso_alphas, lasso_err, label="Lasso")
    ax[3].axhline(rmse(ols), c="grey", ls="--", label="OLS"); ax[3].set_title("Test RMSE vs alpha (U-shape)")
    ax[3].legend()
    for a in ax: a.set_xlabel("alpha (regularisation strength)")
    fig.tight_layout(); fig.savefig(OUT / "regularization_paths.png", dpi=110); plt.close(fig)

    # ---- 4. CV-chosen alpha + feature recovery ---------------------------------------------
    print("\n[4] Cross-validated regularisation (alpha chosen ONLY from training data)")
    true_support = set(np.flatnonzero(beta))
    rows = {"OLS": (None, ols),
            "RidgeCV": (None, None), "LassoCV": (None, None), "ElasticNetCV": (None, None)}
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 60), fit_intercept=False).fit(Xtr, ytr)
    lasso = LassoCV(cv=5, fit_intercept=False, random_state=0, max_iter=50000).fit(Xtr, ytr)
    enet = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 0.95], cv=5, fit_intercept=False, random_state=0,
                        max_iter=50000).fit(Xtr, ytr)
    rows["RidgeCV"] = (ridge.alpha_, ridge.coef_)
    rows["LassoCV"] = (lasso.alpha_, lasso.coef_)
    rows["ElasticNetCV"] = (f"{enet.alpha_:.3f}, l1_ratio={enet.l1_ratio_}", enet.coef_)
    print(f"    {'model':<14}{'alpha':>22}{'test RMSE':>11}{'#nonzero':>10}{'true feats found':>18}{'false feats':>12}")
    for name, (a, w) in rows.items():
        sel = set(np.flatnonzero(np.abs(w) > 1e-8))
        a_txt = "-" if a is None else (f"{a:.3f}" if isinstance(a, float) else str(a))
        if name in ("OLS", "RidgeCV"):      # these keep every feature - "recovery" is meaningless
            print(f"    {name:<14}{a_txt:>22}{rmse(w):>11.3f}{len(sel):>10}{'n/a':>18}{'':>12}")
        else:
            print(f"    {name:<14}{a_txt:>22}{rmse(w):>11.3f}{len(sel):>10}{len(sel & true_support):>15}/{K_TRUE}{len(sel - true_support):>12}")
        log_experiment(PROJECT, name, name, {"test_rmse": rmse(w), "n_nonzero": len(sel)},
                       params={"alpha": str(a)}, dataset_name="synthetic_p50")
    print("    Lasso performs feature selection (few non-zeros) but may keep a few false features when features are correlated.")

    # ---- 5. correlated features --------------------------------------------------------------
    print("\n[5] Two almost-identical features (x2 = x1 + tiny noise); true model y = x1 + x2 (total weight = 2)")
    print("    coefficients (x1, x2) on 5 different random samples:")
    makers = {"OLS": lambda: LinearRegression(), "Ridge(a=10)": lambda: Ridge(alpha=10),
              "Lasso(a=0.5)": lambda: Lasso(alpha=0.5), "ElasticNet(a=0.5,l1=0.5)": lambda: ElasticNet(alpha=0.5, l1_ratio=0.5)}
    for name, mk in makers.items():
        cells = []
        for seed in range(5):
            r = np.random.default_rng(seed)
            x1 = r.normal(size=300); x2 = x1 + r.normal(0, 0.05, 300)
            Xc = StandardScaler().fit_transform(np.c_[x1, x2]); yc = x1 + x2 + r.normal(0, 0.5, 300)
            c = mk().fit(Xc, yc).coef_
            cells.append(f"({c[0]:.2f},{c[1]:.2f})")
        print(f"    {name:<26}" + "  ".join(cells))
    print("    OLS and Lasso split the weight ARBITRARILY between the twins (different every sample -> unstable);")
    print("    Ridge and Elastic Net share it consistently (the totals differ from 2 because penalties shrink coefficients).")

    # ---- 6. logistic regression -------------------------------------------------------------
    print("\n[6] Logistic regression on breast-cancer (30 features): L1 vs L2 as C varies (small C = strong penalty)")
    Xb, yb = load_breast_cancer(return_X_y=True)
    print(f"    {'C':>8}{'L2 AUC':>10}{'L2 #nz':>8}{'L1 AUC':>10}{'L1 #nz':>8}")
    for C in (0.003, 0.01, 0.03, 0.1, 1, 10):
        out = []
        for pen in ("l2", "l1"):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty=pen, solver="liblinear", max_iter=5000))
            auc = cross_val_score(clf, Xb, yb, cv=5, scoring="roc_auc").mean()
            nz = int(np.sum(clf.fit(Xb, yb)[-1].coef_ != 0))
            out += [auc, nz]
        print(f"    {C:>8}{out[0]:>10.3f}{out[1]:>8}{out[2]:>10.3f}{out[3]:>8}")
    print("    L1 zeroes out features as C shrinks (built-in feature selection); L2 keeps all 30 but shrinks them.")
    print(f"\nPlot saved: {OUT / 'regularization_paths.png'}")


if __name__ == "__main__":
    main()
