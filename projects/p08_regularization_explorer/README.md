# Topic 8 — Regularization (L1 and L2)

**Project: Regularization Path Explorer** · `python projects/p08_regularization_explorer/main.py`

---

## 1. The five-question framework

| Question | Answer |
|---|---|
| **What is it?** | Adding a *penalty on model complexity* to the training objective, so the model must trade fit against simplicity. |
| **Why do we need it?** | Unconstrained models (many features, few rows, correlated features) get huge, unstable coefficients that fit noise. Regularisation trades a little bias for a big reduction in variance (topics 2–3). |
| **How does it work?** | `minimise  loss(w) + λ · penalty(w)`. **L2 (Ridge)** penalty = Σwⱼ² shrinks all weights smoothly. **L1 (Lasso)** penalty = Σ\|wⱼ\| shrinks *and* drives some weights to exactly zero. |
| **When do I use it?** | Almost always with linear/logistic models, especially when `p` is large relative to `n`, features are correlated, or you want automatic feature selection (L1). |
| **Limitations?** | Requires scaled features and a tuned λ. Adds bias. L1 is unstable among correlated features. It does nothing for *un*derfitting (it makes it worse). |

## 2. Deep dive

### The objective functions
* **Ridge:** `Σ(yᵢ − xᵢw)² + λ‖w‖₂²` → closed form `w = (XᵀX + λI)⁻¹Xᵀy`. Adding `λI` makes the matrix invertible even when features are collinear or `p > n` — a big reason Ridge is numerically stable.
* **Lasso:** `(1/2n)Σ(yᵢ − xᵢw)² + λ‖w‖₁` → no closed form; solved by **coordinate descent** using the **soft-threshold** operator `S(ρ, λ) = sign(ρ)·max(|ρ| − λ, 0)`. If a feature's correlation with the current residual is smaller than λ, its weight becomes *exactly* zero.
* **Elastic Net:** mixes both: `λ[ α‖w‖₁ + (1−α)/2 ‖w‖₂² ]` — sparsity from L1, stability with correlated features from L2.
* The **bias/intercept is not penalised.**

### Why L1 gives zeros and L2 doesn't (geometry)
Minimising the loss subject to `penalty(w) ≤ t`. The L2 constraint region is a *circle/sphere* — the loss contours touch it at a generic point, so no coordinate is exactly zero. The L1 region is a *diamond* with corners on the axes — the contours are likely to touch at a corner, where some coordinates are zero.

Probabilistic view (nice to mention): **Ridge = MAP estimate with a Gaussian prior on weights; Lasso = Laplace prior.**

### Ridge vs Lasso vs Elastic Net

| | Ridge (L2) | Lasso (L1) | Elastic Net |
|---|---|---|---|
| Coefficients | shrunk, never exactly 0 | many exactly 0 | some 0, groups kept together |
| Feature selection | no | yes | yes |
| Correlated features | shares weight fairly | picks arbitrary one(s), unstable | keeps them together |
| `p > n` | fine | selects at most `n` features | fine |
| Closed form | yes | no | no |
| Typical use | many small effects | few big effects, want sparse model | correlated groups + sparsity |

### Practicalities (interviewers love these)
1. **Scale features first** — the penalty treats all weights equally; unscaled features are penalised unfairly.
2. **Choose λ by cross-validation** (`RidgeCV`, `LassoCV`, `ElasticNetCV`, or `GridSearchCV`). λ ↑ = simpler model = more bias, less variance.
3. **scikit-learn naming:** `alpha` in `Ridge/Lasso` is λ; in `LogisticRegression` the knob is `C = 1/λ` — **small C = strong regularisation**. `LogisticRegression` applies L2 by default.
4. Standardise, then fit; interpret coefficients on the standardised scale (or convert back).
5. Regularisation ≠ selecting features on the full data: do selection inside CV.

### Regularisation beyond linear models
* **Neural nets:** weight decay (L2; use decoupled `AdamW`), **dropout**, **early stopping**, data augmentation, batch-norm side effects.
* **Trees/boosting:** `max_depth`, `min_samples_leaf`, pruning (`ccp_alpha`), **learning rate + shrinkage**, `subsample`, `colsample_bytree`, L1/L2 leaf penalties in XGBoost/LightGBM.
* **SVM:** `C` (margin softness).
* **Bayesian:** priors.

## 3. What the project demonstrates (verified outputs)

Synthetic regression: `n_train=100`, `p=50` correlated features, only 8 truly relevant, noise σ=3 (so the best possible test RMSE ≈ 3.0).

* **From-scratch Ridge** (closed form) matches scikit-learn to **1.8e-15**. **From-scratch Lasso** (coordinate descent + soft-thresholding) matches to **2.1e-10** with the same 15 non-zero coefficients.
* **Unregularised OLS overfits:** test RMSE **4.35**.
* **Cross-validated penalties** (λ chosen from training data only):

| Model | Test RMSE | # non-zero coefficients | Note |
|---|---|---|---|
| OLS | 4.351 | 50 | no regularisation |
| RidgeCV (α≈7.0) | 3.916 | 50 | all weights shrunk |
| LassoCV (α≈0.23) | **3.741** | 20 | kept 7 of 8 true features + 13 spurious |
| ElasticNetCV | 3.746 | 21 | nearly identical to Lasso here |

  Lasso does *feature selection*, but with correlated features and only 100 rows it is not perfect — it missed one true feature and kept some false ones.
* **Correlated twins** (`x2 ≈ x1`, true weights 1 and 1): across 5 different samples OLS and Lasso split the weight **arbitrarily and differently each time** (e.g. `(0.49, 1.57)`, `(1.16, 0.62)` for OLS), while Ridge (≈ `(1.0, 1.0)`) and Elastic Net (≈ equal pairs) share it consistently.
* **Logistic regression on breast-cancer** (30 features): as `C` shrinks, L1 keeps only 4 features at `C=0.01` with AUC 0.982 versus 0.992 for L2 with all 30; at `C=0.003` L1 zeroes everything (AUC 0.5) — *too much* regularisation.
* Plot `outputs/regularization_paths.png` shows the four classic curves: Ridge path, Lasso path, number of non-zeros, and U-shaped test error.

## 4. Interview questions with model answers

**Q1. What are L1 and L2 regularisation and how do they prevent overfitting?**
Penalties added to the loss: L1 = λΣ|w|, L2 = λΣw². They constrain weight magnitude so the model can't rely on large, noise-fitting coefficients: variance falls, bias rises slightly, and generalisation usually improves. L2 shrinks smoothly; L1 also produces sparse solutions.

**Q2. Why does L1 produce sparsity and L2 not?**
Geometry (diamond vs circle constraint) or calculus: the L1 gradient is a constant ±λ regardless of the weight's size, so it can push a weight all the way to zero, whereas the L2 gradient shrinks proportionally to the weight and only approaches zero asymptotically.

**Q3. When do you choose Lasso, Ridge or Elastic Net?**
Ridge: many features with small effects, multicollinearity. Lasso: you believe few features matter and want interpretability/selection. Elastic Net: correlated groups of features and you still want sparsity, or `p ≫ n`.

**Q4. Why standardise features before regularising?**
The penalty is scale-dependent: a feature measured in small units needs a large coefficient and would be penalised more, so results would depend on units.

**Q5. What happens as λ → 0 and λ → ∞?**
λ → 0: ordinary least squares (high variance). λ → ∞: all weights → 0, the model predicts the mean (high bias).

**Q6. How do you pick λ?**
Cross-validation over a log-spaced grid; inspect the regularisation path; use the "one-standard-error rule" to prefer the simplest model within one std of the best.

**Q7. How does regularisation relate to the bias–variance trade-off?**
It's a knob on it: more regularisation = more bias, less variance.

**Q8. Does regularisation help an underfit model?**
No, it worsens it. Reduce regularisation or add capacity.

**Q9. Regularisation in a neural network?**
Weight decay, dropout, early stopping, data augmentation, smaller network, batch norm; label smoothing for classifiers.

**Q10. Is intercept regularised?**
No — shrinking it toward zero would make predictions depend on where the target is centred.

## 5. Common traps

* Forgetting to scale features.
* Reading Lasso's "selected" features as causal/stable when features are correlated.
* Confusing `alpha` (bigger = stronger) with `C` (bigger = *weaker*).
* Tuning λ on the test set.
* Using L1 selection on all data before cross-validation.

---

## 6. Project: Regularization Path Explorer — roadmap

| Phase | Time | Tasks | Done when |
|---|---|---|---|
| **1. Derive** | 2 h | On paper: derive Ridge's closed form and the Lasso coordinate update (soft-thresholding) | Notes with both derivations |
| **2. Read the scratch code** | 2 h | Trace `lasso_coordinate_descent` on a 2-feature example by hand | Matches the code's output |
| **3. Break it** | 2 h | Skip the `StandardScaler`; give one feature ×1000 scale. What happens to Ridge/Lasso coefficients? | Explain in your own words |
| **4. Real data** | 3 h | Load the `houses` table (`get_dataset('houses')`); fit polynomial features (degree 3) + Ridge/Lasso with `RidgeCV/LassoCV`; compare test RMSE and selected terms | Table + short interpretation |
| **5. Path animation** | 2 h | Plot coefficient paths with `sklearn.linear_model.lasso_path`; label the first 5 features to enter | Labelled plot |
| **6. Stability selection** | 3 h | Refit Lasso on 100 bootstrap samples; report how often each feature is selected. Compare with the true support | Selection-frequency bar chart |
| **7. Neural-net regularisation** | 3 h | Train an MLP with/without weight decay, dropout, early stopping (PyTorch/Keras or `MLPClassifier(alpha=...)`) | Overfitting curves compared |

**Stretch:** implement Ridge with gradient descent and show it converges to the closed form; adaptive/relaxed Lasso; group Lasso.

**Resume bullet:** *"Implemented Ridge (closed form) and Lasso (coordinate descent) from scratch, verified to 1e-10 against scikit-learn, and analysed regularisation paths, sparsity and correlated-feature behaviour under cross-validated tuning."*
