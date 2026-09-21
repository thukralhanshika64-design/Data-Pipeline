# Topic 9 — Ensemble Learning (Bagging, Boosting, Random Forest, Gradient Boosting)

**Project: Ensemble Arena** · `python projects/p09_ensemble_arena/main.py` (≈ 50 s)

---

## 1. The five-question framework

| Question | Answer |
|---|---|
| **What is it?** | Combining many models ("weak/base learners") into one stronger predictor. |
| **Why do we need it?** | A single model is either high-variance (deep tree) or high-bias (stump). Combining many well-chosen models can cut variance (bagging), cut bias (boosting), or blend different inductive biases (stacking/voting). Ensembles dominate tabular-data benchmarks. |
| **How does it work?** | *Parallel:* train independent models on resampled data and average (bagging, random forest). *Sequential:* each model corrects the previous ones (boosting). *Meta:* learn how to combine models (stacking). |
| **When do I use it?** | Tabular data where accuracy matters and interpretability/latency are secondary; whenever a single tree overfits. |
| **Limitations?** | Slower and larger (topic 10), harder to interpret, boosting can overfit noisy labels and needs tuning, no free lunch — on simple/linear problems a linear model may tie. |

## 2. Deep dive

### Why averaging works
If `B` predictors each have variance σ² and pairwise correlation ρ, their average has variance `ρσ² + (1 − ρ)σ²/B`. Averaging kills the second term; **lowering ρ** (making models *different*) kills the first. Bias of the average equals the average bias — so bagging is for *low-bias, high-variance* learners (deep trees).

### Bagging (Bootstrap AGGregatING)
1. Draw `B` bootstrap samples (n rows *with replacement*, ≈ 63.2 % unique rows each).
2. Train one model (usually a deep tree) per sample.
3. Average (regression) or majority/probability vote (classification).

The ≈ 36.8 % rows *left out* of each bootstrap are **out-of-bag (OOB)** — a free validation set (`oob_score=True`).

### Random Forest = bagging + feature randomness
At each split only a random subset of features (`max_features`; `√p` classification, `p/3` regression as folklore defaults) is considered. That **decorrelates the trees** (lower ρ), which is the whole trick. Properties: hard to overfit by adding trees, minimal tuning, parallel, handles mixed features, gives feature importances (impurity-based ones are biased towards continuous/high-cardinality features → check with permutation importance/SHAP). Extra Trees adds random split thresholds for even more decorrelation.

### Boosting
Train learners **sequentially**, each focusing on what the ensemble still gets wrong.

* **AdaBoost:** re-weights training rows — misclassified rows get higher weight; learners (usually stumps) get a vote weight based on accuracy. Sensitive to label noise/outliers (they get ever-higher weights).
* **Gradient Boosting:** stage-wise gradient descent *in function space*. Start with a constant `F₀`. For `m = 1..M`: compute the **negative gradient** of the loss w.r.t. current predictions (for squared error that is simply the *residual* `y − F_{m−1}(x)`), fit a small tree to it, update `F_m = F_{m−1} + η·tree_m`. Because it works for any differentiable loss (log-loss, Huber, quantile, ranking) it is very general.
* Key knobs: `n_estimators`, **learning rate η** (small η + more trees = better but slower), `max_depth` (3–8; boosting wants *weak* learners), `subsample` (stochastic GB), `colsample`, regularisation (L1/L2 on leaf weights, `min_child_weight`, `gamma`), **early stopping**.
* Modern implementations: **XGBoost, LightGBM, CatBoost** (histogram binning, second-order gradients, native missing-value and categorical handling); scikit-learn's `HistGradientBoosting*` is the built-in equivalent.

### Stacking and voting
* **Voting:** average probabilities (soft) or labels (hard) from different model families.
* **Stacking:** out-of-fold predictions of base models become features for a meta-learner (often a regularised linear model). Gains are usually small and cost complexity — use at the end of a competition, rarely in production.

### Side-by-side

| | Bagging / Random Forest | Boosting |
|---|---|---|
| Training | parallel, independent | sequential, dependent |
| Reduces | **variance** | **bias** (and some variance) |
| Base learner | deep, low-bias trees | shallow, weak trees |
| Overfitting by adding rounds | rarely | yes → needs early stopping |
| Noise sensitivity | robust | more sensitive (especially AdaBoost) |
| Tuning effort | low | high |
| Typical accuracy on tabular data | very good | often best |

## 3. What the project demonstrates (verified outputs)

**From scratch vs scikit-learn** (house price, test RMSE; lower is better):

| Model | RMSE |
|---|---|
| single deep tree | 47,560 |
| shallow tree (depth 3) | 51,626 |
| Bagging — *my* implementation (100 trees) | 34,902 |
| `BaggingRegressor` (sklearn) | 33,760 |
| Random forest | 34,154 |
| Gradient boosting — *my* implementation | 33,844 |
| `GradientBoostingRegressor` (sklearn) | 33,870 |

Both scratch versions are ~25 lines and land close to the library.

**Classification arena** (churn, 5-fold CV ROC-AUC ± std):

| Model | AUC | std |
|---|---|---|
| Logistic regression (reference) | **0.7755** | 0.013 |
| AdaBoost (stumps) | 0.7712 | 0.013 |
| Gradient boosting | 0.7687 | 0.008 |
| Voting (LR + RF + HGB) | 0.7636 | 0.009 |
| Extra trees | 0.7601 | 0.011 |
| Stacking | 0.7548 | 0.007 |
| Random forest | 0.7548 | 0.007 |
| HistGradientBoosting | 0.7400 | 0.008 |
| Decision tree, depth 4 | 0.7347 | 0.016 |
| Bagging of unpruned trees | 0.7172 | 0.006 |
| Decision tree, unpruned | **0.5909** | 0.016 |

Read it carefully: (a) the unpruned tree is terrible (0.59) — extreme variance; bagging it lifts AUC by 0.13 — that's variance reduction in action; (b) **the plain logistic regression is at the top**, because this dataset's churn mechanism is nearly linear-logistic. Ensembles are not automatically better — that's why the reference baseline is in the table; (c) many gaps are within one std. (Project 10 uses a non-linear variant of the data where ensembles clearly win.)

**Random forest extras:** OOB accuracy 0.727 vs held-out test accuracy 0.754 — the free OOB estimate is in the right neighbourhood.

**Boosting: learning rate vs number of trees** (validation AUC):

| η | best round | best AUC | AUC after 400 rounds |
|---|---|---|---|
| 0.3 | 11 | 0.7821 | 0.7181 |
| 0.1 | 32 | 0.7857 | 0.7578 |
| 0.03 | 138 | 0.7877 | 0.7792 |

A high learning rate peaks early and then overfits badly; a small one peaks later and degrades slowly — hence "small η + many trees + early stopping".

## 4. Interview questions with model answers

**Q1. What is ensemble learning? Difference between bagging, boosting, random forest, gradient boosting?**
Combining models for better performance. Bagging trains independent models on bootstrap samples and averages them (variance ↓). Random forest = bagging of trees + random feature subsets per split (decorrelates trees). Boosting trains models sequentially, each correcting the previous errors (bias ↓); gradient boosting does this by fitting each new tree to the negative gradient (residuals) of the loss.

**Q2. Why does a random forest use random feature subsets?**
To decorrelate trees. Without it, strong features make all trees similar (high ρ) and averaging helps less.

**Q3. Bagging vs boosting — when to prefer each?**
Bagging/RF for a robust, low-tuning baseline, noisy data, parallel training, high-variance base learners. Boosting when you need maximum accuracy and can tune, with clean-ish labels.

**Q4. How does gradient boosting work? Why "gradient"?**
Iteratively adds trees fitted to the negative gradient of the loss w.r.t. current predictions; for squared loss that's the residual. It's gradient descent in function space.

**Q5. How do you prevent boosting from overfitting?**
Small learning rate, shallow trees, early stopping on a validation set, subsampling rows/columns, L1/L2 leaf regularisation, `min_child_weight`, limiting the number of rounds.

**Q6. What's OOB error?**
Each tree ignores ≈ 37 % of the rows; predicting those rows with the trees that didn't see them gives an unbiased-ish validation estimate without a separate hold-out.

**Q7. Random forest vs gradient boosting: which is faster to train/tune?**
RF trains in parallel with few hyper-parameters; boosting is sequential (though histogram methods are fast) and needs more tuning.

**Q8. Can ensembles hurt?**
Yes: added latency/size/complexity, harder debugging and explanation, and no gain if base models are highly correlated or the problem is simple.

**Q9. What is stacking and what's the leakage risk?**
A meta-learner trained on base models' predictions. The base predictions used to train it must be **out-of-fold**, otherwise the meta-learner sees overfit predictions.

**Q10. XGBoost vs LightGBM vs CatBoost?**
All gradient boosting with engineering differences: XGBoost = level-wise growth with strong regularisation options; LightGBM = leaf-wise growth + histograms (fastest on large data, can overfit small data); CatBoost = ordered boosting and native categorical handling.

## 5. Common traps

* Bagging *shallow* trees (little to reduce) or boosting *deep* trees (already overfit).
* Trusting impurity-based feature importances.
* Not using early stopping with boosting.
* Concluding "ensembles win" without a simple baseline.
* Stacking with in-sample base predictions.

---

## 6. Project: Ensemble Arena — roadmap

| Phase | Time | Tasks | Done when |
|---|---|---|---|
| **1. Read scratch code** | 2 h | Line-by-line: `SimpleBagging`, `SimpleGradientBoosting`. Write in a comment what each line does to bias/variance | Notes |
| **2. Extend scratch code** | 3 h | Add feature subsampling to `SimpleBagging` to turn it into a random forest; add a `subsample` option to boosting; test vs sklearn | RMSE within ~5 % of sklearn |
| **3. AdaBoost from scratch** | 3 h | Implement AdaBoost.M1 with decision stumps: sample weights, `α = ½ ln((1−ε)/ε)`. Compare with `AdaBoostClassifier` | Same AUC ±0.01 |
| **4. Tune properly** | 3 h | `RandomizedSearchCV` on HistGB/LightGBM (`learning_rate`, `max_depth`, `min_samples_leaf`, `l2_regularization`) with early stopping inside CV; record every run in the DB | Best config + leaderboard SQL |
| **5. Non-linear data** | 1 h | Switch to `get_dataset('customers_nl')` (non-linear churn rules) and rerun the arena. Who wins now? | Written explanation |
| **6. Interpretation** | 2 h | Permutation importance vs impurity importance for the forest; explain any disagreement | Two bar charts |
| **7. XGBoost/LightGBM** | 2 h | `pip install xgboost lightgbm`; benchmark against `HistGradientBoosting` on time and AUC | Table |

**Stretch:** monotonic constraints; Isolation Forest for anomaly detection; a stacking ensemble with proper out-of-fold predictions coded by hand.

**Resume bullet:** *"Implemented bagging and gradient boosting from scratch (within ~3.5 % of scikit-learn's RMSE) and benchmarked 11 ensemble/baseline models with cross-validated AUC, OOB estimates and learning-rate/early-stopping analysis."*
