# Topic 7 — Cross-Validation

**Project: CV Playground** · `python projects/p07_cv_playground/main.py`

---

## 1. The five-question framework

| Question | Answer |
|---|---|
| **What is it?** | Repeatedly splitting the data into training and validation parts, training on one and scoring on the other, and averaging — to estimate how a model will perform on unseen data. |
| **Why do we need it?** | A single train/test split gives a *noisy* estimate (it depends on which rows landed where) and wastes data. Cross-validation uses every row for both training and validation and gives you a mean **and** a spread. |
| **How does it work?** | **K-Fold:** split into K folds; for each fold train on K−1, validate on the held-out one; average the K scores. |
| **When do I use it?** | Model selection, hyper-parameter tuning, and performance estimation — especially on small/medium datasets. Prefer a simple hold-out when data is huge and training is very expensive. |
| **Limitations?** | K× the training cost. Breaks (gives over-optimistic numbers) if the split ignores structure in the data: time order, groups, duplicates, or if preprocessing leaks. |

## 2. Deep dive

### K-Fold step by step
5-fold: shuffle → cut into 5 parts → 5 rounds, each part is the validation set once → report mean ± std. Typical K = 5 or 10. Larger K → less bias in the estimate, more variance/cost; **leave-one-out** (K = n) is the extreme (expensive, high-variance estimate).

### Splitter cheat-sheet (choose by the *structure of your data*)

| Situation | Use | Why |
|---|---|---|
| Regression, i.i.d. rows | `KFold(shuffle=True)` | baseline |
| **Classification / imbalanced** | `StratifiedKFold` | keeps class ratios in every fold (plain K-Fold can give a fold with zero positives) |
| **Time series** | `TimeSeriesSplit` (expanding window) | always train on the past, test on the future |
| **Repeated rows per entity** (patients, users, devices) | `GroupKFold` / `StratifiedGroupKFold` | keeps an entity entirely in train *or* validation |
| Tiny dataset, want stability | `RepeatedStratifiedKFold` | several random K-Fold rounds |
| Tuning + honest estimate | **Nested CV** | inner loop tunes, outer loop measures |

### Cross-validation vs simple train/test split
* Split is fast and fine for big data; but with 500 rows, the score depends heavily on the luck of the split.
* CV shows the **variance** of your estimate; differences smaller than the fold-to-fold std are noise.
* Keep a **final test set** that CV/tuning never touches; CV is for development.

### The most important idea: leakage
Anything that *learns from data* — scaling, imputation, encoding, feature selection, PCA, resampling — must be **fitted inside each training fold only**. Put it all in a `Pipeline` and pass the pipeline to `cross_val_score`. Fitting it on the full data first contaminates the validation folds.

### Hyper-parameter tuning bias & nested CV
`GridSearchCV.best_score_` is the *maximum* over many candidates — an optimistically biased estimate (you picked the luckiest). **Nested CV** wraps the whole search inside an outer CV loop and reports the outer scores. After that, refit the winner on all data.

### After CV
Train the final model on the **entire** training set with the chosen hyper-parameters; CV models are for estimation, not deployment.

## 3. What the project demonstrates (verified outputs)

| Experiment | Result |
|---|---|
| **A. Hold-out is a lottery** (500 rows, logistic regression) | 100 different 80/20 splits produce accuracies from **0.630 to 0.780** (range 0.15); 5-fold CV gives 0.702 ± 0.013 |
| **B. Stratification** (1 % positives, 5 folds) | Plain KFold: positives per fold `[2, 4, 1, 0, 3]` — a fold with **zero** positives. Stratified: `[2, 2, 2, 2, 2]` |
| **C. Time series** (trend + seasonality, random forest) | Shuffled KFold R² **0.972**; `TimeSeriesSplit` R² **0.297** — the shuffled score was leaked future information (and a tree can't extrapolate a trend) |
| **D. Grouped data** (labels random *per customer*) | KFold accuracy **0.963** (memorised customers); `GroupKFold` **0.516** (the truth: coin flip) |
| **E. Pipeline leakage** (pure-noise data, 2000 features) | Feature selection *before* CV: AUC **0.881** ("signal" in noise!); *inside* CV: **0.540** |
| **F. Nested CV** (150 rows, 78-candidate grid) | Grid-search best score **0.674**; nested-CV estimate **0.582 ± 0.086** |

Mind the scale of each effect: F's gap is large because the dataset is tiny and the grid big; on large datasets it shrinks.

## 4. Interview questions with model answers

**Q1. Why do we use cross-validation? Explain K-Fold and when you'd prefer it over a train-test split.**
It gives a more reliable and less wasteful estimate of generalisation than one split, and a measure of its variability. K-Fold trains K models, each validated on a different fold, and averages. I prefer it on small/medium data, for model selection/tuning; with millions of rows a single hold-out is usually enough and much cheaper.

**Q2. How do you choose K?**
5–10 is the standard trade-off: small K = pessimistic (less training data per fold) and cheap; large K = closer to using all data but costlier and higher-variance. Use repeated CV for small data.

**Q3. What is stratified K-Fold and when is it required?**
Each fold preserves the class distribution. Needed for classification, essential when imbalanced.

**Q4. How do you cross-validate a time series?**
Never shuffle. Use forward-chaining splits (`TimeSeriesSplit`) so training always precedes validation; consider a gap between them if features look back in time; evaluate on the most recent period.

**Q5. What is data leakage in CV? Give an example.**
Validation information influencing training. E.g. scaling or selecting features on the full dataset before splitting; duplicate/related rows across folds; target encoding without out-of-fold fitting. Fix: pipelines, group-aware splits, time-aware splits.

**Q6. What is nested CV and why?**
The inner loop selects hyper-parameters; the outer loop scores the *whole selection procedure* on data the search never saw, removing the optimism of `best_score_`.

**Q7. Cross-validation gives 0.85 ± 0.10. Is model A (0.86) better than model B (0.85)?**
No evidence — the difference is far smaller than the noise. Use repeated CV or paired comparisons on the same folds, and consider a significance test or practical thresholds.

**Q8. Can you use the CV models as the production ensemble?**
You *can* average them, but the standard practice is to refit one model on all training data using the selected hyper-parameters.

**Q9. Does CV replace the test set?**
No. Once you've tuned against CV scores, they're optimistic; keep an untouched test set for the final number.

## 5. Common traps

* Random KFold on time-ordered or grouped data.
* Preprocessing outside the pipeline.
* Reporting `best_score_` from a grid search as the expected performance.
* Comparing models on different splits.
* Looking at only the mean and ignoring the std.

---

## 6. Project: CV Playground — roadmap

| Phase | Time | Tasks | Done when |
|---|---|---|---|
| **1. Implement K-Fold by hand** | 2 h | Write `my_kfold(X, y, model, k)` with NumPy indexing only; compare with `cross_val_score` | Scores identical for the same folds |
| **2. Run the six experiments** | 1 h | Run `main.py`; for each of A–F write a two-sentence "what went wrong / what fixes it" | Notes file |
| **3. Repeat & vary** | 2 h | Vary dataset size in exp. A (200, 500, 5000): how does the hold-out range shrink? Vary the grid size in F | Two small plots |
| **4. Add your own leak** | 2 h | Create a leak of your own (e.g. duplicate rows across folds; target encoding on all data); show inflated CV, then fix it | Before/after numbers |
| **5. Paired comparison** | 2 h | Compare two models on the same 10×5 repeated folds; run a paired t-test / Wilcoxon on fold differences | A defensible "A > B" or "no difference" |
| **6. CV for the time series** | 3 h | Use the `transactions` table ordered by time: `TimeSeriesSplit` with a `gap`; plot per-fold recall over time to look for drift | Drift visible |
| **7. Persist** | 1 h | Log each fold's score to the `metrics` table (`split='cv'`) and query mean/std per model in SQL | SQL query works |

**Stretch:** blocked/purged CV for financial data; conformal prediction; `sklearn.model_selection.cross_validate` with several scorers and `return_train_score=True` to see overfitting per fold.

**Resume bullet:** *"Built a validation-strategy testbed demonstrating and fixing six failure modes (hold-out variance, missing stratification, temporal and group leakage, pipeline leakage, tuning optimism) with nested cross-validation."*
