# Topic 4 — Feature Engineering

**Project: Feature Factory** · `python projects/p04_feature_factory/main.py`

---

## 1. The five-question framework

| Question | Answer |
|---|---|
| **What is it?** | Turning raw data into inputs that make the pattern easier for a model to learn: cleaning, encoding, transforming, combining, selecting. |
| **Why do we need it?** | Models only see numbers. Good features encode domain knowledge, expose non-linear or ratio relationships, and handle messy reality (missing values, categories, dates). On tabular data, features usually matter more than the choice of algorithm. |
| **How does it work?** | A reproducible transformation **pipeline**, fitted on training data only, applied identically at training and serving time. |
| **When do I use it?** | Always for tabular data; the amount depends on the model (linear/NN/KNN need much more care than tree ensembles). |
| **Limitations?** | Every feature is a hypothesis — some don't help (measure them!). Fitted transformations can **leak** test information if done outside the CV loop. Complex features hurt maintainability and can break at serving time. |

## 2. Deep dive

### Numerical features

| Technique | What it does | When |
|---|---|---|
| **Standardisation** `(x−μ)/σ` | mean 0, std 1 | linear/logistic regression, SVM, k-NN, neural nets, PCA, regularised models (penalties assume comparable scales). *Not needed for trees.* |
| **Min-max scaling** | squeezes to [0,1] | bounded inputs, image pixels, some NNs. Sensitive to outliers |
| **Robust scaling** (median/IQR) | ignores outliers | heavy-tailed data |
| **Log / sqrt / Box-Cox / Yeo-Johnson** | reduces right skew, turns multiplicative effects into additive | prices, counts, durations |
| **Clipping / winsorising** | caps extremes | obvious data errors, outlier-sensitive models |
| **Binning** (`KBinsDiscretizer`, thresholds like `tenure<=6`) | non-linear steps for linear models; robustness | when the effect is a threshold, not a slope |
| **Ratios / differences / aggregates** | `total/tenure`, `debt/income` | domain-driven, often the strongest features |
| **Polynomial / interactions** | `a·b`, `a²` | linear models (trees find interactions themselves) |
| **Missing values** | median/mean/mode imputation, model-based (KNN/iterative), **plus a missing-indicator flag** | *missingness can itself carry signal* |

### Categorical features

| Technique | Notes |
|---|---|
| **One-hot** | default for low cardinality; creates `k` columns; use `handle_unknown="ignore"` |
| **Ordinal** | only when categories have a real order (`low<medium<high`); giving arbitrary integers to nominal categories tricks linear models into a fake order |
| **Target encoding** | replace category by (smoothed) mean of target — compact for high cardinality, but **leaks the label** unless computed out-of-fold. scikit-learn's `TargetEncoder` cross-fits internally |
| **Frequency / count encoding** | category → how common it is |
| **Hashing** | fixed-size, no vocabulary; collisions accepted |
| **Embeddings** | learned dense vectors (neural nets), for very high cardinality |
| **Rare-label grouping** | lump categories below a threshold into "other" |

### Dates and cycles
Extract year/month/day-of-week/hour, "days since event", holiday flags. For **cyclical** values use `sin(2πm/12)` and `cos(2πm/12)` so December and January are neighbours. Domain flags (e.g. "signed up in Q4") can beat generic transforms if you know the mechanism.

### Feature selection
* **Filter:** variance threshold, correlation, mutual information, χ² — fast, model-agnostic.
* **Wrapper:** recursive feature elimination — expensive, model-aware.
* **Embedded:** L1 regularisation (topic 8), tree importances.
* Selection is *part of the model*: do it **inside** cross-validation (topic 7 shows selection on all data finding "signal" in pure noise).

### The rules that keep you honest
1. **Split first, then fit transformers on train only** (scalers, imputers, encoders, selectors). Put everything in a `Pipeline` + `ColumnTransformer` so cross-validation refits them per fold.
2. Stateless row-wise features (ratios, logs, date parts) are safe to compute before the split.
3. Same code path for training and serving — a feature store or a saved pipeline object.
4. **Ablate:** add feature groups one at a time and compare CV score ± std.

## 3. What the project demonstrates (verified outputs)

An **ablation ladder** on the churn data, 5-fold CV ROC-AUC (`^` = gain larger than fold noise, `~` = within noise):

| Feature set | LogReg AUC | HistGB AUC |
|---|---|---|
| L0 raw numerics + one-hot | 0.7755 | 0.7323 |
| L1 + log/ratio/flag/interaction/ordinal/missing-indicators | 0.7750 (`~`) | 0.7399 (`~`) |
| L2 + date features | 0.7813 (`~`) | 0.7448 (`^`) |
| L3 + `city` target-encoded | **0.7915 (`^`)** | 0.7557 (`^`) |
| L3b + `city` one-hot (alternative) | 0.7919 (`^`) | **0.7580 (`^`)** |

Honest take-aways:

* Not every clever transform pays: the L1 bundle was **within noise** for both models. That is exactly why you ablate.
* The gains came from information tied to how the data was generated: the **`city`** effect (a high-cardinality categorical; helped both models) and the date group with its **Q4-signup** flag (a clear gain for HistGB, but only `~` for LogReg) — i.e. categorical/domain information, not generic transforms. The same feature group can help one model family and not another, so ablate per model.
* Target encoding **tied** one-hot with 30 cities. Its advantage is expected at high cardinality; in a quick test with 500 cities (`get_dataset("customers", refresh=True, n_cities=500)`) it edged one-hot (0.7725 vs 0.7676 AUC for LogReg) but within noise. Test it yourself before claiming it wins.
* The engineered matrix is saved to the `customer_features` DB table — a minimal **feature store**.

## 4. Interview questions with model answers

**Q1. What is feature engineering and which techniques would you use for numerical and categorical features?**
Creating/transforming inputs to improve learnability. Numerical: impute (+ missing flag), scale (for scale-sensitive models), log/Box-Cox for skew, clip outliers, bin thresholds, ratios and interactions. Categorical: one-hot for low cardinality, ordinal only for ordered data, target/frequency encoding or embeddings for high cardinality (target encoding out-of-fold to avoid leakage). Dates: parts, cyclical encodings, elapsed time.

**Q2. Which models need feature scaling?**
Distance- and gradient-based/penalised models: k-NN, SVM, k-means, PCA, neural nets, linear/logistic regression with regularisation. Tree-based models are invariant to monotonic transforms of features.

**Q3. What is target leakage and how can feature engineering cause it?**
Using information unavailable at prediction time (e.g. `cancellation_date` for churn, or target encoding computed on the full dataset). Prevent by thinking "was this known at prediction time?", by fitting encoders inside CV folds, and by time-aware splits.

**Q4. One-hot vs label/ordinal vs target encoding?**
One-hot: nominal, few levels, no false order, but wide. Ordinal: ordered categories only. Target encoding: many levels, compact, needs smoothing and out-of-fold fitting to avoid overfitting/leakage.

**Q5. How do you handle missing values?**
First understand *why* (MCAR/MAR/MNAR). Impute (median, mode, model-based), add a missing indicator, or let the algorithm handle it (HistGradientBoosting, XGBoost do natively). Fit imputers on train only. Dropping rows is a last resort.

**Q6. How do you know a new feature is worth keeping?**
Ablation with cross-validation (mean and std), check stability across folds/time, check it exists at serving time and isn't leaky, weigh maintenance cost.

**Q7. Why encode month as sin/cos?**
So the model sees December→January as adjacent; a raw 1–12 integer places them 11 apart.

**Q8. Curse of dimensionality after one-hot on a 10 000-level categorical?**
Sparse, huge, overfits. Use hashing, target/frequency encoding, grouping of rare levels, or embeddings.

## 5. Common traps

* Fitting the scaler/imputer/encoder on the full dataset before splitting.
* Assuming more features = better.
* Ordinal-encoding nominal data for linear models.
* Reporting an improvement smaller than the CV standard deviation.
* Training features computed differently from serving features (train/serve skew).

---

## 6. Project: Feature Factory — roadmap

| Phase | Time | Tasks | Done when |
|---|---|---|---|
| **1. Explore** | 2 h | Load `customers` from SQL; histogram every numeric column; `df.isna().mean()`; check skew of `monthly_charges`; churn rate per category (`GROUP BY contract_type`) | EDA notebook/markdown of 5 findings |
| **2. Baseline** | 1 h | Reproduce L0; record CV AUC ± std | Baseline logged in DB |
| **3. Engineer** | 3 h | Read `add_features` and `build_preprocessor`. Add 3 features of your own (e.g. `support_calls_per_month`, `is_high_price`, `tenure_bucket`). Ablate each individually | Table with `^ / ~ / v` per feature |
| **4. Encoders** | 2 h | Compare one-hot, ordinal, frequency, target encoding for `city`; then rerun with `n_cities=500` | Conclusion supported by numbers |
| **5. Custom transformer** | 2 h | Write a scikit-learn `BaseEstimator, TransformerMixin` class (e.g. `RareCategoryGrouper`) and drop it into the pipeline | Passes `cross_val_score` |
| **6. Selection** | 2 h | Compare mutual-information filter, L1 (`SelectFromModel`) and RFE — all *inside* the pipeline | Fewer features, same AUC |
| **7. Serve-safe** | 2 h | `joblib.dump` the full pipeline; load in a new process and predict a raw row (no manual preprocessing) | Train/serve parity proven |

**Stretch:** real Kaggle "Telco Customer Churn" loaded into the same DB table; SHAP values to see which engineered features the model actually uses.

**Resume bullet:** *"Designed leakage-safe scikit-learn feature pipelines (imputation with missing indicators, cyclical date features, target encoding) and quantified each feature group's contribution through cross-validated ablation; persisted engineered features to a SQL feature-store table."*
