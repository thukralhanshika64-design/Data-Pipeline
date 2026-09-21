# Topic 5 — Classification vs Regression

**Project: Twin Predictors** · `python projects/p05_twin_predictors/main.py`

---

## 1. The five-question framework

| Question | Answer |
|---|---|
| **What is it?** | Both are supervised learning. **Classification** predicts a *category*; **regression** predicts a *continuous number*. |
| **Why does the distinction matter?** | It decides the model's output layer, the loss function, the metrics, the baseline and even how you split data. Mixing them up gives meaningless numbers (e.g. RMSE on class labels). |
| **How do they differ mechanically?** | Classification outputs class scores/probabilities (softmax/sigmoid) trained with cross-entropy; regression outputs a real number trained with squared/absolute/Huber loss. |
| **When do I use which?** | Look at the *decision* being made, not just the column type: "will they leave?" → classification; "how much will they spend?" → regression. Sometimes you can frame the same business problem either way (see the bridge below). |
| **Limitations?** | Classification throws away magnitude; regression assumes the number is meaningful and the error is symmetric. Both fail under distribution shift and leakage. |

## 2. Deep dive

### Flavours
* **Binary** classification (churn/no churn), **multiclass** (one label out of many), **multilabel** (several labels at once — each is a separate yes/no), **ordinal** (ordered classes like ratings 1–5: classification with order, or regression rounded).
* Regression: point prediction, **quantile** regression (predict the 90th percentile — for capacity planning), count regression (Poisson), positive-skewed targets (log-transform, Gamma).

### Loss functions

| Task | Typical loss | Notes |
|---|---|---|
| Binary classification | log-loss (binary cross-entropy) | gives calibrated-ish probabilities; hinge for SVM |
| Multiclass | categorical cross-entropy | softmax output |
| Regression | MSE (L2), MAE (L1), Huber | MSE → conditional **mean**, MAE → conditional **median**, Huber = MSE near zero, MAE for outliers |

### Metrics

**Classification** (topic 6 goes deeper)

| Metric | Formula / meaning |
|---|---|
| Accuracy | (TP+TN)/all — misleading when classes are imbalanced |
| Precision | TP/(TP+FP) — of those I flagged, how many were right |
| Recall (sensitivity) | TP/(TP+FN) — of the real positives, how many did I catch |
| F1 | harmonic mean of precision and recall |
| ROC-AUC | probability a random positive is ranked above a random negative; threshold-free |
| Log-loss / Brier | quality of the *probabilities* |
| Confusion matrix | the four counts behind everything above |

**Regression**

| Metric | Formula | Property |
|---|---|---|
| MAE | mean(\|y−ŷ\|) | same units as y; robust to outliers; easy to explain |
| MSE | mean((y−ŷ)²) | penalises big misses heavily; differentiable; units² |
| RMSE | √MSE | units of y; RMSE ≥ MAE always; a large RMSE/MAE ratio means a few big errors dominate |
| R² | 1 − SSE/SST | share of variance explained vs predicting the mean; can be negative on test data |
| Adjusted R² | penalises extra predictors | compares models with different numbers of features |
| MAPE | mean(\|y−ŷ\|/\|y\|) | scale-free, but explodes near y=0 and is asymmetric |
| RMSLE | RMSE of log(1+y) | penalises *relative* error; good for skewed targets |

### Baselines (always!)
`DummyClassifier` (predict the majority class/prior) and `DummyRegressor` (predict the mean). If your model can't clearly beat them, you don't have a model yet.

### Classification produces a *score*, the threshold makes the *decision*
The default 0.5 is arbitrary. Moving it trades precision for recall (topic 6). Choose it from the *costs* of the two error types.

### Linear regression assumptions (classic interview item — "LINE")
**L**inearity of the relationship, **I**ndependent errors, **N**ormally distributed residuals (matters for inference, not for prediction), **E**qual variance (homoscedasticity). Plus: no severe multicollinearity. Check with residual plots, not just R².

### The bridge between the two
* Turn regression into classification by **thresholding a continuous prediction** ("is price > $200k?") — you keep magnitude information and can retune the cut-off later.
* Turn a classification problem into regression by predicting a **risk score/probability** — that's what logistic regression is.
* **Bucketing** a continuous target into classes loses information and creates boundary artefacts; do it only if the decision truly is categorical.

## 3. What the project demonstrates (verified outputs)

**Classification — churn** (25 % held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (prior) | 0.708 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.757 | 0.633 | 0.397 | 0.488 | 0.776 |
| Random forest | 0.730 | 0.565 | 0.332 | 0.418 | 0.749 |

Moving logistic regression's threshold: 0.5 → precision 0.63 / recall 0.40; 0.35 → 0.52 / 0.63; **0.25 → 0.47 / 0.80 (F1 0.589)**. Same model, three different products.

**Regression — house price**

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Dummy (mean) | 58,546 | 85,603 | ≈ 0 | 43.0 % |
| Linear regression | 21,031 | 40,808 | 0.773 | 13.5 % |
| Linear regression on log(price) | **20,239** | 49,258 | 0.669 | **11.0 %** |
| Random forest | 20,044 | **40,081** | **0.781** | 11.8 % |

* The **log-target** model has *lower* MAE and MAPE but *higher* RMSE and lower R². The data contains a few extreme outliers (1 % luxury/data-entry cases), which dominate squared error in dollar terms. A likely contributor is that back-transforming a log-scale prediction gives a *median*-like estimate — I did not isolate the cause, so verifying it is a good exercise (phase 5). **Which metric is "right" depends on whether big misses are especially costly.**
* Random forest RMSE/MAE = 2.0 (Gaussian errors would give ≈ 1.25): a few huge errors dominate RMSE.
* **Bridge:** for "is this house in the top 25 % by price?", regress-then-threshold (AUC 0.971) is almost as good as a dedicated classifier (0.976).

## 4. Interview questions with model answers

**Q1. How do classification and regression differ? Which evaluation metrics for each?**
Discrete label vs continuous value. Classification: confusion-matrix-based precision/recall/F1, ROC-AUC/PR-AUC, log-loss, chosen by class balance and error costs. Regression: MAE, MSE/RMSE, R², sometimes MAPE/RMSLE, chosen by the units, outlier sensitivity and business cost of errors.

**Q2. Why can't you use accuracy for regression, or MSE for classification labels?**
Accuracy needs exact matches, which a continuous prediction almost never has. MSE on hard labels ignores calibration and gives no probability information; use cross-entropy/Brier on probabilities.

**Q3. Why is logistic *regression* a classification algorithm?**
It regresses the log-odds (a continuous quantity) on the features; the sigmoid turns it into a probability; a threshold turns that into a class.

**Q4. MAE vs RMSE — when do you choose which?**
RMSE when large errors are disproportionately bad (safety margins, inventory stock-outs) and the errors are roughly Gaussian; MAE when you want a robust, interpretable "typical error" or the data has outliers. MSE-trained models estimate the conditional mean; MAE-trained models the median.

**Q5. R² is negative on the test set — what does it mean?**
The model predicts worse than a constant equal to the training mean. Check for leakage in training, distribution shift, or heavy overfitting.

**Q6. Would you treat "predict the customer's rating (1–5)" as classification or regression?**
It's ordinal. Multiclass classification ignores order; plain regression treats distances as equal. Options: ordinal regression, or regression then round/threshold. Pick by how errors should be penalised (is 1 vs 5 worse than 1 vs 2? yes → use order).

**Q7. You have a skewed positive target (prices). What do you do?**
Consider a log transform (or a Gamma/Tweedie objective), evaluate in original units *and* on the log scale (RMSLE), check outliers, and be explicit about which metric the business cares about.

## 5. Common traps

* Reporting only accuracy for imbalanced classification.
* Reporting only R² (says nothing about the size of errors in real units).
* Back-transforming a log-target prediction with `exp` and forgetting it estimates the *median*, not the mean.
* Using MAPE when the target can be zero or near zero.
* Random-splitting time-ordered data (topic 7).

---

## 6. Project: Twin Predictors — roadmap

| Phase | Time | Tasks | Done when |
|---|---|---|---|
| **1. Pipeline** | 2 h | Read the `ColumnTransformer` + `Pipeline` code; run both tasks | You can explain each step |
| **2. Classifier upgrade** | 2 h | Add gradient boosting, calibrate probabilities (`CalibratedClassifierCV`), plot ROC and PR curves | ROC/PR plots saved |
| **3. Cost-based threshold** | 2 h | Assume a retention offer costs $20 and a churner is worth $300; pick the threshold that maximises expected profit (see project 6 for the method) | Threshold + $ impact |
| **4. Regressor upgrade** | 3 h | Add a Huber/quantile regressor (`GradientBoostingRegressor(loss='quantile')`) and predict the 10th/50th/90th percentile → prediction intervals. Check coverage on test | ~80 % of actual prices fall inside the interval |
| **5. Residual analysis** | 2 h | Plot residuals vs predicted, vs `sqft`, by `neighborhood`. Where is the model biased? | 3 findings written down |
| **6. Serve both** | 3 h | Wrap the two pipelines in one FastAPI app with `/churn` and `/price` endpoints (copy the pattern from `p10/serve.py`) | Both endpoints answer curl requests |

**Stretch:** multiclass version (churn reason), ordinal regression on ratings, monotonic constraints (price must not fall as `sqft` rises) in HistGradientBoosting.

**Resume bullet:** *"Built paired classification (churn) and regression (house price) pipelines with baselines, threshold analysis and outlier-aware metric selection; exposed predictions to a SQL table."*
