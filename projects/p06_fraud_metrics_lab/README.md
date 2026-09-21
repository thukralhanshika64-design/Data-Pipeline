# Topic 6 — Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, MAE/MSE/RMSE)

**Project: Fraud Metrics Lab** · `python projects/p06_fraud_metrics_lab/main.py`

---

## 1. The five-question framework

| Question | Answer |
|---|---|
| **What is it?** | Quantifying how good predictions are, using metrics that reflect what errors *cost* in the real problem. |
| **Why do we need it?** | The wrong metric optimises the wrong thing. A fraud model with 99 % accuracy can catch zero fraud. |
| **How does it work?** | Compare predictions to truth on *held-out* data (topic 7), summarise with one or more metrics, pick a decision threshold from costs. |
| **When do I use each?** | See the decision table in section 2. |
| **Limitations?** | Every metric hides something; a single number rarely captures the business goal. Metrics computed on a leaky or unrepresentative split are meaningless. |

## 2. Deep dive

### The confusion matrix — the source of everything

|  | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | TP | FN (miss) |
| **Actually negative** | FP (false alarm) | TN |

* **Accuracy** = (TP+TN)/N
* **Precision** = TP/(TP+FP) — "when I raise an alarm, how often is it real?"
* **Recall / sensitivity / TPR** = TP/(TP+FN) — "of all real cases, how many do I catch?"
* **Specificity** = TN/(TN+FP); **FPR** = 1 − specificity
* **F1** = 2PR/(P+R) — harmonic mean; punishes imbalance between P and R. **Fβ** weights recall β× more than precision (F2 when misses are worse).

### Decision table: which metric when?

| Metric | Use when | Avoid when |
|---|---|---|
| **Accuracy** | classes balanced and error costs equal | imbalanced data (the *accuracy paradox*) |
| **Precision** | false alarms are expensive (spam filter deleting real mail, manual review capacity is limited) | you can't afford misses |
| **Recall** | misses are expensive (cancer screening, fraud, safety) | alarms are costly and you'd flood the team |
| **F1** | you need one number balancing P and R on an imbalanced problem | costs of FP and FN are very different (use Fβ or costs) |
| **ROC-AUC** | you care about *ranking quality* over all thresholds; roughly balanced classes | heavy imbalance — it can look great while precision is poor |
| **PR-AUC (average precision)** | heavy imbalance, positives are what you care about | you care equally about negatives |
| **Log-loss / Brier** | you use the *probabilities* (pricing, risk, triage) | you only need class labels |
| **MAE** | you want a robust, interpretable typical error; outliers exist | large errors are disproportionately harmful |
| **MSE / RMSE** | large errors are much worse than small ones; errors ≈ Gaussian | heavy-tailed noise or outliers you can't fix |
| **R²** | comparing to a mean-only baseline | as the sole metric (no units) |

### ROC vs PR — why imbalance breaks ROC-AUC
FPR = FP/(FP+TN). When negatives outnumber positives 100:1, a *large number of false positives* can still be a *tiny FPR*, so the ROC curve looks excellent. Precision uses FP directly, so the PR curve shows the pain. The PR-AUC baseline for a random model equals the positive rate (≈ 0.01 here), not 0.5.

### Choosing the decision threshold
The default 0.5 is not sacred. Compute expected cost per threshold:
`cost(t) = FN(t)·cost_of_miss + FP(t)·cost_of_false_alarm`, choose the minimiser **on validation data**, then report the result on untouched test data.

### Calibration
A calibrated model's "0.3" means ≈ 30 % of such cases are positive. Class-weighting, resampling (SMOTE) and many tree ensembles distort probabilities while leaving *ranking* (AUC) intact. If you use probabilities for decisions, recalibrate (Platt scaling/isotonic) on held-out data and check a reliability curve + Brier score.

### Multiclass averaging
**Macro** = mean of per-class scores (treats classes equally — exposes weak minority classes), **micro** = pool all decisions (dominated by big classes), **weighted** = macro weighted by class size.

### Regression metrics recap
* RMSE ≥ MAE always; the ratio tells you about tail errors.
* A constant prediction minimises **MSE at the mean** and **MAE at the median**.
* One outlier can dominate RMSE: 20 errors of 5 → MAE = RMSE = 5; make one error 200 and MAE = 14.8 but **RMSE = 45.0**.

## 3. What the project demonstrates (verified outputs)

Synthetic card transactions, ~1 % fraud, **time-based** split (train on the first 70 % of the timeline, test on the last 30 %; 125 test frauds).

| Model (threshold 0.5) | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Dummy "never fraud" | **0.990** | 0.000 | **0.000** | 0.000 | 0.500 | 0.010 |
| LogReg (balanced) | 0.970 | 0.255 | 0.960 | 0.403 | 0.996 | 0.885 |
| Random forest (balanced) | 0.995 | 0.790 | 0.752 | 0.770 | 0.996 | 0.858 |
| HistGB (balanced) | 0.989 | 0.479 | 0.816 | 0.604 | 0.977 | 0.728 |

* **Accuracy paradox:** the useless dummy has 99 % accuracy.
* **Same ROC-AUC, different products:** LogReg and random forest tie on ROC-AUC (0.996) yet one has 25 % precision and 96 % recall while the other has 79 % / 75 % at the same threshold.
* **Business cost with honest tuning.** Missing a fraud costs its amount; each manual review costs $5. Model and threshold were chosen on a **validation slice**, then evaluated on test: approve-everything **$19,737** → default 0.5 threshold **$2,351** → validation-chosen threshold (0.87) **$1,181**. Picking the threshold *on the test set* would have given $1,161 — the honest procedure lost almost nothing. (My first draft of this script tuned on test — a classic evaluation leak; the shipped version fixes it.)
* **Calibration:** balanced LogReg's mean predicted probability is **0.051** vs a true rate of 0.010 (Brier 0.0218 vs 0.0029 for the plain model), while ROC-AUC is unchanged.
* **Caveat:** this synthetic fraud is much *easier* than real fraud (ROC-AUC 0.996!). Expect real data to be far messier; the *lessons* transfer, the numbers don't.

## 4. Interview questions with model answers

**Q1. When would you use accuracy, precision, recall, F1, ROC-AUC, MAE/RMSE?**
Accuracy on balanced data with symmetric costs. Precision when false alarms are costly; recall when misses are. F1 for one balanced number on imbalanced data. ROC-AUC for threshold-free ranking quality (balanced-ish data); PR-AUC when positives are rare. MAE for robust typical error; RMSE when big errors are especially bad.

**Q2. 99 % accuracy on a 1 %-positive problem — good?**
Not by itself. A majority-class predictor achieves that. Look at recall/precision/PR-AUC and the confusion matrix, and compare with the baseline.

**Q3. Explain ROC-AUC intuitively.**
The probability that a randomly chosen positive gets a higher score than a randomly chosen negative. 0.5 = random ranking, 1.0 = perfect.

**Q4. Why prefer PR-AUC for imbalanced data?**
ROC uses FPR, which is diluted by the huge negative class; PR uses precision, which reflects the false alarms users actually experience.

**Q5. How do you pick the classification threshold?**
From costs/capacity: minimise expected cost, or fix recall and maximise precision, or fix review capacity (top-k). Choose on validation data, confirm on test.

**Q6. What is calibration and why might a high-AUC model be badly calibrated?**
Agreement between predicted probabilities and observed frequencies. AUC only depends on ranking, so any monotonic distortion (class weights, boosting, SMOTE) keeps AUC but breaks probabilities.

**Q7. F1 vs Fβ? Micro vs macro?**
Fβ tilts toward recall (β>1) or precision (β<1). Micro pools every prediction (dominated by frequent classes); macro averages per-class scores (shows minority-class performance).

**Q8. MAE vs RMSE?**
RMSE squares errors → sensitive to outliers, estimates the conditional mean; MAE is linear → robust, estimates the median. RMSE/MAE ≫ 1.25 signals heavy-tailed errors.

**Q9. Your offline AUC is 0.92, online lift is nil. Why?**
Leakage, train/serve skew, distribution shift, metric mismatch with the business KPI, feedback loops, or the population you score differs from the one you evaluated on.

## 5. Common traps

* Selecting the model *and* threshold on the test set (optimistic).
* Comparing models at threshold 0.5 when they are calibrated differently.
* Using `class_weight='balanced'` or SMOTE and then quoting the probabilities as real.
* Random-splitting transactions that have a time order.
* Reporting AUC without the class balance.

---

## 6. Project: Fraud Metrics Lab — roadmap

| Phase | Time | Tasks | Done when |
|---|---|---|---|
| **1. Metric drill** | 2 h | Compute TP/FP/FN/TN, precision, recall, F1 **by hand** from a 2×2 matrix and confirm with `sklearn.metrics` | Matches to 3 decimals |
| **2. Run and annotate** | 1 h | Run `main.py`; annotate each table row with "why is this number what it is?" | Notes file |
| **3. Threshold lab** | 2 h | Change `REVIEW_COST` to $1, $25, $100 — how does the optimal threshold move? Add a **capacity constraint** ("we can review 100 alerts/day") | Two decision rules compared |
| **4. Calibration** | 2 h | Wrap the balanced LogReg in `CalibratedClassifierCV`; plot reliability curves before/after | Brier improves, AUC unchanged |
| **5. Resampling** | 3 h | Compare `class_weight`, random undersampling and SMOTE (`imbalanced-learn`, applied **inside** the pipeline) on PR-AUC | Table of 3 strategies |
| **6. Harder data** | 2 h | Make fraud harder: increase overlap in `make_transactions`; watch ROC-AUC vs PR-AUC diverge | Realistic PR-AUC ≈ 0.3–0.6 |
| **7. Monitoring** | 3 h | Simulate a week of drift (fraud pattern changes); compute weekly recall/precision with a rolling window and log to the DB | Time-series plot from SQL |

**Stretch:** lift/gain charts; cost-sensitive learning (weights = transaction amount); precision@k and recall@k; conformal prediction.

**Resume bullet:** *"Built a fraud-evaluation harness on a 1 %-positive stream (time-based split, PR-AUC, calibration, cost-optimal thresholding chosen on validation data), cutting simulated loss from $19.7k to $1.2k."*
