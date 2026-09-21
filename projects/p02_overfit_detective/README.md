# Topic 2 — Overfitting vs Underfitting

**Project: Overfit Detective** · `python projects/p02_overfit_detective/main.py`

---

## 1. The five-question framework

| Question | Answer |
|---|---|
| **What is it?** | **Overfitting:** the model learns the training data's noise and quirks, so it scores well on training data and badly on new data. **Underfitting:** the model is too simple to capture the real pattern, so it scores badly on both. |
| **Why do we need to care?** | The only thing that matters in production is performance on *unseen* data. A model that memorised its training set is useless (or dangerous) there. |
| **How do we detect it?** | Compare training vs validation error; plot validation curves (complexity) and learning curves (data size). |
| **When does it happen?** | Overfit: flexible model + little data + noisy labels + many features + long training. Underfit: rigid model, weak features, too much regularisation, too little training. |
| **Limitations of the fixes?** | Every fix trades one problem for the other; each has a cost (more data = money, regularisation = tuning, simpler model = bias). |

## 2. Deep dive

### The picture to keep in your head
As model complexity rises, **training error keeps falling**, but **validation error falls, bottoms out, then rises**. The sweet spot is the validation minimum. Left of it you underfit; right of it you overfit.

```
error
 |\                         validation
 | \____                   /
 |      \___ ___ _______ /
 |          \___ training
 +---------------------------------> model complexity
  underfit   good fit    overfit
```

### Causes of overfitting
1. Model too flexible for the amount of data (deep tree, high-degree polynomial, huge network).
2. Too few training rows relative to features (`p ≫ n`).
3. Noisy or mislabelled data — the model faithfully learns the noise.
4. Training too long (neural nets, boosting keep fitting noise).
5. **Leakage** (topic 7) — looks like a great model, then collapses in production. Different cause, same symptom.
6. **Over-tuning against the validation set** — hundreds of experiments on one validation set overfit *the validation set*. Keep a final untouched test set.

### Detection: the diagnosis table

| Training error | Validation error | Diagnosis | Learning curve looks like |
|---|---|---|---|
| High | High (≈ training) | **Underfitting / high bias** | both curves converge *high*; **more data won't help** |
| Low | High (big gap) | **Overfitting / high variance** | large gap that *shrinks as data grows*; **more data helps** |
| Low | Low (≈ training) | Good fit | both converge low |

*Use the noise floor as reference:* if you know irreducible noise σ² (or a good baseline, or human-level performance), "high" means "well above that".

### Fixes

| Problem | Remedies |
|---|---|
| **Overfitting** | more data / augmentation · simpler model (lower degree, shallower tree, fewer units) · **regularisation** (L1/L2, dropout, weight decay — topic 8) · **early stopping** · pruning (`max_depth`, `min_samples_leaf`, `ccp_alpha`) · **ensembling/bagging** (topic 9) · feature selection · cleaner labels · cross-validated hyper-parameter search |
| **Underfitting** | more flexible model · better features / interactions (topic 4) · **reduce** regularisation · train longer · boosting |

### Early stopping
For iterative learners (boosting, neural nets), watch validation error each round and stop when it stops improving (`n_iter_no_change`). In the project, gradient boosting's *training* error kept falling to ~0.0007 while validation error was best after **22 trees** and was about twice as bad by 600 trees.

## 3. What the project demonstrates (verified outputs)

* **[A] Validation curve.** Polynomial degree 1 underfits (train 0.27, val 0.31 vs noise floor 0.09). Degree 3–4 is the sweet spot (val ≈ 0.10). Degree 12 overfits (train 0.037, val 0.43); degree 15 explodes (val ≈ 100).
* **[B] Learning curves.** Degree-1: both curves converge high — adding data cannot fix it. Degree-12: the train/validation gap is enormous at n=25 and vanishes by n=320 — *this* model needed more data.
* **[C] Five fixes for the same overfit model** (test MSE): baseline 0.435 → Ridge α=1 **0.19** → simpler degree-4 **0.093** → 10× more data **0.087**. Note that Ridge *raises training error* (0.037 → 0.156) yet lowers test error — the whole point of regularisation.
* **[D] Early stopping** as described above.
* **[E] Real data (breast cancer).** Unrestricted decision tree: train accuracy 1.00, test ≈ 0.91; a depth-5 tree is as good or better on test.

## 4. Interview questions with model answers

**Q1. What causes overfitting and how do you detect/reduce it?**
Model capacity too high relative to data/noise. Detect: big gap between training and validation metrics, validation curve that turns upward, learning curve with a persistent gap. Reduce: more data, simpler model, regularisation, early stopping, dropout, pruning, ensembling, better validation practice.

**Q2. Training accuracy 99 %, validation 70 %. What do you do?**
First confirm it's *real* overfitting, not a bug: check for leakage, duplicates across splits, distribution shift between splits. Then plot learning curves; if the gap shrinks with data, collect more/augment; else regularise/simplify and check the feature set.

**Q3. Training and validation accuracy are both 60 %. What now?**
Underfitting (or a genuinely hard/noisy problem). Compare with a baseline and the noise floor. Add features, increase capacity, reduce regularisation, train longer, check the labels are learnable at all.

**Q4. Can regularisation fix underfitting?**
No — it *increases* bias. It is a tool against variance. If you underfit, lower it.

**Q5. Why does more data help overfitting but not underfitting?**
Overfitting is a variance problem: more samples pin down the estimate. Underfitting is a bias problem: the hypothesis class can't represent the truth no matter how much data you give it.

**Q6. How can a model overfit the validation set?**
By repeated tuning: each decision uses validation feedback, so information leaks. Use nested CV or a held-out test set touched once.

**Q7. Random Forest trees are deep and overfit individually — why is the forest fine?**
Averaging many decorrelated high-variance trees cuts variance while bias stays low (see topic 3 & 9).

## 5. Common traps

* Calling a model "good" from training metrics.
* Tuning on the test set, then reporting the test score.
* Interpreting a big gap as overfitting when the real cause is a train/validation distribution shift or leakage.
* Fixing overfitting by removing data or features blindly, without checking learning curves first.

---

## 6. Project: Overfit Detective — roadmap

| Phase | Time | Tasks | Done when |
|---|---|---|---|
| **1. Read & run** | 1 h | Run `main.py`; open the 5 PNGs in `outputs/`; map each plot to the diagnosis table above | You can explain each plot in one sentence |
| **2. Change the noise** | 1 h | Set `NOISE = 0.6` and `1.0`; how does the best degree change? Set n_train to 15 / 100 | You can predict the direction before running |
| **3. Swap the model** | 2 h | Replace polynomials by KNN (`k`) and by a Random Forest (`max_depth`, `min_samples_leaf`). Build a validation curve for each | Three complexity knobs, one plot each |
| **4. Neural-net version** | 3 h | Train a small MLP (`sklearn.neural_network.MLPClassifier` or PyTorch) on the breast-cancer data; plot train/val loss per epoch; add dropout/weight-decay and early stopping | Overfitting visible and then fixed |
| **5. Automated diagnoser** | 2 h | Turn `diagnose()` into a function that takes `learning_curve` output and prints "collect more data / add capacity / regularise" | Works on 3 datasets |
| **6. Log & report** | 1 h | Log each fix to the DB (`log_experiment`) and query the leaderboard: `SELECT * FROM experiments e JOIN metrics m ON m.experiment_id=e.id WHERE project='p02_overfit_detective'` | One-page findings |

**Stretch goals:** label-noise experiment (flip 10 % of labels and watch a deep tree memorise them); double-descent demo (very high-degree polynomials with ridge).

**Resume bullet:** *"Built a diagnostic toolkit (validation curves, learning curves, early stopping) that identifies high-bias vs high-variance models and quantifies the effect of five mitigation strategies."*
