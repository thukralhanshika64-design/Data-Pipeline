# Topic 3 — Bias–Variance Trade-off

**Project: Bias-Variance Simulator** · `python projects/p03_bias_variance_simulator/main.py`

---

## 1. The five-question framework

| Question | Answer |
|---|---|
| **What is it?** | A decomposition of a model's expected prediction error into **bias** (systematic error from wrong assumptions), **variance** (sensitivity to the particular training set) and **irreducible noise**. |
| **Why do we need it?** | It explains *why* models fail and tells you which lever to pull: bias problems need more expressive models/features; variance problems need more data, simpler models, regularisation or averaging. |
| **How does it work?** | For squared loss: `E[(y − f̂(x))²] = Bias² + Variance + σ²`. Making a model more flexible lowers bias and raises variance. |
| **When do I use it?** | Every time you diagnose a model (topic 2), choose complexity, choose between bagging and boosting (topic 9), or set regularisation strength (topic 8). |
| **Limitations?** | The clean additive decomposition is exact only for squared error. You can only *measure* it when you know the true function (simulation) — on real data you *infer* it from learning curves. In very over-parameterised deep nets, the classic U-shape can break ("double descent"). |

## 2. Deep dive

### Definitions
Imagine you could redraw the training set many times from the same population and refit the model each time. At a fixed input `x` you'd get a *cloud of predictions* `f̂(x)`.

* **Bias(x) = E[f̂(x)] − f(x)** — how far the *average* prediction is from the truth. High bias = the model class can't represent the truth (a straight line through a sine wave). *Systematic* error.
* **Variance(x) = Var[f̂(x)]** — how much predictions jump around between training sets. High variance = the model chases noise in each sample. *Instability.*
* **Noise σ²** — randomness in `y` itself. No model can remove it. It's your error floor.

Dartboard picture: **low bias / low variance** = tight cluster on the bullseye; **high bias / low variance** = tight cluster off-centre; **low bias / high variance** = scattered around the bullseye; **high bias / high variance** = scattered and off-centre.

### Derivation sketch (worth being able to write)
Let `y = f(x) + ε`, `E[ε]=0`, `Var[ε]=σ²`, and ε independent of `f̂`. Then

```
E[(y − f̂)²] = E[(f + ε − f̂)²]
            = E[(f − f̂)²] + σ²                      (cross term vanishes: E[ε]=0, independence)
            = (f − E[f̂])² + E[(f̂ − E[f̂])²] + σ²    (add and subtract E[f̂])
            =     Bias²    +      Variance     + σ²
```

### Where models sit

| Model | Typical bias | Typical variance | Knob that moves along the curve |
|---|---|---|---|
| Linear/logistic regression | high (if truth non-linear) | low | add polynomial/interaction terms, regularisation `α`/`C` |
| k-NN | low for small k | **high for small k** | `k` (bigger = simpler) |
| Decision tree | low if deep | **high if deep** | `max_depth`, `min_samples_leaf` |
| Random forest | ≈ tree's | much lower than a tree | number of trees, `max_features` |
| Gradient boosting | reduces bias step by step | grows with rounds | rounds, learning rate, depth, `subsample` |
| Neural network | low | can be high | width/depth, weight decay, dropout, early stopping |

### What moves what

| Action | Bias | Variance |
|---|---|---|
| More training data | ~ | ↓ |
| More complex model | ↓ | ↑ |
| More regularisation | ↑ | ↓ |
| Add informative features | ↓ | slight ↑ |
| Add noisy features | ~ | ↑ |
| Bagging / random forest | ~ | ↓↓ |
| Boosting | ↓↓ | mild ↑ |

## 3. What the project demonstrates (verified outputs)

The script uses a **known** truth `f(x)=sin(2πx)`, noise σ=0.3 (σ²=0.09), 40 training points, and 200 independent training sets per model setting.

* **The identity checks out empirically.** Polynomial degree 3: bias² 0.0035 + variance 0.0102 + σ² 0.09 = **0.1037**; independently measured test MSE = **0.1034**. This holds across every setting in the tables (usually to the 3rd decimal).
* **Polynomials:** degree 1 → bias² 0.178 (underfit); degree 3–6 → both tiny; degree 12 → variance ≈ 124 (wildly unstable).
* **k-NN:** k=1 → variance 0.094, bias² ≈ 0; k=30 → bias² 0.295. Complexity = 1/k.
* **Trees:** depth 1 has the highest bias² (0.062); deeper trees drive bias² to ≈ 0.0005 while variance rises to ≈ 0.094.
* **Bagging:** 50 bagged deep trees cut variance from **0.0943 to 0.0458** while bias² stays ≈ 0.0005–0.0007. This is *the* reason random forests work.

## 4. Interview questions with model answers

**Q1. Explain the bias–variance trade-off.**
Expected squared error splits into bias², variance and irreducible noise. Simple models have high bias and low variance; complex models the reverse. Total error is minimised at an intermediate complexity. In practice I locate it with validation curves and cross-validation.

**Q2. Your model has high bias. What do you do? High variance?**
High bias: richer model, better features, less regularisation, boosting. High variance: more data, regularisation, simpler model, bagging/ensembles, feature selection, early stopping, dropout.

**Q3. Why does bagging reduce variance but not bias?**
Averaging `B` identically-distributed predictors with pairwise correlation ρ has variance `ρσ² + (1−ρ)σ²/B`. Averaging shrinks the second term; bias is the average of the individual biases, so it is unchanged. Random forests additionally decorrelate trees (random feature subsets) to reduce ρ.

**Q4. Why can boosting reduce bias?**
Each new weak learner fits what the current ensemble gets wrong, so the ensemble can represent progressively more complex functions.

**Q5. Does k in k-NN follow the trade-off?**
Yes: k=1 has near-zero bias but high variance (fits every noisy point); large k averages many neighbours (low variance) but blurs real structure (high bias).

**Q6. Can you compute bias and variance on real data?**
Not exactly — the true function is unknown. Approximate with bootstrap resamples (variance across refits) and use learning curves and train-vs-validation gaps as proxies.

**Q7. Does the trade-off always hold? (senior-level)**
For classic models, yes. Very large over-parameterised networks can show "double descent": test error falls, rises near the interpolation threshold, then falls again. Implicit regularisation from SGD is part of the explanation. The intuition still guides practice, but don't treat the U-curve as a law.

## 5. Common traps

* Saying "bias" when you mean *dataset* bias (sampling/fairness) — the statistical term is different; clarify.
* Claiming "more data always fixes it" (only variance).
* Forgetting the noise term: you can't drive error below σ².
* Applying the squared-loss decomposition literally to accuracy/log-loss.

---

## 6. Project: Bias-Variance Simulator — roadmap

| Phase | Time | Tasks | Done when |
|---|---|---|---|
| **1. Run & read** | 1 h | Run `main.py`; read `simulate()` line by line. Where does each of `bias2`, `var`, `emp` come from? | You can reproduce the decomposition on paper |
| **2. Vary the world** | 1–2 h | Change `SIGMA` (0.1, 0.6), `N_TRAIN` (15, 200). Predict then check: where does the best degree move? | Written prediction vs result table |
| **3. New model families** | 2 h | Add Ridge with different `α` on degree-10 polynomials, and RandomForest with `max_features`. Plot bias²/variance vs `α`. | Regularisation shown moving the curve |
| **4. Boosting** | 2 h | Add `GradientBoostingRegressor` vs number of rounds; show bias² falling and variance rising. | Bias↓/variance↑ visible |
| **5. Per-x view** | 2 h | Instead of averaging over x, plot bias²(x) and variance(x) separately — where is variance worst? (Hint: edges.) | Explanation of edge effects |
| **6. Real-data proxy** | 2 h | On `houses`, use 100 bootstrap refits of a tree vs a forest and compare prediction variance on a fixed test set. | Forest variance < tree variance |

**Stretch:** classification bias-variance with 0-1 loss (Domingos' decomposition); double-descent experiment with min-norm least squares.

**Resume bullet:** *"Implemented a Monte-Carlo bias–variance decomposition framework verifying `Bias² + Variance + σ²` against measured test error across polynomial, k-NN, tree and bagged models."*
