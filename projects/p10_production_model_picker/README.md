# Topic 10 — ML Model Selection for Production

**Project: Production Model Picker** (benchmark → scorecard → registry → API)
`python projects/p10_production_model_picker/main.py` (≈ 20 s)
`uvicorn projects.p10_production_model_picker.serve:app --port 8000`

---

## 1. The five-question framework

| Question | Answer |
|---|---|
| **What is it?** | Choosing which trained model to deploy when several look good offline — by weighing accuracy against interpretability, latency, cost, scalability, maintenance and business constraints. |
| **Why do we need it?** | The best offline metric is rarely the best *product*. A 0.4 % AUC gain isn't worth a 50× slower, unexplainable, hard-to-maintain model — unless money says otherwise. |
| **How does it work?** | (1) Establish performance is *statistically* comparable, (2) measure every non-accuracy criterion, (3) apply hard **gates** from requirements, (4) score the survivors with weights that reflect business priorities, (5) run a pilot/shadow test, (6) register, deploy, monitor. |
| **When do I use it?** | Before any launch, and again whenever requirements or data change. |
| **Limitations?** | Weights are subjective; scores can be sensitive to noisy measurements; offline metrics may not predict online business impact — validate with an experiment. |

## 2. Deep dive

### Step 0 — start from requirements, not algorithms
Ask: What decision does the prediction drive? What is the cost of each error type? Batch or real-time? Latency budget? Must decisions be explained (regulation, customer support)? How often does data change? Who maintains the model? What compute/memory budget exists? What happens when the model is wrong or unavailable (fallback)?

### The criteria

| Criterion | What to measure | Why it can dominate |
|---|---|---|
| **Performance** | CV mean **and std**, PR-AUC/cost-based metric, per-segment performance, calibration | is the difference bigger than the noise? |
| **Interpretability** | inherent (linear, small tree) vs post-hoc (SHAP) | regulated domains (lending, healthcare), debugging, stakeholder trust |
| **Latency** | p50/**p95/p99** per request, on production-like hardware, including preprocessing | real-time APIs have hard SLAs; averages hide tail latency |
| **Throughput / scalability** | rows/s in batch, horizontal scaling, memory | nightly scoring of 100 M rows |
| **Training cost & speed** | wall-clock, $ per retrain, hardware | frequent retraining, hyper-parameter search budgets |
| **Model size** | artefact MB, RAM at inference | edge/mobile, cold starts, serverless |
| **Maintenance & simplicity** | number of moving parts, dependencies, feature pipeline complexity, team skills | the model that nobody can debug at 3 a.m. is a liability |
| **Robustness & fairness** | behaviour under drift, missing inputs, subgroup metrics | risk, compliance |
| **Business fit** | offline metric → expected $ impact | ultimately the only thing that matters |

### Decision procedure
1. **Baseline first** (majority class, last-value, logistic regression). Always keep a simple model in the running.
2. **Tie detection:** models whose CV score is within ~1 std of the best are statistically indistinguishable → don't pick on decimals.
3. **Gates** (pass/fail): e.g. "p95 latency ≤ 12 ms", "must be explainable", "artefact ≤ 100 MB". Disqualify before scoring.
4. **Weighted scorecard** among survivors. Score *noisy* measurements (latency!) against budgets, not by min-max across candidates (see the note in section 3).
5. **Sensitivity analysis:** vary the weights; if the winner flips easily, the decision is a judgement call — say so.
6. **Pilot:** shadow mode/canary/A-B test on real traffic; compare the *business KPI*, not just AUC.
7. **Register, version, deploy with rollback, monitor** (below).

### From notebook to production (what "production" adds)
* **Model registry:** versioned artefacts with stage (`staging → production → archived`), the experiment that produced them and its metrics (this repo: table `model_registry`).
* **Serving:** wrap the *entire pipeline* (preprocessing + model) in one artefact; validate inputs (Pydantic); return model version; log every request (`prediction_log`).
* **Monitoring:** latency percentiles, error rates, **data drift** (feature distributions vs training: PSI, KS-test), **prediction drift**, **performance decay** once labels arrive, alerts.
* **Retraining:** scheduled or triggered by drift; champion/challenger; rollback plan.
* **Reproducibility:** pinned dependencies, seeds, data snapshot/version, experiment tracking (MLflow / this DB).
* **Train/serve skew:** the #1 silent killer — identical feature code paths.

## 3. What the project demonstrates (verified outputs)

Seven candidates on a **non-linear** churn dataset (`customers_nl`) — 5-fold CV AUC, test AUC, and measured cost:

| Model | CV AUC ± std | Test AUC | Size | Notes |
|---|---|---|---|---|
| Logistic regression | 0.8045 ± 0.010 | 0.797 | 5 KB | can't represent the non-linear rules |
| Decision tree (depth 5) | 0.8747 ± 0.010 | 0.886 | 10 KB | small, readable, nearly as accurate |
| Random forest | 0.8798 ± 0.008 | 0.878 | 8.4 MB | slowest at inference (≈ 14–19 ms p95 here) |
| Gradient boosting | **0.8836 ± 0.009** | 0.887 | 204 KB | best CV AUC |
| HistGradientBoosting | 0.8755 ± 0.014 | 0.867 | 542 KB | |
| k-NN (k=25) | 0.8406 ± 0.005 | 0.828 | 1.1 MB | |
| MLP (64, 32) | 0.8546 ± 0.011 | 0.868 | 86 KB | least interpretable |

* **Statistical tie:** the best CV AUC (gradient boosting) is within one std of random forest and HistGB; the depth-5 tree is 0.009 below it — practically the same. Accuracy alone can't decide.
* **Profiles produce different winners** (same measurements, different priorities):
  * `realtime_api` (12 ms p95 budget, small team) → **decision tree** (random forest disqualified by the latency gate).
  * `batch_max_accuracy` (performance weight 0.85) → **gradient boosting**.
  * `regulated_lending` (must be explainable; gate requires interpretability ≥ 4) → **decision tree** (the boosted/forest/MLP/k-NN models are disqualified).
* **Sensitivity sweep** (performance weight 0 → 1, other weights in fixed ratio; identical in two repeated runs): logistic regression up to 0.1, **decision tree from 0.2 to 0.6, gradient boosting from 0.7**. The decision flips at *two* points, and where exactly depends on the weights and gates you chose — the takeaway is the *shape* of the decision, not the decimals.
* **On the near-linear dataset** (`--dataset customers`) logistic regression wins **every** profile and the sweep never flips: when a simple model dominates, the trade-off disappears. *A senior answer: "it depends on how big the accuracy gap is."*
* **A bug I hit while building this — worth remembering.** My first scorecard normalised latency with min-max across candidates. Single-row latencies were 3–9 ms and noisy, but min-max stretched a 2–3 ms wobble into a big score gap, moving a flip point from 0.7 to 0.4 between two runs. Fix: (a) more robust measurement (warm-up, median of block p95s) and (b) score latency against the SLA budget (full marks up to half the budget). *Never let measurement noise decide a production choice.*
* Timings depend on your machine — the *relative* pattern (forest ≫ others) is what to expect, not the exact milliseconds.
* **Serving:** `serve.py` loads the registry's production model, validates input (422 on bad requests), returns `churn_probability` and the model version, and logs each call to `prediction_log`.

## 4. Interview questions with model answers

**Q1. Several models perform well. How do you decide which one goes to production?**
Confirm the differences are beyond noise (CV mean ± std). Then compare on what production needs: interpretability, p95 latency, throughput, training cost, model size, maintenance and robustness, filtering by hard requirements first and weighing the rest by business priorities. Run a sensitivity check, then a shadow/A-B pilot on the real KPI before full rollout. And keep the simplest model that meets the requirement.

**Q2. Gradient boosting has +0.5 % AUC over logistic regression. Ship it?**
Depends on the value of 0.5 %: quantify in $ (e.g. at 1 M scored customers, extra catches × margin) versus added cost (latency, infra, explainability work, maintenance). If the gain is within CV noise, prefer the simpler model.

**Q3. How do you measure latency properly?**
On production-like hardware, including preprocessing and serialisation, after warm-up, single-request and batch, reporting p50/p95/p99 rather than the mean; under realistic concurrency.

**Q4. The model needs to be explainable to regulators. Options?**
Inherently interpretable models (regularised logistic regression, small trees, GAMs/EBMs, scorecards); or complex model + post-hoc explanations (SHAP, LIME) — but be aware regulators may not accept approximations for adverse-action reasons.

**Q5. How do you monitor a model in production?**
Operational (latency, errors, throughput), data drift (PSI/KS on inputs), prediction drift, and — as labels arrive — live performance by segment; plus alerts, dashboards and a retraining/rollback playbook.

**Q6. What is train/serve skew and how do you prevent it?**
Features computed differently online vs offline. Prevent with one shared feature pipeline (a serialised pipeline object or feature store), input schema validation and offline replay tests.

**Q7. Offline AUC improved but the online A/B test shows no lift. Why?**
Metric–KPI mismatch, leakage in the offline evaluation, distribution shift, feedback loops, position/selection effects, or the improvement was below the noise.

**Q8. When do you retrain?**
On a schedule, on drift alerts, or when labelled performance decays; always via champion/challenger with a rollback.

**Q9. How would you choose between a deep network and gradient boosting for tabular data?**
Default to gradient boosting: strong, fast to train, needs less data and tuning. Consider deep learning for very large data, unstructured inputs (text/images), multi-task or embedding-heavy problems, or when it already fits the serving stack.

**Q10. How do you handle the cold-start of a brand-new model with no production data?**
Shadow mode (predict but don't act), replay historical data, canary release to small traffic, compare with the incumbent on the same requests.

## 5. Common traps

* Picking the top-AUC model from a single run.
* Optimising an offline metric that isn't the business KPI.
* Ignoring tail latency, memory and deployment complexity.
* Weights chosen after seeing which model they favour (decide the weights *before* the benchmark).
* No fallback or rollback plan.
* Forgetting that retraining costs recur.

---

## 6. Project: Production Model Picker — roadmap

| Phase | Time | Tasks | Done when |
|---|---|---|---|
| **1. Run & inspect** | 1 h | `python main.py`; open `ml_lab.db` and run the queries in `database/queries.sql` (leaderboard, registry) | You can find your model version in SQL |
| **2. Own profile** | 2 h | Add a profile for a business you know (e.g. "mobile_edge": size gate ≤ 1 MB, no external libs). Predict the winner, then run | Prediction vs result |
| **3. Sensitivity of everything** | 3 h | Extend `sensitivity()` to sweep the interpretability weight and the latency budget; plot winner regions | 2-D "decision map" |
| **4. Better latency test** | 3 h | Load-test the API with `locust` or `wrk` (concurrent users); record p50/p95/p99 from the client side | Latency table incl. concurrency |
| **5. Docker** | 3 h | Write a `Dockerfile` (python-slim, copy the model artefact, run uvicorn); run the container locally | `docker run` answers `/predict` |
| **6. Monitoring** | 4 h | Write `monitor.py`: reads `prediction_log`, computes latency percentiles and PSI of `monthly_charges` vs the training data; flag drift over 0.2 | Drift report; simulate drift by sending shifted requests |
| **7. Champion/challenger** | 3 h | Register a second model as `staging`; route 10 % of traffic to it in `serve.py`; compare predictions/latency from the log table | Promotion via SQL/`register_model(stage='production')` |
| **8. CI** | 2 h | GitHub Actions: install requirements, run `python run_all.py 8 3`, run a smoke test against `TestClient` | Green pipeline |

**Stretch:** MLflow instead of the homemade registry; ONNX export for faster inference; SHAP explanations endpoint; quantile-based prediction intervals; cost-per-1000-predictions estimate.

**Resume bullet:** *"Built an end-to-end model-selection and serving pipeline: benchmarked 7 models on CV performance, latency, size and interpretability, applied business-profile gates and weighted scoring with sensitivity analysis, registered versions in a SQL model registry and served the winner through a FastAPI service with prediction logging."*
