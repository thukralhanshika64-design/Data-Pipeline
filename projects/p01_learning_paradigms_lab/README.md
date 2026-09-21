# Topic 1 — Supervised vs Unsupervised vs Reinforcement Learning

**Project: Learning Paradigms Lab** · `python projects/p01_learning_paradigms_lab/main.py`

---

## 1. The five-question framework

| Question | Answer |
|---|---|
| **What is it?** | Three families of learning, distinguished by *what feedback the learner gets*. |
| **Why do we need it?** | Different business problems come with different data: labelled history (predict), raw records (discover), or an environment to interact with (decide). Choosing the wrong family is the most expensive mistake — it happens before any modelling. |
| **How does it work?** | Supervised: minimise a loss between predictions and known labels. Unsupervised: optimise a structural objective (compactness, variance captured, density). Reinforcement: maximise cumulative reward through trial and error. |
| **When do I use it?** | See the decision table below. |
| **Limitations?** | Supervised needs (expensive) labels and inherits their bias. Unsupervised has no ground truth to validate against. RL needs many interactions, a well-designed reward, and is hard to debug. |

## 2. Deep dive

### Supervised learning
You have pairs `(x, y)` and learn `f(x) ≈ y`. Two sub-types: **classification** (discrete `y`) and **regression** (continuous `y`). Training minimises a loss (cross-entropy, squared error…) and *generalisation* is measured on unseen data. Everything in topics 2–10 is mostly about doing this well.

*Real example:* churn prediction — features are tenure, price, support calls; label is "left within 90 days".

### Unsupervised learning
You only have `x`. Main tasks:

* **Clustering** (K-Means, hierarchical, DBSCAN, Gaussian mixtures) — group similar rows.
* **Dimensionality reduction** (PCA, t-SNE/UMAP for visualisation, autoencoders).
* **Density estimation / anomaly detection** (Isolation Forest, one-class methods).
* **Association rules** (market-basket analysis).

**K-Means in one paragraph:** pick `k`, initialise centroids (k-means++ spreads them out), then alternate *assign each point to nearest centroid* and *move each centroid to the mean of its points* until nothing changes. It minimises **inertia** = Σ‖x − centroid‖². It assumes roughly spherical, similarly sized clusters and uses Euclidean distance, so **features must be scaled** (otherwise the feature with the largest units decides everything).

**Choosing k without labels:** elbow plot (inertia vs k — look for a bend), silhouette score (−1…1, how much closer a point is to its own cluster than the next one), stability across random seeds/bootstrap resamples, and — most important — *is the segmentation actionable for the business?* In the project all silhouette scores are ≈0.25–0.27: the data has weak natural cluster structure, and the honest answer is "k is a business choice here", not "the metric found the truth".

### Reinforcement learning
An **agent** observes a **state**, takes an **action**, gets a **reward**, and the environment moves to a new state. The goal is a **policy** π(a|s) that maximises the expected *discounted return* Σ γᵗ rₜ.

Key ideas interviewers probe:

* **Exploration vs exploitation** — trying unknown options (might be better) vs using the best-known option (safe now).
* **Delayed reward / credit assignment** — the move that lost the chess game may have been 30 moves ago.
* **Markov Decision Process (MDP)** — the formal model; the *Bellman equation* relates a state's value to its successors' values.
* **Q-learning** (off-policy, tabular): `Q(s,a) ← Q(s,a) + α [ r + γ·maxₐ' Q(s',a') − Q(s,a) ]`.
* **Policy-gradient / actor-critic / PPO** — learn the policy directly; RLHF for LLMs uses PPO-style methods.

The **multi-armed bandit** is RL with *one state*: pick an arm, see a reward. It is the workhorse of production "RL" (ad/email/creative selection, A/B testing that adapts):

| Strategy | Idea | Weakness |
|---|---|---|
| Greedy | always pick the best estimate | can lock onto a wrong arm forever (high variance) |
| ε-greedy | explore with probability ε | keeps exploring at a fixed rate even when sure |
| UCB | pick `mean + c·√(ln t / n)` — optimism under uncertainty | `c` must be scaled to the reward range (see below) |
| Thompson sampling | sample from each arm's posterior, pick the max | needs a probabilistic model per arm |

**Regret** = (reward you would have got always playing the best arm) − (reward you got). Lower is better.

### Which one do I use?

| Situation | Family |
|---|---|
| I have historical outcomes and want to predict them for new cases | Supervised |
| I have no labels and want structure, segments, anomalies, or compressed features | Unsupervised |
| My actions change what happens next, and I learn from outcomes of my own actions | Reinforcement |
| Few labels, many unlabelled rows | Semi-supervised / self-supervised (pre-train on unlabelled, fine-tune on labelled) |

## 3. What the project demonstrates (verified outputs)

* **Supervised:** logistic regression on 4 features reaches ROC-AUC ≈ 0.75 versus 0.5 for chance. Accuracy (≈0.75) barely beats the "always say no churn" baseline (≈0.71) — your first taste of why accuracy misleads (topic 6).
* **Unsupervised:** K-Means segments customers; the high-charge/high-support segment churns at 40 % versus 22 % for the other segment — a label used *only to interpret* clusters, never to build them. Results are written back to a DB table `customer_segments`.
* **Reinforcement:** five email variants with hidden click rates (2%–5%), 30 random seeds per strategy. In the run: random ≈ 140 regret, greedy ≈ 166 **but with std ≈ 125** (some seeds lock onto a bad arm, some get lucky), ε-greedy ≈ 51, textbook UCB ≈ 117, **UCB with `c=0.2` ≈ 45**.
  *Lesson:* the textbook UCB constant assumes rewards spread over [0,1]; with click rates of a few percent it over-explores. Hyper-parameters of exploration must match the reward scale. Also: **never judge a bandit from a single run.**

## 4. Interview questions with model answers

**Q1. Difference between supervised, unsupervised and reinforcement learning? Give an example of each.**
Supervised learns from labelled examples (spam filter). Unsupervised finds structure in unlabelled data (customer segmentation). Reinforcement learns a policy by interacting with an environment and receiving rewards (dynamic pricing, game-playing, adaptive email selection). The difference is the feedback: correct answers, none, or a scalar reward that may be delayed.

**Q2. How do you evaluate a clustering when there are no labels?**
Internal metrics (silhouette, Davies–Bouldin, inertia/elbow), stability under resampling, cluster profiling (are clusters distinct *and* interpretable?), and downstream utility (does targeting by segment improve a KPI in an experiment?). If labels exist for a subset, external metrics like adjusted Rand index.

**Q3. Why must you scale features before K-Means?**
K-Means uses Euclidean distance; a feature in dollars (0–10 000) would dominate one in years (0–70). Standardise so every feature contributes comparably.

**Q4. What is the exploration–exploitation trade-off? How do you handle it?**
Exploiting the current best option earns reward now; exploring gathers information that may find something better. ε-greedy, UCB and Thompson sampling manage it; regret quantifies the cost. Thompson/UCB are usually more sample-efficient than fixed-ε.

**Q5. When would you choose a bandit over an A/B test?**
When the cost of showing a bad variant is high or continuous, and you want traffic to shift toward winners while learning. A/B tests give cleaner inference (fixed allocation, easy hypothesis testing); bandits optimise reward during the experiment but complicate statistical analysis.

**Q6. Is "self-supervised learning" supervised or unsupervised?**
It builds labels *from the data itself* (predict the masked word, the next token, a rotated image's rotation). Technically no human labels, so it is often grouped under unsupervised, but the training loop is supervised. LLM pre-training is self-supervised.

**Follow-ups to prepare:** K-Means failure cases (non-convex clusters → DBSCAN; different densities → GMM), how PCA differs from clustering, what a discount factor γ does, on-policy vs off-policy, why RL needs a simulator.

## 5. Common traps

* Presenting cluster "discoveries" that are just an artefact of unscaled features.
* Using labels to *tune* an unsupervised model, then claiming it is unsupervised.
* Concluding from a single bandit/RL run.
* Choosing RL when a supervised model on logged data would do (RL is usually the last resort, not the first).

---

## 6. Project: Learning Paradigms Lab — roadmap

**Goal:** solve three sub-problems of one telecom business with each paradigm and be able to explain the differences with your own plots.

| Phase | Time | Tasks | Done when |
|---|---|---|---|
| **0. Setup** | 30 min | `pip install -r requirements.txt`; `python -m common.db` (creates DB + seeds data); open `ml_lab.db` in DB Browser for SQLite or `sqlite3` and run `SELECT * FROM customers LIMIT 5;` | You can query the tables |
| **1. Supervised** | 2 h | Read `supervised()`. Add 2 more features (contract, internet). Add a Random Forest; compare AUC. | You can explain why accuracy ≈ baseline but AUC > 0.5 |
| **2. Unsupervised** | 2 h | Read `unsupervised()`. Try k=3,4,5 manually and profile segments. Replace K-Means with a Gaussian Mixture or DBSCAN. Add PCA to 2-D and scatter-plot segments. | You can name each segment in business language |
| **3. RL** | 3 h | Read `run_bandit`. Implement **Thompson sampling** (Beta posteriors). Compare to UCB over 30 seeds. | Thompson's regret plot added to `rl_regret.png` |
| **4. Q-learning (stretch)** | 3 h | Build a 5×5 grid-world, implement tabular Q-learning with ε-decay; plot episode reward. | Agent reaches the goal reliably |
| **5. Write-up** | 1 h | One-page README: "which paradigm for which problem and why". | Ready to present |

**Stretch goals:** contextual bandit (LinUCB) using customer features; semi-supervised churn (label only 5 % of rows, use self-training).

**Resume bullet:** *"Built a comparative ML lab covering supervised churn prediction, K-Means customer segmentation (results persisted to SQL) and bandit-based campaign optimisation; quantified exploration strategies over 30 simulated runs."*
