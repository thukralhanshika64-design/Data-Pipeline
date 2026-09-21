-- =====================================================================================
--  Useful queries for the ML lab database (SQLite dialect; every query here was run).
--  Try them:   sqlite3 -header -column ml_lab.db < database/queries.sql
--  or open ml_lab.db in "DB Browser for SQLite" / DBeaver / VS Code SQLite extension.
--  Run the projects first (python run_all.py) so the tables have data.
-- =====================================================================================

-- 1. What is in the lab? --------------------------------------------------------------
SELECT name, n_rows, n_cols FROM datasets ORDER BY name;

-- 2. Experiments per project ----------------------------------------------------------
SELECT project, COUNT(*) AS runs
FROM experiments
GROUP BY project
ORDER BY project;

-- 3. Leaderboard: best cross-validated ROC-AUC across all projects --------------------
SELECT e.project, e.run_name, e.model_name, ROUND(m.metric_value, 4) AS cv_auc
FROM experiments e
JOIN metrics m ON m.experiment_id = e.id
WHERE m.metric_name = 'cv_roc_auc_mean' AND m.split = 'cv'
ORDER BY m.metric_value DESC
LIMIT 10;

-- 4. Best run per project for one metric (window function) ----------------------------
SELECT project, run_name, model_name, ROUND(metric_value, 4) AS roc_auc
FROM (
    SELECT e.project, e.run_name, e.model_name, m.metric_value,
           ROW_NUMBER() OVER (PARTITION BY e.project ORDER BY m.metric_value DESC) AS rn
    FROM experiments e JOIN metrics m ON m.experiment_id = e.id
    WHERE m.metric_name = 'roc_auc'
) t
WHERE rn = 1
ORDER BY project;

-- 5. Pivot: one row per run with train vs test RMSE (tall metrics table -> wide) ------
SELECT e.run_name,
       ROUND(MAX(CASE WHEN m.split = 'train' AND m.metric_name = 'mse' THEN m.metric_value END), 3) AS train_mse,
       ROUND(MAX(CASE WHEN m.split = 'test'  AND m.metric_name = 'mse' THEN m.metric_value END), 3) AS test_mse
FROM experiments e JOIN metrics m ON m.experiment_id = e.id
WHERE e.project = 'p02_overfit_detective'
GROUP BY e.id, e.run_name
HAVING train_mse IS NOT NULL;

-- 6. Model registry: what is in production? ------------------------------------------
SELECT model_name, version, stage, artifact_path, created_at
FROM model_registry
ORDER BY model_name, version DESC;

-- 7. Which experiment produced the production model, and how good was it? ------------
SELECT r.model_name, r.version, e.run_name, e.params, ROUND(m.metric_value, 4) AS test_auc
FROM model_registry r
JOIN experiments e ON e.id = r.experiment_id
JOIN metrics m ON m.experiment_id = e.id AND m.metric_name = 'roc_auc'
WHERE r.stage = 'production';

-- 8. Business questions on the raw data ----------------------------------------------
-- churn rate by contract type
SELECT contract_type, COUNT(*) AS customers, ROUND(AVG(churned), 3) AS churn_rate
FROM customers
GROUP BY contract_type
ORDER BY churn_rate DESC;

-- data quality: missing values
SELECT SUM(total_charges IS NULL) AS missing_total_charges,
       SUM(payment_method IS NULL) AS missing_payment_method,
       COUNT(*) AS n
FROM customers;

-- K-Means segments (table written by project 1) joined back to the customers
SELECT s.segment, COUNT(*) AS size,
       ROUND(AVG(c.monthly_charges), 1) AS avg_monthly,
       ROUND(AVG(c.churned), 3) AS churn_rate
FROM customer_segments s
JOIN customers c ON c.customer_id = s.customer_id
GROUP BY s.segment;

-- fraud rate per merchant category and hour bucket
SELECT merchant_category, COUNT(*) AS txns, ROUND(100.0 * AVG(is_fraud), 2) AS fraud_pct
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_pct DESC;

-- average price per neighbourhood (regression sanity check)
SELECT neighborhood, COUNT(*) AS houses, ROUND(AVG(price)) AS avg_price
FROM houses
GROUP BY neighborhood
ORDER BY avg_price DESC;

-- 9. Monitoring the served model (needs the API to have received requests) ------------
--    (SQLite has no percentile function; on PostgreSQL use percentile_cont(0.95) WITHIN GROUP)
SELECT model_name, model_version, COUNT(*) AS calls,
       ROUND(AVG(latency_ms), 2) AS avg_ms, ROUND(MAX(latency_ms), 2) AS max_ms,
       ROUND(AVG(prediction), 3) AS avg_predicted_churn
FROM prediction_log
GROUP BY model_name, model_version;

-- p95 latency in plain SQL (works in SQLite): the value at the 95th-percentile rank
SELECT latency_ms AS approx_p95_ms
FROM prediction_log
ORDER BY latency_ms
LIMIT 1 OFFSET (SELECT CAST(0.95 * COUNT(*) AS INTEGER) FROM prediction_log);

-- input drift check: mean of a request feature vs the training population (SQLite JSON1)
--    PostgreSQL:  AVG((features::json->>'monthly_charges')::float)
SELECT ROUND(AVG(json_extract(features, '$.monthly_charges')), 2) AS served_avg_monthly,
       (SELECT ROUND(AVG(monthly_charges), 2) FROM customers)   AS training_avg_monthly
FROM prediction_log;

-- 10. Feature store: engineered features written by project 4 --------------------------
SELECT COUNT(*) AS rows_, COUNT(*) FILTER (WHERE churned = 1) AS churners
FROM customer_features;
