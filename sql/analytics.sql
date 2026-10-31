-- Reusable analytical queries. Each block starts with "-- name: <query_name>".
-- Parameters use SQLAlchemy/SQLite named binds (:model_version). Loaded by src/database.py.
-- All rates are OBSERVED HISTORICAL associations, not causal effects.

-- name: portfolio_summary
-- Headline numbers for the cleaned portfolio.
SELECT COUNT(*)                                   AS n_customers,
       SUM(default_next_month)                    AS n_defaults,
       ROUND(AVG(default_next_month), 4)          AS default_rate,
       ROUND(AVG(limit_bal), 0)                   AS avg_credit_limit,
       ROUND(AVG(age), 1)                         AS avg_age,
       ROUND(AVG(bill_amt1), 0)                   AS avg_latest_bill
FROM clean_credit_clients;

-- name: split_summary
-- Rows and default rate per train/val/test split (checks the stratification).
SELECT split, COUNT(*) AS n, ROUND(AVG(default_next_month), 4) AS default_rate
FROM clean_credit_clients
GROUP BY split
ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'val' THEN 2 ELSE 3 END;

-- name: default_rate_by_education
SELECT CASE education WHEN 1 THEN 'Graduate school' WHEN 2 THEN 'University'
                      WHEN 3 THEN 'High school' ELSE 'Other' END AS education_level,
       COUNT(*) AS n, ROUND(AVG(default_next_month), 4) AS default_rate
FROM clean_credit_clients
GROUP BY education
ORDER BY education;

-- name: default_rate_by_marriage
SELECT CASE marriage WHEN 1 THEN 'Married' WHEN 2 THEN 'Single' ELSE 'Other' END AS marital_status,
       COUNT(*) AS n, ROUND(AVG(default_next_month), 4) AS default_rate
FROM clean_credit_clients
GROUP BY marriage
ORDER BY marriage;

-- name: default_rate_by_age_band
SELECT CASE WHEN age < 25 THEN '18-24' WHEN age < 35 THEN '25-34' WHEN age < 45 THEN '35-44'
            WHEN age < 55 THEN '45-54' ELSE '55+' END AS age_band,
       COUNT(*) AS n, ROUND(AVG(default_next_month), 4) AS default_rate
FROM clean_credit_clients
GROUP BY age_band
ORDER BY MIN(age);

-- name: default_rate_by_limit_band
SELECT CASE WHEN limit_bal < 50000 THEN '1. <50k' WHEN limit_bal < 100000 THEN '2. 50k-100k'
            WHEN limit_bal < 200000 THEN '3. 100k-200k' WHEN limit_bal < 400000 THEN '4. 200k-400k'
            ELSE '5. 400k+' END AS limit_band,
       COUNT(*) AS n, ROUND(AVG(default_next_month), 4) AS default_rate
FROM clean_credit_clients
GROUP BY limit_band
ORDER BY limit_band;

-- name: default_rate_by_latest_repayment_status
-- pay_1: -2 no consumption, -1 paid in full, 0 revolving, 1..9 months of payment delay.
SELECT pay_1 AS latest_repayment_status, COUNT(*) AS n,
       ROUND(AVG(default_next_month), 4) AS default_rate
FROM clean_credit_clients
GROUP BY pay_1
ORDER BY pay_1;

-- name: default_rate_by_months_delayed
-- Repayment-history group: how many of the last 6 months had a payment delay (status >= 1).
SELECT (pay_1 >= 1) + (pay_2 >= 1) + (pay_3 >= 1) + (pay_4 >= 1) + (pay_5 >= 1) + (pay_6 >= 1)
           AS months_delayed_of_6,
       COUNT(*) AS n, ROUND(AVG(default_next_month), 4) AS default_rate
FROM clean_credit_clients
GROUP BY months_delayed_of_6
ORDER BY months_delayed_of_6;

-- name: risk_band_summary
-- Predicted risk bands vs OBSERVED outcomes for held-out records scored by a model version.
SELECT p.risk_band, COUNT(*) AS n,
       ROUND(AVG(p.predicted_probability), 4) AS avg_predicted_probability,
       ROUND(AVG(c.default_next_month), 4)    AS observed_default_rate,
       SUM(p.flagged_for_review)              AS n_flagged
FROM predictions p
JOIN clean_credit_clients c ON c.customer_id = p.customer_id
WHERE p.model_version = :model_version AND p.source = 'heldout_test'
GROUP BY p.risk_band
ORDER BY avg_predicted_probability;

-- name: flagged_accounts
-- Highest-probability held-out accounts flagged for analyst review.
SELECT p.customer_id, ROUND(p.predicted_probability, 4) AS predicted_probability, p.risk_band,
       c.pay_1, c.limit_bal, c.default_next_month AS observed_outcome
FROM predictions p
JOIN clean_credit_clients c ON c.customer_id = p.customer_id
WHERE p.model_version = :model_version AND p.source = 'heldout_test' AND p.flagged_for_review = 1
ORDER BY p.predicted_probability DESC
LIMIT 50;

-- name: recent_prediction_log
SELECT pred_id, model_version, customer_id, source, ROUND(predicted_probability, 4) AS probability,
       threshold, flagged_for_review, risk_band, predicted_at
FROM predictions
WHERE source <> 'heldout_test'
ORDER BY pred_id DESC
LIMIT 200;
