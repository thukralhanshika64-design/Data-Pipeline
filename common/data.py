"""
Synthetic-but-realistic datasets.

Why synthetic?
  * Every project runs offline and is 100% reproducible (fixed seeds).
  * We control the "ground truth", so we can PROVE why a technique helps
    (e.g. churn depends on log(monthly_charges) -> log-transform helps).
  * You can swap in a real dataset (Kaggle Telco Churn, IEEE-CIS Fraud,
    Ames Housing) later by loading it into the same DB table names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
# 1) Customers  -> churn classification (projects 1, 4, 5, 7, 10)
# --------------------------------------------------------------------------- #
def make_customers(n: int = 5000, seed: int = 42, n_cities: int = 30, nonlinear: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cities = [f"city_{i:03d}" for i in range(n_cities)]  # high-cardinality categorical
    city_effect = dict(zip(cities, rng.normal(0, 0.6, len(cities))))

    signup = pd.Timestamp("2018-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 6, n), unit="D"
    )
    ref_date = pd.Timestamp("2024-01-01")
    tenure = np.clip(((ref_date - signup).days // 30).astype(int), 1, None)

    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.35, 0.20])
    payment = rng.choice(["credit_card", "bank_transfer", "e-check", "mailed_check"], n)
    city = rng.choice(cities, n)

    base_price = pd.Series(internet).map({"fiber": 80, "dsl": 55, "none": 25}).to_numpy()
    monthly = (base_price * rng.lognormal(0, 0.25, n)).round(2)  # right-skewed
    total = (monthly * tenure * rng.uniform(0.9, 1.1, n)).round(2)
    support_calls = rng.poisson(np.where(internet == "fiber", 2.0, 1.0))

    # Ground-truth churn mechanism (the model must rediscover this)
    logit = (
        -1.2
        + 1.0 * (contract == "month-to-month")
        - 0.9 * (contract == "two-year")
        - 0.03 * tenure
        + 1.1 * np.log(monthly / 50.0)          # log-scale effect
        + 0.25 * support_calls
        + 0.45 * (payment == "e-check")
        + 0.35 * np.isin(signup.month, [11, 12])  # seasonal signup effect
        + pd.Series(city).map(city_effect).to_numpy()
    )
    if nonlinear:   # threshold/XOR-style rules that a linear model cannot represent but tree ensembles can
        hi_price, many_calls = monthly > 80, support_calls >= 3
        logit = 0.5 * logit - 0.9 + (
            3.0 * (hi_price ^ many_calls)                           # XOR: exactly one of the two
            + 2.2 * ((tenure <= 8) & (contract == "month-to-month"))
            - 2.0 * ((tenure > 36) & (internet == "dsl"))
        )
    churn = rng.binomial(1, _sigmoid(logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "signup_date": signup,
            "tenure_months": tenure,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "city": city,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_calls": support_calls,
            "churned": churn,
        }
    )
    # Realistic mess: missing values
    miss = rng.random(n) < 0.03
    df.loc[miss, "total_charges"] = np.nan
    miss2 = rng.random(n) < 0.02
    df.loc[miss2, "payment_method"] = None
    return df


# --------------------------------------------------------------------------- #
# 2) Transactions -> fraud (imbalanced) (projects 6, 7)
# --------------------------------------------------------------------------- #
def make_transactions(n: int = 40000, fraud_rate: float = 0.01, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate
    f = is_fraud.astype(int)

    amount = rng.lognormal(np.where(is_fraud, 4.6, 3.6), 0.9).round(2)
    hour = np.where(
        is_fraud,
        rng.choice(24, n, p=_hour_probs(night_heavy=True)),
        rng.choice(24, n, p=_hour_probs(night_heavy=False)),
    )
    df = pd.DataFrame(
        {
            "txn_id": [f"T{i:07d}" for i in range(n)],
            "timestamp": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(np.sort(rng.integers(0, 90 * 24 * 3600, n)), unit="s"),
            "amount": amount,
            "hour": hour,
            "merchant_category": rng.choice(
                ["grocery", "electronics", "travel", "restaurant", "fuel", "online_retail"], n
            ),
            "is_foreign": rng.binomial(1, np.where(is_fraud, 0.45, 0.05)),
            "distance_from_home_km": np.abs(rng.normal(np.where(is_fraud, 120, 15), 40)).round(1),
            "card_age_days": np.clip(rng.normal(np.where(is_fraud, 300, 900), 350), 1, None).astype(int),
            "txn_count_last_24h": rng.poisson(np.where(is_fraud, 3.5, 1.2)),
            "is_fraud": f,
        }
    )
    return df


def _hour_probs(night_heavy: bool):
    p = np.ones(24)
    if night_heavy:
        p[[0, 1, 2, 3, 4, 23]] = 4.0
    else:
        p[[8, 9, 12, 13, 17, 18, 19]] = 3.0
    return p / p.sum()


# --------------------------------------------------------------------------- #
# 3) Houses -> regression (projects 5, 8)
# --------------------------------------------------------------------------- #
def make_houses(n: int = 3000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    neighborhoods = {"downtown": 1.5, "riverside": 1.3, "midtown": 1.15, "suburb_a": 1.0,
                     "suburb_b": 0.95, "old_town": 0.9, "outskirts": 0.75, "industrial": 0.6}
    hood = rng.choice(list(neighborhoods), n)
    sqft = np.clip(rng.lognormal(7.3, 0.35, n), 400, 6000).round(0)
    bedrooms = np.clip((sqft / 600 + rng.normal(0, 0.7, n)).round(), 1, 7).astype(int)
    bathrooms = np.clip((bedrooms * 0.6 + rng.normal(0, 0.5, n)).round(), 1, 5).astype(int)
    age = rng.integers(0, 80, n)
    dist = np.abs(rng.normal(8, 5, n)).round(1)
    garage = rng.binomial(1, 0.6, n)

    mult = pd.Series(hood).map(neighborhoods).to_numpy()
    price = (
        (110 * sqft * mult) * (1 - 0.004 * age) * (1 - 0.015 * np.minimum(dist, 30))
        + 9000 * bathrooms + 14000 * garage
    ) * rng.lognormal(0, 0.10, n)
    outliers = rng.random(n) < 0.01  # a few luxury / data-entry outliers
    price = np.where(outliers, price * 2.5, price).round(-2)

    return pd.DataFrame(
        {
            "house_id": np.arange(n),
            "sqft": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "dist_to_center_km": dist,
            "has_garage": garage,
            "neighborhood": hood,
            "price": price,
        }
    )
