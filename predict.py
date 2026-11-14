"""Predict how many days a claim will take to close.

Two estimators, both trained only on claims that had closed before a cutoff date:
- Baseline: median days by adjuster team and amount band (transparent lookup).
- Model: gradient boosting regressor on features known when the claim is reported.

The cutoff moves with the simulated clock (first day of the current month), so a
prediction never uses outcomes from the future.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CAT = ["adjuster_team", "city", "claim_type", "policy_type", "risk_band", "channel"]
NUM = ["claim_amount", "report_lag"]
FEATURES = CAT + NUM
# Never used as features (they reveal the answer): status, paid_amount, closed_date.


@dataclass
class Predictor:
    cutoff: pd.Timestamp
    n_train: int
    baseline: pd.Series
    team_median: pd.Series
    overall_median: float
    model: Pipeline | None

    def predict_baseline(self, df: pd.DataFrame) -> np.ndarray:
        keys = list(zip(df["adjuster_team"], df["amount_band"].astype(str)))
        out = np.array([self.baseline.get(k, np.nan) for k in keys], dtype=float)
        fallback = df["adjuster_team"].map(self.team_median).fillna(self.overall_median)
        return np.where(np.isnan(out), fallback.to_numpy(dtype=float), out)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or df.empty:
            return self.predict_baseline(df)
        return np.clip(self.model.predict(df[FEATURES]), 1, None)


def _closed_before(claims: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return claims[claims["closed_date"].notna() & (claims["closed_date"] < cutoff)]


def _make_model() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
        ("num", "passthrough", NUM),
    ])
    reg = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=200, learning_rate=0.05, max_depth=3,
        min_samples_leaf=15, categorical_features=list(range(len(CAT))), random_state=0)
    return Pipeline([("pre", pre), ("reg", reg)])


def train(claims: pd.DataFrame, cutoff: pd.Timestamp, use_model: bool = True) -> Predictor:
    hist = _closed_before(claims, cutoff)
    baseline = hist.groupby(["adjuster_team", hist["amount_band"].astype(str)])["days_to_close"].median()
    team_median = hist.groupby("adjuster_team")["days_to_close"].median()
    overall = float(hist["days_to_close"].median()) if len(hist) else 17.0
    model = None
    if use_model and len(hist) >= 150:
        model = _make_model().fit(hist[FEATURES], hist["days_to_close"])
    return Predictor(cutoff, len(hist), baseline, team_median, overall, model)


def evaluate(claims: pd.DataFrame, split: pd.Timestamp) -> dict:
    """Time-based test: train on claims closed before `split`, test on claims reported after it."""
    pred = train(claims, split)
    test = claims[(claims["reported_date"] >= split) & claims["closed_date"].notna()]
    y = test["days_to_close"].to_numpy()
    naive = np.full_like(y, pred.overall_median, dtype=float)
    return {
        "split": split.date().isoformat(),
        "n_train": pred.n_train,
        "n_test": len(test),
        "mae_overall_median": float(np.mean(np.abs(y - naive))),
        "mae_baseline": float(np.mean(np.abs(y - pred.predict_baseline(test)))),
        "mae_model": float(np.mean(np.abs(y - pred.predict(test)))),
    }


def feature_importance(claims: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 0) -> pd.Series:
    """Permutation importance (increase in absolute error when a feature is shuffled)."""
    from sklearn.inspection import permutation_importance

    pred = train(claims, cutoff)
    test = claims[(claims["reported_date"] >= cutoff) & claims["closed_date"].notna()]
    if pred.model is None or len(test) < 30:
        return pd.Series(dtype=float)
    r = permutation_importance(pred.model, test[FEATURES], test["days_to_close"],
                               scoring="neg_mean_absolute_error", n_repeats=10, random_state=seed)
    return pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)


if __name__ == "__main__":
    from data import load

    _, c, _ = load()
    for s in ["2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01"]:
        print(evaluate(c, pd.Timestamp(s)))
    print(feature_importance(c, pd.Timestamp("2026-04-01")).round(2))
