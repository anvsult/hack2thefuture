"""Business rules. Pure functions: cleaned data + a date in, alerts out. No UI, no network."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import open_on, policy_wait
from predict import Predictor

TEAM_LEADS = {
    "Team A": "Team A lead (Toronto, Ottawa)",
    "Team B": "Team B lead (Calgary)",
    "Team C": "Team C lead (Montreal, Quebec City)",
}
RENEWAL_WINDOW_DAYS = 90


@dataclass
class Thresholds:
    warn_days: int = 20
    critical_days: int = 30
    review_sla_days: int | None = None  # None = derive from history


def renewal_uplift(policies: pd.DataFrame, claims: pd.DataFrame, cutoff_days: int = 30) -> dict:
    """Renewal rate for policies whose longest claim wait was within vs beyond `cutoff_days`.

    Uses only policies whose renewal outcome is known (term ended before the extract date).
    """
    pw = policy_wait(claims)
    known = policies[policies["outcome_known"]].join(pw, on="policy_id", how="inner")
    fast = known[known["max_wait"] <= cutoff_days]
    slow = known[known["max_wait"] > cutoff_days]
    return {
        "fast_rate": float(fast["renewed"].mean()),
        "slow_rate": float(slow["renewed"].mean()),
        "fast_n": len(fast),
        "slow_n": len(slow),
        "uplift": float(fast["renewed"].mean() - slow["renewed"].mean()),
    }


def review_sla(claims: pd.DataFrame, cutoff: pd.Timestamp, q: float = 0.75) -> int:
    """75th percentile of close times for $10k+ claims closed before `cutoff`."""
    hist = claims[claims["large"] & claims["closed_date"].notna() & (claims["closed_date"] < cutoff)]
    return int(round(hist["days_to_close"].quantile(q))) if len(hist) >= 10 else 50


def detect(claims: pd.DataFrame, today: pd.Timestamp, predictor: Predictor,
           th: Thresholds, uplift: float) -> pd.DataFrame:
    """Return one row per (claim, rule) that fires on `today`."""
    op = claims[open_on(claims, today)].copy()
    if op.empty:
        return pd.DataFrame()
    op["age"] = (today - op["reported_date"]).dt.days
    op["predicted"] = np.round(predictor.predict(op)).astype(int)
    op["days_to_renewal"] = (op["term_end"] - today).dt.days
    sla = th.review_sla_days or review_sla(claims, predictor.cutoff)

    # Open claims per team today, to suggest where to move work.
    load_by_team = op["adjuster_team"].value_counts().reindex(list(TEAM_LEADS), fill_value=0)

    alerts = []
    for r in op.itertuples(index=False):
        # Rule 1: slow claim. Age is the team-written rule; the prediction gives early warning.
        tier = None
        if r.age >= th.critical_days or r.predicted >= th.critical_days:
            tier = "critical"
        elif r.age >= th.warn_days or r.predicted >= th.warn_days:
            tier = "warn"
        if tier:
            why = (f"open {r.age} days" if r.age >= th.warn_days
                   else f"predicted to take {r.predicted} days")
            alerts.append(_alert(r, "slow_claim", tier, why, load_by_team, uplift))
        # Rule 2: senior-review stall ($10k+ claims open longer than the historical 75th pct).
        if r.large and r.age >= sla:
            alerts.append(_alert(r, "review_stall", "critical",
                                 f"$10k+ claim open {r.age} days (review limit {sla} days)",
                                 load_by_team, uplift))
    out = pd.DataFrame(alerts)
    return out.sort_values(["tier", "premium_at_risk"], ascending=[True, False]).reset_index(drop=True)


def _alert(r, rule: str, tier: str, why: str, load_by_team: pd.Series, uplift: float) -> dict:
    renewal_soon = 0 <= r.days_to_renewal <= RENEWAL_WINDOW_DAYS
    renewal_ahead = r.days_to_renewal > 0
    steps = []
    if rule == "review_stall":
        steps.append("Send to a senior reviewer today")
    elif r.adjuster_team == "Team C":
        other = load_by_team.drop("Team C").idxmin()
        steps.append(f"Reassign to {other} (lowest open backlog: {load_by_team[other]})")
    else:
        steps.append("Fast-track: assign a named adjuster and set a close date")
    if renewal_soon:
        steps.append(f"Retention: call the customer, renewal in {r.days_to_renewal} days")
    return {
        "claim_id": r.claim_id,
        "policy_id": r.policy_id,
        "rule": rule,
        "tier": tier,
        "reason": why,
        "age": r.age,
        "predicted": r.predicted,
        "team": r.adjuster_team,
        "city": r.city,
        "claim_type": r.claim_type,
        "claim_amount": r.claim_amount,
        "annual_premium": r.annual_premium,
        "days_to_renewal": r.days_to_renewal,
        # Only a future renewal can be lost; estimate = premium x renewal-rate gap.
        "premium_at_risk": round(r.annual_premium * uplift, 2) if renewal_ahead and tier == "critical" else 0.0,
        "owner": TEAM_LEADS[r.adjuster_team],
        "next_step": "; ".join(steps),
    }


def premium_at_risk_total(alerts: pd.DataFrame) -> float:
    """Sum per policy (a policy with two alerts counts once)."""
    if alerts.empty:
        return 0.0
    return float(alerts.groupby("policy_id")["premium_at_risk"].max().sum())


if __name__ == "__main__":
    from data import EXTRACT_DATE, load
    from predict import train

    p, c, _ = load()
    u = renewal_uplift(p, c)
    print(u)
    today = EXTRACT_DATE
    pred = train(c, today.replace(day=1), use_model=False)
    a = detect(c, today, pred, Thresholds(), u["uplift"])
    print("open:", open_on(c, today).sum(), "alerts:", len(a), "sla:", review_sla(c, pred.cutoff))
    print(a.groupby(["rule", "tier", "team"]).size())
    print("premium at risk:", premium_at_risk_total(a))
    print(a.head(8)[["claim_id", "rule", "tier", "reason", "team", "premium_at_risk", "next_step"]].to_string())
