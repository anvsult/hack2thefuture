"""Load and clean Millennium Mutual policies and claims.

Every cleaning step appends an entry to a log so the app can show what changed and why.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
EXTRACT_DATE = pd.Timestamp("2026-10-31")
REVIEW_THRESHOLD = 10_000  # claims at or above this go to senior review

CITY_MAP = {
    "toronto": "Toronto",
    "montreal": "Montreal",
    "montréal": "Montreal",
    "mtl": "Montreal",
    "ottawa": "Ottawa",
    "calgary": "Calgary",
    "quebec city": "Quebec City",
    "québec city": "Quebec City",
    "quebec": "Quebec City",
    "qc city": "Quebec City",
}


def _parse_dates(s: pd.Series) -> tuple[pd.Series, int]:
    """Parse ISO (yyyy-mm-dd) dates, falling back to dd/mm/yyyy. Returns (dates, n_fallback)."""
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    alt = s.notna() & iso.isna()
    if alt.any():
        parts = s[alt].str.split("/", expand=True).astype(int)
        # Day comes first: the first part goes above 12, the second part never does.
        assert (parts[1] <= 12).all(), "slash dates are not dd/mm/yyyy"
    dmy = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    out = iso.fillna(dmy)
    assert out[s.notna()].notna().all(), "unparseable dates remain"
    return out, int(alt.sum())


def load(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (policies, claims, cleaning_log)."""
    log: list[dict] = []

    def note(table: str, issue: str, rows: int, fix: str, why: str) -> None:
        log.append({"table": table, "issue": issue, "rows": rows, "fix": fix, "why": why})

    p = pd.read_csv(data_dir / "policies.csv", dtype={"annual_premium": str})
    c = pd.read_csv(data_dir / "claims.csv")

    # --- policies -----------------------------------------------------------------
    raw_city = p["city"]
    key = raw_city.str.strip().str.lower()
    unmapped = set(key) - set(CITY_MAP)
    assert not unmapped, f"unmapped cities: {unmapped}"
    p["city"] = key.map(CITY_MAP)
    note("policies", f"{raw_city.nunique()} spellings of 5 cities (MTL, Montréal, 'Toronto ')",
         int((raw_city != p["city"]).sum()), "Mapped each spelling to one of 5 city names",
         "Split spellings break city and team totals")

    prem_txt = p["annual_premium"].str.contains(r"[$,]", na=False)
    p["annual_premium"] = pd.to_numeric(
        p["annual_premium"].str.replace(r"[$,]", "", regex=True), errors="raise")
    note("policies", "Premium stored as text like '$1,046.62'", int(prem_txt.sum()),
         "Removed $ and commas, converted to number", "Text values cannot be summed")

    p["start_date"], n = _parse_dates(p["start_date"])
    note("policies", "start_date in dd/mm/yyyy instead of yyyy-mm-dd", n,
         "Parsed both formats (verified day-first)", "Wrong dates give wrong policy terms")

    bad_age = p["customer_age"] < 16
    p.loc[bad_age, "customer_age"] = np.nan
    note("policies", "Customer age below 16 (mostly 0)", int(bad_age.sum()),
         "Set to unknown", "Not a valid age for a policyholder")

    zero_prem = p["annual_premium"] <= 0
    p["premium_valid"] = ~zero_prem
    note("policies", "Annual premium of $0", int(zero_prem.sum()),
         "Kept, but excluded from loss-ratio math", "A zero premium breaks ratios")

    p["term_end"] = p["start_date"] + pd.Timedelta(days=365)
    p["renewed"] = p["renewal_status"].eq("Renewed")
    p["outcome_known"] = p["renewal_status"].isin(["Renewed", "Not Renewed", "Cancelled"])

    # --- claims -------------------------------------------------------------------
    dups = c.duplicated().sum()
    c = c.drop_duplicates()
    assert not c["claim_id"].duplicated().any()
    note("claims", "Exact duplicate claim rows", int(dups), "Removed duplicates",
         "Duplicates double-count claims and payouts")

    c["loss_date"], n = _parse_dates(c["loss_date"])
    note("claims", "loss_date in dd/mm/yyyy instead of yyyy-mm-dd", n,
         "Parsed both formats (verified day-first)", "Wrong dates give wrong wait times")
    for col in ("reported_date", "closed_date"):
        c[col], _ = _parse_dates(c[col])

    neg = c["reported_date"] < c["loss_date"]
    c["report_lag"] = (c["reported_date"] - c["loss_date"]).dt.days.clip(lower=0)
    note("claims", "Reported before the loss date", int(neg.sum()),
         "Kept; report lag clipped to 0", "Likely a data-entry slip; wait time uses reported date")

    no_amt = c["claim_amount"].isna()
    note("claims", "Missing claim_amount", int(no_amt.sum()),
         "Kept and flagged; excluded from the $10k review rule",
         "They still count as waiting claims")

    c["days_to_close"] = (c["closed_date"] - c["reported_date"]).dt.days
    c["large"] = c["claim_amount"] >= REVIEW_THRESHOLD
    c["amount_band"] = pd.cut(c["claim_amount"], [0, 2500, 5000, 10_000, np.inf],
                              labels=["<2.5k", "2.5-5k", "5-10k", "10k+"], right=False)
    note("claims", "Denied claims", int(c["status"].eq("Denied").sum()),
         "Kept in wait-time statistics", "The customer still waited for an answer")

    c = c.merge(
        p[["policy_id", "city", "policy_type", "risk_band", "channel", "broker_code",
           "annual_premium", "start_date", "term_end", "renewal_status"]],
        on="policy_id", how="left", validate="many_to_one")
    assert c["city"].notna().all(), "orphan claims"

    return p, c, log


def open_on(claims: pd.DataFrame, today: pd.Timestamp) -> pd.Series:
    """Boolean mask: claim is open (reported, not yet closed) on `today`."""
    return (claims["reported_date"] <= today) & (
        claims["closed_date"].isna() | (claims["closed_date"] > today))


def policy_wait(claims: pd.DataFrame) -> pd.DataFrame:
    """Per policy: claims during the first term, longest wait, and whether a claim was open at term end."""
    first = claims[claims["reported_date"] < claims["term_end"]].copy()
    end = first["closed_date"].fillna(EXTRACT_DATE)
    first["wait"] = (end - first["reported_date"]).dt.days
    first["open_at_term_end"] = end > first["term_end"]
    return first.groupby("policy_id").agg(
        n_claims=("claim_id", "count"),
        max_wait=("wait", "max"),
        open_at_term_end=("open_at_term_end", "any"),
    )


def renewal_by_wait(policies: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Renewal rate by longest claim wait, for policies whose renewal outcome is known."""
    pw = policy_wait(claims)
    known = policies[policies["outcome_known"]].join(pw, on="policy_id")
    known["bucket"] = pd.cut(known["max_wait"], [-1, 10, 20, 30, 10_000],
                             labels=["0-10 days", "11-20 days", "21-30 days", "30+ days"])
    known["bucket"] = known["bucket"].cat.add_categories("No claim").fillna("No claim")
    out = known.groupby("bucket", observed=False).agg(
        policies=("policy_id", "count"), renewal_rate=("renewed", "mean"),
        premium=("annual_premium", "sum"))
    return out.reindex(["No claim", "0-10 days", "11-20 days", "21-30 days", "30+ days"])


if __name__ == "__main__":
    pol, clm, lg = load()
    print(pd.DataFrame(lg).to_string())
    print(pol.shape, clm.shape)
