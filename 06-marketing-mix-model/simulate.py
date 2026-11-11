"""Simulated brand data for the marketing mix model, with a known truth.

No company publishes weekly promotional spend by channel, so marketing mix
work cannot be shown on public data. Instead this builds a realistic brand:
a branded type 2 diabetes injectable sold in ten US regions over three years
(156 weeks), promoted through six channels, with the things that really move
pharma demand alongside it: an annual Q1 dip when deductibles reset, a slow
underlying trend, a competitor launch and a formulary win at a large PBM.

Because the data is simulated, the true return of every channel is known, so
the model's estimates can be checked against the answer. The truth uses a Hill
saturation curve and a 12-week carry-over window while the model uses a
different curve family and an 8-week window, so the model is not simply told
the right functional form.

Terms
-----
NRx            new prescriptions; the outcome a brand team plans around.
Adstock        carry-over: part of this week's promotion still works next week.
               Geometric adstock keeps a fixed share (the decay rate) each week.
Saturation     diminishing returns: each extra dollar adds less than the last.
ROI            incremental net revenue divided by spend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20251231
WEEKS = 156
START = "2023-01-02"
VALUE_PER_NRX = 3_000  # illustrative net revenue per new prescription, including 12 months of refills

REGIONS = {  # share of national demand
    "Northeast": 0.17, "Mid-Atlantic": 0.12, "Southeast": 0.15, "Florida": 0.08, "Great Lakes": 0.12,
    "Plains": 0.06, "Texas": 0.10, "Mountain": 0.06, "California": 0.09, "Pacific NW": 0.05,
}

# Weekly national spend level, true effect size and shape for each channel.
#   beta  : weekly NRx at full saturation, national
#   decay : share of effect carried to the next week
#   half  : per-capita spend (relative to the channel's average) at half saturation
#   shape : Hill steepness
CHANNELS = {
    "Field rep details":     dict(spend=1_500_000, beta=1_550, decay=0.55, half=1.1, shape=1.6, kind="HCP"),
    "Speaker programs":      dict(spend=250_000,   beta=330,   decay=0.70, half=1.0, shape=1.8, kind="HCP"),
    "HCP digital & email":   dict(spend=200_000,   beta=380,   decay=0.35, half=0.9, shape=1.4, kind="HCP"),
    "DTC television":        dict(spend=1_800_000, beta=1_200, decay=0.80, half=1.3, shape=2.0, kind="Consumer"),
    "DTC social & video":    dict(spend=500_000,   beta=460,   decay=0.45, half=1.0, shape=1.5, kind="Consumer"),
    "Paid search":           dict(spend=300_000,   beta=390,   decay=0.15, half=0.8, shape=1.3, kind="Consumer"),
}

COMPETITOR_WEEK = 70   # a competing brand launches; baseline demand falls
FORMULARY_WEEK = 105   # preferred status won at a large PBM; baseline demand rises


def geometric_adstock(x: np.ndarray, decay: float, window: int) -> np.ndarray:
    w = decay ** np.arange(window)
    w = w / w.sum()
    out = np.zeros_like(x, dtype=float)
    for lag, wl in enumerate(w):
        out[lag:] += wl * x[: len(x) - lag]
    return out


def hill(x: np.ndarray, half: float, shape: float) -> np.ndarray:
    return x ** shape / (x ** shape + half ** shape)


def _spend_patterns(rng: np.random.Generator) -> dict[str, np.ndarray]:
    t = np.arange(WEEKS)
    pats = {}
    # Reps: steady, with summer and holiday troughs and a sales-force expansion in year 3
    pats["Field rep details"] = 1 - 0.18 * np.isin(t % 52, list(range(26, 34)) + [51, 0]) + 0.15 * (t >= 110)
    # Speaker programs: bunched around conference season, paused in summer
    pats["Speaker programs"] = 0.6 + 0.8 * np.isin(t % 52, list(range(8, 20)) + list(range(36, 46))) * rng.uniform(0.7, 1.3, WEEKS)
    # HCP digital: grows steadily as budgets move to digital
    pats["HCP digital & email"] = 0.7 + 0.6 * t / WEEKS
    # DTC TV: on/off flights of 6-8 weeks
    tv = np.zeros(WEEKS)
    week = 2
    while week < WEEKS:
        length = rng.integers(6, 9)
        tv[week: week + length] = rng.uniform(1.4, 2.2)
        week += length + rng.integers(4, 9)
    pats["DTC television"] = tv
    # Social: always on, with bursts that overlap TV flights only partly
    pats["DTC social & video"] = 0.7 + 0.5 * (np.roll(tv, 3) > 0) + rng.normal(0, 0.12, WEEKS)
    # Search: follows consumer interest (TV flights lift it) plus noise
    pats["Paid search"] = 0.8 + 0.25 * (tv > 0) + rng.normal(0, 0.1, WEEKS)
    for k in pats:
        p = np.clip(pats[k], 0, None)
        pats[k] = p / p.mean()
    return pats


def simulate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weekly data by region, true channel contributions and ROI)."""
    rng = np.random.default_rng(SEED)
    t = np.arange(WEEKS)
    dates = pd.date_range(START, periods=WEEKS, freq="W-MON")
    pats = _spend_patterns(rng)

    # Regional allocation differs by channel, which is what lets a model separate them.
    tilt = {c: rng.lognormal(0, 0.35, len(REGIONS)) for c in CHANNELS}
    season = -0.10 * np.isin(t % 52, range(0, 9)) + 0.03 * np.sin(2 * np.pi * t / 52)  # Q1 deductible reset
    trend = 0.0018 * t
    events = -0.07 * (t >= COMPETITOR_WEEK) + 0.09 * (t >= FORMULARY_WEEK)
    base_national = 9_000  # weekly NRx with no promotion at all

    frames, contrib_rows = [], []
    for ri, (region, share) in enumerate(REGIONS.items()):
        reg_noise = rng.normal(0, 0.03)
        base = base_national * share * (1 + reg_noise) * (1 + trend + season + events)
        y = base.copy()
        row = {"week": dates, "region": region, "region_share": share, "baseline_true": base}
        for c, p in CHANNELS.items():
            alloc = tilt[c][ri] / np.sum(tilt[c] * np.array(list(REGIONS.values()))) * share
            noise = 1 + rng.normal(0, 0.06, WEEKS) if c != "DTC television" else 1
            spend = np.clip(p["spend"] * alloc * pats[c] * noise, 0, None)
            per_cap = spend / (p["spend"] * share)  # spend relative to the region's fair share
            eff = p["beta"] * share * hill(geometric_adstock(per_cap, p["decay"], 12), p["half"], p["shape"])
            y += eff
            row[f"spend::{c}"] = spend
            row[f"true::{c}"] = eff
        y = y * rng.lognormal(0, 0.035, WEEKS)
        row["nrx"] = y
        frames.append(pd.DataFrame(row))

    df = pd.concat(frames, ignore_index=True)
    for c in CHANNELS:
        spend = df[f"spend::{c}"].sum()
        inc = df[f"true::{c}"].sum()
        contrib_rows.append({"channel": c, "kind": CHANNELS[c]["kind"], "spend": spend, "incremental_nrx": inc,
                             "true_roi": inc * VALUE_PER_NRX / spend, "true_decay": CHANNELS[c]["decay"]})
    truth = pd.DataFrame(contrib_rows)
    df["competitor_launched"] = (df.groupby("region").cumcount() >= COMPETITOR_WEEK).astype(int)
    df["formulary_win"] = (df.groupby("region").cumcount() >= FORMULARY_WEEK).astype(int)
    return df, truth


def true_response(channel: str, weekly_national_spend: np.ndarray) -> np.ndarray:
    """True steady-state weekly NRx from a constant national spend level (all regions at fair share)."""
    p = CHANNELS[channel]
    per_cap = weekly_national_spend / p["spend"]
    return p["beta"] * hill(per_cap, p["half"], p["shape"])
