"""Simulated omnichannel HCP journeys with a known truth.

Open Payments shows only one promotional channel (meals and speaker fees).
Rep visits, emails, webinars and digital ads are not public anywhere, so the
multi-channel half of this project uses a simulation in which every channel's
true effect is known. That lets each method (attribution, uplift modelling,
a learning bandit) be graded against the answer instead of being trusted on
faith.

The world: 20,000 prescribers over 52 weeks, deciding whether to start a new
diabetes brand. Each week a prescriber who has not yet adopted has a chance of
writing a first prescription. Promotion raises that chance, with:
  * carry-over:  a touch keeps working for a few weeks (stock decays by half a week);
  * segments:    digital-first, traditional and low-engagement prescribers respond
                 differently to each channel;
  * synergy:     an email shortly after a rep visit works better than either alone;
  * fatigue:     more than three brand emails in four weeks starts to hurt.

Terms
-----
Adoption       the prescriber's first prescription of the brand (a "conversion").
Path           the ordered list of touches a prescriber received before adopting.
Uplift         the change in a prescriber's adoption chance caused by a channel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["Rep visit", "Rep-triggered email", "Brand email", "Webinar", "Speaker program", "Endemic banner"]
SEGMENTS = ["Digital-first", "Traditional", "Low-engagement"]
COST = {"Rep visit": 250, "Rep-triggered email": 5, "Brand email": 2, "Webinar": 40, "Speaker program": 350, "Endemic banner": 6}
VALUE_PER_ADOPTION = 25_000  # illustrative first-year net revenue from a prescriber who adopts (about ten patients)

# True log-odds effect of a full unit of "stock" for each channel and segment.
EFFECT = pd.DataFrame({
    "Digital-first":  [0.45, 0.55, 0.40, 0.45, 0.50, 0.14],
    "Traditional":    [0.85, 0.35, 0.06, 0.25, 0.85, 0.03],
    "Low-engagement": [0.40, 0.15, 0.03, 0.10, 0.35, 0.02],
}, index=CHANNELS)
SYNERGY = 0.35        # extra log-odds when rep and email stock are both present
FATIGUE = 0.18        # log-odds lost per brand email above 3 in the last 4 weeks
DECAY = 0.5           # weekly carry-over of every channel's stock
BASE = -5.9           # weekly log-odds with no promotion


def population(n: int = 20_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seg = rng.choice(SEGMENTS, n, p=[0.30, 0.45, 0.25])
    spec = rng.choice(["Primary care", "Endocrinology", "NP / PA"], n, p=[0.60, 0.15, 0.25])
    decile = rng.integers(1, 11, n)
    # What a brand team can observe: proxies for the segment, not the segment itself
    dig = np.clip(np.where(seg == "Digital-first", 0.65, np.where(seg == "Traditional", 0.35, 0.25)) + rng.normal(0, 0.15, n), 0, 1)
    opens = np.clip(np.where(seg == "Digital-first", 0.35, np.where(seg == "Traditional", 0.15, 0.06)) + rng.normal(0, 0.07, n), 0, 1)
    rep_access = np.clip(np.where(seg == "Traditional", 0.75, np.where(seg == "Digital-first", 0.45, 0.25)) + rng.normal(0, 0.12, n), 0, 1)
    return pd.DataFrame({"hcp": np.arange(n), "segment": seg, "specialty": spec, "decile": decile,
                         "digital_affinity": dig.round(3), "email_open_rate": opens.round(3), "rep_access": rep_access.round(3),
                         "email_opt_in": rng.random(n) < 0.72})


def exposure_plan(pop: pd.DataFrame, weeks: int, rng: np.random.Generator, email_holdout: np.ndarray | None = None) -> np.ndarray:
    """Touches under the brand's current rule-based plan, shape (weeks, n, channels)."""
    n = len(pop)
    dz = (pop["decile"].to_numpy() - 1) / 9
    p = np.zeros((n, len(CHANNELS)))
    p[:, 0] = (0.03 + 0.10 * dz) * pop["rep_access"].to_numpy()                      # reps chase volume
    p[:, 2] = np.where(pop["email_opt_in"].to_numpy(), 0.45, 0.0)                     # brand email to opt-ins
    p[:, 3] = 0.03 + 0.03 * pop["digital_affinity"].to_numpy()
    p[:, 4] = 0.004 + 0.02 * dz ** 2                                                  # speaker programs for top deciles
    p[:, 5] = 0.05 + 0.25 * pop["digital_affinity"].to_numpy()
    if email_holdout is not None:
        p[email_holdout, 2] = 0.0
    X = (rng.random((weeks, n, len(CHANNELS))) < p[None, :, :]).astype(np.int8)
    # Rep-triggered email: sent the week after a rep visit, to half of opted-in prescribers
    trig = np.zeros_like(X[:, :, 0])
    trig[1:] = X[:-1, :, 0] * (rng.random((weeks - 1, n)) < 0.5) * pop["email_opt_in"].to_numpy()[None, :]
    X[:, :, 1] = trig
    return X


def run(pop: pd.DataFrame, X: np.ndarray, rng_seed: int = 5, drop: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate adoption. Returns (adoption week or -1, weekly hazard matrix).

    `drop` removes one channel's touches (same random numbers), which gives the
    true incremental effect of that channel."""
    rng = np.random.default_rng(rng_seed)
    weeks, n, C = X.shape
    U = rng.random((weeks, n))  # common random numbers across counterfactuals
    eff, dz = segment_effects(pop)
    Xd = X.copy()
    if drop is not None:
        Xd[:, :, CHANNELS.index(drop)] = 0
    stock = np.zeros((n, C))
    email_hist = np.zeros((weeks, n))
    adopted = np.full(n, -1)
    for t in range(weeks):
        stock = stock * DECAY + Xd[t]
        email_hist[t] = Xd[t, :, 2]
        recent = email_hist[max(0, t - 3): t + 1].sum(axis=0)
        h = hazard(stock, recent, eff, dz)
        new = (adopted < 0) & (U[t] < h)
        adopted[new] = t
    return adopted, None


def segment_effects(pop: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    seg_idx = pop["segment"].map({s: i for i, s in enumerate(SEGMENTS)}).to_numpy()
    return EFFECT.to_numpy()[:, seg_idx].T, (pop["decile"].to_numpy() - 1) / 9


def hazard(stock: np.ndarray, recent_emails: np.ndarray, eff: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Weekly adoption probability given each prescriber's promotion stock."""
    z = (BASE + 0.9 * dz + (eff * np.minimum(stock, 2)).sum(axis=1)
         + SYNERGY * np.minimum(stock[:, 0], 1) * np.minimum(stock[:, 1] + stock[:, 2], 1)
         - FATIGUE * np.maximum(recent_emails - 3, 0))
    return 1 / (1 + np.exp(-z))


def paths(X: np.ndarray, adopted: np.ndarray) -> list[list[str]]:
    """Ordered channel touches before adoption (or over the whole year if no adoption)."""
    weeks, n, C = X.shape
    out = []
    for i in range(n):
        end = adopted[i] if adopted[i] >= 0 else weeks - 1
        seq = []
        for t in range(end + 1):
            for c in np.nonzero(X[t, i])[0]:
                seq.append(CHANNELS[c])
        out.append(seq)
    return out
