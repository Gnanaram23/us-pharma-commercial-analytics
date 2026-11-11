"""A Bayesian marketing mix model with carry-over and diminishing returns.

    NRx(region, week) / region size
        = region baseline + trend + seasonality + known events
        + sum over channels of  beta_c * saturation_c( adstock_c( spend ) )
        + noise

* Adstock: geometric, 8-week window, decay rate learned per channel.
* Saturation: logistic, (1 - e^(-lam x)) / (1 + e^(-lam x)), lam learned per channel.
* Spend enters per head of regional demand and relative to the channel's average,
  so one curve describes every region.
* Fitted with the No-U-Turn Sampler (NUTS) in PyMC, which returns thousands of
  plausible parameter sets rather than one, so every ROI comes with a range.

Optionally, the result of a lift test (a geo experiment that measured one
channel's ROI directly) can be added as extra evidence. This is "calibration":
the experiment pins down the channel that observational data alone cannot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

WINDOW = 8


def design(df: pd.DataFrame, channels: list[str]) -> dict:
    regions = list(dict.fromkeys(df["region"]))
    weeks = sorted(df["week"].unique())
    R, T, C = len(regions), len(weeks), len(channels)
    share = df.groupby("region")["region_share"].first().loc[regions].to_numpy()
    spend = np.stack([df.pivot(index="week", columns="region", values=f"spend::{c}")[regions].to_numpy().T
                      for c in channels], axis=-1)  # (R, T, C)
    nat_mean = spend.sum(axis=0).mean(axis=0)  # national mean weekly spend per channel
    x = spend / (nat_mean[None, None, :] * share[:, None, None])
    xlag = np.zeros((R, T, C, WINDOW))
    for lag in range(WINDOW):
        xlag[:, lag:, :, lag] = x[:, : T - lag, :]
    y = df.pivot(index="week", columns="region", values="nrx")[regions].to_numpy().T  # (R, T)
    ybar = y.sum(axis=0).mean()
    t = np.arange(T)
    four = np.column_stack([f(2 * np.pi * k * t / 52) for k in (1, 2) for f in (np.sin, np.cos)])
    q1 = np.isin(t % 52, range(0, 9)).astype(float)
    ev = df.groupby("week")[["competitor_launched", "formulary_win"]].first().to_numpy().astype(float)
    return dict(regions=regions, weeks=weeks, channels=channels, share=share, spend=spend, nat_mean=nat_mean,
                xlag=xlag, y=y, y_norm=y / (share[:, None] * ybar), ybar=ybar, t=t / T, four=four, q1=q1, events=ev)


def build(d: dict, train_weeks: int | None = None, lift: dict | None = None) -> pm.Model:
    """lift = {"channel": name, "roi": measured ROI, "se": standard error, "value": $ per NRx}"""
    R, T, C = d["spend"].shape
    tt = slice(0, train_weeks or T)
    coords = {"region": d["regions"], "channel": d["channels"], "lag": np.arange(WINDOW)}
    with pm.Model(coords=coords) as model:
        # Media
        decay = pm.Beta("decay", 2, 2, dims="channel")
        lam = pm.Gamma("lam", 2, 1.5, dims="channel")
        beta = pm.HalfNormal("beta", 0.15, dims="channel")
        w = decay[:, None] ** pt.arange(WINDOW)[None, :]
        w = w / w.sum(axis=1, keepdims=True)
        ad = (pt.as_tensor(d["xlag"]) * w[None, None, :, :]).sum(axis=-1)  # (R, T, C)
        sat = (1 - pt.exp(-lam * ad)) / (1 + pt.exp(-lam * ad))
        media = sat * beta  # (R, T, C)
        # Baseline
        b0 = pm.Normal("baseline_mu", 0.85, 0.2)
        b_sd = pm.HalfNormal("baseline_sd", 0.05)
        b_r = pm.Normal("baseline", b0, b_sd, dims="region")
        trend = pm.Normal("trend", 0, 0.3)
        four = pm.Normal("fourier", 0, 0.05, shape=d["four"].shape[1])
        q1 = pm.Normal("q1_reset", 0, 0.1)
        ev = pm.Normal("events", 0, 0.15, shape=2)
        base = (b_r[:, None] + trend * d["t"][None, :] + pt.dot(d["four"], four)[None, :]
                + q1 * d["q1"][None, :] + pt.dot(d["events"], ev)[None, :])
        mu = base + media.sum(axis=-1)
        sigma = pm.HalfNormal("sigma", 0.05)
        pm.Normal("obs", mu[:, tt], sigma, observed=d["y_norm"][:, tt])
        # Contribution in NRx and ROI per channel (over the training window)
        nrx = media[:, tt, :] * (d["share"][:, None, None] * d["ybar"])
        contrib = pm.Deterministic("contribution_nrx", nrx.sum(axis=(0, 1)), dims="channel")
        if lift is not None:
            ci = d["channels"].index(lift["channel"])
            spend_c = d["spend"][:, tt, ci].sum()
            roi_c = contrib[ci] * lift["value"] / spend_c
            pm.Normal("lift_test", roi_c, lift["se"], observed=lift["roi"])
    return model


def fit(model: pm.Model, draws: int = 1000, tune: int = 1500, seed: int = 7):
    with model:
        return pm.sample(draws=draws, tune=tune, chains=4, cores=2, target_accept=0.95,
                         random_seed=seed, progressbar=False)


def posterior(idata, max_draws: int | None = None):
    """Posterior draws stacked into one 's' dimension (works across ArviZ versions)."""
    p = idata.posterior
    ds = p.to_dataset() if hasattr(p, "to_dataset") else p
    ds = ds.stack(s=("chain", "draw"))
    if max_draws and ds.sizes["s"] > max_draws:
        ds = ds.isel(s=np.linspace(0, ds.sizes["s"] - 1, max_draws).astype(int))
    return ds


def predict(idata, d: dict, max_draws: int = 800):
    """Posterior draws of NRx (draws, R, T) and of each channel's NRx (draws, R, T, C)."""
    post = posterior(idata, max_draws)
    decay, lam, beta = (post[v].values.T for v in ("decay", "lam", "beta"))  # (S, C)
    S = decay.shape[0]
    w = decay[:, :, None] ** np.arange(WINDOW)[None, None, :]
    w = w / w.sum(axis=2, keepdims=True)  # (S, C, L)
    ad = np.einsum("rtcl,scl->srtc", d["xlag"], w)
    sat = (1 - np.exp(-lam[:, None, None, :] * ad)) / (1 + np.exp(-lam[:, None, None, :] * ad))
    media = (sat * beta[:, None, None, :]).sum(axis=-1)
    b_r = post["baseline"].values.T  # (S, R)
    four = post["fourier"].values.T
    ev = post["events"].values.T
    base = (b_r[:, :, None] + post["trend"].values[:, None, None] * d["t"][None, None, :]
            + (four @ d["four"].T)[:, None, :] + post["q1_reset"].values[:, None, None] * d["q1"][None, None, :]
            + (ev @ d["events"].T)[:, None, :])
    return (base + media) * d["share"][None, :, None] * d["ybar"], sat * beta[:, None, None, :] * d["share"][None, :, None, None] * d["ybar"]


def response(idata, d: dict, ci: int, weekly_spend: np.ndarray) -> np.ndarray:
    """Posterior steady-state weekly national NRx at constant spend levels, shape (draws, len(spend))."""
    post = posterior(idata)
    lam = post["lam"].values[ci]
    beta = post["beta"].values[ci]
    x = weekly_spend / d["nat_mean"][ci]
    sat = (1 - np.exp(-lam[:, None] * x[None, :])) / (1 + np.exp(-lam[:, None] * x[None, :]))
    return sat * beta[:, None] * d["ybar"]
