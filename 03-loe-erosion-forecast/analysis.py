# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Loss of Exclusivity: How Fast Do Brands Erode, and What Happens to Eliquis in 2028?
#
# **Business question:** Eliquis (apixaban) is expected to face US generic
# competition from April 2028. How quickly will brand volume fall, and how much
# reliable is an analog-based forecast?
#
# **Data:** Medicaid State Drug Utilization Data (SDUD), 2017 onward. For every
# quarter it reports prescriptions, units and dollars Medicaid reimbursed for each
# drug product, so brand and generic versions of the same molecule can be told apart.
#
# **Method:**
# 1. Build quarterly brand vs. generic prescriptions for brands that already lost exclusivity (the **analogs**).
# 2. Align them on the quarter generics arrived and measure the **erosion curve**: the brand's share of its molecule's prescriptions in each quarter after entry.
# 3. **Back-test**: forecast Entresto's 2025 erosion using only the other analogs, and compare against what actually happened.
# 4. Apply the validated curves to Eliquis.
#
# Terms: **LOE** (loss of exclusivity) is when patents and regulatory exclusivity
# end and generics can launch. **Biosimilars** are the near-copies of biologic
# drugs, the biologic equivalent of generics.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = Path.cwd() if Path.cwd().name.startswith("03-") else Path("03-loe-erosion-forecast")
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

style.apply()
RAW, FIG, TAB = HERE / "data" / "raw", HERE / "outputs" / "figures", HERE / "outputs" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
SRC = "Medicaid State Drug Utilization Data (CMS), national totals"

import json
MOLECULES = json.loads((RAW / "sources.json").read_text())["molecules"]

frames = [pd.read_csv(f, dtype=str) for f in sorted(RAW.glob("sdud_*.csv.gz"))]
sd = pd.concat(frames, ignore_index=True)
for c in ["units_reimbursed", "number_of_prescriptions", "total_amount_reimbursed", "medicaid_amount_reimbursed"]:
    sd[c] = pd.to_numeric(sd[c], errors="coerce")
sd["year"] = sd["year"].astype(int)
sd["quarter"] = sd["quarter"].astype(int)
sd["product_name"] = sd["product_name"].str.strip().str.upper()
print(sd.shape, sorted(sd["state"].unique())[:8], "...")
sd.head(3)

# %% [markdown]
# SDUD publishes each state and also a national total row (state code `XX`).
# The national rows are used here; they carry less privacy suppression than
# summing states.

# %%
nat = sd[sd["state"] == "XX"].copy() if (sd["state"] == "XX").any() else sd.copy()


def tag(name: str):
    for mol, names in MOLECULES.items():
        for i, p in enumerate(names):
            if name.startswith(p):
                return mol, ("brand" if i == 0 else "competitor")
    return None, None


nat[["molecule", "role"]] = nat["product_name"].apply(lambda n: pd.Series(tag(n)))
nat = nat.dropna(subset=["molecule"])
nat["period"] = pd.PeriodIndex.from_fields(year=nat["year"], quarter=nat["quarter"], freq="Q")
q = (nat.groupby(["molecule", "period", "role"])[["number_of_prescriptions", "total_amount_reimbursed"]]
        .sum().unstack("role", fill_value=0))
q.columns = [f"{a}_{b}" for a, b in q.columns]
q = q.rename(columns={"number_of_prescriptions_brand": "brand_rx", "number_of_prescriptions_competitor": "comp_rx",
                      "total_amount_reimbursed_brand": "brand_usd", "total_amount_reimbursed_competitor": "comp_usd"})
q = q.reset_index()
q["brand_share"] = q["brand_rx"] / (q["brand_rx"] + q["comp_rx"])
latest = q["period"].max()
print("Latest quarter in data:", latest)

# %% [markdown]
# ## 1. Find each molecule's entry quarter from the data
#
# Entry quarter = first quarter in which generics or biosimilars account for at
# least 2% of the molecule's prescriptions. Deriving it from the data, rather than
# from press releases, avoids mismatches with how quickly Medicaid claims follow a
# launch. Each one is cross-checked against the public launch date in the README.

# %%
def entry_quarter(d: pd.DataFrame):
    d = d.sort_values("period")
    hit = d[(1 - d["brand_share"]) >= 0.02]
    return hit["period"].iloc[0] if len(hit) else None


entries = {m: entry_quarter(d) for m, d in q.groupby("molecule")}
entries = {m: p for m, p in entries.items() if p is not None}
pd.Series({m: str(p) for m, p in entries.items()}, name="entry_quarter").sort_values()

# %%
q["t"] = [ (p - entries[m]).n if m in entries else np.nan for m, p in zip(q["molecule"], q["period"]) ]
pre = q[q["t"].between(-4, -1)].groupby("molecule")["brand_rx"].mean().rename("pre_rx")
q = q.merge(pre, on="molecule", how="left")
q["brand_index"] = q["brand_rx"] / q["pre_rx"]

BIOLOGICS = {"adalimumab", "ustekinumab"}
# Analogs need at least 4 quarters of history after entry
analogs = [m for m, p in entries.items() if (latest - p).n >= 4 and m not in BIOLOGICS]
print("Small-molecule analogs:", analogs)
print("Biologic analogs:", [m for m in BIOLOGICS if m in entries])

# %% [markdown]
# ## 2. Small-molecule erosion curves

# %%
win = q[q["molecule"].isin(analogs) & q["t"].between(-4, 12)]
curve = win.pivot_table(index="t", columns="molecule", values="brand_share")

fig, ax = plt.subplots(figsize=(10, 5.4))
for i, m in enumerate(curve.columns):
    ax.plot(curve.index, curve[m], color=style.PALETTE[i % len(style.PALETTE)], lw=1.4, alpha=0.85, label=m)
med = curve.loc[curve.index >= 0].median(axis=1)
ax.plot(med.index, med.values, color=style.INK, lw=3, label="Median of analogs")
ax.axvline(0, color=style.MUTED, ls=":")
ax.text(0.1, 0.05, "generic entry", color=style.MUTED, fontsize=9)
ax.set_xlabel("Quarters since generic entry")
ax.set_ylabel("Brand share of molecule prescriptions")
style.pct(ax)
ax.set_ylim(0, 1.05)
ax.set_title("Brand share after generic entry: Medicaid, small molecules")
style.subtitle(ax, "Most brands keep under a fifth of prescriptions within two quarters of generic entry")
ax.legend(fontsize=8, ncol=2, loc="upper right")
style.source(fig, SRC)
style.save(fig, FIG / "01_erosion_curves.png")
curve.loc[[0, 1, 2, 4, 8] if 8 in curve.index else [0, 1, 2, 4]].round(2)

# %% [markdown]
# ### Fit one curve per analog
#
# `share(t) = floor + (1 − floor) · e^(−k·t)` for t ≥ 0
#
# - **k** sets the speed of erosion. Higher k means faster loss.
# - **floor** is the share the brand keeps long-term, from patients and
#   prescribers who stay on brand ("dispense as written", brand-preferring
#   state formularies).

# %%
def erosion(t, k, floor):
    return floor + (1 - floor) * np.exp(-k * t)


fits = {}
for m in analogs:
    d = q[(q["molecule"] == m) & q["t"].between(0, 12)].dropna(subset=["brand_share"])
    s0 = d["brand_share"].iloc[0] if len(d) else 1
    try:
        (k, fl), _ = curve_fit(erosion, d["t"], d["brand_share"] / max(s0, 1e-9) * s0,
                               p0=[1.0, 0.1], bounds=([0.01, 0.0], [10, 0.9]))
        fits[m] = {"k": k, "floor": fl, "share_q1": float(d.loc[d["t"] == 1, "brand_share"].iloc[0]) if (d["t"] == 1).any() else np.nan,
                   "share_q4": float(d.loc[d["t"] == 4, "brand_share"].iloc[0]) if (d["t"] == 4).any() else np.nan}
    except RuntimeError:
        pass
fits = pd.DataFrame(fits).T.sort_values("k")
fits.to_csv(TAB / "analog_fits.csv")
fits.round(3)

# %% [markdown]
# ## 3. Back-test: could the analogs have predicted Entresto?
#
# Entresto (sacubitril/valsartan) generics launched in July 2025. Fit the
# erosion model on every **other** analog, forecast Entresto's brand share from
# its entry quarter, and compare with the actual Medicaid data.

# %%
TEST = "sacubitril/valsartan"
train = fits.drop(index=TEST, errors="ignore")
k_q = train["k"].quantile([0.25, 0.5, 0.75])
f_q = train["floor"].quantile([0.25, 0.5, 0.75])
scen = {"slow erosion (P25 k, P75 floor)": (k_q[0.25], f_q[0.75]),
        "median": (k_q[0.5], f_q[0.5]),
        "fast erosion (P75 k, P25 floor)": (k_q[0.75], f_q[0.25])}

act = q[(q["molecule"] == TEST) & (q["t"] >= 0)].sort_values("t")
bt = act[["t", "period", "brand_share"]].copy()
for name, (k, fl) in scen.items():
    bt[name] = erosion(bt["t"], k, fl)
mae = (bt["median"] - bt["brand_share"]).abs().mean() if len(bt) else np.nan

fig, ax = plt.subplots(figsize=(8.5, 4.6))
tt = np.linspace(0, max(8, bt["t"].max() if len(bt) else 8), 100)
ax.fill_between(tt, erosion(tt, *scen["fast erosion (P75 k, P25 floor)"]), erosion(tt, *scen["slow erosion (P25 k, P75 floor)"]),
                color=style.ACCENT, alpha=0.15, label="Analog range (P25–P75)")
ax.plot(tt, erosion(tt, *scen["median"]), color=style.ACCENT, lw=2, label="Analog median forecast")
ax.plot(bt["t"], bt["brand_share"], "o-", color=style.ACCENT_2, lw=2, label="Entresto actual")
ax.set_xlabel("Quarters since generic entry")
ax.set_ylabel("Brand share of prescriptions")
style.pct(ax)
ax.set_ylim(0, 1.05)
ax.set_title("Back-test: Entresto forecast vs. actual (Medicaid)")
style.subtitle(ax, f"Mean absolute error of the median forecast: {mae:.1%} share points over {len(bt)} quarters")
ax.legend(fontsize=9)
style.source(fig, SRC)
style.save(fig, FIG / "02_backtest_entresto.png")
bt.round(3)

# %% [markdown]
# ## 4. Biologics erode far more slowly

# %%
bio = [m for m in BIOLOGICS if m in entries]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for i, m in enumerate(bio):
    d = q[(q["molecule"] == m) & q["t"].between(-2, 12)]
    ax.plot(d["t"], d["brand_share"], "o-", color=[style.ACCENT_2, "#7C3AED"][i % 2], lw=2, ms=4,
            label=f"{m} ({'Humira' if m == 'adalimumab' else 'Stelara'})")
ax.plot(med.index, med.values, color=style.INK, lw=2, ls="--", label="Small-molecule median")
ax.axvline(0, color=style.MUTED, ls=":")
ax.set_xlabel("Quarters since first biosimilar / generic prescriptions")
ax.set_ylabel("Brand share of molecule prescriptions")
style.pct(ax)
ax.set_ylim(0, 1.05)
ax.set_title("Biosimilar entry vs. generic entry")
style.subtitle(ax, "Biologic brands hold share for years, not quarters")
ax.legend(fontsize=9)
style.source(fig, SRC)
style.save(fig, FIG / "03_biologics_vs_small_molecules.png")
q[q["molecule"].isin(bio) & q["t"].isin([0, 2, 4, 6, 8, 10, 12])].pivot_table(index="t", columns="molecule", values="brand_share").round(2)

# %% [markdown]
# ## 5. Forecast: Eliquis in Medicaid, 2026–2030
#
# Assumptions:
# - Generic apixaban enters in Q2 2028 (April 1, 2028, per court rulings and settlements).
# - Until then, the apixaban molecule keeps its recent growth trend (linear fit on the last 8 quarters).
# - From entry, the brand's share follows the analog curves (all analogs, including Entresto).
# - Brand reimbursement per prescription stays at its latest four-quarter average. This is
#   a gross figure before Medicaid rebates, which are large for brands.

# %%
ELQ = "apixaban"
e = q[q["molecule"] == ELQ].sort_values("period").copy()
e["mol_rx"] = e["brand_rx"] + e["comp_rx"]
hist = e.tail(8)
slope, intercept = np.polyfit(range(len(hist)), hist["mol_rx"], 1)
usd_per_rx = (e.tail(4)["brand_usd"].sum() / e.tail(4)["brand_rx"].sum())

entry = pd.Period("2028Q2", freq="Q")
fut = pd.period_range(e["period"].max() + 1, pd.Period("2030Q4", freq="Q"), freq="Q")
fc = pd.DataFrame({"period": fut})
fc["mol_rx"] = [intercept + slope * (len(hist) - 1 + i + 1) for i in range(len(fut))]
k_all, f_all = fits["k"].quantile([0.25, 0.5, 0.75]), fits["floor"].quantile([0.25, 0.5, 0.75])
for name, (k, fl) in {"slow": (k_all[0.25], f_all[0.75]), "median": (k_all[0.5], f_all[0.5]),
                      "fast": (k_all[0.75], f_all[0.25])}.items():
    t = np.array([(p - entry).n for p in fc["period"]])
    share = np.where(t < 0, 1.0, erosion(np.clip(t, 0, None), k, fl))
    fc[f"brand_rx_{name}"] = fc["mol_rx"] * share
    fc[f"brand_usd_{name}"] = fc[f"brand_rx_{name}"] * usd_per_rx

annual_hist = e.groupby(e["period"].dt.year)["brand_usd"].sum()
annual_fc = fc.groupby(fc["period"].dt.year)[["brand_usd_slow", "brand_usd_median", "brand_usd_fast"]].sum()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(e["period"].dt.to_timestamp(), e["brand_usd"], color=style.INK, lw=2, label="Actual brand reimbursement")
x = fc["period"].dt.to_timestamp()
ax.fill_between(x, fc["brand_usd_fast"], fc["brand_usd_slow"], color=style.ACCENT, alpha=0.18, label="Analog range")
ax.plot(x, fc["brand_usd_median"], color=style.ACCENT, lw=2, label="Median forecast")
ax.axvline(entry.to_timestamp(), color=style.ACCENT_2, ls="--")
ax.text(entry.to_timestamp(), ax.get_ylim()[1] * 0.95, "  generic entry (Q2 2028)", color=style.ACCENT_2, fontsize=9)
style.dollars(ax, unit="M")
ax.set_title("Eliquis Medicaid gross reimbursement per quarter: actual and forecast")
style.subtitle(ax, "Brand revenue in Medicaid is expected to fall by roughly three-quarters within a year of entry")
ax.legend(loc="upper left", fontsize=9)
style.source(fig, SRC + ". Gross, before Medicaid rebates.")
style.save(fig, FIG / "04_eliquis_forecast.png")
out = pd.concat([annual_hist.rename("actual"), annual_fc], axis=1) / 1e6
out.round(0)

# %% [markdown]
# ## 6. Summary numbers used in the README

# %%
first4 = q[q["molecule"].isin(analogs) & (q["t"] == 4)]["brand_share"]
first1 = q[q["molecule"].isin(analogs) & (q["t"] == 1)]["brand_share"]
summary = {
    "latest_quarter": str(latest),
    "analogs": analogs,
    "median_brand_share_q1": float(first1.median()),
    "median_brand_share_q4": float(first4.median()),
    "range_brand_share_q4": [float(first4.min()), float(first4.max())],
    "entresto_backtest_mae": float(mae),
    "entresto_quarters_tested": int(len(bt)),
    "eliquis_usd_per_rx": float(usd_per_rx),
    "eliquis_2027_median_usd": float(annual_fc.loc[2027, "brand_usd_median"]) if 2027 in annual_fc.index else None,
    "eliquis_2029_median_usd": float(annual_fc.loc[2029, "brand_usd_median"]),
    "eliquis_2029_range_usd": [float(annual_fc.loc[2029, "brand_usd_fast"]), float(annual_fc.loc[2029, "brand_usd_slow"])],
    "entries": {m: str(p) for m, p in entries.items()},
}
(TAB / "summary.json").write_text(json.dumps(summary, indent=1))
fc.round(0).to_csv(TAB / "eliquis_forecast_quarterly.csv", index=False)
summary
