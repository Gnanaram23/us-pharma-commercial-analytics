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
# # GLP-1 Market Sizing and Prescriber Targeting in Medicare Part D
#
# **Business question:** A GLP-1 brand team has a fixed field force. Where is the
# remaining growth in Medicare, and which prescribers should reps call on first?
#
# **Data:** CMS Medicare Part D Prescribers, by Provider and Drug (2022–2024) and
# by Geography and Drug (2018–2024). Every row is a real prescriber or state, drug
# and year, as published by CMS.
#
# **Terms used:**
# - **GLP-1** (glucagon-like peptide-1) agonists lower blood sugar and body weight:
#   Ozempic, Rybelsus and Wegovy (semaglutide), Mounjaro and Zepbound (tirzepatide,
#   which also acts on the GIP receptor), Trulicity, Victoza and others.
# - **SGLT2** inhibitors (Jardiance, Farxiga) and **DPP-4** inhibitors (Januvia,
#   Tradjenta) are the other modern, branded type 2 diabetes classes. A prescriber
#   writing any of the three is treating type 2 diabetes with modern therapy, which
#   makes them a GLP-1 prospect.
# - **Claim**: one prescription fill paid by a Part D plan.
# - **Gross drug cost**: what plans and patients paid at the pharmacy, *before*
#   manufacturer rebates. It overstates manufacturer revenue but is the best public
#   measure of market size.
# - CMS hides any prescriber–drug row with fewer than 11 claims to protect patient
#   privacy, so very low-volume prescribers are missing.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path.cwd() if Path.cwd().name.startswith("01-") else Path("01-glp1-market-and-targeting")
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

style.apply()
RAW, FIG, TAB = HERE / "data" / "raw", HERE / "outputs" / "figures", HERE / "outputs" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
SRC = "CMS Medicare Part D Prescribers by Provider and Drug / by Geography and Drug"

CLASS_TERMS = {
    "GLP-1": ["semaglutide", "tirzepatide", "dulaglutide", "liraglutide", "exenatide", "lixisenatide"],
    "SGLT2": ["empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin", "bexagliflozin"],
    "DPP-4": ["sitagliptin", "linagliptin", "saxagliptin", "alogliptin"],
}


def drug_class(generic: str) -> str:
    g = str(generic).lower()
    for cls, terms in CLASS_TERMS.items():   # GLP-1 first: combos with insulin count as GLP-1
        if any(t in g for t in terms):
            return cls
    return "Other"


def brand_group(brand: str, generic: str) -> str:
    b, g = str(brand).lower(), str(generic).lower()
    for key, label in [("ozempic", "Ozempic"), ("wegovy", "Wegovy"), ("rybelsus", "Rybelsus"),
                       ("mounjaro", "Mounjaro"), ("zepbound", "Zepbound"), ("trulicity", "Trulicity"),
                       ("victoza", "Victoza"), ("saxenda", "Saxenda")]:
        if key in b:
            return label
    if "liraglutide" in g and ("insulin" not in g):
        return "Liraglutide (generic)"
    return "Other GLP-1"


NUM = ["Tot_Clms", "Tot_30day_Fills", "Tot_Day_Suply", "Tot_Drug_Cst", "Tot_Benes", "Tot_Prscrbrs"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    for c in NUM:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["cls"] = df["Gnrc_Name"].map(drug_class)
    df["brand"] = np.where(df["cls"] == "GLP-1",
                           [brand_group(b, g) for b, g in zip(df["Brnd_Name"], df["Gnrc_Name"])], df["Brnd_Name"])
    return df


# %% [markdown]
# ## 1. Market size: Medicare GLP-1 spend has grown several-fold since 2018

# %%
geo = pd.concat([load(f).assign(year=int(f.name[9:13])) for f in sorted(RAW.glob("geo_drug_*.csv.gz"))])
nat = geo[geo["Prscrbr_Geo_Lvl"] == "National"]
trend = nat[nat["cls"] == "GLP-1"].pivot_table(index="year", columns="brand", values="Tot_Drug_Cst", aggfunc="sum").fillna(0)
trend = trend[trend.sum().sort_values(ascending=False).index]
cls_trend = nat.pivot_table(index="year", columns="cls", values=["Tot_Drug_Cst", "Tot_Clms"], aggfunc="sum")

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.stackplot(trend.index, trend.T.values, labels=trend.columns, colors=style.PALETTE[:len(trend.columns)], alpha=0.92)
ax.set_title("Medicare Part D gross spend on GLP-1 agonists, by brand")
first, last = trend.index.min(), trend.index.max()
growth = trend.sum(axis=1)
style.subtitle(ax, f"${growth[first] / 1e9:,.1f}B in {first} to ${growth[last] / 1e9:,.1f}B in {last}"
               f" ({(growth[last] / growth[first]) ** (1 / (last - first)) - 1:.0%} a year)")
style.dollars(ax)
ax.set_xticks(trend.index)
ax.legend(loc="upper left", fontsize=8.5, ncol=2)
style.source(fig, SRC + ". Gross cost before rebates.")
style.save(fig, FIG / "01_glp1_spend_by_brand.png")
(trend / 1e6).round(0)

# %%
share = trend.div(trend.sum(axis=1), axis=0)
share.round(3).T

# %% [markdown]
# Context against the other modern diabetes classes (claims, millions):

# %%
(cls_trend["Tot_Clms"] / 1e6).round(2)

# %% [markdown]
# ## 2. GLP-1 penetration varies sharply by state
#
# **GLP-1 penetration** = GLP-1 claims ÷ all GLP-1 + SGLT2 + DPP-4 claims. It shows
# how much of modern type 2 diabetes treatment already goes to GLP-1s, without
# needing state population data.

# %%
st = geo[(geo["Prscrbr_Geo_Lvl"] == "State") & (geo["year"] == geo["year"].max())]
pen = st.pivot_table(index="Prscrbr_Geo_Desc", columns="cls", values="Tot_Clms", aggfunc="sum").fillna(0)
pen["glp1_share"] = pen["GLP-1"] / pen[["GLP-1", "SGLT2", "DPP-4"]].sum(axis=1)
pen = pen.sort_values("glp1_share")
pen.to_csv(TAB / "state_glp1_penetration.csv")

STATE_ABBR = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
# Tile-grid map: each state is one square, placed roughly where it sits on a map
TILES = {"AK": (0, 0), "ME": (0, 11), "VT": (1, 10), "NH": (1, 11), "WA": (2, 1), "ID": (2, 2), "MT": (2, 3), "ND": (2, 4),
    "MN": (2, 5), "IL": (2, 6), "WI": (2, 7), "MI": (2, 8), "NY": (2, 9), "RI": (2, 10), "MA": (2, 11), "OR": (3, 1),
    "NV": (3, 2), "WY": (3, 3), "SD": (3, 4), "IA": (3, 5), "IN": (3, 6), "OH": (3, 7), "PA": (3, 8), "NJ": (3, 9),
    "CT": (3, 10), "CA": (4, 1), "UT": (4, 2), "CO": (4, 3), "NE": (4, 4), "MO": (4, 5), "KY": (4, 6), "WV": (4, 7),
    "VA": (4, 8), "MD": (4, 9), "DE": (4, 10), "AZ": (5, 2), "NM": (5, 3), "KS": (5, 4), "AR": (5, 5), "TN": (5, 6),
    "NC": (5, 7), "SC": (5, 8), "DC": (5, 9), "OK": (6, 4), "LA": (6, 5), "MS": (6, 6), "AL": (6, 7), "GA": (6, 8),
    "HI": (7, 0), "TX": (7, 4), "FL": (7, 8)}

vals = {STATE_ABBR[s]: v for s, v in pen["glp1_share"].items() if s in STATE_ABBR}
lo, hi = min(vals.values()), max(vals.values())
cmap = plt.get_cmap("YlGnBu")
fig, ax = plt.subplots(figsize=(10, 6.2))
for abbr, (r, c) in TILES.items():
    v = vals.get(abbr)
    color = cmap(0.15 + 0.85 * (v - lo) / (hi - lo)) if v is not None else "#F3F4F6"
    ax.add_patch(plt.Rectangle((c, -r), 0.94, 0.94, color=color))
    ax.text(c + 0.47, -r + 0.58, abbr, ha="center", va="center", fontsize=9, fontweight="bold",
            color="white" if v is not None and (v - lo) / (hi - lo) > 0.55 else style.INK)
    if v is not None:
        ax.text(c + 0.47, -r + 0.26, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                color="white" if (v - lo) / (hi - lo) > 0.55 else style.INK)
ax.set_xlim(-0.2, 12.2)
ax.set_ylim(-7.2, 1.1)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"GLP-1 share of modern type 2 diabetes claims by state, {st['year'].iloc[0]}")
style.subtitle(ax, f"From {lo:.0%} to {hi:.0%}: low-share states are where the class still has room to grow")
style.source(fig, SRC + ". Share = GLP-1 / (GLP-1 + SGLT2 + DPP-4) claims.")
style.save(fig, FIG / "02_state_glp1_penetration.png")
pen[["GLP-1", "SGLT2", "DPP-4", "glp1_share"]].round(3).head(10)

# %% [markdown]
# ## 3. Who prescribes: specialty mix and concentration
#
# Endocrinologists are the obvious target, but they write a minority of claims.

# %%
YEAR = 2024
hcp_drug = load(RAW / f"prescriber_drug_{YEAR}.csv.gz")
glp = hcp_drug[hcp_drug["cls"] == "GLP-1"]
spec = glp.groupby("Prscrbr_Type")["Tot_Clms"].sum().sort_values(ascending=False)
spec_share = spec / spec.sum()
top_spec = spec_share.head(8)
other = 1 - top_spec.sum()

fig, ax = plt.subplots(figsize=(9, 4.8))
labels = list(top_spec.index[::-1])
ax.barh(["All other specialties"] + labels, [other] + list(top_spec.values[::-1]),
        color=[style.NEUTRAL] + [style.ACCENT if "Endocrin" in l else "#5EAAA8" for l in labels])
for y, v in enumerate([other] + list(top_spec.values[::-1])):
    ax.text(v, y, f" {v:.0%}", va="center", fontsize=9)
ax.set_title(f"Share of Medicare GLP-1 claims by prescriber specialty, {YEAR}")
style.subtitle(ax, "Primary care and advanced practice providers write most GLP-1 prescriptions")
style.pct(ax, "x")
ax.grid(axis="y", visible=False)
style.source(fig, SRC)
style.save(fig, FIG / "03_specialty_mix.png")
spec_share.head(12).round(3)

# %%
per_hcp = glp.groupby("Prscrbr_NPI")["Tot_Clms"].sum().sort_values(ascending=False)
cum = per_hcp.cumsum() / per_hcp.sum()
n = len(per_hcp)
conc = {f"top {p:.0%} of prescribers": float(cum.iloc[int(n * p) - 1]) for p in (0.05, 0.10, 0.20)}

fig, ax = plt.subplots(figsize=(7.5, 4.6))
x = np.arange(1, n + 1) / n
ax.plot(x, cum.values, color=style.ACCENT, lw=2)
ax.plot([0, 1], [0, 1], color=style.NEUTRAL, ls=":")
for p, v in [(0.1, conc["top 10% of prescribers"]), (0.2, conc["top 20% of prescribers"])]:
    ax.scatter([p], [v], color=style.ACCENT_2, zorder=3)
    ax.annotate(f"top {p:.0%} write {v:.0%}", (p, v), xytext=(p + 0.06, v - 0.08), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=style.MUTED))
ax.set_title("Concentration of GLP-1 prescribing")
style.subtitle(ax, f"{n:,} prescribers with at least one reportable GLP-1 drug row in {YEAR}")
style.pct(ax, "x"); style.pct(ax, "y")
ax.set_xlabel("Prescribers, ranked by GLP-1 claims")
ax.set_ylabel("Cumulative share of claims")
style.source(fig, SRC)
style.save(fig, FIG / "04_prescriber_concentration.png")
conc

# %% [markdown]
# ## 4. Segment prescribers by potential and adoption
#
# The universe is every prescriber with reportable claims for any GLP-1, SGLT2 or
# DPP-4 drug in 2024. Two measures define the segments:
#
# - **Potential**: decile of the prescriber's total modern diabetes claims
#   (GLP-1 + SGLT2 + DPP-4). Decile 10 is the busiest 10% of prescribers.
# - **Adoption**: the prescriber's GLP-1 share of those claims.
#
# | Segment | Rule | What reps should do |
# |---|---|---|
# | **Champions** | Decile 8–10, adoption at or above the universe median | Protect and retain |
# | **Growth targets** | Decile 8–10, adoption below the median | First call priority: big diabetes practice, under-using GLP-1s |
# | **Steady adopters** | Decile 1–7, adoption at or above median | Lower-cost channels (email, virtual) |
# | **Low priority** | Decile 1–7, adoption below median | Non-personal promotion only |
#
# Rule-based segments are used instead of clustering because a field team needs
# to understand, and trust, why a doctor is on their list.

# %%
hcp = (hcp_drug[hcp_drug["cls"].isin(["GLP-1", "SGLT2", "DPP-4"])]
       .pivot_table(index="Prscrbr_NPI", columns="cls", values="Tot_Clms", aggfunc="sum", fill_value=0))
info = hcp_drug.drop_duplicates("Prscrbr_NPI").set_index("Prscrbr_NPI")[["Prscrbr_Type", "Prscrbr_City", "Prscrbr_State_Abrvtn"]]
hcp = hcp.join(info)
hcp["modern_claims"] = hcp[["GLP-1", "SGLT2", "DPP-4"]].sum(axis=1)
hcp["glp1_share"] = hcp["GLP-1"] / hcp["modern_claims"]
hcp["decile"] = pd.qcut(hcp["modern_claims"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
median_share = hcp["glp1_share"].median()
high = hcp["decile"] >= 8
adopt = hcp["glp1_share"] >= median_share
hcp["segment"] = np.select([high & adopt, high & ~adopt, ~high & adopt],
                           ["Champions", "Growth targets", "Steady adopters"], "Low priority")

seg = hcp.groupby("segment").agg(prescribers=("modern_claims", "size"),
                                 modern_claims=("modern_claims", "sum"),
                                 glp1_claims=("GLP-1", "sum"))
seg["share_of_prescribers"] = seg["prescribers"] / seg["prescribers"].sum()
seg["share_of_modern_claims"] = seg["modern_claims"] / seg["modern_claims"].sum()
seg["glp1_share"] = seg["glp1_claims"] / seg["modern_claims"]
seg = seg.loc[["Champions", "Growth targets", "Steady adopters", "Low priority"]]
print(f"Universe: {len(hcp):,} prescribers; median GLP-1 share {median_share:.0%}")
seg.round(3)

# %%
sample = hcp.sample(min(len(hcp), 25000), random_state=7)
seg_colors = {"Champions": style.ACCENT, "Growth targets": style.ACCENT_2,
              "Steady adopters": "#5EAAA8", "Low priority": style.NEUTRAL}
fig, ax = plt.subplots(figsize=(9, 5.6))
for s, c in seg_colors.items():
    d = sample[sample["segment"] == s]
    ax.scatter(d["modern_claims"], d["glp1_share"], s=6, alpha=0.35, color=c, label=s, rasterized=True)
d8 = hcp.loc[hcp["decile"] == 8, "modern_claims"].min()
ax.axvline(d8, color=style.INK, lw=0.8, ls="--")
ax.axhline(median_share, color=style.INK, lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xlabel("Modern type 2 diabetes claims per prescriber, 2024 (log scale)")
ax.set_ylabel("GLP-1 share of those claims")
style.pct(ax)
ax.set_title("Prescriber segmentation: potential vs. GLP-1 adoption")
style.subtitle(ax, "Bottom-right quadrant = high-volume prescribers still favouring other classes")
ax.legend(loc="upper left", markerscale=3, fontsize=9)
style.source(fig, SRC + ". Random sample of 25,000 prescribers plotted.")
style.save(fig, FIG / "05_segmentation.png")

# %% [markdown]
# ## 5. Size the opportunity in the growth-target segment
#
# If each growth target moved to the **median GLP-1 share of champions**, how many
# extra GLP-1 claims would that be? Value is shown at the 2024 average gross cost
# per GLP-1 claim, so it is an upper bound on gross sales, not net revenue.

# %%
champ_share = hcp.loc[hcp["segment"] == "Champions", "glp1_share"].median()
cost_per_claim = glp["Tot_Drug_Cst"].sum() / glp["Tot_Clms"].sum()
tg = hcp[hcp["segment"] == "Growth targets"].copy()
tg["incremental_claims"] = ((champ_share - tg["glp1_share"]).clip(lower=0) * tg["modern_claims"])
tg["incremental_gross_cost"] = tg["incremental_claims"] * cost_per_claim
tg = tg.sort_values("incremental_claims", ascending=False)
tg["cum_share_of_opportunity"] = tg["incremental_claims"].cumsum() / tg["incremental_claims"].sum()
opp = {
    "growth_targets": len(tg),
    "champion_median_share": round(float(champ_share), 3),
    "incremental_claims": int(tg["incremental_claims"].sum()),
    "avg_gross_cost_per_glp1_claim": round(float(cost_per_claim), 0),
    "incremental_gross_cost_usd": int(tg["incremental_gross_cost"].sum()),
    "targets_for_half_of_opportunity": int((tg["cum_share_of_opportunity"] <= 0.5).sum() + 1),
}
opp

# %%
tspec = (tg.groupby("Prscrbr_Type")["incremental_claims"].sum().sort_values(ascending=False) / tg["incremental_claims"].sum())
tspec.head(8).round(3)

# %% [markdown]
# Target list: the 1,000 growth targets with the largest opportunity. Names are
# left out on purpose; the NPI (National Provider Identifier) links back to the
# public CMS record for anyone who needs it.

# %%
cols = ["Prscrbr_Type", "Prscrbr_City", "Prscrbr_State_Abrvtn", "decile", "modern_claims", "GLP-1", "SGLT2", "DPP-4",
        "glp1_share", "incremental_claims", "incremental_gross_cost"]
target_list = tg.head(1000)[cols].round({"glp1_share": 3, "incremental_claims": 0, "incremental_gross_cost": 0})
target_list.to_csv(TAB / "growth_target_list_top1000.csv")
seg.to_csv(TAB / "segment_summary.csv")
target_list.head(15)

# %% [markdown]
# ## 6. Brand view: where Mounjaro is under-indexed
#
# A second lens for one brand team. Among prescribers writing at least 50 GLP-1
# claims, what share goes to Mounjaro (tirzepatide)? High-volume GLP-1 writers who
# rarely use Mounjaro are conversion targets for its team. The same logic could
# be run for any brand.

# %%
gl = glp.pivot_table(index="Prscrbr_NPI", columns="brand", values="Tot_Clms", aggfunc="sum", fill_value=0)
gl["total"] = gl.sum(axis=1)
gl = gl[gl["total"] >= 50]
gl["mounjaro_share"] = gl.get("Mounjaro", 0) / gl["total"]
bins = pd.cut(gl["mounjaro_share"], [-0.001, 0.0, 0.1, 0.25, 0.5, 1.0],
              labels=["0%", "1–10%", "10–25%", "25–50%", ">50%"])
dist = gl.groupby(bins, observed=False)["total"].agg(["size", "sum"])
dist.columns = ["prescribers", "glp1_claims"]
dist["share_of_prescribers"] = dist["prescribers"] / dist["prescribers"].sum()

fig, ax = plt.subplots(figsize=(8, 4.4))
ax.bar(dist.index.astype(str), dist["share_of_prescribers"],
       color=[style.ACCENT_2, style.ACCENT_2, "#5EAAA8", style.ACCENT, style.ACCENT])
for i, v in enumerate(dist["share_of_prescribers"]):
    ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
ax.set_title(f"Mounjaro share of each prescriber's GLP-1 claims, {YEAR}")
style.subtitle(ax, f"{len(gl):,} prescribers with 50+ GLP-1 claims")
style.pct(ax)
ax.set_xlabel("Mounjaro share of the prescriber's GLP-1 claims")
ax.set_ylabel("Share of prescribers")
style.source(fig, SRC)
style.save(fig, FIG / "06_mounjaro_share_distribution.png")
dist.round(3)

# %% [markdown]
# ## 7. Momentum: which prescribers are growing GLP-1 use fastest?

# %%
def glp_by_npi(year):
    d = load(RAW / f"prescriber_drug_{year}.csv.gz")
    return d[d["cls"] == "GLP-1"].groupby("Prscrbr_NPI")["Tot_Clms"].sum()

hist = pd.concat({y: glp_by_npi(y) for y in (2022, 2023, 2024)}, axis=1).fillna(0)
both = hist[(hist[2023] > 0) & (hist[2024] > 0)]
yoy = (hist[2024].sum() / hist[2023].sum() - 1)
new_writers = int(((hist[2023] == 0) & (hist[2024] > 0)).sum())
new_writer_claims = hist.loc[(hist[2023] == 0) & (hist[2024] > 0), 2024].sum() / hist[2024].sum()
growth_src = {
    "glp1_claims_2022": int(hist[2022].sum()), "glp1_claims_2023": int(hist[2023].sum()),
    "glp1_claims_2024": int(hist[2024].sum()), "yoy_growth_2024": round(float(yoy), 3),
    "new_reportable_writers_2024": new_writers, "share_of_2024_claims_from_new_writers": round(float(new_writer_claims), 3),
    "growth_from_existing_writers": round(float((both[2024].sum() - both[2023].sum()) / (hist[2024].sum() - hist[2023].sum())), 3),
}
growth_src

# %% [markdown]
# ## 8. Summary numbers used in the README

# %%
summary = {
    "spend_first_year": int(first), "spend_last_year": int(last),
    "glp1_spend_first": float(growth[first]), "glp1_spend_last": float(growth[last]),
    "glp1_spend_cagr": float((growth[last] / growth[first]) ** (1 / (last - first)) - 1),
    "state_share_min": float(lo), "state_share_max": float(hi),
    "state_lowest": pen.index[0], "state_highest": pen.index[-1],
    "endocrinology_share": float(spec_share[[s for s in spec_share.index if "Endocrin" in s]].sum()),
    "top10pct_prescriber_share": conc["top 10% of prescribers"],
    "universe_prescribers": int(len(hcp)), "median_glp1_share": float(median_share),
    **{f"seg_{k.lower().replace(' ', '_')}": int(v) for k, v in seg["prescribers"].items()},
    **opp, **growth_src,
    "mounjaro_low_share_prescribers": float(dist.loc[["0%", "1–10%"], "share_of_prescribers"].sum()),
}
pd.Series(summary).to_json(TAB / "summary.json", indent=1)
pd.Series(summary)
