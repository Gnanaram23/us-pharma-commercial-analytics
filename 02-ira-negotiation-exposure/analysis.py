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
# # Medicare Price Negotiation: Which Manufacturers Are Most Exposed, and Who Is Next?
#
# **Business question:** The Inflation Reduction Act (IRA) lets Medicare set a
# "maximum fair price" (MFP) for high-spend drugs. For a manufacturer's market
# access and finance teams: how much Medicare revenue is already under negotiated
# prices, who carries the most, and which drugs are likely to be picked in the
# next selection round (February 2027)?
#
# **Data:**
# - CMS Medicare Part D and Part B Spending by Drug (annual, by manufacturer)
# - CMS lists of selected drugs and their negotiated prices (`negotiation_lists.py`)
# - openFDA Drugs@FDA approval history (application type, first approval, generic approvals)
#
# **Terms:**
# - **Part D** covers drugs patients collect at a pharmacy. **Part B** covers drugs
#   given in a clinic or hospital outpatient setting (infusions, injections).
# - **IPAY** (initial price applicability year) is the year a negotiated price takes effect.
# - **Gross spend** is what Medicare plans and patients paid before manufacturer
#   rebates. It overstates manufacturer revenue, which is why section 3 calibrates
#   to CMS's own net savings estimates.

# %%
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path.cwd() if Path.cwd().name.startswith("02-") else Path("02-ira-negotiation-exposure")
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import style  # noqa: E402
from negotiation_lists import SELECTED, CMS_CYCLE_SAVINGS  # noqa: E402

style.apply()
SHARED = HERE.parent / "data" / "shared"
RAW, FIG, TAB = HERE / "data" / "raw", HERE / "outputs" / "figures", HERE / "outputs" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
SRC = "CMS Medicare Part D and Part B Spending by Drug; CMS negotiation announcements"


def load_spend(path: Path, part: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    for c in df.columns:
        if re.match(r"(Tot_|Avg_)", c):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["part"] = part
    df["mftr"] = df["Mftr_Name"].str.strip()
    df["is_overall"] = df["mftr"].str.lower() == "overall"
    return df


d = load_spend(SHARED / "partd_spending_by_drug.csv.gz", "D")
b = load_spend(SHARED / "partb_spending_by_drug.csv.gz", "B")
YEARS = sorted(int(c[-4:]) for c in d.columns if re.fullmatch(r"Tot_Spndng_\d{4}", c))
Y = YEARS[-1]
print(f"Part D rows {len(d):,}, Part B rows {len(b):,}; spend years {YEARS}")

sel = pd.DataFrame(SELECTED, columns=["cycle", "drug", "match", "list_price", "mfp", "condition"])
sel["discount"] = 1 - sel["mfp"] / sel["list_price"]


def selected_drug(brand: str):
    bl = str(brand).lower()
    for r in sel.itertuples():
        if any(bl.startswith(m) for m in r.match):
            return r.drug
    return None


for df in (d, b):
    df["selected"] = df["Brnd_Name"].map(selected_drug)

# %% [markdown]
# ## 1. How much Medicare spend is already in negotiation?

# %%
spend_col = f"Tot_Spndng_{Y}"
tot = {p: df.loc[df["is_overall"], spend_col].sum() for p, df in (("D", d), ("B", b))}
by_drug = pd.concat([df[df["is_overall"] & df["selected"].notna()].groupby("selected")[spend_col].sum().rename(p)
                     for p, df in (("D", d), ("B", b))], axis=1).fillna(0)
by_drug["total"] = by_drug.sum(axis=1)
by_drug = by_drug.join(sel.set_index("drug")[["cycle", "discount", "condition"]])
by_cycle = by_drug.groupby("cycle")[["D", "B", "total"]].sum()
by_cycle["share_of_partD"] = by_cycle["D"] / tot["D"]
print(f"Total {Y} gross spend: Part D ${tot['D'] / 1e9:,.1f}B, Part B ${tot['B'] / 1e9:,.1f}B")
(by_cycle / [1e9, 1e9, 1e9, 1]).round(3)

# %%
fig, ax = plt.subplots(figsize=(10, 6.4))
plot = by_drug.sort_values("total")
colors = plot["cycle"].map({2026: style.ACCENT, 2027: "#2563EB", 2028: style.ACCENT_2})
ax.barh(plot.index, plot["total"], color=colors)
for y, v in enumerate(plot["total"]):
    ax.text(v, y, f" ${v / 1e9:,.1f}B", va="center", fontsize=8)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (style.ACCENT, "#2563EB", style.ACCENT_2)]
ax.legend(handles, ["Price from 2026", "Price from 2027", "Price from 2028 (being negotiated)"], loc="lower right", fontsize=9)
style.dollars(ax, "x")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", labelsize=8.5)
ax.set_title(f"Medicare gross spend on the 40 negotiated drugs, {Y}")
style.subtitle(ax, f"${by_cycle['total'].sum() / 1e9:,.0f}B across three cycles; Part B and Part D combined")
style.source(fig, SRC)
style.save(fig, FIG / "01_selected_drug_spend.png")

# %% [markdown]
# ## 2. Headline price cuts versus net impact
#
# CMS publishes each MFP next to the drug's **list price**. The list-price
# discounts look dramatic (38% to 85%), but manufacturers already paid large
# rebates on many of these drugs, so the true revenue loss is much smaller. CMS's
# own estimate of **net** savings is the better guide:
#
# | Cycle | Gross Part D cost (base year) | CMS net savings | Net savings ÷ gross |
# |---|---|---|---|
# | IPAY 2026 (10 drugs) | $56.2B (2023) | ~$6B (22% of net) | ~11% |
# | IPAY 2027 (15 drugs) | $42.5B (2024) | ~$12B (44% of net) | ~28% |
#
# Each drug's net impact is estimated by sharing out CMS's cycle total in
# proportion to *gross spend × list-price discount*. A drug with more spend and a
# deeper cut takes more of the loss. Lacking drug-level rebate data, this is the
# most defensible public method; it assumes rebate depth is similar within a cycle.

# %%
rows = []
for cyc, (savings, pct, base_year, gross_base) in CMS_CYCLE_SAVINGS.items():
    sub = by_drug[by_drug["cycle"] == cyc].copy()
    base = d[d["is_overall"] & d["selected"].isin(sub.index)].groupby("selected")[f"Tot_Spndng_{base_year}"].sum()
    weight = base * sub["discount"]
    alloc = savings * weight / weight.sum()            # CMS savings shared by spend x discount
    ratio = alloc / base                               # net impact as share of gross, per drug
    sub["partD_gross_base"] = base
    sub["net_impact_ratio"] = ratio
    sub["est_annual_net_impact"] = ratio * sub["D"]    # applied to latest-year Part D spend
    rows.append(sub)
imp = pd.concat(rows)
imp["list_reduction_at_gross"] = imp["D"] * imp["discount"]
imp.sort_values("est_annual_net_impact", ascending=False).round(3).head(25)

# %% [markdown]
# ## 3. Manufacturer exposure
#
# Spend is assigned to manufacturers using CMS's manufacturer-level rows. Cycle 3
# (IPAY 2028) prices are not yet public, so its exposure is a range: the 25th to
# 75th percentile of the net-impact ratios seen in cycles 1 and 2.

# %%
def mftr_spend(df):
    x = df[~df["is_overall"] & df["selected"].notna()]
    return x.groupby(["selected", "mftr"])[spend_col].sum()

ms = pd.concat([mftr_spend(d), mftr_spend(b)]).groupby(level=[0, 1]).sum().rename("spend").reset_index()
ms = ms.merge(sel[["drug", "cycle"]], left_on="selected", right_on="drug")
ratio_by_drug = imp["net_impact_ratio"]
lo_r, mid_r, hi_r = ratio_by_drug.quantile([0.25, 0.5, 0.75])
ms["ratio_mid"] = ms["selected"].map(ratio_by_drug).fillna(mid_r)
ms["ratio_lo"] = ms["selected"].map(ratio_by_drug).fillna(lo_r)
ms["ratio_hi"] = ms["selected"].map(ratio_by_drug).fillna(hi_r)
for s in ("mid", "lo", "hi"):
    ms[f"impact_{s}"] = ms["spend"] * ms[f"ratio_{s}"]

# Consolidate CMS labeler names to parent companies for readability
PARENT = [("bristol", "Bristol Myers Squibb"), ("squibb", "Bristol Myers Squibb"), ("pfizer", "Pfizer"),
          ("novo", "Novo Nordisk"), ("lilly", "Eli Lilly"), ("boehringer", "Boehringer Ingelheim"),
          ("janssen", "Johnson & Johnson"), ("johnson", "Johnson & Johnson"), ("astrazeneca", "AstraZeneca"),
          ("merck", "Merck"), ("novartis", "Novartis"), ("sandoz", "Novartis"), ("amgen", "Amgen"), ("abbvie", "AbbVie"),
          ("allergan", "AbbVie"), ("pharmacyclics", "AbbVie"), ("glaxo", "GSK"), ("gsk", "GSK"), ("astellas", "Astellas"),
          ("celgene", "Bristol Myers Squibb"), ("teva", "Teva"), ("salix", "Bausch Health"), ("bausch", "Bausch Health"),
          ("gilead", "Gilead"), ("takeda", "Takeda"), ("genentech", "Roche"), ("eisai", "Eisai"), ("otsuka", "Otsuka"),
          ("ucb", "UCB"), ("immunex", "Amgen"), ("ironwood", "AbbVie"), ("sanofi", "Sanofi")]


def parent(name: str) -> str:
    n = name.lower()
    return next((p for k, p in PARENT if k in n), name)


ms["parent"] = ms["mftr"].map(parent)
mx = ms.groupby("parent").agg(spend=("spend", "sum"), impact_mid=("impact_mid", "sum"),
                              impact_lo=("impact_lo", "sum"), impact_hi=("impact_hi", "sum"),
                              drugs=("selected", lambda s: ", ".join(sorted(set(s)))))
mx = mx.sort_values("impact_mid", ascending=False)
mx.to_csv(TAB / "manufacturer_exposure.csv")
(mx[["spend", "impact_lo", "impact_mid", "impact_hi"]] / 1e9).round(2).join(mx["drugs"]).head(12)

# %%
top = mx.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.barh(top.index, top["impact_mid"], color=style.ACCENT, label="Estimated annual net revenue impact")
ax.errorbar(top["impact_mid"], range(len(top)),
            xerr=[top["impact_mid"] - top["impact_lo"], top["impact_hi"] - top["impact_mid"]],
            fmt="none", ecolor=style.INK, elinewidth=1, capsize=3)
for y, (v, s) in enumerate(zip(top["impact_mid"], top["spend"])):
    ax.text(v, y + 0.28, f" ${v / 1e9:,.1f}B net on ${s / 1e9:,.0f}B gross", va="center", fontsize=8, color=style.MUTED)
style.dollars(ax, "x")
ax.grid(axis="y", visible=False)
ax.set_title("Estimated annual Medicare net revenue impact by manufacturer")
style.subtitle(ax, f"All three cycles at {Y} volumes; bars show range where cycle-3 prices are not yet known")
style.source(fig, SRC + ". Net impact calibrated to CMS net savings estimates.")
style.save(fig, FIG / "02_manufacturer_exposure.png")

# %% [markdown]
# ## 4. Who is next? Screening for the February 2027 selection (IPAY 2029)
#
# Eligibility rules applied, from the statute and CMS guidance:
# 1. **Time on market:** at least 7 years since first approval for a small molecule
#    (NDA) or 11 years for a biologic (BLA), counted to the selection date (Feb 1, 2027).
# 2. **No generic or biosimilar on the market.** Approved, marketed generics
#    (ANDAs in Drugs@FDA) are excluded. Biosimilar competition is flagged by hand.
# 3. **Not already selected.**
# 4. Ranked by combined Part B + Part D gross spend. CMS picks up to 15 drugs.
#
# Not modelled: the orphan-drug exclusion (widened in July 2025 to cover drugs with
# several orphan indications) and plasma-derived products. These are flagged for
# manual review instead of guessed.
#
# **Validation first.** Before trusting the screen for 2029, run it as if it were
# January 2025 against Part D only, and compare with the 15 drugs CMS actually picked for IPAY 2027.

# %%
fda = pd.read_csv(RAW / "fda_approvals.csv.gz")
fda["first_approval"] = pd.to_datetime(fda["first_approval"].astype(str), format="%Y%m%d", errors="coerce")

overall = pd.concat([d[d["is_overall"]], b[b["is_overall"]]])
cand = (overall.groupby(["Brnd_Name", "Gnrc_Name"])[[f"Tot_Spndng_{y}" for y in YEARS]].sum().reset_index())
cand = cand.merge(fda.drop(columns=["Gnrc_Name"], errors="ignore"), on="Brnd_Name", how="left")
cand["selected"] = cand["Brnd_Name"].map(selected_drug)
cand["sel_cycle"] = cand["selected"].map(sel.set_index("drug")["cycle"])

# Biologics with marketed US biosimilars as of late 2026 (manual flag)
BIOSIMILAR_COMPETITION = {"humira", "stelara", "remicade", "neulasta", "neupogen", "herceptin", "avastin",
                          "rituxan", "lantus", "epogen", "procrit", "prolia", "xgeva", "eylea", "soliris", "tysabri",
                          "actemra", "simponi", "lucentis", "nplate"}


def screen(df, selection_date, spend_year, exclude_cycles, parts_col_note=""):
    x = df.copy()
    x["spend"] = x[f"Tot_Spndng_{spend_year}"]
    years_on_market = (pd.Timestamp(selection_date) - x["first_approval"]).dt.days / 365.25
    need = np.where(x["app_type"] == "BLA", 11, 7)
    x["years_on_market"] = years_on_market.round(1)
    x["eligible_time"] = years_on_market >= need
    x["has_generic"] = x["anda_marketed_count"].fillna(0) > 0
    x["biosimilar"] = x["Brnd_Name"].str.lower().str.split().str[0].isin(BIOSIMILAR_COMPETITION)
    x["already_selected"] = x["sel_cycle"].isin(exclude_cycles)
    ok = x["fda_found"].fillna(False).astype(bool) & x["eligible_time"] & ~x["has_generic"] & ~x["biosimilar"] & ~x["already_selected"]
    return x[ok].sort_values("spend", ascending=False)


# Back-test 1: IPAY 2027 selection (Jan 2025), Part D only, using CY2024 spend
partd_only = cand[cand["Brnd_Name"].isin(d.loc[d["is_overall"], "Brnd_Name"])].copy()
partd_only[[f"Tot_Spndng_{y}" for y in YEARS]] = (
    d[d["is_overall"]].groupby("Brnd_Name")[[f"Tot_Spndng_{y}" for y in YEARS]].sum()
    .reindex(partd_only["Brnd_Name"]).values)
bt = screen(partd_only, "2025-02-01", 2024, exclude_cycles=[2026])
bt_groups = bt.assign(group=bt["selected"].fillna(bt["Gnrc_Name"])).drop_duplicates("group").head(15)
actual_2027 = set(sel.loc[sel["cycle"] == 2027, "drug"])
hits = set(bt_groups["selected"].dropna()) & actual_2027
print(f"Back-test IPAY 2027: screen's top 15 contains {len(hits)} of the 15 drugs CMS selected")
bt_groups[["Brnd_Name", "Gnrc_Name", "spend", "app_type", "years_on_market", "selected"]].assign(
    spend=lambda x: (x["spend"] / 1e9).round(2))

# %% [markdown]
# The misses are informative: they show what the public data cannot see
# (CMS groups all brands of one active moiety, uses a November–October spend
# window, and applies exclusions such as orphan status). The screen is a
# shortlist for analysts to review, not an oracle.

# %%
bt2 = screen(cand, "2026-02-01", 2024, exclude_cycles=[2026, 2027])
bt2_groups = bt2.assign(group=bt2["selected"].fillna(bt2["Gnrc_Name"])).drop_duplicates("group").head(15)
actual_2028 = set(sel.loc[sel["cycle"] == 2028, "drug"])
hits2 = set(bt2_groups["selected"].dropna()) & actual_2028
print(f"Back-test IPAY 2028 (Part B + D): screen's top 15 contains {len(hits2)} of the 15 drugs CMS selected")

# %% [markdown]
# ### Forecast: likely candidates for the February 2027 selection (IPAY 2029)

# %%
nxt = screen(cand, "2027-02-01", Y, exclude_cycles=[2026, 2027, 2028])
nxt = nxt.assign(group=nxt["Gnrc_Name"]).drop_duplicates("group").head(25)
nxt["rank"] = range(1, len(nxt) + 1)
watch = nxt[["rank", "Brnd_Name", "Gnrc_Name", "spend", "app_type", "first_approval", "years_on_market", "sponsor"]].copy()
watch["spend_$B"] = (watch.pop("spend") / 1e9).round(2)
watch.to_csv(TAB / "ipay2029_watchlist.csv", index=False)
watch

# %%
w = watch.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(w["Brnd_Name"], w["spend_$B"] * 1e9, color=[style.ACCENT_2 if t == "BLA" else style.ACCENT for t in w["app_type"]])
for y, (v, yrs) in enumerate(zip(w["spend_$B"], w["years_on_market"])):
    ax.text(v * 1e9, y, f" ${v:,.1f}B · {yrs:.0f} yrs on market", va="center", fontsize=8.5)
style.dollars(ax, "x")
ax.grid(axis="y", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (style.ACCENT, style.ACCENT_2)]
ax.legend(handles, ["Small molecule (7-yr rule)", "Biologic (11-yr rule)"], loc="lower right", fontsize=9)
ax.set_title("Screened candidates for the February 2027 selection")
style.subtitle(ax, f"Top 15 eligible drugs by {Y} Medicare Part B + D gross spend, before orphan review")
style.source(fig, SRC + "; openFDA Drugs@FDA for approval dates and generics")
style.save(fig, FIG / "03_ipay2029_candidates.png")

# %% [markdown]
# ## 5. Summary numbers used in the README

# %%
import json
summary = {
    "spend_year": Y,
    "partD_total": float(tot["D"]), "partB_total": float(tot["B"]),
    "selected_spend_by_cycle": {int(k): float(v) for k, v in by_cycle["total"].items()},
    "selected_partD_share_by_cycle": {int(k): float(v) for k, v in by_cycle["share_of_partD"].items()},
    "net_impact_total_mid": float(mx["impact_mid"].sum()),
    "net_impact_total_range": [float(mx["impact_lo"].sum()), float(mx["impact_hi"].sum())],
    "top_manufacturers": {k: float(v) for k, v in mx["impact_mid"].head(5).items()},
    "backtest_2027_hits": len(hits), "backtest_2027_hit_names": sorted(hits),
    "backtest_2028_hits": len(hits2), "backtest_2028_hit_names": sorted(hits2),
    "watchlist_top10": watch["Brnd_Name"].head(10).tolist(),
}
(TAB / "summary.json").write_text(json.dumps(summary, indent=1))
summary
