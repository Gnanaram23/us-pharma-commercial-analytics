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
# # Obesity Drug Pipeline: Competitive Intelligence
#
# **Question:** Who will be competing in the US obesity market by 2030, with what
# mechanisms, and when do the key readouts land?
#
# **Data:** every interventional obesity or overweight trial registered on
# ClinicalTrials.gov that started in 2015 or later, plus FDA approval records.
#
# **Method:**
# 1. Map free-text trial interventions to named molecules and mechanisms (`molecules.py`).
# 2. Find each molecule's most advanced phase and its Phase 3 primary completion dates.
# 3. Apply historical phase-success rates to estimate how many new entrants to expect each year.
#
# Terms: a **Phase 3** trial is the large confirmatory study a sponsor files on.
# **Primary completion** is the date the last patient gives data for the main
# endpoint, so it is the earliest point a topline readout can come.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path.cwd() if Path.cwd().name.startswith("04-") else Path("04-obesity-pipeline-intelligence")
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import style  # noqa: E402
from molecules import classify, APPROVED_US_OBESITY, APPROVED_US_OTHER  # noqa: E402

style.apply()
RAW, FIG, TAB = HERE / "data" / "raw", HERE / "outputs" / "figures", HERE / "outputs" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
SRC = "ClinicalTrials.gov API v2, trials starting 2015 or later"

trials = pd.read_csv(RAW / "trials.csv.gz")
ivs = pd.read_csv(RAW / "interventions.csv.gz")
print(f"{len(trials):,} interventional trials, {len(ivs):,} intervention records")
trials.head(3)

# %% [markdown]
# ## 1. Map interventions to molecules
#
# Only drug and biologic interventions are kept. Each intervention's name, other
# names and description are searched for known molecule names and code numbers.

# %%
drug_ivs = ivs[ivs["type"].isin(["DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"])].copy()
drug_ivs["text"] = drug_ivs[["name", "other_names"]].fillna("").agg(" ".join, axis=1)

rows = []
for r in drug_ivs.itertuples():
    for mol, sponsor, mech in classify(r.text):
        rows.append({"nct_id": r.nct_id, "molecule": mol, "mol_sponsor": sponsor, "mechanism": mech})
hits = pd.DataFrame(rows).drop_duplicates()
# Molecules found only through their INN stem take the trial's lead sponsor
hits = hits.merge(trials[["nct_id", "sponsor"]], on="nct_id", how="left")
hits["mol_sponsor"] = np.where(hits["mol_sponsor"] == "unlisted", hits["sponsor"], hits["mol_sponsor"])
hits = hits.drop(columns="sponsor").drop_duplicates()

trials["start"] = pd.to_datetime(trials["start_date"], errors="coerce")
trials["pcd"] = pd.to_datetime(trials["primary_completion_date"], errors="coerce")
trials["start_year"] = trials["start"].dt.year
trials["phase"] = (trials["phases"].fillna("")
                   .str.replace("EARLY_PHASE1", "PHASE1")
                   .str.split("|").apply(lambda ps: max([p for p in ps if p] or ["NA"])))

m = hits.merge(trials, on="nct_id")
matched_share = drug_ivs["nct_id"].isin(hits["nct_id"]).groupby(drug_ivs["nct_id"]).any().mean()
print(f"{hits['nct_id'].nunique():,} trials matched to a named molecule "
      f"({matched_share:.0%} of trials that test a drug)")

# %% [markdown]
# Unmatched drug interventions in **industry** trials are listed so nothing important is silently dropped.

# %%
ind = trials.loc[trials["sponsor_class"] == "INDUSTRY", "nct_id"]
unmatched = (drug_ivs[drug_ivs["nct_id"].isin(ind) & ~drug_ivs["nct_id"].isin(hits["nct_id"])]
             .groupby("name").size().sort_values(ascending=False))
unmatched.head(25)

# %% [markdown]
# ## 2. Industry investment has shifted from single GLP-1 agonists to next-generation mechanisms
#
# Industry-sponsored trial starts per year, by mechanism. One trial can test more
# than one mechanism (for example a head-to-head study), so it can count twice.

# %%
ind_m = m[m["sponsor_class"] == "INDUSTRY"]
by_year = (ind_m[(ind_m["start_year"] >= 2015) & (ind_m["start_year"] <= ind_m["start_year"].max())]
           .drop_duplicates(["nct_id", "mechanism"])
           .pivot_table(index="start_year", columns="mechanism", values="nct_id", aggfunc="count", fill_value=0))
order = by_year.sum().sort_values(ascending=False).index
by_year = by_year[order]

fig, ax = plt.subplots(figsize=(10, 5.2))
bottom = np.zeros(len(by_year))
for i, col in enumerate(by_year.columns):
    ax.bar(by_year.index, by_year[col], bottom=bottom, label=col, color=style.PALETTE[i % len(style.PALETTE)], width=0.75)
    bottom += by_year[col].values
ax.set_title("Industry obesity trial starts by mechanism")
style.subtitle(ax, "Multi-agonist, amylin and oral GLP-1 programmes now make up a large share of new trials")
ax.set_ylabel("Trials started")
ax.legend(loc="upper left", fontsize=8.5, ncol=2)
ax.set_xticks(by_year.index)
ax.tick_params(axis="x", rotation=0)
style.source(fig, SRC + f". {int(by_year.index.max())} includes trials with planned start dates later in the year.")
style.save(fig, FIG / "01_trial_starts_by_mechanism.png")
by_year

# %% [markdown]
# ## 3. The pipeline: most advanced phase for every named molecule
#
# Industry-sponsored trials only, so academic studies of approved drugs do not
# inflate the pipeline. A molecule counts as **US-relevant** if at least one of
# its Phase 2 or 3 trials has a US site; the rest are **China-first** or
# ex-US programmes, which matter as licensing targets rather than near-term
# US competitors.

# %%
PHASE_RANK = {"PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4}
live = m[~m["overall_status"].isin(["WITHDRAWN"]) & (m["sponsor_class"] == "INDUSTRY")].copy()
live["phase_rank"] = live["phase"].map(PHASE_RANK).fillna(0)
live["us_site"] = live["countries"].fillna("").str.contains("United States")
live["cn_site"] = live["countries"].fillna("").str.contains("China")
# Not-yet-recruiting trials sometimes lack a completion date: assume 24 months after start
live["pcd_est"] = live["pcd"].fillna(live["start"] + pd.DateOffset(months=24))

def ph3(s):
    return live.loc[s.index, "phase_rank"] == 3

pipe = (live.groupby(["molecule", "mol_sponsor", "mechanism"])
            .agg(max_phase=("phase_rank", "max"),
                 trials=("nct_id", "nunique"),
                 ph3_trials=("phase_rank", lambda s: int((s == 3).sum())),
                 ph3_enrolment=("enrollment", lambda s: int(s[ph3(s)].sum())),
                 last_ph3_pcd=("pcd_est", lambda s: s[ph3(s)].max()),
                 us_relevant=("us_site", lambda s: bool(s[live.loc[s.index, "phase_rank"] >= 2].any())),
                 us_max_phase=("phase_rank", lambda s: s[live.loc[s.index, "us_site"]].max() if live.loc[s.index, "us_site"].any() else 0),
                 last_us_ph3_pcd=("pcd_est", lambda s: s[ph3(s) & live.loc[s.index, "us_site"]].max()),
                 china_trials=("cn_site", "sum"))
            .reset_index())
pipe["status"] = np.select(
    [pipe["molecule"].isin(APPROVED_US_OBESITY), pipe["molecule"].isin(APPROVED_US_OTHER),
     pipe["mol_sponsor"].str.contains("discontinued")],
    ["Approved (US obesity)", "Approved (US diabetes only)", "Discontinued"], "In development")
pipe = pipe.sort_values(["status", "max_phase", "ph3_enrolment"], ascending=[True, False, False])
pipe.to_csv(TAB / "pipeline_by_molecule.csv", index=False)
dev = pipe[pipe["status"] == "In development"]
print(f"In development: {len(dev)} molecules; {int((dev['max_phase'] == 3).sum())} in Phase 3 somewhere, "
      f"{int((dev['us_max_phase'] == 3).sum())} in Phase 3 with US sites")
pipe

# %% [markdown]
# ## 4. Phase 3 scale: who is spending to win
#
# Patients enrolled in industry Phase 3 obesity trials started since 2020, by
# lead sponsor. Enrolment is a reasonable proxy for development spend, since
# Phase 3 cost scales with patient numbers.

# %%
ph3 = trials[(trials["sponsor_class"] == "INDUSTRY") & (trials["phase"] == "PHASE3") & (trials["start_year"] >= 2020)
             & ~trials["overall_status"].isin(["WITHDRAWN"])]
def short(name: str) -> str:
    name = name.split(",")[0]
    for x in [" and Company", " A/S", " Inc.", " Co. Ltd.", " Pharmaceuticals", " (Suzhou)", " (China)", " Limited", " Co.", " GmbH"]:
        name = name.replace(x, "")
    return name.replace("Hoffmann-La Roche", "Roche").strip()


ph3 = ph3.assign(sponsor=ph3["sponsor"].map(short))
spon = (ph3.groupby("sponsor").agg(trials=("nct_id", "count"), patients=("enrollment", "sum"))
           .sort_values("patients", ascending=False))
top = spon.head(10).iloc[::-1]

fig, ax = plt.subplots(figsize=(9, 5))
colors = [style.ACCENT if s in ("Eli Lilly and Company", "Novo Nordisk A/S") else style.NEUTRAL for s in top.index]
ax.barh(top.index, top["patients"], color=colors)
for y, (p, t) in enumerate(zip(top["patients"], top["trials"])):
    ax.text(p, y, f"  {p:,.0f} pts / {t} trials", va="center", fontsize=9, color=style.INK)
ax.set_title("Phase 3 obesity enrolment by sponsor, trials started 2020+")
style.subtitle(ax, "Two companies account for most late-stage patients")
ax.set_xlabel("Patients enrolled (planned or actual)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.grid(axis="y", visible=False)
ax.set_xlim(0, top["patients"].max() * 1.35)
style.source(fig, SRC + "; industry-sponsored, excluding withdrawn trials")
style.save(fig, FIG / "02_phase3_enrolment_by_sponsor.png")
lead_share = spon["patients"].head(2).sum() / spon["patients"].sum()
print(f"Top two sponsors hold {lead_share:.0%} of Phase 3 patients")
spon.head(12)

# %% [markdown]
# ## 5. Readout calendar: Phase 3 primary completions still ahead
#
# Each dot is an industry Phase 3 trial with US sites, for a molecule still in
# development. The dot sits at the trial's primary completion date (actual dates
# for finished trials, sponsor estimates for ongoing ones).

# %%
today = pd.Timestamp.today().normalize()
us_dev = set(dev.loc[dev["us_max_phase"] == 3, "molecule"])
cal = live[(live["phase_rank"] == 3) & live["us_site"] & live["molecule"].isin(us_dev) & live["pcd"].notna()
           & (live["pcd"] >= pd.Timestamp("2023-01-01"))].copy()
mol_order = cal.groupby("molecule")["pcd"].min().sort_values().index.tolist()

fig, ax = plt.subplots(figsize=(10, max(3.5, 0.42 * len(mol_order) + 1.5)))
mechs = sorted(cal["mechanism"].unique())
cmap = {mk: style.PALETTE[i % len(style.PALETTE)] for i, mk in enumerate(mechs)}
for y, mol in enumerate(mol_order):
    d = cal[cal["molecule"] == mol]
    ax.plot([d["pcd"].min(), d["pcd"].max()], [y, y], color=style.GRID, lw=3, zorder=1)
    ax.scatter(d["pcd"], [y] * len(d), color=cmap[d["mechanism"].iloc[0]], s=42, zorder=2,
               edgecolor="white", linewidth=0.6)
    ax.text(d["pcd"].max() + pd.Timedelta(days=40), y, f"{len(d)}", va="center", fontsize=8.5, color=style.MUTED)
ax.axvline(today, color=style.ACCENT_2, lw=1.2, ls="--")
ax.text(today, len(mol_order) - 0.4, " today", color=style.ACCENT_2, fontsize=9)
ax.set_yticks(range(len(mol_order)))
ax.set_yticklabels(mol_order)
ax.invert_yaxis()
ax.set_title("Phase 3 readout calendar for obesity candidates")
style.subtitle(ax, "Dots = Phase 3 primary completion dates; number = Phase 3 trials")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in cmap.items()]
ax.legend(handles=handles, loc="upper left", fontsize=8.5, bbox_to_anchor=(1.01, 1.0), title="Mechanism", title_fontsize=9)
style.source(fig, SRC + "; industry Phase 3 trials with US sites, primary completion 2023 or later")
style.save(fig, FIG / "03_phase3_readout_calendar.png")

# %% [markdown]
# ## 6. How many new entrants should we expect, and when?
#
# Historical success rates for metabolic drugs (BIO, Biomedtracker and Amplion,
# *Clinical Development Success Rates 2006–2015*):
#
# | Transition | Metabolic | All indications |
# |---|---|---|
# | Phase 1 → 2 | 61.1% | 63.2% |
# | Phase 2 → 3 | 45.2% | 30.7% |
# | Phase 3 → filing | 71.4% | 58.1% |
# | Filing → approval | 77.8% | 85.3% |
#
# These rates come from all metabolic drugs. Incretin-based drugs act on a
# validated mechanism and have succeeded far more often, so the base case uses
# the metabolic rates and an upside case raises Phase 3 → filing to 90%.
#
# Timing assumption: filing about 6 months after the last Phase 3 primary
# completion, then a 10-month standard FDA review (about 16 months in total).
# Phase 2 assets are assumed to need about 3.5 more years before approval.
# The question is about US competitors, so each molecule is staged by its most
# advanced trial **with US sites**, and timed off those trials' completion dates.
# A molecule in Phase 3 in China but only Phase 2 in the US counts as Phase 2.

# %%
P = {"base": {1: .611 * .452 * .714 * .778, 2: .452 * .714 * .778, 3: .714 * .778},
     "upside": {1: .611 * .452 * .90 * .90, 2: .452 * .90 * .90, 3: .90 * .90}}

cand = pipe[(pipe["status"] == "In development") & (pipe["us_max_phase"].between(2, 3))].copy()
cand["p_base"] = cand["us_max_phase"].map(P["base"])
cand["p_upside"] = cand["us_max_phase"].map(P["upside"])
lag3 = pd.DateOffset(months=16)
cand["exp_approval"] = np.where(
    cand["us_max_phase"] == 3,
    (cand["last_us_ph3_pcd"] + lag3).dt.year,
    today.year + 4)
cand["exp_approval"] = cand["exp_approval"].clip(lower=today.year)
cand = cand.sort_values(["us_max_phase", "exp_approval", "ph3_enrolment"], ascending=[False, True, False])
cand.to_csv(TAB / "launch_probability.csv", index=False)

years = range(today.year, today.year + 6)
exp = pd.DataFrame({
    "year": years,
    "base": [cand.loc[cand["exp_approval"] <= y, "p_base"].sum() for y in years],
    "upside": [cand.loc[cand["exp_approval"] <= y, "p_upside"].sum() for y in years],
})
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(exp["year"], exp["base"], marker="o", color=style.ACCENT, label="Base: historical metabolic success rates")
ax.plot(exp["year"], exp["upside"], marker="o", color=style.ACCENT_2, ls="--", label="Upside: incretin-like late-stage success")
for x, b in zip(exp["year"], exp["base"]):
    ax.text(x, b, f"{b:.1f}  ", ha="right", va="bottom", fontsize=9, color=style.ACCENT)
ax.set_title("Expected new US obesity approvals from today's pipeline (cumulative)")
style.subtitle(ax, "Probability-weighted count of pipeline molecules reaching approval")
ax.set_ylabel("Expected molecules")
ax.set_xticks(list(years))
ax.legend(loc="upper left", fontsize=9)
style.source(fig, SRC + "; phase success rates from BIO/Biomedtracker/Amplion (2016)")
style.save(fig, FIG / "04_expected_new_entrants.png")
cand[["molecule", "mol_sponsor", "mechanism", "us_max_phase", "ph3_trials", "ph3_enrolment",
      "last_us_ph3_pcd", "p_base", "p_upside", "exp_approval"]].head(30)

# %% [markdown]
# ## 7. Where the trials run: the rise of China-based development

# %%
ind_tr = trials[(trials["sponsor_class"] == "INDUSTRY") & trials["start_year"].between(2015, today.year)].copy()
ind_tr["china"] = ind_tr["countries"].fillna("").str.contains("China")
ind_tr["us"] = ind_tr["countries"].fillna("").str.contains("United States")
geo = ind_tr.groupby("start_year")[["china", "us"]].mean()

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(geo.index, geo["us"], marker="o", color=style.NEUTRAL, label="Has a US site")
ax.plot(geo.index, geo["china"], marker="o", color=style.ACCENT_2, label="Has a site in China")
ax.set_title("Share of industry obesity trials by site location")
style.subtitle(ax, "China's share of new industry trials has climbed")
style.pct(ax)
ax.set_ylim(0, 1)
ax.legend(loc="center left")
style.source(fig, SRC + "; trials with no listed sites count as neither")
style.save(fig, FIG / "05_trial_geography.png")
geo.round(2)

# %% [markdown]
# Late-stage programmes with Phase 3 running outside the US but not yet in the US
# (mostly China-first). These are the assets US companies have been licensing.

# %%
china_first = dev[(dev["max_phase"] == 3) & (dev["us_max_phase"] < 3)][["molecule", "mol_sponsor", "mechanism", "us_max_phase", "ph3_trials", "ph3_enrolment"]]
china_first.to_csv(TAB / "china_first_programmes.csv", index=False)
china_first

# %% [markdown]
# ## 8. Advanced analytics extension: Bayesian success rates and Monte Carlo simulation
#
# Section 6 gives a single expected number of new approvals. Decision makers
# need a **range**: how likely is it that the market gets 5 new competitors by
# 2030, or 15? Two improvements:
#
# 1. **Learn obesity-specific success rates from this data (Bayesian updating).**
#    The industry-wide metabolic rates are the *prior*: the belief before looking at
#    obesity data. The obesity programmes in this dataset then show how often
#    molecules actually moved from Phase 2 to Phase 3, and from Phase 3 to approval.
#    Combining the two gives a *posterior*, a Beta distribution that is pulled
#    towards the obesity evidence as more of it accumulates. The prior counts as
#    10 molecules' worth of evidence.
# 2. **Simulate 10,000 possible futures (Monte Carlo).** In each run, every
#    pipeline molecule succeeds or fails at a probability drawn from the
#    posterior, and its approval date slips at random (0–12 months for Phase 3
#    programmes, 0–30 months for Phase 2, because registry dates usually slip).
#    Counting approvals in each run gives a full distribution.

# %%
rng = np.random.default_rng(7)
mol_hist = (live.groupby("molecule")
                .agg(first_ph2=("start", lambda s: s[live.loc[s.index, "phase_rank"] == 2].min()),
                     first_ph3=("start", lambda s: s[live.loc[s.index, "phase_rank"] == 3].min()))
                .join(pipe.drop_duplicates("molecule").set_index("molecule")["status"]))

# Phase 2 -> 3: molecules whose first Phase 2 started 2015-2021 (enough time to progress)
c2 = mol_hist[mol_hist["first_ph2"].between("2015-01-01", "2021-12-31")]
s2 = int(c2["first_ph3"].notna().sum()); n2 = len(c2)
# Phase 3 -> approval: molecules whose first Phase 3 started 2015-2021
c3 = mol_hist[mol_hist["first_ph3"].between("2015-01-01", "2021-12-31")]
ok3 = c3["status"].str.startswith("Approved")
fail3 = c3["status"].eq("Discontinued")
s3, f3 = int(ok3.sum()), int(fail3.sum())          # still-running programmes are left out
PRIOR_N = 10
prior = {"ph2": .452, "ph3": .714 * .778}
post = {"ph2": (prior["ph2"] * PRIOR_N + s2, (1 - prior["ph2"]) * PRIOR_N + (n2 - s2)),
        "ph3": (prior["ph3"] * PRIOR_N + s3, (1 - prior["ph3"]) * PRIOR_N + f3)}
bayes = pd.DataFrame({
    "prior mean": [prior["ph2"], prior["ph3"]],
    "obesity evidence": [f"{s2} of {n2} progressed", f"{s3} approved, {f3} discontinued"],
    "posterior mean": [post["ph2"][0] / sum(post["ph2"]), post["ph3"][0] / sum(post["ph3"])],
}, index=["Phase 2 → Phase 3", "Phase 3 → approval"])
bayes

# %%
N_SIM = 10_000
years = list(range(today.year, 2032))
base_year = cand["exp_approval"].astype(float).values
is_ph3 = (cand["us_max_phase"] == 3).values
mech_arr = cand["mechanism"].values
p2 = rng.beta(*post["ph2"], N_SIM)
p3 = rng.beta(*post["ph3"], N_SIM)
counts = np.zeros((N_SIM, len(years)))
mech_hits = {m: np.zeros(N_SIM, dtype=bool) for m in set(mech_arr)}
for i in range(N_SIM):
    p = np.where(is_ph3, p3[i], p2[i] * p3[i])
    win = rng.random(len(p)) < p
    # Phase 3 dates slip 0-12 months; Phase 2 timing is far less certain, so 0-30 months
    when = base_year + np.where(is_ph3, rng.uniform(0, 1, len(p)), rng.uniform(0, 2.5, len(p)))
    for j, y in enumerate(years):
        counts[i, j] = (win & (when < y + 1)).sum()
    for m in mech_hits:
        mech_hits[m][i] = (win & (when < 2031) & (mech_arr == m)).any()

sim = pd.DataFrame({"P10": np.percentile(counts, 10, axis=0), "Median": np.median(counts, axis=0),
                    "P90": np.percentile(counts, 90, axis=0)}, index=years)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.fill_between(years, sim["P10"], sim["P90"], color=style.ACCENT, alpha=0.18, label="80% of simulated futures")
ax.plot(years, sim["Median"], color=style.ACCENT, marker="o", lw=2.2, label="Median")
ax.plot(exp["year"], exp["base"], color=style.MUTED, ls="--", lw=1.4, label="Section 6 point estimate")
for x, m_, lo_, hi in zip(years, sim["Median"], sim["P10"], sim["P90"]):
    if x >= 2029:
        ax.text(x, hi + 0.4, f"{m_:.0f} ({lo_:.0f}–{hi:.0f})", ha="center", fontsize=8.5, color=style.INK)
ax.set_xticks(years)
ax.set_ylabel("New US obesity approvals (cumulative)")
ax.set_title("How many new competitors by 2030? 10,000 simulated futures")
style.subtitle(ax, "Bayesian success rates (metabolic prior + obesity evidence) with random date slips")
ax.legend(loc="upper left", fontsize=9)
style.source(fig, SRC + "; BIO/Biomedtracker/Amplion prior")
style.save(fig, FIG / "06_monte_carlo_entrants.png")
sim.round(1)

# %%
# %% [markdown]
# **Caution on the obesity evidence:** the Phase 3 record (5 approvals, no
# Phase 3 discontinuations) comes from a mechanism class that has been unusually
# successful, and failed programmes are less likely to be fully registered. The
# prior keeps the estimate from swinging too far, but the simulated range should
# be read as optimistic-leaning.

# %%
p_mech = pd.Series({m: v.mean() for m, v in mech_hits.items()}).sort_values(ascending=False)
p5_2030 = float((counts[:, years.index(2030)] >= 5).mean())
p10_2030 = float((counts[:, years.index(2030)] >= 10).mean())
print(f"Probability of at least 5 new approvals by end-2030: {p5_2030:.0%}; at least 10: {p10_2030:.0%}")
p_mech.rename("P(at least one new approval by end-2030)").round(2)

# %% [markdown]
# ## 9. A documented negative result: can trial text reveal the mechanism?
#
# About 40% of industry trials that test a drug use code numbers the molecule
# table cannot map ("ZT006", "HDM1005", "MWN109"). Most are early-stage, often
# China-origin assets, and exactly the ones a business-development team wants to
# spot early. Their trial records, however, usually describe the drug in words:
# "a novel GLP-1/GIP dual receptor agonist", "long-acting amylin analogue".
#
# **Approach (natural language processing):**
# - **Training data:** industry trials whose drug *was* mapped by the molecule
#   table, labelled with that mechanism. The text is the trial title, the
#   sponsor's brief summary, keywords, and every intervention name, alias and
#   description.
# - **Text features:** TF-IDF, which turns text into word and two-word-phrase
#   weights. Words frequent in one record but rare across all records get the
#   highest weight.
# - **Model:** a regularised logistic regression that predicts the mechanism
#   from those weights.
# - **Validation:** cross-validation **grouped by molecule**. The model is always
#   tested on molecules it never saw, so it cannot score well by memorising drug
#   names; it has to read the mechanism language. That is the situation it faces
#   with unmapped codes.
#
# A large language model might do better, but it would need a paid API and
# would be harder to reproduce.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

iv_text = (drug_ivs.assign(desc=ivs.loc[drug_ivs.index, "description"].fillna(""))
           .groupby("nct_id")[["name", "other_names", "desc"]]
           .agg(lambda s: " ".join(s.fillna("").astype(str))).agg(" ".join, axis=1))
tt = trials.set_index("nct_id")
text = tt["title"].fillna("") + " " + iv_text.reindex(tt.index).fillna("")
for extra in ["brief_summary", "keywords"]:          # richer text where the extract has it
    if extra in tt:
        text = text + " " + tt[extra].fillna("").str.replace("|", " ", regex=False)

ind_ids = set(trials.loc[trials["sponsor_class"] == "INDUSTRY", "nct_id"])
lab = hits[hits["nct_id"].isin(ind_ids)]
lab = lab[~lab["molecule"].isin(APPROVED_US_OBESITY | APPROVED_US_OTHER) | lab.groupby("nct_id")["molecule"].transform("size").eq(1)]
one = lab.groupby("nct_id").filter(lambda g: g["mechanism"].nunique() == 1).drop_duplicates("nct_id")
counts = one["mechanism"].value_counts()
keep = counts[counts >= 15].index
one = one[one["mechanism"].isin(keep)]
X_text, y_mech, groups = text.reindex(one["nct_id"]).values, one["mechanism"].values, one["molecule"].values
print(f"{len(one)} labelled industry trials across {len(keep)} mechanisms and {one['molecule'].nunique()} molecules")


def text_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"),
                         LogisticRegression(C=5, max_iter=3000, class_weight="balanced"))


pred = np.empty(len(one), dtype=object)
for tr, te in GroupKFold(n_splits=5).split(X_text, y_mech, groups):
    pred[te] = text_model().fit(X_text[tr], y_mech[tr]).predict(X_text[te])
majority = pd.Series(y_mech).value_counts().index[0]
nlp_eval = pd.Series({"Accuracy": accuracy_score(y_mech, pred), "Macro F1": f1_score(y_mech, pred, average="macro"),
                      "Majority-class baseline accuracy": (y_mech == majority).mean()})
nlp_eval.round(3)

# %% [markdown]
# **Result: it does not work, and the reason is useful.** On molecules it has
# not seen, the classifier does *worse* than always guessing the commonest
# mechanism. Registry text rarely states the mechanism ("Study of LY3437943 in
# adults with obesity"); the cues the model can find are route ("oral") and
# sponsor habits, not biology. Mapping new code numbers to mechanisms still needs
# curated sources (company pipelines, publications), which is why
# `molecules.py` is maintained by hand. The attempt is kept here so the
# conclusion can be checked rather than taken on trust.

# %% [markdown]
# ## 10. Summary numbers used in the README

# %%
summary = {
    "trials_total": len(trials),
    "industry_trials": int((trials["sponsor_class"] == "INDUSTRY").sum()),
    "named_molecules": int(pipe["molecule"].nunique()),
    "in_development": int(len(dev)),
    "in_development_ph3": int((dev["max_phase"] == 3).sum()),
    "in_development_ph3_us": int((dev["us_max_phase"] == 3).sum()),
    "in_development_ph2_us": int((dev["us_max_phase"] == 2).sum()),
    "china_first_ph3": int(len(china_first)),
    "china_site_share_2015": float(geo.loc[2015, "china"]) if 2015 in geo.index else None,
    "china_site_share_2025": float(geo.loc[2025, "china"]) if 2025 in geo.index else None,
    "top2_ph3_patient_share": round(float(lead_share), 3),
    "bayes": bayes.drop(columns="obesity evidence").round(3).to_dict(orient="index"),
    "bayes_evidence": bayes["obesity evidence"].to_dict(),
    "mc_2030": sim.loc[2030].round(1).to_dict(), "mc_p_ge5_2030": p5_2030, "mc_p_ge10_2030": p10_2030,
    "mc_p_mechanism": p_mech.round(3).to_dict(),
    "nlp_labelled_trials": int(len(one)), "nlp_eval": nlp_eval.round(4).to_dict(),
    "expected_new_by_2030_base": round(float(exp.loc[exp["year"] == 2030, "base"].iloc[0]), 1) if 2030 in exp["year"].values else None,
    "expected_new_by_2030_upside": round(float(exp.loc[exp["year"] == 2030, "upside"].iloc[0]), 1) if 2030 in exp["year"].values else None,
}
pd.Series(summary).to_json(TAB / "summary.json", indent=1)
pd.Series(summary)
