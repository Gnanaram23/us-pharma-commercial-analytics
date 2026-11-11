# US Pharma Commercial Analytics

Seven end-to-end analytics case studies on the US pharmaceutical market. Each
takes a question a brand, market access, strategy or analytics team would
actually ask, and works through to a recommendation. Five run on free public
government data. Two (marketing mix and parts of omnichannel) use simulations
with a known answer, because the data they need is never published, and that
lets every method be graded against the truth instead of trusted.

## Commercial strategy and market access

| # | Project | Business question | Headline finding |
|---|---|---|---|
| 01 | [**GLP-1 market sizing & prescriber targeting**](01-glp1-market-and-targeting/) | Where is GLP-1 growth left in Medicare, and which prescribers should reps call on first? | Spend grew 40% a year to $27.9B (2024). About 33,000 high-volume primary care prescribers under-use GLP-1s, worth about 4M extra claims a year; half of that sits with 6,700 prescribers. |
| 02 | [**Medicare price negotiation (IRA) exposure**](02-ira-negotiation-exposure/) | Which manufacturers lose most from negotiated prices, and which drugs are next? | The 40 selected drugs took $126B of Medicare spend; estimated net impact is $21–25B a year, led by Novo Nordisk. A transparent screen recovered 11 of 15 actual selections in two back-tests and ranks Keytruda first for the February 2027 selection. |
| 03 | [**Loss-of-exclusivity erosion forecast**](03-loe-erosion-forecast/) | How fast do brands erode after generic entry, and what happens to Eliquis in 2028? | Medicaid shows two regimes: most brands fall to about 20% share within a year, some keep two-thirds. A December 2025 free-supply deal points Eliquis towards brand retention. |
| 04 | [**Obesity pipeline competitive intelligence**](04-obesity-pipeline-intelligence/) | Who will compete in US obesity by 2030, and when are the key readouts? | About 10 new US approvals expected by 2030, almost all in 2029–30. Lilly and Novo run 48% of Phase 3 patients, but survival analysis shows they win on scale, not speed. |

## Marketing science and experimentation

| # | Project | Business question | Headline finding |
|---|---|---|---|
| 05 | [**Omnichannel HCP engagement**](05-omnichannel-hcp-engagement/) | Does promotion reach the prescribers who matter, and what should each one get next? | The most active brand reaches 57% of top-decile GLP-1 prescribers. Every standard attribution method misplaces about 30% of credit. Uplift targeting and a learning next-best-channel policy (+14% vs rules) beat hand-written plans. |
| 06 | [**Marketing mix model**](06-marketing-mix-model/) | What does each channel return, and where should the next dollar go? | A Bayesian MMM recovers the field force's ROI almost exactly, and lift tests fix the overstated social return. Its budget plan promises $25M a year; the truth delivers $5.6M, so moves must be tested first. |
| 07 | [**Test and control experiments**](07-test-and-control-experiments/) | What happens when Medicaid drops obesity drug coverage, and how should a state pilot be sized? | Ending coverage cut use 75–93% in four states; North Carolina recovered only 57% after restarting. A post-only pilot cannot see a 30% lift; DiD with pre-period adjustment detects 10% with 12 states. |

Each project ends with recommendations and an honest list of limitations.
Back-tests against real outcomes are included wherever a forecast is made, and
what did **not** work is reported as plainly as what did.

## Methods, and where to find them

| Method | Where | What it found |
|---|---|---|
| Gradient-boosted trees with grouped cross-validation | 01, 02, 03 | Beats volume ranking for next-year growth (AUC 0.82 vs 0.78); does not beat persistence for LOE, a useful negative result |
| SHAP explanations and reason codes | 01 | New-to-class prescribers and multi-brand users score highest; large existing panels score lower |
| Growth-curve back-testing (exponential, logistic, Gompertz) | 01 | All three under-forecast 2023–24 by a fifth; the forecast is shown as the range across them |
| Bayesian hierarchical mixture model (PyMC) | 03 | Two erosion regimes with probabilities; all held-out quarters inside the 80% band |
| Split conformal prediction | 03 | 80% intervals cover 78–79% on held-out molecules |
| Bayesian updating and Monte Carlo (10,000 runs) | 04 | Median 11 new obesity approvals by 2030 (80% range 7–14) |
| Kaplan–Meier and Cox proportional hazards | 04 | Only phase and trial size change durations; sponsor does not |
| Text mining of trial records | 04 | Obesity-related indications rose from 2% to 9% of industry trial starts |
| Propensity score matching with balance and pre-trend checks | 05 | +2.4 points of Mounjaro share for newly reached prescribers, partly reps following interest |
| Markov-chain and Shapley attribution, graded against a known truth | 05 | Path methods misallocate about 30% of credit, however sophisticated |
| X-learner uplift modelling | 05 | Top half by uplift captures 78% of email's effect, against 67% by response |
| Thompson sampling and a gradient-boosted contextual bandit | 05 | Coarse Thompson sampling loses to rules; the contextual bandit beats them by 14% |
| Bayesian MMM: adstock, saturation, regional baselines, lift-test calibration | 06 | Calibration puts the true ROI inside the 90% interval for all six channels |
| Budget optimisation under posterior uncertainty, checked against truth | 06 | The optimiser overstates its gain more than four times when a curve shape is wrong |
| Synthetic control, DiD, placebo (permutation) tests | 07 | Coverage removal effects of 75–93%, far outside the placebo range |
| Event study with two-way fixed effects | 07 | No spillover into diabetes GLP-1 use |
| Power simulation on real state noise, CUPED-style adjustment | 07 | The analysis method matters more than the number of test states |

## AI workflow: briefs that cannot invent numbers

Every project also has short briefs for two readers, a brand lead and an
analytics reviewer, drafted by an open-weights language model
([`common/ai_briefs.py`](common/ai_briefs.py), run by the
[AI briefs workflow](.github/workflows/ai-briefs.yml) on GitHub Actions with
Ollama). A language model can write a fluent sentence around a number it made
up, so no draft is trusted:

1. The model gets a numbered fact sheet built from the project's `summary.json` and must cite the fact behind every number, like "57% [F18]".
2. A checker extracts every number from the draft and matches it to the cited fact, allowing only rounding, percentages, units ($M, $B, k) and a dropped minus sign.
3. Unsupported numbers go back to the model for a rewrite. Anything still unsupported after the last attempt is removed, and the removal is logged.
4. A right number can sit in a wrong sentence ("North Carolina had the largest drop" when Pennsylvania's was larger). A second pass reads every sentence against the project's write-up. Flagged sentences go back to the drafter, with the reason, for one rewrite; anything still flagged is removed. That pass is also a language model, so it reduces errors rather than ruling them out, and every removal is logged with its reason.

Each project's `outputs/briefs/` folder holds the briefs, a copy with
citations, and a JSON audit of every number. The checker has its own tests
([`tests/`](tests/)).

## How each project is organised

```
0X-project-name/
├── README.md          the business question, findings and recommendation (start here)
├── analysis.ipynb     the full analysis with every chart and table, rendered by GitHub
├── analysis.py        the same analysis as a plain Python script (source of truth)
├── fetch_data.py      downloads the raw public data (simulate.py / journeys.py where data is simulated)
├── data/raw/          the extracts used, committed so results can be checked
└── outputs/
    ├── figures/       charts used in the README
    ├── tables/        result tables (CSV) and summary.json
    └── briefs/        AI-drafted briefs with a number-by-number audit
```

## Data sources

All data is public and free to use. Nothing in this repository comes from an
employer or a client.

| Source | Publisher | Used in |
|---|---|---|
| Medicare Part D Prescribers, by Provider and Drug / by Geography and Drug | CMS, data.cms.gov | 01 |
| Medicare Part D and Part B Spending by Drug | CMS, data.cms.gov | 02 |
| Medicare Drug Price Negotiation selected drugs and negotiated prices | CMS | 02 |
| Drugs@FDA (approval dates, generic approvals) | FDA, open.fda.gov | 02, 04 |
| State Drug Utilization Data | CMS / Medicaid, data.medicaid.gov | 03 |
| ClinicalTrials.gov API v2 | National Library of Medicine | 04 |
| Open Payments general payments | CMS, openpaymentsdata.cms.gov | 05 |
| State Drug Utilization Data (GLP-1s, all states, 2021–2026) | CMS / Medicaid, data.medicaid.gov | 07 |

## Reproduce the results

```bash
pip install -r requirements.txt

# 1. Download data (or run the "Fetch public data" GitHub Action, which does the same)
python -m common.fetch_shared
python 01-glp1-market-and-targeting/fetch_data.py
python 02-ira-negotiation-exposure/fetch_data.py
python 03-loe-erosion-forecast/fetch_data.py
python 04-obesity-pipeline-intelligence/fetch_data.py
python 05-omnichannel-hcp-engagement/fetch_data.py
python 07-test-and-control-experiments/fetch_data.py

# 2. Run every analysis and rebuild the notebooks
python -m common.build_notebooks

# 3. Optional: draft and check the AI briefs (needs Ollama running locally)
ollama pull qwen2.5:14b
python -m common.ai_briefs
python -m pytest tests
```

The download step looks up each dataset's current address in the publisher's
catalogue (data.cms.gov/data.json, the data.medicaid.gov metastore), so it
keeps working when a new year of data is published.

## Tools

Python (pandas, NumPy, SciPy, scikit-learn, statsmodels, PyMC, ArviZ, SHAP,
matplotlib), Jupyter, Ollama for open-weights language models, and GitHub
Actions for data refresh and the AI briefs workflow.

## Author

Gnanaram M, solutions consultant working on US pharma and biotech commercial and
market access problems. [LinkedIn](https://linkedin.com/in/gnanaram/) ·
[Product concepts](https://github.com/Gnanaram23/Product-Concepts---Commercial-)
