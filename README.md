# US Pharma Commercial Analytics

Four end-to-end analytics case studies on the US pharmaceutical market, each built
on free public government data. Each one takes a commercial question a brand,
market access or strategy team would actually ask, and works through to a
recommendation.

| # | Project | Business question | Headline finding |
|---|---|---|---|
| 01 | [**GLP-1 market sizing & prescriber targeting**](01-glp1-market-and-targeting/) | Where is GLP-1 growth left in Medicare, and which prescribers should reps call on first? | Spend grew 40% a year to $27.9B (2024). About 33,000 high-volume primary care prescribers under-use GLP-1s, worth about 4M extra claims a year; half of that sits with 6,700 prescribers. |
| 02 | [**Medicare price negotiation (IRA) exposure**](02-ira-negotiation-exposure/) | Which manufacturers lose most from negotiated prices, and which drugs are next? | The 40 selected drugs took $126B of Medicare spend; estimated net impact is $21–25B a year, led by Novo Nordisk. A transparent screen recovered 11 of 15 actual selections in two back-tests and ranks Keytruda first for the February 2027 selection. |
| 03 | [**Loss-of-exclusivity erosion forecast**](03-loe-erosion-forecast/) | How fast do brands erode after generic entry, and what happens to Eliquis in 2028? | Medicaid shows two regimes: most brands fall to about 20% share within a year, some keep two-thirds. A December 2025 free-supply deal points Eliquis towards brand retention. |
| 04 | [**Obesity pipeline competitive intelligence**](04-obesity-pipeline-intelligence/) | Who will compete in US obesity by 2030, and when are the key readouts? | About 10 new US approvals expected by 2030, almost all in 2029–30; Lilly and Novo run 48% of Phase 3 patients; China's share of new trials rose from 0% to 36%. |

Each project ends with recommendations and an honest list of limitations.
Back-tests against real outcomes are included wherever a forecast is made.

### Machine learning and advanced analytics extensions

| # | Extension | Method | What it found |
|---|---|---|---|
| 01 | Next-year growth prediction | Gradient-boosted trees, cross-validated on held-out states | Reaches 44% of next-year growth by calling the top 20% of prescribers, against 40% by volume ranking (AUC 0.82 vs 0.78) |
| 02 | Learning CMS's selection pattern | Logistic regression and gradient boosting, trained on rounds 1–2, tested on round 3 | Pure ML fails (5 of 15); rules plus ML reaches 10 of 15; a 13-drug consensus list for 2027 |
| 03 | State-level early warning after generic entry | Gradient boosting and ridge regression, leave-one-molecule-out | State data halves the error of national curves; ML does not beat simple persistence, a useful negative result |
| 04 | Range of new competitors by 2030 | Bayesian updating of phase-success rates plus Monte Carlo simulation (10,000 runs) | Median 11 new approvals (80% range 7–14); a text classifier for unmapped drug codes was tried and documented as not working |

The extensions report what did **not** work as plainly as what did. Each method
is tested on data it has not seen.

## How each project is organised

```
0X-project-name/
├── README.md          the business question, findings and recommendation (start here)
├── analysis.ipynb     the full analysis with every chart and table, rendered by GitHub
├── analysis.py        the same analysis as a plain Python script (source of truth)
├── fetch_data.py      downloads the raw public data
├── data/raw/          the extracts used, committed so results can be checked
└── outputs/
    ├── figures/       charts used in the README
    └── tables/        result tables (CSV) and summary.json
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

## Reproduce the results

```bash
pip install -r requirements.txt

# 1. Download data (or run the "Fetch public data" GitHub Action, which does the same)
python -m common.fetch_shared
python 01-glp1-market-and-targeting/fetch_data.py
python 02-ira-negotiation-exposure/fetch_data.py
python 03-loe-erosion-forecast/fetch_data.py
python 04-obesity-pipeline-intelligence/fetch_data.py

# 2. Run every analysis and rebuild the notebooks
python -m common.build_notebooks
```

The download step looks up each dataset's current address in the publisher's
catalogue (data.cms.gov/data.json, the data.medicaid.gov metastore), so it
keeps working when a new year of data is published.

## Tools

Python (pandas, NumPy, SciPy, scikit-learn, matplotlib), Jupyter, and GitHub
Actions for automated data refresh. Methods used: gradient-boosted trees, logistic
and ridge regression, k-means clustering, TF-IDF text classification, Bayesian
updating, Monte Carlo simulation, non-linear curve fitting and grouped,
out-of-time and leave-one-group-out cross-validation.

## Author

Gnanaram M, solutions consultant working on US pharma and biotech commercial and
market access problems. [LinkedIn](https://linkedin.com/in/gnanaram/) ·
[Product concepts](https://github.com/Gnanaram23/Product-Concepts---Commercial-)
