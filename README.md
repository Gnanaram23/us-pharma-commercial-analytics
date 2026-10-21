# US Pharma Commercial Analytics

Four end-to-end analytics case studies on the US pharmaceutical market, each built
on free public government data. Each one takes a commercial question a brand,
market access or strategy team would actually ask, and works through to a
recommendation.

<!-- PROJECT_TABLE -->

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

Python (pandas, NumPy, SciPy, scikit-learn, statsmodels, matplotlib), Jupyter,
and GitHub Actions for automated data refresh.

## Author

Gnanaram M, solutions consultant working on US pharma and biotech commercial and
market access problems. [LinkedIn](https://linkedin.com/in/gnanaram/) ·
[Product concepts](https://github.com/Gnanaram23/Product-Concepts---Commercial-)
