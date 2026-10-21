"""Download datasets used by more than one project into data/shared/.

  * Medicare Part D Spending by Drug   (annual, 5 years of history per release)
  * Medicare Part B Spending by Drug   (annual)
  * Medicare Quarterly Part D Spending by Drug (most recent year to date)

Run from the repo root:  python -m common.fetch_shared
"""
from pathlib import Path

from common.sources import cms_api_rows, cms_catalog, cms_dataset, write_csv_gz, write_json

OUT = Path("data/shared")


def latest_year(title: str) -> int:
    years = []
    for d in cms_catalog():
        if d.get("title", "").split(" : ")[0].strip() == title:
            for dist in d.get("distribution", []):
                t = dist.get("temporal") or []
                if t:
                    years.append(int(t[0]["startDate"][:4]))
    return max(years)


def main() -> None:
    meta = {}
    for title, name in [
        ("Medicare Part D Spending by Drug", "partd_spending_by_drug"),
        ("Medicare Part B Spending by Drug", "partb_spending_by_drug"),
        ("Medicare Quarterly Part D Spending by Drug", "partd_quarterly_spending_by_drug"),
    ]:
        year = latest_year(title)
        ds = cms_dataset(title, year)
        print(f"{title} {year}: {ds['api']}")
        rows = list(cms_api_rows(ds["api"]))
        write_csv_gz(rows, OUT / f"{name}.csv.gz")
        meta[name] = {**ds, "data_year": year, "rows": len(rows)}
    write_json(meta, OUT / "sources.json")


if __name__ == "__main__":
    main()
