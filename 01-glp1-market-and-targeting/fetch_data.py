"""Pull Medicare Part D prescribing data for incretin and adjacent diabetes drugs.

Outputs (data/raw/):
  prescriber_drug_<year>.csv.gz   one row per prescriber x drug (CMS 'by Provider and Drug')
  geo_drug_<year>.csv.gz          national and state totals per drug (CMS 'by Geography and Drug')

Run from the repo root:  python 01-glp1-market-and-targeting/fetch_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.sources import cms_csv_rows, cms_dataset, write_csv_gz, write_json  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"

# Active ingredients that define each class. Matching is on the CMS generic name.
GLP1 = ["semaglutide", "tirzepatide", "dulaglutide", "liraglutide", "exenatide", "lixisenatide"]
SGLT2 = ["empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin", "bexagliflozin"]
DPP4 = ["sitagliptin", "linagliptin", "saxagliptin", "alogliptin"]

PROVIDER_YEARS = {2024: GLP1 + SGLT2 + DPP4, 2023: GLP1, 2022: GLP1}
GEO_YEARS = range(2018, 2025)

KEEP_COLS = ["Prscrbr_NPI", "Prscrbr_Last_Org_Name", "Prscrbr_First_Name", "Prscrbr_City",
             "Prscrbr_State_Abrvtn", "Prscrbr_Type", "Brnd_Name", "Gnrc_Name", "Tot_Clms",
             "Tot_30day_Fills", "Tot_Day_Suply", "Tot_Drug_Cst", "Tot_Benes"]


def pull(title: str, year: int, terms: list[str], keep_cols=None) -> tuple[list[dict], dict]:
    """Stream the full CMS CSV for one year and keep rows whose generic name matches.

    The provider-level file is about 25 million rows (several GB), so it is read
    line by line rather than loaded; streaming is much faster than paging the API."""
    ds = cms_dataset(title, year)
    print(f"{title} {year}: {ds['csv']}")
    rows = []
    for i, r in enumerate(cms_csv_rows(ds["csv"]), 1):
        g = r.get("Gnrc_Name", "").lower()
        if any(t in g for t in terms):
            rows.append({c: r.get(c) for c in keep_cols} if keep_cols else r)
        if i % 5_000_000 == 0:
            print(f"  {i:,} rows read, {len(rows):,} kept")
    return rows, ds


def main() -> None:
    meta = {}
    for year, terms in PROVIDER_YEARS.items():
        rows, ds = pull("Medicare Part D Prescribers - by Provider and Drug", year, terms, KEEP_COLS)
        write_csv_gz(rows, RAW / f"prescriber_drug_{year}.csv.gz")
        meta[f"prescriber_drug_{year}"] = {**ds, "terms": terms, "rows": len(rows)}
    for year in GEO_YEARS:
        rows, ds = pull("Medicare Part D Prescribers - by Geography and Drug", year, GLP1 + SGLT2 + DPP4)
        write_csv_gz(rows, RAW / f"geo_drug_{year}.csv.gz")
        meta[f"geo_drug_{year}"] = {**ds, "rows": len(rows)}
    write_json(meta, RAW / "sources.json")


if __name__ == "__main__":
    main()
