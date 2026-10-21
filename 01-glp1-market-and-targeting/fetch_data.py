"""Pull Medicare Part D prescribing data for incretin and adjacent diabetes drugs.

Outputs (data/raw/):
  prescriber_drug_<year>.csv.gz   one row per prescriber x drug (CMS 'by Provider and Drug')
  geo_drug_<year>.csv.gz          national and state totals per drug (CMS 'by Geography and Drug')

Run from the repo root:  python 01-glp1-market-and-targeting/fetch_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.sources import cms_dataset, cms_keyword_rows, write_csv_gz, write_json  # noqa: E402

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


def matches(terms):
    return lambda row: any(t in row.get("Gnrc_Name", "").lower() for t in terms)


def pull(title: str, year: int, terms: list[str], key_cols: list[str], keep_cols=None) -> tuple[list[dict], dict]:
    ds = cms_dataset(title, year)
    print(f"{title} {year}")
    seen, rows = set(), []
    for term in terms:
        for r in cms_keyword_rows(ds["api"], term, matches([term])):
            k = tuple(r.get(c) for c in key_cols)
            if k in seen:
                continue
            seen.add(k)
            rows.append({c: r.get(c) for c in keep_cols} if keep_cols else r)
    return rows, ds


def main() -> None:
    meta = {}
    for year, terms in PROVIDER_YEARS.items():
        rows, ds = pull("Medicare Part D Prescribers - by Provider and Drug", year, terms,
                        ["Prscrbr_NPI", "Brnd_Name", "Gnrc_Name"], KEEP_COLS)
        write_csv_gz(rows, RAW / f"prescriber_drug_{year}.csv.gz")
        meta[f"prescriber_drug_{year}"] = {**ds, "terms": terms, "rows": len(rows)}
    for year in GEO_YEARS:
        rows, ds = pull("Medicare Part D Prescribers - by Geography and Drug", year,
                        GLP1 + SGLT2 + DPP4, ["Prscrbr_Geo_Lvl", "Prscrbr_Geo_Cd", "Brnd_Name", "Gnrc_Name"])
        write_csv_gz(rows, RAW / f"geo_drug_{year}.csv.gz")
        meta[f"geo_drug_{year}"] = {**ds, "rows": len(rows)}
    write_json(meta, RAW / "sources.json")


if __name__ == "__main__":
    main()
