"""Pull Medicaid State Drug Utilization Data (SDUD) for GLP-1 drugs, by state.

Used to study a natural experiment: North Carolina Medicaid began covering
GLP-1 drugs for weight management on 1 August 2024, stopped on 1 October 2025
and restarted on 12 December 2025.

Also records total prescriptions per state and quarter across every drug, so
GLP-1 use can be expressed as a share of all Medicaid prescriptions (a
denominator that does not need separate enrolment data).

Output: data/raw/sdud_glp1_<year>.csv.gz and data/raw/sdud_totals_<year>.csv.gz
Run from the repo root:  python 07-test-and-control-experiments/fetch_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.sources import medicaid_sdud_filter_with_totals, medicaid_sdud_urls, write_csv_gz, write_json  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
YEARS = range(2021, 2027)

# SDUD product names (upper case, truncated to about 10 characters).
PREFIXES = ["WEGOVY", "ZEPBOUND", "SAXENDA", "OZEMPIC", "MOUNJARO", "RYBELSUS", "TRULICITY",
            "VICTOZA", "LIRAGLUTID", "BYDUREON", "BYETTA", "EXENATIDE"]

KEEP = ["utilization_type", "state", "ndc", "year", "quarter", "suppression_used", "product_name",
        "units_reimbursed", "number_of_prescriptions", "total_amount_reimbursed"]


def main() -> None:
    urls = medicaid_sdud_urls()
    meta = {}
    for year in YEARS:
        if year not in urls:
            print(f"SDUD {year}: not published, skipping")
            continue
        print(f"SDUD {year}: {urls[year]}")
        rows, totals = medicaid_sdud_filter_with_totals(urls[year], PREFIXES)
        write_csv_gz([{k: r.get(k) for k in KEEP} for r in rows], RAW / f"sdud_glp1_{year}.csv.gz")
        write_csv_gz(totals, RAW / f"sdud_totals_{year}.csv.gz")
        meta[year] = {"url": urls[year], "glp1_rows": len(rows), "total_rows": len(totals)}
        print(f"  kept {len(rows):,} GLP-1 rows; {len(totals):,} state-quarter totals")
    write_json({"prefixes": PREFIXES, "files": meta}, RAW / "sources.json")


if __name__ == "__main__":
    main()
