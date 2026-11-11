"""Pull CMS Open Payments general payments for diabetes and obesity brands.

Open Payments (the Sunshine Act database) lists every payment or transfer of
value a drug or device maker gives a US physician or other prescriber: meals,
speaker fees, consulting, travel, education. Each record names the prescriber
(NPI), the company, the product it promoted and the nature of the payment.

The yearly files are 6-8 GB, so this streams them and keeps only payments tied
to the GLP-1, SGLT2 and DPP-4 brands that project 01 studies, aggregated to
prescriber x month x company x product x payment type.

Output: data/raw/openpayments_<year>.csv.gz
Run from the repo root:  python 05-omnichannel-hcp-engagement/fetch_data.py [year ...]
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.sources import openpayments_general_urls, stream_csv, write_csv_gz, write_json  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"

BRANDS = {
    # GLP-1 and GIP/GLP-1
    "OZEMPIC": "GLP-1", "RYBELSUS": "GLP-1", "WEGOVY": "GLP-1", "VICTOZA": "GLP-1", "SAXENDA": "GLP-1",
    "MOUNJARO": "GLP-1", "ZEPBOUND": "GLP-1", "TRULICITY": "GLP-1", "BYDUREON": "GLP-1", "BYETTA": "GLP-1",
    "XULTOPHY": "GLP-1", "SOLIQUA": "GLP-1",
    # SGLT2
    "JARDIANCE": "SGLT2", "FARXIGA": "SGLT2", "INVOKANA": "SGLT2", "STEGLATRO": "SGLT2",
    "SYNJARDY": "SGLT2", "XIGDUO": "SGLT2", "GLYXAMBI": "SGLT2", "TRIJARDY": "SGLT2", "QTERN": "SGLT2",
    # DPP-4
    "JANUVIA": "DPP-4", "JANUMET": "DPP-4", "TRADJENTA": "DPP-4", "JENTADUETO": "DPP-4",
}

PRODUCT_COLS = [f"name_of_drug_or_biological_or_device_or_medical_supply_{i}" for i in range(1, 6)]


def match_brand(row: dict) -> str | None:
    for col in PRODUCT_COLS:
        name = (row.get(col) or "").strip().upper()
        if not name:
            continue
        for brand in BRANDS:
            if name.startswith(brand) or f" {brand}" in name:
                return brand
    return None


def fetch_year(year: int, url: str) -> int:
    agg = defaultdict(lambda: [0, 0.0])
    seen = kept = 0
    for row in stream_csv(url):
        seen += 1
        if seen % 2_000_000 == 0:
            print(f"  {year}: {seen:,} rows read, {kept:,} kept")
        brand = match_brand(row)
        if not brand:
            continue
        npi = (row.get("covered_recipient_npi") or "").strip()
        if not npi:
            continue
        kept += 1
        date = (row.get("date_of_payment") or "").strip()  # MM/DD/YYYY
        month = f"{date[6:10]}-{date[0:2]}" if len(date) >= 10 else ""
        key = (
            npi,
            (row.get("covered_recipient_type") or "").replace("Covered Recipient ", "").strip(),
            (row.get("covered_recipient_specialty_1") or "").strip(),
            (row.get("recipient_state") or "").strip(),
            (row.get("applicable_manufacturer_or_applicable_gpo_making_payment_name") or "").strip(),
            brand,
            BRANDS[brand],
            (row.get("nature_of_payment_or_transfer_of_value") or "").strip(),
            month,
        )
        try:
            amount = float(row.get("total_amount_of_payment_usdollars") or 0)
        except ValueError:
            amount = 0.0
        a = agg[key]
        a[0] += 1
        a[1] += amount
    cols = ["npi", "recipient_type", "specialty", "state", "manufacturer", "brand", "drug_class", "nature", "month"]
    rows = [dict(zip(cols, k), payments=v[0], amount=round(v[1], 2)) for k, v in agg.items()]
    write_csv_gz(rows, RAW / f"openpayments_{year}.csv.gz")
    print(f"  {year}: {seen:,} rows read, {kept:,} matched, {len(rows):,} aggregated rows written")
    return seen


def main() -> None:
    years = [int(a) for a in sys.argv[1:]] or [2022, 2023, 2024]
    urls = openpayments_general_urls(years)
    meta = {}
    for y in years:
        if y not in urls:
            print(f"Open Payments {y}: not in catalogue, skipping")
            continue
        print(f"Open Payments {y}: {urls[y]}")
        meta[y] = {"url": urls[y], "rows_read": fetch_year(y, urls[y])}
    write_json({"brands": BRANDS, "files": meta}, RAW / f"sources_{'_'.join(map(str, years))}.json")


if __name__ == "__main__":
    main()
