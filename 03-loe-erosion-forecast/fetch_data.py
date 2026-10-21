"""Pull Medicaid State Drug Utilization Data (SDUD) for brands that lost exclusivity.

SDUD reports, for every state and quarter, the units, prescriptions and dollars
Medicaid reimbursed for each NDC (11-digit drug product code). Brand and generic
versions of the same molecule appear under different product names, which lets
us measure how fast a brand loses volume once generics or biosimilars launch.

Output: data/raw/sdud_<year>.csv.gz (only the molecules below)
Run from the repo root:  python 03-loe-erosion-forecast/fetch_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.sources import medicaid_sdud_filter, medicaid_sdud_urls, write_csv_gz, write_json  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
YEARS = range(2017, 2027)

# Product-name prefixes as they appear in SDUD (upper case, often truncated).
# Brand name first, then the generic / biosimilar names that compete with it.
MOLECULES = {
    "pregabalin":           ["LYRICA", "PREGABALIN"],
    "dimethyl fumarate":    ["TECFIDERA", "DIMETHYL F"],
    "varenicline":          ["CHANTIX", "VARENICLIN"],
    "fingolimod":           ["GILENYA", "FINGOLIMOD"],
    "teriflunomide":        ["AUBAGIO", "TERIFLUNOM"],
    "lurasidone":           ["LATUDA", "LURASIDONE"],
    "lisdexamfetamine":     ["VYVANSE", "LISDEXAMFE"],
    "mirabegron":           ["MYRBETRIQ", "MIRABEGRON"],
    "ticagrelor":           ["BRILINTA", "TICAGRELOR"],
    "sacubitril/valsartan": ["ENTRESTO", "SACUBITRIL"],
    "apixaban":             ["ELIQUIS", "APIXABAN"],
    "rivaroxaban":          ["XARELTO", "RIVAROXABA"],
    "adalimumab":           ["HUMIRA", "AMJEVITA", "CYLTEZO", "HYRIMOZ", "HADLIMA", "YUSIMRY", "HULIO",
                             "YUFLYMA", "IDACIO", "ABRILADA", "SIMLANDI", "ADALIMUMAB"],
    "ustekinumab":          ["STELARA", "WEZLANA", "SELARSDI", "PYZCHIVA", "OTULFI", "IMULDOSA",
                             "YESINTEK", "STEQEYMA", "STARJEMZA", "USTEKINUMA"],
}


def main() -> None:
    prefixes = [p for v in MOLECULES.values() for p in v]
    urls = medicaid_sdud_urls()
    meta = {}
    for year in YEARS:
        if year not in urls:
            print(f"SDUD {year}: not published, skipping")
            continue
        print(f"SDUD {year}: {urls[year]}")
        rows = medicaid_sdud_filter(urls[year], prefixes)
        write_csv_gz(rows, RAW / f"sdud_{year}.csv.gz")
        meta[year] = {"url": urls[year], "rows": len(rows)}
    write_json({"molecules": MOLECULES, "files": meta}, RAW / "sources.json")


if __name__ == "__main__":
    main()
