"""Look up FDA approval history for the highest-spend Medicare drugs.

Needs data/shared/ first (python -m common.fetch_shared).

For each of the top brands by Medicare Part B + Part D spend, openFDA Drugs@FDA
gives the application type (NDA = small molecule, BLA = biologic), the original
and original approval date. With the time on market, these set Medicare
negotiation eligibility. Whether a generic is actually on the market is taken
from the CMS spending data in the analysis, which is more reliable than FDA
approval records (many approved generics are blocked by patents).

Output: data/raw/fda_approvals.csv.gz
Run from the repo root:  python 02-ira-negotiation-exposure/fetch_data.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.sources import fda_applications, write_csv_gz  # noqa: E402

HERE = Path(__file__).parent
SHARED = HERE.parent / "data" / "shared"
TOP_N = 300


def top_brands() -> pd.DataFrame:
    frames = []
    for part, f in [("D", "partd_spending_by_drug.csv.gz"), ("B", "partb_spending_by_drug.csv.gz")]:
        df = pd.read_csv(SHARED / f, dtype=str)
        if "Mftr_Name" in df:          # Part D: one 'Overall' row per drug plus one per manufacturer
            df = df[df["Mftr_Name"].str.strip().str.lower() == "overall"]
        else:                          # Part B: one row per billing code; tidy brand names like 'Orencia*'
            df = df.dropna(subset=["Gnrc_Name"])
            df["Brnd_Name"] = df["Brnd_Name"].str.replace(r"\*|\(.*?\)", "", regex=True).str.strip()
            df["Gnrc_Name"] = df["Gnrc_Name"].str.replace("*", "", regex=False).str.strip()
        spend_col = sorted(c for c in df.columns if re.fullmatch(r"Tot_Spndng_\d{4}", c))[-1]
        df["spend"] = pd.to_numeric(df[spend_col], errors="coerce")
        frames.append(df[["Brnd_Name", "Gnrc_Name", "spend"]].assign(part=part))
    allp = pd.concat(frames)
    return (allp.groupby(["Brnd_Name", "Gnrc_Name"], as_index=False)["spend"].sum()
                .sort_values("spend", ascending=False).head(TOP_N))


def clean(name: str) -> str:
    # CMS brand names carry suffixes like 'Ozempic' or 'Trelegy Ellipta'; FDA uses upper case
    return re.sub(r"[^A-Za-z0-9 \-]", " ", name).strip().upper()


def summarise(brand: str, generic: str) -> dict:
    # Search the product-level brand name first: some Drugs@FDA records have no 'openfda' block
    name = clean(brand)
    apps = fda_applications(f'products.brand_name:"{name}"') or fda_applications(f'openfda.brand_name:"{name}"')
    if not apps:
        first = name.split()[0]
        apps = fda_applications(f'products.brand_name:"{first}"') if len(first) > 3 else []
    out = {"Brnd_Name": brand, "Gnrc_Name": generic, "fda_found": bool(apps)}
    innovator = [a for a in apps if a.get("application_number", "").startswith(("NDA", "BLA"))]
    if innovator:
        dates, nums, sponsors = [], [], set()
        for a in innovator:
            nums.append(a["application_number"])
            sponsors.add(a.get("sponsor_name", ""))
            for s in a.get("submissions", []):
                if s.get("submission_type") == "ORIG" and s.get("submission_status") == "AP":
                    dates.append(s.get("submission_status_date"))
        out.update({
            "applications": ";".join(sorted(set(nums))),
            "app_type": "BLA" if any(n.startswith("BLA") for n in nums) else "NDA",
            "first_approval": min(d for d in dates if d) if any(dates) else None,
            "sponsor": ";".join(sorted(s for s in sponsors if s)),
        })
    return out


def main() -> None:
    brands = top_brands()
    rows = []
    for i, r in enumerate(brands.itertuples(), 1):
        rows.append({**summarise(r.Brnd_Name, r.Gnrc_Name), "medicare_spend": r.spend})
        if i % 25 == 0:
            print(f"  {i}/{len(brands)} brands looked up")
    write_csv_gz(rows, HERE / "data" / "raw" / "fda_approvals.csv.gz")


if __name__ == "__main__":
    main()
