"""Pull every interventional obesity / overweight trial from ClinicalTrials.gov,
plus FDA approval records for marketed anti-obesity medicines.

Outputs (data/raw/):
  trials.csv.gz          one row per trial
  interventions.csv.gz   one row per trial x intervention
  fda_obesity.json       Drugs@FDA records for approved obesity drugs

Run from the repo root:  python 04-obesity-pipeline-intelligence/fetch_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.sources import ctgov_studies, fda_applications, write_csv_gz, write_json  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"

QUERY = {
    "query.cond": "obesity OR overweight",
    "filter.advanced": "AREA[StudyType]INTERVENTIONAL AND AREA[StartDate]RANGE[2015-01-01,MAX]",
}
APPROVED = ["WEGOVY", "ZEPBOUND", "SAXENDA", "QSYMIA", "CONTRAVE", "IMCIVREE", "XENICAL"]


def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
    return default if d is None else d


def main() -> None:
    trials, interventions = [], []
    for s in ctgov_studies(QUERY):
        p = s.get("protocolSection", {})
        nct = g(p, "identificationModule", "nctId")
        sponsor = g(p, "sponsorCollaboratorsModule", "leadSponsor", default={})
        design = g(p, "designModule", default={})
        status = g(p, "statusModule", default={})
        locs = g(p, "contactsLocationsModule", "locations", default=[])
        trials.append({
            "nct_id": nct,
            "title": g(p, "identificationModule", "briefTitle"),
            "acronym": g(p, "identificationModule", "acronym"),
            "sponsor": sponsor.get("name"),
            "sponsor_class": sponsor.get("class"),
            "collaborators": "; ".join(c.get("name", "") for c in
                                       g(p, "sponsorCollaboratorsModule", "collaborators", default=[])),
            "phases": "|".join(design.get("phases", []) or []),
            "allocation": g(design, "designInfo", "allocation"),
            "primary_purpose": g(design, "designInfo", "primaryPurpose"),
            "enrollment": g(design, "enrollmentInfo", "count"),
            "overall_status": status.get("overallStatus"),
            "start_date": g(status, "startDateStruct", "date"),
            "primary_completion_date": g(status, "primaryCompletionDateStruct", "date"),
            "primary_completion_type": g(status, "primaryCompletionDateStruct", "type"),
            "completion_date": g(status, "completionDateStruct", "date"),
            "conditions": "|".join(g(p, "conditionsModule", "conditions", default=[])),
            "countries": "|".join(sorted({l.get("country", "") for l in locs if l.get("country")})),
            "has_results": s.get("hasResults"),
            "brief_summary": (g(p, "descriptionModule", "briefSummary") or "")[:3000],
            "keywords": "|".join(g(p, "conditionsModule", "keywords", default=[]) or []),
        })
        for iv in g(p, "armsInterventionsModule", "interventions", default=[]):
            interventions.append({
                "nct_id": nct,
                "type": iv.get("type"),
                "name": iv.get("name"),
                "other_names": "|".join(iv.get("otherNames", []) or []),
                "description": (iv.get("description") or "")[:500],
            })
    print(f"ClinicalTrials.gov: {len(trials):,} trials, {len(interventions):,} interventions")
    write_csv_gz(trials, RAW / "trials.csv.gz")
    write_csv_gz(interventions, RAW / "interventions.csv.gz")

    fda = {b: fda_applications(f'openfda.brand_name:"{b}"', limit=10) for b in APPROVED}
    write_json(fda, RAW / "fda_obesity.json")


if __name__ == "__main__":
    main()
