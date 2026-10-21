"""Helpers for pulling public US drug data.

Every function here talks to a free, public government API or file:
  * data.cms.gov        Medicare Part D and Part B datasets
  * data.medicaid.gov   Medicaid State Drug Utilization Data (SDUD)
  * api.fda.gov         openFDA Drugs@FDA approvals
  * clinicaltrials.gov  ClinicalTrials.gov API v2

Dataset IDs are looked up from each publisher's catalogue at run time, so the
scripts keep working when CMS or Medicaid publish a new year of data.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import time
from pathlib import Path
from typing import Iterable, Iterator

import requests

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "us-pharma-commercial-analytics (portfolio project; public data)"


def get(url: str, params: dict | None = None, stream: bool = False, tries: int = 5) -> requests.Response:
    """GET with simple exponential back-off. Raises after `tries` failures."""
    for attempt in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=180, stream=stream)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} on {r.url}")
            r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == tries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt + 1} in {wait}s: {exc}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- CMS (Medicare)

_CMS_CATALOG: list[dict] | None = None


def cms_catalog() -> list[dict]:
    global _CMS_CATALOG
    if _CMS_CATALOG is None:
        _CMS_CATALOG = get("https://data.cms.gov/data.json").json()["dataset"]
    return _CMS_CATALOG


def cms_dataset(title: str, year: int) -> dict:
    """Find the API and CSV links for one year of a CMS dataset.

    CMS publishes each year as its own catalogue entry, titled
    '<title> : <date>', where the date falls in the data year.
    """
    for d in cms_catalog():
        t = d.get("title", "")
        if t.split(" : ")[0].strip() != title:
            continue
        for dist in d.get("distribution", []):
            temporal = dist.get("temporal") or []
            start = temporal[0].get("startDate", "") if temporal else ""
            if start.startswith(str(year)):
                api = next((x["accessURL"] for x in d["distribution"]
                            if x.get("format") == "API" and x.get("accessURL")), None)
                csv_url = next((x["downloadURL"] for x in d["distribution"]
                                if x.get("format") == "CSV" and x.get("downloadURL")), None)
                return {"title": t, "api": api, "csv": csv_url, "modified": dist.get("modified")}
    raise KeyError(f"No CMS dataset '{title}' for {year}")


def cms_api_rows(api_url: str, params: dict | None = None, page: int = 5000) -> Iterator[dict]:
    """Page through a data.cms.gov data-api endpoint."""
    offset = 0
    while True:
        q = dict(params or {})
        q.update({"size": page, "offset": offset})
        rows = get(api_url, q).json()
        if not rows:
            return
        yield from rows
        if len(rows) < page:
            return
        offset += page


def cms_keyword_rows(api_url: str, keyword: str, keep) -> list[dict]:
    """Full-text search a CMS dataset, then keep only rows passing `keep(row)`.

    The keyword search is broad (it matches any column), so the local filter
    is what guarantees correctness.
    """
    out = [r for r in cms_api_rows(api_url, {"keyword": keyword}) if keep(r)]
    print(f"    keyword={keyword!r}: {len(out):,} rows kept")
    return out


def cms_csv_rows(csv_url: str) -> Iterator[dict]:
    """Stream a (possibly multi-GB) CMS CSV row by row without holding it in memory."""
    r = get(csv_url, stream=True)
    lines = (line.decode("utf-8-sig", "replace") for line in r.iter_lines())
    yield from csv.DictReader(lines)


# --------------------------------------------------------------- Medicaid SDUD

def medicaid_sdud_urls() -> dict[int, str]:
    """Map year -> download URL for every State Drug Utilization Data file."""
    items = get("https://data.medicaid.gov/api/1/metastore/schemas/dataset/items").json()
    urls = {}
    for it in items:
        m = re.fullmatch(r"State Drug Utilization Data (\d{4})", it.get("title", "").strip())
        if not m:
            continue
        for dist in it.get("distribution", []):
            u = dist.get("downloadURL") or (dist.get("data") or {}).get("downloadURL")
            if u:
                urls[int(m.group(1))] = u
    return dict(sorted(urls.items()))


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def medicaid_sdud_filter(url: str, prefixes: Iterable[str]) -> list[dict]:
    """Stream one SDUD year and keep rows whose product name starts with a prefix.

    SDUD product names are short and upper-case (e.g. 'ENTRESTO', 'SACUBITRIL'),
    so prefix matching is reliable.
    """
    prefixes = tuple(p.upper() for p in prefixes)
    r = get(url, stream=True)
    lines = (line.decode("utf-8-sig", "replace") for line in r.iter_lines())
    reader = csv.reader(lines)
    header = [_norm(h) for h in next(reader)]
    i_name = header.index("product_name")
    keep = []
    for row in reader:
        if len(row) == len(header) and row[i_name].strip().upper().startswith(prefixes):
            keep.append(dict(zip(header, row)))
    return keep


# --------------------------------------------------------------------- openFDA

def fda_applications(search: str, limit: int = 100) -> list[dict]:
    """Query openFDA Drugs@FDA. Returns [] when nothing matches (openFDA answers 404)."""
    try:
        r = SESSION.get("https://api.fda.gov/drug/drugsfda.json",
                        params={"search": search, "limit": limit}, timeout=60)
    except requests.RequestException:
        return []
    if r.status_code == 404:
        return []
    if r.status_code != 200:
        print(f"  openFDA {r.status_code} for search={search!r}")
        return []
    time.sleep(0.3)  # stay well under openFDA's unauthenticated rate limit
    return r.json().get("results", [])


# --------------------------------------------------------- ClinicalTrials.gov

def ctgov_studies(params: dict) -> Iterator[dict]:
    """Page through ClinicalTrials.gov API v2 /studies."""
    token = None
    while True:
        q = dict(params)
        q["pageSize"] = 1000
        if token:
            q["pageToken"] = token
        data = get("https://clinicaltrials.gov/api/v2/studies", q).json()
        yield from data.get("studies", [])
        token = data.get("nextPageToken")
        if not token:
            return


# ----------------------------------------------------------------------- I/O

def write_csv_gz(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty file {path}")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with gzip.open(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows):,} rows)")


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    print(f"  wrote {path}")
