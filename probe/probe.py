"""One-off probe: discover dataset endpoints and schemas from public sources."""
import json, urllib.request, urllib.parse, pathlib, traceback
OUT = pathlib.Path("probe/out"); OUT.mkdir(parents=True, exist_ok=True)
def get(url, n=None):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 analytics-portfolio"})
    with urllib.request.urlopen(req, timeout=120) as r:
        b = r.read() if n is None else r.read(n)
    return b
def save(name, obj):
    (OUT / name).write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=1)[:400000])
log = []
# 1. CMS catalogue
try:
    cat = json.loads(get("https://data.cms.gov/data.json"))
    keep = []
    for d in cat.get("dataset", []):
        t = d.get("title", "")
        if any(k in t for k in ["Part D", "Part B Spending", "Prescriber"]):
            keep.append({"title": t, "identifier": d.get("identifier"),
                         "dist": [{k: x.get(k) for k in ("title", "format", "accessURL", "downloadURL", "temporal", "modified")} for x in d.get("distribution", [])][:40]})
    save("cms_catalog.json", keep); log.append(f"cms ok {len(keep)}")
except Exception: log.append("cms FAIL " + traceback.format_exc()[-500:])
# 2. sample rows from each CMS API distribution of interest
try:
    for d in keep:
        apis = [x for x in d["dist"] if (x.get("format") == "API" or "data-api" in str(x.get("accessURL")))]
        if not apis: continue
        u = apis[0]["accessURL"]
        sep = "&" if "?" in u else "?"
        try:
            s = json.loads(get(u + sep + "size=3"))
            save("sample_" + "".join(c for c in d["title"] if c.isalnum())[:60] + ".json", {"url": u, "rows": s})
        except Exception as e: log.append(f"sample fail {d['title']}: {e}")
    # filter test on prescriber-by-drug
    pd_ = [d for d in keep if d["title"].strip() == "Medicare Part D Prescribers - by Provider and Drug"]
    if pd_:
        u = [x for x in pd_[0]["dist"] if x.get("format") == "API"][0]["accessURL"]
        for q in ["filter[Gnrc_Name]=Semaglutide&size=2", "filter[Brnd_Name]=Ozempic&size=2", "keyword=Ozempic&size=2"]:
            try: save("filtertest_" + q.split("=")[0].replace("[","_").replace("]","") + ".json", {"q": q, "rows": json.loads(get(u + "?" + q))})
            except Exception as e: log.append(f"filter fail {q}: {e}")
        try: save("stats_prescriber.json", json.loads(get(u.replace("/data", "/data-viewer/stats") + "?filter[Gnrc_Name]=Semaglutide")))
        except Exception as e: log.append(f"stats fail {e}")
except Exception: log.append("cms sample FAIL " + traceback.format_exc()[-500:])
# 3. Medicaid metastore
try:
    items = json.loads(get("https://data.medicaid.gov/api/1/metastore/schemas/dataset/items"))
    sd = [{"id": i.get("identifier"), "title": i.get("title"), "dist": [x.get("downloadURL") or x.get("data", {}).get("downloadURL") for x in i.get("distribution", [])]} for i in items if "Drug Utilization" in i.get("title", "") or "NADAC" in i.get("title", "")]
    save("medicaid_catalog.json", sd); log.append(f"medicaid ok {len(sd)}")
    sid = [s for s in sd if s["title"].strip().endswith("2023")][0]["id"]
    q = "https://data.medicaid.gov/api/1/datastore/query/%s/0?limit=3&conditions[0][property]=product_name&conditions[0][value]=HUMIRA&conditions[0][operator]=starts%%20with" % sid
    try: save("medicaid_sample.json", json.loads(get(q)))
    except Exception as e:
        log.append(f"medicaid query fail {e}")
        save("medicaid_sample.json", json.loads(get("https://data.medicaid.gov/api/1/datastore/query/%s/0?limit=3" % sid)))
except Exception: log.append("medicaid FAIL " + traceback.format_exc()[-500:])
# 4. openFDA + CT.gov + Orange Book/Purple
for name, u in [("openfda_label", "https://api.fda.gov/drug/drugsfda.json?search=openfda.brand_name:%22MOUNJARO%22&limit=1"),
                ("ctgov", "https://clinicaltrials.gov/api/v2/studies?query.cond=obesity&query.intr=GLP-1&pageSize=2&countTotal=true"),
                ("cms_ira_page", "https://www.cms.gov/inflation-reduction-act-and-medicare/medicare-drug-price-negotiation")]:
    try:
        b = get(u, 300000); save(name + (".json" if "api" in u else ".html"), b.decode("utf-8", "ignore")); log.append(f"{name} ok")
    except Exception as e: log.append(f"{name} fail {e}")
save("log.txt", "\n".join(log)); print("\n".join(log))
