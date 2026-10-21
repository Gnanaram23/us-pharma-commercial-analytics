import json, urllib.request, urllib.parse, pathlib, time
OUT = pathlib.Path("probe/out"); OUT.mkdir(parents=True, exist_ok=True)
api = "https://data.cms.gov/data-api/v1/dataset/9552739e-3d05-4c1b-8eff-ecabf391e2e5/data"
def get(u):
    t=time.time()
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent":"x"}), timeout=300) as r: b=r.read()
    return b, round(time.time()-t,1)
res = {}
tests = {
 "keyword_sema": "keyword=semaglutide&size=5",
 "keyword_sema_off": "keyword=semaglutide&size=5000&offset=0",
 "filter_eq": "filter[Gnrc_Name]=Semaglutide&size=5",
 "filter_brand": "filter[Brnd_Name]=Ozempic&size=5",
 "filter_cond_contains": "filter[f1][condition][path]=Gnrc_Name&filter[f1][condition][operator]=CONTAINS&filter[f1][condition][value]=glutide&size=5",
 "filter_cond_in": "filter[f1][condition][path]=Gnrc_Name&filter[f1][condition][operator]=IN&filter[f1][condition][value][1]=Semaglutide&filter[f1][condition][value][2]=Tirzepatide&size=5",
}
for k,q in tests.items():
    try:
        b,t = get(api+"?"+q); rows=json.loads(b)
        res[k] = {"secs":t, "n":len(rows), "gn":[r.get("Gnrc_Name") for r in rows[:5]], "bn":[r.get("Brnd_Name") for r in rows[:5]]}
    except Exception as e: res[k] = {"err": str(e)}
for k,q in {"stats_keyword": "keyword=semaglutide", "stats_filter": "filter[Gnrc_Name]=Semaglutide", "stats_all": ""}.items():
    try:
        b,t = get(api.replace("/data","/data-viewer/stats")+"?"+q); res[k] = {"secs":t, "body": json.loads(b)}
    except Exception as e: res[k] = {"err": str(e)}
# also stats endpoint variant
try:
    b,t = get("https://data.cms.gov/data-api/v1/dataset/9552739e-3d05-4c1b-8eff-ecabf391e2e5/data/stats?keyword=semaglutide"); res["stats2"] = {"secs":t,"body":json.loads(b)}
except Exception as e: res["stats2"]={"err":str(e)}
(OUT/"api_tests.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res)[:3000])
