"""Novelty check: OpenAlex title+abstract counts and records for each claimed contribution."""
import csv, json, sys, time, urllib.parse, urllib.request, urllib.error
def get(url):
    for k in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(5 * (k + 1))
    raise RuntimeError('rate limited')
from pathlib import Path
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LST = '("land surface temperature" OR "surface urban heat island" OR "urban heat island")'
SHAP = '(SHAP OR Shapley)'
Q = {
 "A_lst_shap": f"{LST} AND {SHAP}",
 "B_lst_shap_spatialcv": f"{LST} AND {SHAP} AND (\"spatial cross-validation\" OR \"spatial block\" OR \"block cross-validation\" OR \"spatially independent\" OR \"spatial leave\" OR \"spatial k-fold\")",
 "C_lst_shap_terrain": f"{LST} AND {SHAP} AND (terrain OR topography OR topographic OR elevation OR mountainous OR hilly)",
 "D_lst_shap_stability": f"{LST} AND {SHAP} AND (stability OR robustness OR \"sensitivity analysis\" OR consistency)",
 "E_cool_island": "(\"urban cool island\" OR \"surface urban cool island\" OR \"heat sink\" OR \"negative surface urban heat island\" OR \"oasis effect\") AND (\"land surface temperature\" OR \"surface urban heat island\")",
 "F_pak_lst_ml": f"{LST} AND (Pakistan OR Islamabad OR Rawalpindi OR Lahore OR Karachi OR Peshawar OR Faisalabad OR Multan OR Quetta) AND (SHAP OR Shapley OR XGBoost OR \"random forest\" OR \"machine learning\" OR explainable)",
 "G_isb_rwp_lst": "(\"land surface temperature\" OR \"urban heat island\" OR \"surface temperature\") AND (Islamabad OR Rawalpindi)",
 "H_lst_ml_spatialcv_overestimate": f"{LST} AND (\"machine learning\" OR \"random forest\" OR XGBoost) AND (\"spatial cross-validation\" OR \"spatial autocorrelation\" OR \"spatial block\") AND (overestimat OR optimistic OR inflated OR leakage)",
}
def abstract(inv):
    if not inv: return ""
    return " ".join(w for _, w in sorted((p, w) for w, ps in inv.items() for p in ps))
def fetch(q, cap):
    rows, cur, total = [], "*", 0
    while cur:
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"title_and_abstract.search:{q},publication_year:2018-2026", "per-page": 200, "cursor": cur, "mailto": "noreply@example.org"})
        d = get(url)
        total = d["meta"]["count"]
        for w in d["results"]:
            src = (w.get("primary_location") or {}).get("source") or {}
            rows.append({"year": w.get("publication_year"), "doi": w.get("doi") or "", "title": (w.get("title") or "").replace("\n"," "),
                         "venue": src.get("display_name") or "", "cited_by": w.get("cited_by_count"), "abstract": abstract(w.get("abstract_inverted_index"))})
        cur = d["meta"].get("next_cursor")
        if not d["results"] or len(rows) >= cap: break
        time.sleep(0.2)
    return total, rows
def years(q):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"title_and_abstract.search:{q},publication_year:2018-2026", "group_by": "publication_year"})
    d = get(url)
    return {g["key"]: g["count"] for g in d["group_by"]}
for name, q in Q.items():
    if (OUT / f'{name}.csv').exists() and name in ('A_lst_shap','B_lst_shap_spatialcv'): continue
    time.sleep(2)
    cap = 50 if name == "A_lst_shap" else 1000
    total, rows = fetch(q, cap)
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["year","doi","title","venue","cited_by","abstract"]); w.writeheader(); w.writerows(rows)
    print(f"{name}: {total} hits; by year {dict(sorted(years(q).items()))}")
