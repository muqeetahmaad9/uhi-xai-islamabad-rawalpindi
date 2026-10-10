"""Reproducible literature search via the OpenAlex API (title + abstract matching).

Writes one CSV per query (literature/search_<name>.csv) and a counts log
(literature/search_log.csv) with the exact query strings and run date, for the
PRISMA-style flow reported in the paper.
"""
import csv
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
YEARS = "2018-2026"

UHI = '("land surface temperature" OR "surface urban heat island" OR "urban heat island" OR "surface temperature")'
PAK = "(Pakistan OR Islamabad OR Rawalpindi OR Lahore OR Karachi OR Peshawar OR Faisalabad OR Multan)"
ML = ('("machine learning" OR "random forest" OR XGBoost OR LightGBM OR "gradient boosting" OR SHAP '
      'OR "Shapley" OR explainable OR interpretable OR "deep learning" OR "neural network" OR "support vector")')
XAI = '(SHAP OR "Shapley" OR "explainable" OR "interpretable machine learning")'

QUERIES = {
    "Q1_pakistan_uhi_ml": f"{UHI} AND {PAK} AND {ML}",
    "Q2_isb_rwp_uhi_any": f"{UHI} AND (Islamabad OR Rawalpindi)",
    "Q3_global_lst_xai": '("land surface temperature" OR "urban heat island") AND ' + XAI,
}


def abstract(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def fetch(query, max_records=None):
    rows, cursor = [], "*"
    filt = f"title_and_abstract.search:{query},publication_year:{YEARS}"
    while cursor:
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": filt, "per-page": 200, "cursor": cursor})
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        total = d["meta"]["count"]
        for w in d["results"]:
            src = (w.get("primary_location") or {}).get("source") or {}
            rows.append({
                "openalex_id": w["id"], "doi": w.get("doi") or "", "year": w.get("publication_year"),
                "type": w.get("type"), "title": (w.get("title") or "").replace("\n", " "),
                "venue": src.get("display_name") or "", "cited_by": w.get("cited_by_count"),
                "abstract": abstract(w.get("abstract_inverted_index")),
            })
        cursor = d["meta"].get("next_cursor")
        if not d["results"] or (max_records and len(rows) >= max_records):
            break
        time.sleep(0.2)
    return total, rows


def main():
    log = []
    for name, q in QUERIES.items():
        cap = 200 if name.startswith("Q3") else None   # Q3 is a prevalence count only
        total, rows = fetch(q, cap)
        with open(OUT / f"search_{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["openalex_id"])
            w.writeheader()
            w.writerows(rows)
        log.append({"query": name, "string": q, "years": YEARS, "fields": "title+abstract",
                    "hits": total, "saved": len(rows), "run_utc": dt.datetime.now(dt.UTC).isoformat()})
        print(f"{name}: {total} hits, {len(rows)} saved")
    with open(OUT / "search_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)


if __name__ == "__main__":
    main()
