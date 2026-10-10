"""Step 0 — build the study-area boundary from a pinned geoBoundaries release.

Output: data/boundary/aoi_isb_rwp.geojson (EPSG:4326), one feature per stratum,
plus aoi_metadata.json recording source, commit, license and areas.
Runs locally; no Earth Engine access needed.
"""
import json
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402

MIN_PART_M2 = 10_000  # polygon parts smaller than 0.01 km2 are treated as slivers


def main():
    C.BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    raw = C.BOUNDARY_DIR / "geoBoundaries-PAK-ADM3.geojson"
    if not raw.exists():
        print(f"Downloading {C.GB_ADM3_URL}")
        urllib.request.urlretrieve(C.GB_ADM3_URL, raw)

    adm3 = gpd.read_file(raw)
    sel = adm3[adm3["shapeName"].isin(C.AOI_UNITS)].copy()
    missing = set(C.AOI_UNITS) - set(sel["shapeName"])
    if missing or len(sel) != len(C.AOI_UNITS):
        raise SystemExit(f"Expected exactly one feature per unit; missing={missing}, found={len(sel)}")

    sel["stratum"] = sel["shapeName"].map(C.AOI_UNITS)
    utm = sel.to_crs(C.CRS)
    # Drop digitising slivers (the source Rawalpindi polygon has a ~200 m2 detached part).
    parts = utm.explode(index_parts=False)
    dropped = parts[parts.area < MIN_PART_M2]
    utm = parts[parts.area >= MIN_PART_M2].dissolve(by="shapeName", as_index=False, aggfunc="first")
    utm["geometry"] = utm.geometry.simplify(C.AOI_SIMPLIFY_M, preserve_topology=True)
    utm["area_km2"] = utm.area / 1e6

    union_area = utm.union_all().area / 1e6
    overlap = utm.area.sum() / 1e6 - union_area
    out = utm[["shapeName", "shapeID", "stratum", "area_km2", "geometry"]].to_crs("EPSG:4326")
    out.to_file(C.AOI_FILE, driver="GeoJSON")

    meta = {
        "source": "geoBoundaries gbOpen PAK ADM3",
        "commit": C.GB_COMMIT,
        "url": C.GB_ADM3_URL,
        "license": "ODbL 1.0 (per geoBoundaries metadata for PAK ADM3)",
        "simplify_tolerance_m": C.AOI_SIMPLIFY_M,
        "dropped_slivers_m2": dropped.area.round(1).tolist(),
        "units": out.drop(columns="geometry").to_dict(orient="records"),
        "union_area_km2": round(union_area, 2),
        "overlap_km2": round(overlap, 4),
        "bounds_utm": list(utm.total_bounds),
    }
    (C.BOUNDARY_DIR / "aoi_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
