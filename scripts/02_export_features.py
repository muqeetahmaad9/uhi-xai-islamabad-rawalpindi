"""Step 2 — build and export the 90 m LST + predictor stack for each period.

Periods: each year in config.YEARS plus the pooled period (all years).
Each period is exported to Google Drive as one multiband GeoTIFF whose band
order is config.FEATURE_BANDS. A manifest (dates used, image counts, grid,
config snapshot) is written to outputs/exports_manifest_<period>.json.

Usage:
  python scripts/02_export_features.py --dry-run          # build + manifest only
  python scripts/02_export_features.py                    # all periods
  python scripts/02_export_features.py --periods 2024 pooled
"""
import argparse
import json
import sys
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import gee_lib as G  # noqa: E402


def landsat_reference_transform(aoi):
    """crsTransform of a Landsat ST_B10 band delivered in C.CRS (defines grid alignment)."""
    imgs = G.landsat_raw(aoi, f"{C.YEARS[0]}-01-01", f"{C.YEARS[-1] + 1}-01-01").limit(25)
    projs = imgs.toList(25).map(lambda i: ee.Image(i).select("ST_B10").projection()).getInfo()
    for p in projs:
        if p["crs"] == C.CRS:
            return p["transform"]
    raise SystemExit(f"No Landsat scene in {C.CRS} found among {[p['crs'] for p in projs]}; "
                     "set grid alignment manually.")


def build_period(period, years, aoi, grid, landsat30, static):
    start, end = f"{years[0]}-01-01", f"{years[-1] + 1}-01-01"

    daily = G.landsat_dates(aoi, start, end)
    used = daily.filter(ee.Filter.gte("aoi_clear_fraction", C.MIN_AOI_CLEAR_FRACTION))
    lst = G.to_grid(G.lst_composite(used), landsat30, grid)

    s2, s2_native, n_s2 = G.s2_indices(aoi, start, end)
    dw, dw_native = G.dynamic_world(aoi, start, end)

    sun_az = ee.Number(used.aggregate_mean("sun_azimuth"))
    sun_el = ee.Number(used.aggregate_mean("sun_elevation"))
    slope, cos_i = G.terrain_features(static["elev"], sun_az, sun_el)

    stack = ee.Image.cat([
        lst,
        G.to_grid(s2, s2_native, grid),
        G.to_grid(dw, dw_native, grid),
        static["wc"], static["bld"],
        static["elev"], slope, cos_i,
        static["gsw"], static["dist"],
        static["stratum"],
    ]).select(C.FEATURE_BANDS).toFloat()
    stack = stack.updateMask(static["stratum"].mask())

    manifest = ee.Dictionary({
        "period": period,
        "landsat_dates_all": daily.aggregate_array("date"),
        "landsat_clear_fraction_all": daily.aggregate_array("aoi_clear_fraction"),
        "landsat_dates_used": used.aggregate_array("date"),
        "landsat_spacecraft_used": used.aggregate_array("spacecraft"),
        "mean_sun_azimuth": sun_az,
        "mean_sun_elevation": sun_el,
        "n_sentinel2_images": n_s2,
    }).getInfo()
    return stack, manifest


def static_layers(aoi, aoi_fc, grid):
    wc, wc_native = G.worldcover_fractions()
    ob, ob_native, n_ob = G.open_buildings(aoi)
    ob90 = G.to_grid(ob, ob_native, grid)
    frac = ob90.select("bld_frac")
    height = (ob90.select("bld_h_x_frac").divide(frac.max(1e-6))
              .where(frac.eq(0), 0).rename("bld_height"))      # 0 m where a cell has no buildings
    elev, dem_native = G.dem(aoi)
    gsw, dist = G.water_features(aoi, grid)
    return {
        "wc": G.to_grid(wc, wc_native, grid),
        "bld": ee.Image.cat([frac, height]),
        "elev": G.to_grid(elev.toFloat(), dem_native, grid),
        "gsw": gsw, "dist": dist,
        "stratum": G.strata_image(aoi_fc, grid),
        "n_open_buildings_tiles": n_ob,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", nargs="*", default=[str(y) for y in C.YEARS] + ["pooled"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    G.init()
    aoi_fc, aoi = G.load_aoi()
    meta = json.loads((C.BOUNDARY_DIR / "aoi_metadata.json").read_text())

    ref = landsat_reference_transform(aoi)
    grid, grid_transform = G.grid_projection(meta["bounds_utm"], ref)
    landsat30 = ee.Projection(C.CRS, ref)
    static = static_layers(aoi, aoi_fc, grid)
    n_ob = static.pop("n_open_buildings_tiles").getInfo()
    if n_ob == 0:
        raise SystemExit(f"No Open Buildings tiles for {C.OB_YEAR} over the AOI.")

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    region = aoi.bounds()
    for period in args.periods:
        years = C.YEARS if period == "pooled" else [int(period)]
        stack, manifest = build_period(period, years, aoi, grid, landsat30, static)
        manifest.update({
            "grid_crs": C.CRS, "grid_transform": grid_transform, "landsat_reference_transform": ref,
            "band_order": C.FEATURE_BANDS, "n_open_buildings_tiles": n_ob,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(C).items()
                       if k.isupper()},
        })
        name = f"features_{C.YEARS[0]}_{C.YEARS[-1]}" if period == "pooled" else f"features_{period}"
        (C.OUTPUT_DIR / f"exports_manifest_{name}.json").write_text(json.dumps(manifest, indent=2, default=str))
        print(f"{name}: {len(manifest['landsat_dates_used'])} Landsat dates used "
              f"of {len(manifest['landsat_dates_all'])}; S2 images: {manifest['n_sentinel2_images']}")
        if args.dry_run:
            continue
        task = ee.batch.Export.image.toDrive(
            image=stack, description=name, folder=C.DRIVE_FOLDER, fileNamePrefix=name,
            region=region, crs=C.CRS, crsTransform=grid_transform, maxPixels=1e10,
            fileFormat="GeoTIFF", formatOptions={"noData": C.NODATA},
        )
        task.start()
        print(f"  started export task {task.id}")


if __name__ == "__main__":
    main()
