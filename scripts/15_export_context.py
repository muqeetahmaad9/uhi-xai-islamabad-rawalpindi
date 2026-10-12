"""Step 15 — export population and MODIS day/night LST on the 90 m analysis grid.

Adds two layers that the Landsat stack cannot provide:
* population (who is exposed): GHS-POP R2023A, epoch 2025 (main) and WorldPop 2020
  (constrained-free 100 m, PAK; sensitivity), as persons per 90 m cell;
* MODIS LST v061 at 1 km (MOD11A1 Terra ~10:30/22:30, MYD11A1 Aqua ~13:30/01:30),
  pre-monsoon (config.SEASON_MONTHS) and all-year means for config.YEARS, so that the
  daytime Landsat SUHII can be compared with night and with other studies' periods.

MODIS quality: mandatory QA bits 0–1 in {0, 1} and LST error flag bits 6–7 <= 1 (<= 2 K).
MODIS values are sampled onto the 90 m grid by nearest neighbour (each 90 m cell carries
its 1 km pixel value); the 1 km support is stated wherever these layers are used.

Usage:
  python scripts/15_export_context.py --dry-run     # check datasets, print counts
  python scripts/15_export_context.py               # start the Drive export
  python scripts/15_export_context.py --direct      # or fetch the pixels directly (no Drive queue)
"""
import argparse
import json
import sys
from pathlib import Path

import ee
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import gee_lib as G  # noqa: E402

NAME = f"context_{C.YEARS[0]}_{C.YEARS[-1]}"
GHS_POP = "JRC/GHSL/P2023A/GHS_POP"
GHS_EPOCH = 2025
WORLDPOP = "WorldPop/GP/100m/pop"
WORLDPOP_YEAR = 2020
MODIS = {"mod": "MODIS/061/MOD11A1", "myd": "MODIS/061/MYD11A1"}
MAX_LST_ERR_FLAG = 1          # bits 6–7: 0 = <=1 K, 1 = <=2 K


def pop_per_cell(img, grid):
    """Counts on a ~100 m native grid -> persons per 90 m cell (density aggregation keeps totals)."""
    native = img.projection()
    dens = img.divide(ee.Image.pixelArea()).unmask(0).toFloat()           # persons / m2
    return G.to_grid(dens, native, grid).multiply(C.GRID_RES ** 2)


def modis_mean(key, part, start, end, months=None):
    col = ee.ImageCollection(MODIS[key]).filterDate(start, end)
    if months:
        col = col.filter(G.season_filter(months))

    def good(img):
        qc = img.select(f"QC_{part}")
        ok = qc.bitwiseAnd(3).lte(1).And(qc.rightShift(6).bitwiseAnd(3).lte(MAX_LST_ERR_FLAG))
        lst = img.select(f"LST_{part}_1km").multiply(0.02).subtract(273.15)
        return lst.updateMask(ok).toFloat()

    good_col = col.map(good)
    return good_col.mean(), good_col.count(), col.size()


def direct_download(stack, band_names, group=6):
    """Fetch the stack with computePixels on exactly the grid of the pooled features GeoTIFF."""
    with rasterio.open(C.EXPORT_DIR / f"features_{C.YEARS[0]}_{C.YEARS[-1]}.tif") as ref:
        profile, t = ref.profile, ref.transform
    grid = {"dimensions": {"width": profile["width"], "height": profile["height"]},
            "affineTransform": {"scaleX": t.a, "shearX": t.b, "translateX": t.c,
                                "shearY": t.d, "scaleY": t.e, "translateY": t.f},
            "crsCode": C.CRS}
    img = stack.unmask(C.NODATA)
    arrays = []
    for i in range(0, len(band_names), group):
        names = band_names[i:i + group]
        a = ee.data.computePixels({"expression": img.select(names), "fileFormat": "NUMPY_NDARRAY", "grid": grid})
        arrays += [a[n].astype("float32") for n in names]
        print(f"  fetched {', '.join(names)}")
    profile.update(count=len(arrays), dtype="float32", nodata=C.NODATA, compress="deflate")
    out = C.EXPORT_DIR / f"{NAME}.tif"
    with rasterio.open(out, "w", **profile) as dst:
        for k, (a, n) in enumerate(zip(arrays, band_names), 1):
            dst.write(a, k); dst.set_band_description(k, n)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--direct", action="store_true", help="fetch pixels directly instead of a Drive export")
    args = ap.parse_args()

    G.init()
    aoi_fc, aoi = G.load_aoi()
    man = json.loads((C.OUTPUT_DIR / f"exports_manifest_features_{C.YEARS[0]}_{C.YEARS[-1]}.json").read_text())
    transform = man["grid_transform"]
    grid = ee.Projection(man["grid_crs"], transform)
    stratum = G.strata_image(aoi_fc, grid)

    ghs = ee.Image(f"{GHS_POP}/{GHS_EPOCH}").select("population_count")
    wp = (ee.ImageCollection(WORLDPOP).filter(ee.Filter.eq("country", "PAK"))
          .filter(ee.Filter.eq("year", WORLDPOP_YEAR)).first().select("population"))
    bands = [pop_per_cell(ghs, grid).rename("pop_ghs2025"), pop_per_cell(wp, grid).rename("pop_wp2020")]

    start, end = f"{C.YEARS[0]}-01-01", f"{C.YEARS[-1] + 1}-01-01"
    info = {}
    for period, months in (("pm", C.SEASON_MONTHS), ("yr", None)):
        for key in MODIS:
            for part in ("Day", "Night"):
                mean, n, size = modis_mean(key, part, start, end, months)
                tag = f"{key}_{part.lower()}_{period}"
                bands += [mean.rename(tag).reproject(grid), n.rename(f"n_{tag}").toFloat().reproject(grid)]
                info[f"images_{tag}"] = size

    stack = ee.Image.cat(bands).toFloat().updateMask(stratum.mask())
    band_names = stack.bandNames().getInfo()
    totals = stack.select(["pop_ghs2025", "pop_wp2020"]).reduceRegion(
        ee.Reducer.sum(), aoi, crs=man["grid_crs"], crsTransform=transform, maxPixels=1e9)
    summary = ee.Dictionary(info).combine(totals).getInfo()
    summary.update({"name": NAME, "bands": band_names, "grid_transform": transform,
                    "ghs_pop": f"{GHS_POP}/{GHS_EPOCH}", "worldpop": f"{WORLDPOP} PAK {WORLDPOP_YEAR}",
                    "modis_qc": "mandatory QA <= 1 and LST error flag <= 1 (<= 2 K)"})
    (C.OUTPUT_DIR / f"exports_manifest_{NAME}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return
    if args.direct:
        direct_download(stack, band_names)
        return
    task = ee.batch.Export.image.toDrive(
        image=stack, description=NAME, folder=C.DRIVE_FOLDER, fileNamePrefix=NAME,
        region=aoi.bounds(), crs=man["grid_crs"], crsTransform=transform, maxPixels=1e10,
        fileFormat="GeoTIFF", formatOptions={"noData": C.NODATA})
    task.start()
    print(f"started export task {task.id}")


if __name__ == "__main__":
    main()
