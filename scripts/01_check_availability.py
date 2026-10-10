"""Step 1 — verify Landsat clear-sky availability before committing to the design.

For every year in config.YEARS and every season in config.COMPARISON_SEASONS it
lists acquisition dates with their AOI clear fraction and AOI median LST, and for
the study season it adds ERA5-Land context at overpass and per-pixel statistics
(clear-observation count, ST_QA uncertainty).

Outputs (outputs/availability/):
  dates_all_seasons.csv      one row per acquisition date (paper Table 3 source)
  pixel_stats_premonsoon.csv per-year percentiles of clear-obs count and uncertainty
"""
import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import gee_lib as G  # noqa: E402

PROPS = ["date", "spacecraft", "wrs_path", "n_tiles", "sun_azimuth", "sun_elevation",
         "scene_cloud_cover_mean", "aoi_clear_fraction", "aoi_median_lst"]
PCTS = [5, 25, 50, 75, 95]


def date_table(daily, aoi, with_era5):
    def to_feat(img):
        d = img.toDictionary(PROPS)
        if with_era5:
            d = d.combine(G.era5_at(img, aoi))
        return ee.Feature(None, d)
    fc = ee.FeatureCollection(daily.map(to_feat))
    return pd.DataFrame([f["properties"] for f in fc.getInfo()["features"]])


def pixel_stats(daily, aoi):
    sel = daily.filter(ee.Filter.gte("aoi_clear_fraction", C.MIN_AOI_CLEAR_FRACTION))
    img = ee.Image.cat([
        sel.select("lst").count().rename("n_clear"),
        sel.select("lst_unc").median().rename("unc_median"),
        sel.select("lst_unc").reduce(ee.Reducer.percentile([95])).rename("unc_p95"),
    ])
    stats = img.reduceRegion(ee.Reducer.percentile(PCTS), aoi, scale=C.GRID_RES, crs=C.CRS,
                             maxPixels=1e9).getInfo()
    return stats, sel.size().getInfo()


def main():
    G.init()
    _, aoi = G.load_aoi()
    out = C.OUTPUT_DIR / "availability"
    out.mkdir(parents=True, exist_ok=True)

    rows, pix = [], []
    for year in C.YEARS:
        for season, months in C.COMPARISON_SEASONS.items():
            is_study = months == C.SEASON_MONTHS
            daily = G.landsat_dates(aoi, f"{year}-01-01", f"{year + 1}-01-01", months)
            df = date_table(daily, aoi, with_era5=is_study)
            df["year"], df["season"] = year, season
            rows.append(df)
            n_ok = int((df["aoi_clear_fraction"] >= C.MIN_AOI_CLEAR_FRACTION).sum()) if len(df) else 0
            print(f"{year} {season:12s} dates={len(df):3d}  usable(>= {C.MIN_AOI_CLEAR_FRACTION:.0%} clear)={n_ok}")
            if is_study:
                stats, n_sel = pixel_stats(daily, aoi)
                stats.update(year=year, n_dates_used=n_sel)
                pix.append(stats)

    dates = pd.concat(rows, ignore_index=True)
    if "temperature_2m" in dates:
        t = dates["temperature_2m"] - 273.15
        td = dates["dewpoint_temperature_2m"] - 273.15
        dates["era5_t2m_c"] = t
        # Magnus formula (Alduchov & Eskridge coefficients)
        dates["era5_rh_pct"] = 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
        dates["era5_wind10_ms"] = np.hypot(dates["u_component_of_wind_10m"], dates["v_component_of_wind_10m"])
    dates.to_csv(out / "dates_all_seasons.csv", index=False)
    pd.DataFrame(pix).to_csv(out / "pixel_stats_premonsoon.csv", index=False)

    print("\nPer-year pixel statistics, study season (percentiles across AOI cells):")
    print(pd.DataFrame(pix).T.to_string())
    print(f"\nDecision rule: proceed if the median n_clear >= ~4 per year. "
          f"Use unc_* percentiles to set MAX_ST_UNCERTAINTY_K in config.py.")


if __name__ == "__main__":
    main()
