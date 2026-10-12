# Derived data: Islamabad–Rawalpindi pre-monsoon LST and predictors (90 m)

Derived datasets for the study "Cool by day, hot at night: drivers of land surface temperature and population heat exposure in Islamabad–Rawalpindi, Pakistan, from spatially validated explainable machine learning" (Muqeet Ahmad). Code: https://github.com/muqeetahmaad9/uhi-xai-islamabad-rawalpindi

**Licence:** CC BY 4.0 for these derived products. Source data remain under their providers' licences (USGS Landsat, Copernicus Sentinel-2, Google Dynamic World/Open Buildings/Cloud Score+, ESA WorldCover, Copernicus DEM, JRC GSW, MODIS (NASA LP DAAC), GHS-POP (EC JRC, CC BY 4.0), WorldPop (CC BY 4.0), geoBoundaries ODbL 1.0).

## gee_exports/

Files: features_2023.tif, features_2024.tif, features_2025.tif and features_2023_2025.tif (pooled; primary).
- Grid: 90 m, EPSG:32643 (WGS 84 / UTM 43N), aligned with the Landsat 30 m grid.
- Values: float32. NoData: -9999 (cells outside the study area may also be NaN).
- Period: May–June of the given year(s). LST in °C (Landsat C2 L2 ST_B10, median of clear dates).

| Band | Name |
|---|---|
| 1 | lst |
| 2 | lst_anom |
| 3 | lst_n_obs |
| 4 | lst_sd |
| 5 | lst_unc |
| 6 | ndvi |
| 7 | mndwi |
| 8 | ndbi |
| 9 | bsi |
| 10 | dw_water |
| 11 | dw_trees |
| 12 | dw_grass |
| 13 | dw_flooded_vegetation |
| 14 | dw_crops |
| 15 | dw_shrub_and_scrub |
| 16 | dw_built |
| 17 | dw_bare |
| 18 | dw_snow_and_ice |
| 19 | wc_tree |
| 20 | wc_shrub |
| 21 | wc_grass |
| 22 | wc_crop |
| 23 | wc_built |
| 24 | wc_bare |
| 25 | wc_water |
| 26 | wc_wetland |
| 27 | bld_frac |
| 28 | bld_height |
| 29 | elev |
| 30 | slope |
| 31 | cos_incidence |
| 32 | gsw_occurrence |
| 33 | dist_water |
| 34 | stratum |

## tables/

- features_<period>.parquet: one row per retained 90 m cell, with all bands above plus row, col, x, y (cell centre, EPSG:32643) and stratum_name. Open-water cells and cells with fewer than 3 clear observations are excluded.
- features_2023_2025_splits.parquet: spatial block, split (train/test) and CV fold per cell (5 km blocks; seed 42).
- *_exclusions.csv: counts removed at each filtering step.

## gee_exports/context_2023_2025.tif (added in v1.1.0)

Population and MODIS LST on the same 90 m grid as `features_2023_2025.tif` (EPSG:32643; float32; NoData −9999). Produced by `scripts/15_export_context.py`; metadata in `context_2023_2025_manifest.json`.
- `pop_ghs2025`, `pop_wp2020`: persons per 90 m cell (GHS-POP R2023A epoch 2025; WorldPop 2020 Pakistan), by area-weighted aggregation of density.
- `{mod,myd}_{day,night}_{pm,yr}`: mean MODIS LST v061 (°C) from Terra (`mod`, ~10:30/22:30) or Aqua (`myd`, ~13:30/01:30), for May–June (`pm`) or all months (`yr`) of 2023–2025. Quality: mandatory QA ≤ 1 and LST error ≤ 2 K. Values are the 1 km MODIS pixel value sampled to the 90 m grid by nearest neighbour (1 km support).
- `n_<band>`: number of valid daily observations for the corresponding MODIS band (the paper excludes pixels with < 10).
