# Explainable machine learning of daytime land surface temperature — Islamabad–Rawalpindi, Pakistan

Code and summary results for the study **"Terrain and vegetation, not built-up land, control pre-monsoon daytime land surface temperature in Islamabad–Rawalpindi, Pakistan: a spatially validated explainable machine-learning analysis"** (Muqeet Ahmad).

The workflow uses Google Earth Engine and Python to:
- derive pre-monsoon (May–June 2023–2025) Landsat 8/9 land surface temperature (LST) and multi-source predictors at 90 m;
- model LST with linear regression, an additive spline model, Random Forest and XGBoost under **spatial-block validation**;
- explain the models with **TreeSHAP**, with stability and sensitivity tests.

**Derived data** (90 m GeoTIFF stacks and analysis tables) are in `data/derived/` (described in `data/derived/README.md`) and archived with this repository on Zenodo.

## Repository structure

| Folder | Contents |
|---|---|
| `src/` | `config.py` (every parameter), `gee_lib.py` (Earth Engine functions), `modeling.py`, `figstyle.py` |
| `scripts/` | Numbered pipeline steps `00`–`13` (see below) |
| `data/boundary/` | Study-area boundary (geoBoundaries PAK ADM3, commit 9469f09; ODbL 1.0) |
| `literature/` | Reproducible OpenAlex literature search (query strings, hits, screening notes) |
| `results/` | Snapshot of the numerical outputs reported in the paper (metrics, SHAP importance, robustness tests) |

## Pipeline

| Step | Script | Needs Earth Engine |
|---|---|---|
| 0 | `00_make_aoi.py`: study-area boundary | no |
| 1 | `01_check_availability.py`: clear-sky Landsat dates, ERA5-Land context | yes |
| 2 | `02_export_features.py`: 90 m LST + predictor stacks to Google Drive | yes |
| 2b | `02b_download_exports.py`: download the exports from Drive | Drive |
| 3 | `03_build_ml_table.py`: analysis tables and exclusion log | no |
| 4 | `04_spatial_blocks.py`: variogram-based spatial blocks, test set, CV folds | no |
| 5 | `05_screen_predictors.py`: correlation and VIF screening (training blocks only) | no |
| 6 | `06_train_models.py`: tuning, spatial vs random CV, test evaluation | no |
| 7 | `07_explain_shap.py`: TreeSHAP, dependence, strata, local cases, stability | no |
| 8–10 | `08_`–`10_figures_*.py`: figures | no |
| 11–13 | `11_extra_tests.py`, `12_yearly_stability.py`, `13_sensitivity_hills.py`: robustness | no |

To skip the Earth Engine steps, copy `data/derived/gee_exports/` and `data/derived/tables/` to `data/gee_exports/` and `data/tables/` and start at step 4.

## Setup

```bash
pip install -r requirements.txt
earthengine authenticate                 # only for steps 1–2
export EE_PROJECT=<your-cloud-project>   # Earth Engine-registered Google Cloud project
```

Outputs are written to `outputs/`. The `results/` folder holds the values reported in the paper for comparison.

## Data sources

All inputs are openly available and were accessed through Google Earth Engine:
- Landsat 8/9 Collection 2 Level-2 (USGS);
- Sentinel-2 L2A and Cloud Score+ (Copernicus, Google);
- Dynamic World V1 (Google/WRI);
- ESA WorldCover v200;
- Google Open Buildings 2.5D Temporal;
- Copernicus DEM GLO-30;
- JRC Global Surface Water v1.4;
- ERA5-Land (ECMWF);
- geoBoundaries.

Please respect each provider's licence.

## Licence and citation

- **Code:** MIT licence (see `LICENSE`).
- **Derived data on Zenodo:** CC BY 4.0.

If you use this work, please cite the article (citation to be added on publication) and the Zenodo record. See `CITATION.cff`.
