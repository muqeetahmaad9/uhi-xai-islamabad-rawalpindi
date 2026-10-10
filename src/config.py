"""Central configuration for the Islamabad–Rawalpindi LST / XAI study.

Every methodological parameter lives here so that the paper's Methods section
can be written directly from this file. Values marked "provisional" must be
revisited after inspecting the data (see README, step 1).
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- GEE / I/O
GEE_PROJECT = os.environ.get("EE_PROJECT")          # Google Cloud project registered for Earth Engine
DRIVE_FOLDER = "UHI_ISB_RWP"                         # Google Drive folder for exports

DATA_DIR = ROOT / "data"
BOUNDARY_DIR = DATA_DIR / "boundary"
EXPORT_DIR = DATA_DIR / "gee_exports"                # put files downloaded from Drive here
TABLE_DIR = DATA_DIR / "tables"
OUTPUT_DIR = ROOT / "outputs"

# ---------------------------------------------------------------- study area
# geoBoundaries gbOpen PAK ADM3 (tehsils, census 2017 source; ODbL 1.0), pinned commit.
GB_COMMIT = "9469f09592ced973a3448cf66b6100b741b64c0d"
GB_ADM3_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/"
    f"{GB_COMMIT}/releaseData/gbOpen/PAK/ADM3/geoBoundaries-PAK-ADM3.geojson"
)
# shapeName -> stratum code. Strata are administrative jurisdictions and only a
# proxy for planned (Islamabad) vs organically grown (Rawalpindi) urban fabric.
AOI_UNITS = {"ISLAMABAD": 1, "RAWALPINDI": 2}
STRATUM_NAMES = {1: "Islamabad", 2: "Rawalpindi"}
AOI_FILE = BOUNDARY_DIR / "aoi_isb_rwp.geojson"
AOI_SIMPLIFY_M = 10                                  # vertex simplification tolerance (keeps GEE payload small)

# ---------------------------------------------------------------- grid
CRS = "EPSG:32643"                                   # WGS 84 / UTM 43N (Landsat native here)
GRID_RES = 90                                        # 3 x 3 Landsat pixels; TIRS native footprint ~100 m
NODATA = -9999.0                                     # written for masked cells in exported GeoTIFFs

# ---------------------------------------------------------------- period
YEARS = [2023, 2024, 2025]
SEASON_MONTHS = (5, 6)                               # pre-monsoon: May–June
# Seasons only used for the availability comparison (01_check_availability.py)
COMPARISON_SEASONS = {"premonsoon": (5, 6), "monsoon": (7, 9), "winter": (12, 2), "postmonsoon": (10, 11)}

# ---------------------------------------------------------------- Landsat LST
LANDSAT_COLLECTIONS = ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"]
ST_SCALE, ST_OFFSET = 0.00341802, 149.0              # USGS C2 L2 ST: K = DN * scale + offset
ST_QA_SCALE = 0.01                                   # ST_QA uncertainty, K
# QA_PIXEL bits masked: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow
QA_MASK_BITS = [0, 1, 2, 3, 4, 5]
# Set from 01 output (2026-10-07): per-pixel median ST_QA 1.7–2.5 K (p5–p95 of cells),
# per-pixel 95th percentile up to 4.7 K. 4 K removes only the cloud-edge tail.
MAX_ST_UNCERTAINTY_K = 4.0
MIN_AOI_CLEAR_FRACTION = 0.5                         # a date is used only if >= 50 % of the AOI is clear
MIN_CLEAR_OBS = 3                                    # provisional: min clear observations per cell (per year)

# ---------------------------------------------------------------- Sentinel-2
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CS_BAND = "cs_cdf"
CS_THRESHOLD = 0.60                                  # sensitivity: 0.50 and 0.65
S2_MAX_SCENE_CLOUD = 60                              # coarse pre-filter only; pixel masking does the real work
S2_SCALE = 1e-4

# ---------------------------------------------------------------- land cover / morphology
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops",
            "shrub_and_scrub", "built", "bare", "snow_and_ice"]
WORLDCOVER = "ESA/WorldCover/v200"
WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
              60: "bare", 80: "water", 90: "wetland"}
OPEN_BUILDINGS = "GOOGLE/Research/open-buildings-temporal/v1"
OB_YEAR = 2023                                       # last available year (dataset spans 2016–2023)
OB_PRESENCE_THRESHOLD = 0.5                          # provisional; test 0.4 / 0.6 as sensitivity

# ---------------------------------------------------------------- terrain / water
DEM_COLLECTION = "COPERNICUS/DEM/GLO30_2024_1"       # supersedes deprecated COPERNICUS/DEM/GLO30
GSW_IMAGE = "JRC/GSW1_4/GlobalSurfaceWater"
GSW_OCCURRENCE_THRESHOLD = 50                        # % of observations as water -> "permanent" water
WATER_DISTANCE_CAP_M = 5000
WATER_SEARCH_BUFFER_M = 10000                        # water outside the AOI still counts for distance

# ---------------------------------------------------------------- context (not predictors)
ERA5_LAND = "ECMWF/ERA5_LAND/HOURLY"
ERA5_BANDS = ["temperature_2m", "dewpoint_temperature_2m",
              "u_component_of_wind_10m", "v_component_of_wind_10m"]

# ---------------------------------------------------------------- ML table exclusions
# Open-water cells are kept on maps but excluded from the land-surface model.
WATER_EXCLUDE_GSW_PCT = 50        # cell-mean GSW occurrence >= this -> water
WATER_EXCLUDE_DW_PROB = 0.5       # or seasonal-mean Dynamic World water probability >= this

# ---------------------------------------------------------------- modelling
SEED = 42
TARGET = "lst"                    # robustness target: "lst_anom" (scene-normalised)
# Candidate predictors in PRIORITY order: when two are collinear, the earlier one is
# kept. Order reflects physical interpretability and independence from the
# Sentinel-2 spectra used for the indices (independent products first within a concept).
CANDIDATE_PREDICTORS = [
    "ndvi",            # vegetation abundance
    "bld_frac",        # building cover (Open Buildings; independent of S2 indices)
    "bld_height",      # vertical morphology
    "dw_bare",         # bare soil probability
    "elev", "slope", "cos_incidence",
    "dist_water",
    "mndwi",           # surface wetness / open water
    "dw_built", "ndbi", "bsi", "dw_trees", "dw_crops",
]
CORR_THRESHOLD = 0.80             # |Spearman rho| above which the lower-priority variable is dropped
VIF_THRESHOLD = 5.0
TEST_FRACTION = 0.20              # share of spatial blocks held out as the final test set
N_FOLDS = 5                       # spatial CV folds on the training blocks
BLOCK_SIZE_M = None               # None -> set from residual variogram range (04_spatial_blocks.py)
BLOCK_SIZE_LIMITS_M = (1000, 5000)
TUNE_MAX_ROWS = 60000             # training rows sampled (by block) for hyper-parameter search
N_BOOTSTRAP = 1000                # block bootstrap of test metrics

# ---------------------------------------------------------------- surface UHI / hotspots
# SUHII = mean LST(urban) - mean LST(rural reference), per period.
SUHI_URBAN_BUILT = 0.5            # urban: WorldCover built-up fraction of the 90 m cell >= this
SUHI_RURAL_BUILT = 0.05           # rural reference: built-up fraction < this, not water ...
SUHI_RURAL_ELEV_TOL_M = 100       # ... and elevation within +/- this of the urban median elevation
SUHI_SENSITIVITY = {"urban_built": [0.3, 0.5, 0.7], "elev_tol_m": [50, 100, 200]}
HOTSPOT_RADIUS_M = 1000           # Getis-Ord Gi* neighbourhood (binary, circular window incl. the cell)
HOTSPOT_FDR_Q = 0.05              # Benjamini-Hochberg false discovery rate

# ---------------------------------------------------------------- exported band order
# 02_export_features.py writes exactly these bands, in this order.
FEATURE_BANDS = (
    ["lst", "lst_anom", "lst_n_obs", "lst_sd", "lst_unc"]
    + ["ndvi", "mndwi", "ndbi", "bsi"]
    + [f"dw_{b}" for b in DW_BANDS]
    + [f"wc_{n}" for n in WC_CLASSES.values()]
    + ["bld_frac", "bld_height"]
    + ["elev", "slope", "cos_incidence"]
    + ["gsw_occurrence", "dist_water"]
    + ["stratum"]
)
