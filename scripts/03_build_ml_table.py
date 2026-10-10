"""Step 3 — convert an exported feature GeoTIFF into the analysis table.

Reads data/gee_exports/<name>.tif (downloaded from the Drive folder), keeps
cells inside the AOI, removes open water, cells with too few clear Landsat
observations and cells with missing values, and logs every exclusion.

Outputs (data/tables/):
  <name>.parquet           one row per retained 90 m cell; x, y = cell centre (EPSG:32643)
  <name>_exclusions.csv    counts per exclusion step (reported in the Methods)

Usage:  python scripts/03_build_ml_table.py features_2023_2025
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402


def read_stack(path):
    with rasterio.open(path) as src:
        if src.count != len(C.FEATURE_BANDS):
            raise SystemExit(f"{path.name}: {src.count} bands, expected {len(C.FEATURE_BANDS)}")
        if src.nodata is None:
            raise SystemExit(f"{path.name}: no nodata value set; masked cells would be read as data")
        if src.crs.to_string() != C.CRS:
            raise SystemExit(f"{path.name}: CRS {src.crs}, expected {C.CRS}")
        arr = src.read(masked=True).astype("float64").filled(np.nan)
        rows, cols = np.indices((src.height, src.width))
        xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel(), offset="center")
    df = pd.DataFrame(arr.reshape(arr.shape[0], -1).T, columns=C.FEATURE_BANDS)
    df["row"], df["col"] = rows.ravel(), cols.ravel()
    df["x"], df["y"] = np.asarray(xs), np.asarray(ys)
    return df


def build(df):
    log = []

    def step(name, keep):
        nonlocal df
        log.append({"step": name, "removed": int((~keep).sum()), "remaining": int(keep.sum())})
        df = df[keep]

    log.append({"step": "grid cells in export extent", "removed": 0, "remaining": len(df)})
    step("outside AOI", df["stratum"].notna())
    water = (df["gsw_occurrence"] >= C.WATER_EXCLUDE_GSW_PCT) | (df["dw_water"] >= C.WATER_EXCLUDE_DW_PROB)
    step("open water", ~water)
    step(f"fewer than {C.MIN_CLEAR_OBS} clear Landsat observations",
         df["lst_n_obs"].fillna(0) >= C.MIN_CLEAR_OBS)
    step("missing value in any band", df[C.FEATURE_BANDS].notna().all(axis=1))

    df = df.assign(stratum=df["stratum"].round().astype(int))
    df["stratum_name"] = df["stratum"].map(C.STRATUM_NAMES)
    return df.reset_index(drop=True), pd.DataFrame(log)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    name = sys.argv[1]
    df = read_stack(C.EXPORT_DIR / f"{name}.tif")
    table, log = build(df)
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(C.TABLE_DIR / f"{name}.parquet", index=False)
    log.to_csv(C.TABLE_DIR / f"{name}_exclusions.csv", index=False)
    print(log.to_string(index=False))
    print(table.groupby("stratum_name").size().rename("cells").to_string())


if __name__ == "__main__":
    main()
