"""Shared helpers for the modelling stage (steps 04–07)."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

import config as C


# ------------------------------------------------------------------ paths / io
def run_dir(period, target=None):
    d = C.OUTPUT_DIR / "models" / (f"{period}_{target}" if target else period)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_table(period):
    df = pd.read_parquet(C.TABLE_DIR / f"{period}.parquet")
    splits = C.TABLE_DIR / f"{period}_splits.parquet"
    if splits.exists():
        df = df.merge(pd.read_parquet(splits), on=["row", "col"], how="left", validate="1:1")
    return df


def selected_predictors(period):
    return json.loads((run_dir(period) / "selected_predictors.json").read_text())["selected"]


# ------------------------------------------------------------------ models
def linear_model():
    return make_pipeline(StandardScaler(), LinearRegression())


def additive_spline_model(alpha=1.0, n_knots=8):
    """GAM-type additive model: per-feature cubic B-splines + ridge penalty.

    Without interaction terms the prediction is a sum of one smooth function per
    feature, so each partial effect is exact (up to a constant).
    """
    return make_pipeline(StandardScaler(), SplineTransformer(n_knots=n_knots, degree=3), Ridge(alpha=alpha))


# ------------------------------------------------------------------ metrics
def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    e = p - y
    ss_res, ss_tot = np.sum(e ** 2), np.sum((y - y.mean()) ** 2)
    return {"r2": 1 - ss_res / ss_tot, "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)), "n": int(len(y))}


def block_bootstrap(y, p, blocks, n_boot=C.N_BOOTSTRAP, seed=C.SEED):
    """95 % CIs of test metrics by resampling whole spatial blocks with replacement."""
    rng = np.random.default_rng(seed)
    y, p, blocks = np.asarray(y), np.asarray(p), np.asarray(blocks)
    ub = np.unique(blocks)
    idx = {b: np.flatnonzero(blocks == b) for b in ub}
    out = []
    for _ in range(n_boot):
        take = np.concatenate([idx[b] for b in rng.choice(ub, size=len(ub), replace=True)])
        out.append(metrics(y[take], p[take]))
    df = pd.DataFrame(out)
    return {f"{k}_ci": [float(df[k].quantile(0.025)), float(df[k].quantile(0.975))]
            for k in ["r2", "rmse", "mae", "bias"]}


def morans_i(values, x, y, k=8, n_perm=199, seed=C.SEED):
    """Global Moran's I with row-standardised kNN weights and a permutation p-value."""
    v = np.asarray(values, float)
    z = v - v.mean()
    nn = NearestNeighbors(n_neighbors=k + 1).fit(np.c_[x, y])
    nbr = nn.kneighbors(return_distance=False)[:, 1:]

    def stat(zz):
        return float(np.sum(zz * zz[nbr].mean(axis=1)) / np.sum(zz ** 2))

    obs = stat(z)
    rng = np.random.default_rng(seed)
    perm = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    return {"morans_i": obs, "p_perm": float(p), "k": k, "n_perm": n_perm}


# ------------------------------------------------------------------ rasters
def to_raster(df, value_col, ref_tif):
    """Write table values back onto the export grid (NaN elsewhere)."""
    import rasterio
    with rasterio.open(ref_tif) as src:
        h, w, transform, crs = src.height, src.width, src.transform, src.crs
    arr = np.full((h, w), np.nan, dtype="float32")
    arr[df["row"].to_numpy(), df["col"].to_numpy()] = df[value_col].to_numpy()
    return arr, transform, crs
