"""Step 4 — spatial blocks: block size from the residual variogram, test blocks, CV folds.

1. Empirical variograms of LST and of residuals from a linear model on the candidate
   predictors (design-stage diagnostic). An exponential model gives the practical range.
2. Square blocks of that size (clipped to config.BLOCK_SIZE_LIMITS_M) tile the AOI.
3. About TEST_FRACTION of cells (whole blocks) is held out as the final test set,
   sampled within strata (jurisdiction x urban/non-urban) so every stratum is tested.
4. The remaining blocks are split into N_FOLDS spatial CV folds, balanced by cell count.

Outputs: data/tables/<period>_splits.parquet (row, col, block, split, fold),
outputs/models/<period>/blocks.json, variogram.png, blocks.png

Usage:  python scripts/04_spatial_blocks.py features_2023_2025
        python scripts/04_spatial_blocks.py features_2024 --reuse features_2023_2025
        (--reuse copies another period's blocks/split/folds so models are comparable)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import modeling as M  # noqa: E402

N_VARIO = 6000          # cells sampled for the variogram (about 18 M pairs)
MAX_LAG = 15000
LAG_WIDTH = 250


def variogram(x, y, z, rng):
    i = rng.choice(len(z), size=min(N_VARIO, len(z)), replace=False)
    d = pdist(np.c_[x[i], y[i]])
    g = 0.5 * pdist(z[i, None], "sqeuclidean")
    keep = d <= MAX_LAG
    bins = np.arange(0, MAX_LAG + LAG_WIDTH, LAG_WIDTH)
    nb = len(bins) - 1
    which = np.minimum(np.digitize(d[keep], bins) - 1, nb - 1)
    npairs = np.bincount(which, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        lag = np.bincount(which, weights=d[keep], minlength=nb) / npairs
        gam = np.bincount(which, weights=g[keep], minlength=nb) / npairs
    return lag, gam, npairs


def exp_model(h, nugget, psill, rng_):
    return nugget + psill * (1 - np.exp(-3 * h / rng_))   # rng_ = practical range (95 % of sill)


def fit_range(lag, gam):
    ok = np.isfinite(gam)
    p0 = [gam[ok][0], gam[ok].max() - gam[ok][0], 2000]
    (nug, ps, r), _ = curve_fit(exp_model, lag[ok], gam[ok], p0=p0,
                                bounds=([0, 0, 50], [np.inf, np.inf, 1e5]), maxfev=20000)
    return {"nugget": float(nug), "partial_sill": float(ps), "practical_range_m": float(r)}


def assign(df, block_m, rng):
    x0, y0 = df["x"].min(), df["y"].min()
    df["block"] = (((df["x"] - x0) // block_m).astype(int).astype(str) + "_"
                   + ((df["y"] - y0) // block_m).astype(int).astype(str))
    b = df.groupby("block").agg(n=("x", "size"), stratum=("stratum", lambda s: s.mode().iat[0]),
                                urban=("dw_built", lambda s: (s >= 0.5).mean()))
    b["cls"] = b["stratum"].astype(str) + "_" + np.where(b["urban"] >= 0.3, "urban", "nonurban")
    b["split"], b["fold"] = "train", -1
    for _, g in b.groupby("cls"):
        order = rng.permutation(g.index.to_numpy())
        target, acc = C.TEST_FRACTION * g["n"].sum(), 0
        for blk in order:
            if acc >= target:
                break
            b.loc[blk, "split"] = "test"
            acc += b.loc[blk, "n"]
        # balance remaining blocks of this class across folds by cell count
        load = np.zeros(C.N_FOLDS)
        for blk in [o for o in order if b.loc[o, "split"] == "train"]:
            f = int(np.argmin(load))
            b.loc[blk, "fold"] = f
            load[f] += b.loc[blk, "n"]
    return df.merge(b[["split", "fold", "cls"]], left_on="block", right_index=True), b


def reuse(period, ref):
    df = M.load_table(period).drop(columns=["block", "split", "fold", "cls"], errors="ignore")
    sp = pd.read_parquet(C.TABLE_DIR / f"{ref}_splits.parquet")
    out = df[["row", "col"]].merge(sp, on=["row", "col"], how="left")
    missing = int(out["block"].isna().sum())
    out = out.dropna(subset=["block"])
    out["fold"] = out["fold"].astype(int)
    out.to_parquet(C.TABLE_DIR / f"{period}_splits.parquet", index=False)
    info = json.loads((M.run_dir(ref) / "blocks.json").read_text())
    info.update(period=period, reused_from=ref, cells_without_block=missing,
                cells_by_split=out["split"].value_counts().to_dict())
    (M.run_dir(period) / "blocks.json").write_text(json.dumps(info, indent=2, default=int))
    print(f"{period}: reused blocks from {ref}; {missing} cells had no block (absent in {ref})")


def main():
    period = sys.argv[1]
    if len(sys.argv) == 4 and sys.argv[2] == "--reuse":
        return reuse(period, sys.argv[3])
    rng = np.random.default_rng(C.SEED)
    df = M.load_table(period).drop(columns=["block", "split", "fold", "cls"], errors="ignore")
    out = M.run_dir(period)
    x, y, z = df["x"].to_numpy(), df["y"].to_numpy(), df[C.TARGET].to_numpy()

    X = df[C.CANDIDATE_PREDICTORS].to_numpy()
    resid = z - M.linear_model().fit(X, z).predict(X)
    lag, g_lst, npairs = variogram(x, y, z, rng)
    _, g_res, _ = variogram(x, y, resid, np.random.default_rng(C.SEED))
    fit_lst, fit_res = fit_range(lag, g_lst), fit_range(lag, g_res)

    block_m = C.BLOCK_SIZE_M or float(np.clip(np.ceil(fit_res["practical_range_m"] / 500) * 500,
                                              *C.BLOCK_SIZE_LIMITS_M))
    df, blocks = assign(df, block_m, rng)

    df[["row", "col", "block", "split", "fold", "cls"]].to_parquet(C.TABLE_DIR / f"{period}_splits.parquet",
                                                                   index=False)
    summary = {
        "period": period, "target": C.TARGET, "seed": C.SEED,
        "variogram_lst": fit_lst, "variogram_residual": fit_res,
        "block_size_m": block_m, "block_size_rule": "residual practical range, rounded up to 500 m, "
                                                    f"clipped to {C.BLOCK_SIZE_LIMITS_M}",
        "n_blocks": int(len(blocks)), "n_test_blocks": int((blocks["split"] == "test").sum()),
        "cells_by_split": df["split"].value_counts().to_dict(),
        "cells_by_fold": df[df["split"] == "train"]["fold"].value_counts().sort_index().to_dict(),
        "test_share_by_class": (df.groupby("cls")["split"].apply(lambda s: (s == "test").mean())).round(3).to_dict(),
    }
    (out / "blocks.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))

    fig, ax = plt.subplots(figsize=(6, 4))
    hh = np.linspace(0, MAX_LAG, 300)
    for gam, fit, lab in [(g_lst, fit_lst, "LST"), (g_res, fit_res, "linear-model residual")]:
        ax.plot(lag / 1000, gam, "o", ms=3, label=f"{lab} (range {fit['practical_range_m'] / 1000:.1f} km)")
        ax.plot(hh / 1000, exp_model(hh, *fit.values()), "-", lw=1)
    ax.axvline(block_m / 1000, color="k", ls="--", lw=0.8, label=f"block size {block_m / 1000:.1f} km")
    ax.set_xlabel("lag distance (km)"); ax.set_ylabel("semivariance (°C²)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "variogram.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    tr = df["split"] == "train"
    sc = ax.scatter(df.loc[tr, "x"], df.loc[tr, "y"], c=df.loc[tr, "fold"], s=0.2, cmap="tab10", vmin=0, vmax=9)
    ax.scatter(df.loc[~tr, "x"], df.loc[~tr, "y"], c="k", s=0.2, label="test blocks")
    ax.set_aspect("equal"); ax.set_title(f"Spatial blocks ({block_m / 1000:.1f} km): CV folds and test (black)")
    ax.set_xlabel("Easting (m, EPSG:32643)"); ax.set_ylabel("Northing (m)")
    fig.tight_layout(); fig.savefig(out / "blocks.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
