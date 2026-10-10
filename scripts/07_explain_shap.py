"""Step 7 — SHAP explanations, response shapes, local cases, stability.

* TreeSHAP for XGBoost on every cell (xgboost native pred_contribs, i.e.
  path-dependent TreeSHAP; Lundberg et al. 2020). The `shap` package is not
  required: plots are drawn here, and the random-forest comparison uses
  XGBoost's random-forest mode (XGBRFRegressor), which also has native TreeSHAP.
* Global importance (mean |SHAP|), beeswarm, dependence plots overlaid with the
  additive spline model's partial effects (independent check of response shape).
* Stratum comparison: binned mean SHAP vs feature value, Islamabad vs Rawalpindi.
* Local explanations for representative cells chosen by fixed, a-priori rules.
* SHAP GeoTIFF (one band per feature) for mapping.
* Stability: Spearman correlation of importance rankings across CV-fold refits
  and between XGBoost and RF.

Usage:  python scripts/07_explain_shap.py features_2023_2025 [--target lst_anom]
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import xgboost as xgb  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.base import clone  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import modeling as M  # noqa: E402

N_SAMPLE = 10000      # cells for beeswarm / RF SHAP / stability comparisons

# Representative-cell rules, fixed before looking at SHAP values.
# Each: (label, filter on the table, ranking column for picking the median case)
CASES = [
    ("dense built-up, Rawalpindi", lambda d: (d.stratum == 2) & (d.bld_frac >= d.bld_frac.quantile(0.95))),
    ("planned built-up, Islamabad", lambda d: (d.stratum == 1) & (d.dw_built >= 0.5)
     & d.ndvi.between(d.ndvi.quantile(0.4), d.ndvi.quantile(0.6))),
    ("forested hill", lambda d: (d.elev >= d.elev.quantile(0.95)) & (d.ndvi >= d.ndvi.quantile(0.8))),
    ("bare land", lambda d: d.dw_bare >= d.dw_bare.quantile(0.98)),
    ("cropland", lambda d: d.dw_crops >= d.dw_crops.quantile(0.95)),
    ("near open water", lambda d: d.dist_water <= d.dist_water.quantile(0.02)),
]


def xgb_shap(model, X, chunk=20000, progress=False):
    """Native TreeSHAP in chunks (prints progress/ETA for long runs)."""
    import time
    booster, out, t0 = model.get_booster(), [], time.time()
    for s in range(0, len(X), chunk):
        out.append(booster.predict(xgb.DMatrix(X[s:s + chunk]), pred_contribs=True))
        if progress:
            done = min(s + chunk, len(X)); el = time.time() - t0
            print(f"  SHAP {done:,}/{len(X):,} cells  {el / 60:.1f} min elapsed, "
                  f"~{el / done * (len(X) - done) / 60:.1f} min left", flush=True)
    contrib = np.vstack(out)
    return contrib[:, :-1], contrib[:, -1]          # per-feature SHAP, base value


def importance(sv, feats):
    return pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)


def gam_partial(gam, X, j, grid):
    """Partial effect of feature j from the additive model, centred on the data mean."""
    Xg = np.repeat(X.mean(axis=0, keepdims=True), len(grid), axis=0)
    Xg[:, j] = grid
    pe = gam.predict(Xg)
    Xd = np.repeat(X.mean(axis=0, keepdims=True), len(X), axis=0)
    Xd[:, j] = X[:, j]
    return pe - gam.predict(Xd).mean()


def beeswarm(sv, X, feats, order, path, title):
    """SHAP summary plot: one row per feature, points jittered by local density, coloured by feature rank."""
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(order) + 1.2))
    rng = np.random.default_rng(C.SEED)
    cmap = plt.get_cmap("coolwarm")
    for r, f in enumerate(order[::-1]):
        j = feats.index(f)
        v, x = sv[:, j], X[:, j]
        col = pd.Series(x).rank(pct=True).to_numpy()
        hist, edges = np.histogram(v, bins=80)
        dens = hist[np.clip(np.digitize(v, edges) - 1, 0, len(hist) - 1)] / hist.max()
        y = r + (rng.random(len(v)) - 0.5) * 0.8 * dens
        o = rng.permutation(len(v))
        ax.scatter(v[o], y[o], c=col[o], cmap=cmap, s=2, alpha=0.6, lw=0, rasterized=True)
    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_yticks(range(len(order)), order[::-1]); ax.set_xlabel("SHAP value: contribution to predicted LST (°C)")
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap), ax=ax, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"]); cb.set_label("Feature value (percentile)")
    ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def waterfall(vals, base, data, feats, title, path):
    """Local explanation: base value -> prediction, features sorted by |SHAP|."""
    o = np.argsort(-np.abs(vals))
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(feats) + 1.4))
    cur = base
    for r, k in enumerate(o):
        v = vals[k]
        ax.barh(len(o) - 1 - r, v, left=cur, color="#d7191c" if v > 0 else "#2c7bb6", height=0.6)
        ax.text(cur + v + (0.05 if v >= 0 else -0.05), len(o) - 1 - r, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7)
        cur += v
    ax.set_yticks(range(len(o)), [f"{feats[k]} = {data[k]:.3g}" for k in o][::-1], fontsize=7)
    ax.axvline(base, color="0.5", ls="--", lw=0.7); ax.axvline(cur, color="k", lw=0.8)
    ax.set_xlabel("Predicted LST (°C)")
    ax.set_title(f"{title}\nE[f(x)] = {base:.2f} °C  →  f(x) = {cur:.2f} °C", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("period")
    ap.add_argument("--target", default=C.TARGET)
    a = ap.parse_args()
    out = M.run_dir(a.period, a.target)
    sd = out / "shap"
    sd.mkdir(exist_ok=True)

    bx = joblib.load(out / "models" / "xgb.joblib")
    feats, model = bx["features"], bx["model"]
    df = M.load_table(a.period)
    X = df[feats].to_numpy()
    print(f"Computing SHAP for {len(X):,} cells x {len(feats)} predictors...", flush=True)
    sv, base = xgb_shap(model, X, progress=True)
    shap_df = pd.DataFrame(sv, columns=[f"shap_{f}" for f in feats])
    pd.concat([df[["row", "col", "x", "y", "split", "stratum", "block"]], shap_df], axis=1) \
        .to_parquet(sd / "shap_values_xgb.parquet", index=False)

    imp = importance(sv, feats)
    imp.rename("mean_abs_shap").to_csv(sd / "importance_xgb.csv")
    print("Global importance (XGBoost, mean |SHAP|, °C):\n", imp.round(3).to_string())

    rng = np.random.default_rng(C.SEED)
    samp = rng.choice(len(df), size=min(N_SAMPLE, len(df)), replace=False)

    # ---- global plots
    beeswarm(sv[samp], X[samp], feats, list(imp.index), sd / "beeswarm.png", f"SHAP summary — XGBoost ({a.target})")

    fig, ax = plt.subplots(figsize=(5, 0.35 * len(feats) + 1))
    imp[::-1].plot.barh(ax=ax, color="#4c72b0")
    ax.set_xlabel("mean |SHAP| (°C)"); fig.tight_layout(); fig.savefig(sd / "importance_bar.png", dpi=200)
    plt.close(fig)

    # ---- dependence + additive-model partial effects
    gam_path = out / "models" / "gam.joblib"
    gam = joblib.load(gam_path)["model"] if gam_path.exists() else None
    top = list(imp.index[:6])
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, f in zip(axs.ravel(), top):
        j = feats.index(f)
        ax.scatter(X[samp, j], sv[samp, j], s=2, alpha=0.3, color="#4c72b0", label="SHAP (XGBoost)")
        if gam is not None:
            g = np.linspace(*np.percentile(X[:, j], [1, 99]), 100)
            ax.plot(g, gam_partial(gam, X, j, g), color="#c44e52", lw=1.5, label="additive spline partial effect")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(f); ax.set_ylabel("contribution to LST (°C)")
    axs.ravel()[0].legend(fontsize=7, markerscale=4)
    fig.tight_layout(); fig.savefig(sd / "dependence_top6.png", dpi=200); plt.close(fig)

    # ---- stratum comparison (binned mean SHAP on shared value range)
    rows = []
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, f in zip(axs.ravel(), top):
        j = feats.index(f)
        lo = max(np.percentile(X[df.stratum == s, j], 5) for s in (1, 2))
        hi = min(np.percentile(X[df.stratum == s, j], 95) for s in (1, 2))
        if not lo < hi:
            continue
        edges = np.linspace(lo, hi, 11)
        for s, col in [(1, "#55a868"), (2, "#dd8452")]:
            m = (df.stratum == s).to_numpy() & (X[:, j] >= lo) & (X[:, j] <= hi)
            b = np.digitize(X[m, j], edges[1:-1])
            means = pd.Series(sv[m, j]).groupby(b).mean()
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers[means.index], means.values, "-o", ms=3, color=col, label=C.STRATUM_NAMES[s])
            rows += [{"feature": f, "stratum": C.STRATUM_NAMES[s], "bin_center": centers[k], "mean_shap": v}
                     for k, v in means.items()]
        ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(f); ax.set_ylabel("mean SHAP (°C)")
    axs.ravel()[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(sd / "strata_dependence.png", dpi=200); plt.close(fig)
    pd.DataFrame(rows).to_csv(sd / "strata_dependence.csv", index=False)

    # ---- local explanations
    pred = model.predict(X)
    local = []
    for label, rule in CASES:
        m = rule(df).to_numpy()
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        i = idx[np.argsort(pred[idx])[len(idx) // 2]]          # median-prediction cell of the group
        local.append({"case": label, "row": int(df.row[i]), "col": int(df.col[i]), "x": float(df.x[i]),
                      "y": float(df.y[i]), "observed": float(df[a.target][i]), "predicted": float(pred[i]),
                      "base_value": float(base[i]), **{f: float(sv[i, k]) for k, f in enumerate(feats)}})
        waterfall(sv[i], base[i], X[i], feats, label, sd / f"local_{label.split(',')[0].replace(' ', '_')}.png")
    pd.DataFrame(local).to_csv(sd / "local_cases.csv", index=False)

    # ---- SHAP GeoTIFF
    ref = C.EXPORT_DIR / f"{a.period}.tif"
    if ref.exists():
        bands = [M.to_raster(df.assign(v=sv[:, k]), "v", ref)[0] for k in range(len(feats))]
        _, transform, crs = M.to_raster(df.assign(v=0.0), "v", ref)
        with rasterio.open(sd / "shap_xgb.tif", "w", driver="GTiff", height=bands[0].shape[0],
                           width=bands[0].shape[1], count=len(feats), dtype="float32", crs=crs,
                           transform=transform, nodata=np.nan) as dst:
            for k, b in enumerate(bands, 1):
                dst.write(b, k); dst.set_band_description(k, feats[k - 1])

    # ---- stability
    tr = df[df.split == "train"].reset_index(drop=True)
    ranks = {"xgb_full": imp}
    for f in range(C.N_FOLDS):
        sub = tr[tr.fold != f]
        mf = clone(model).fit(sub[feats].to_numpy(), sub[a.target].to_numpy())
        ranks[f"xgb_fold{f}"] = importance(xgb_shap(mf, X[samp])[0], feats)
    # random-forest model family with native TreeSHAP, using the tuned sklearn-RF feature fraction
    rf_path = out / "models" / "rf.joblib"
    mf_frac = joblib.load(rf_path)["params"].get("max_features", 0.33) if rf_path.exists() else 0.33
    rf = xgb.XGBRFRegressor(n_estimators=300, colsample_bynode=float(mf_frac), subsample=0.632, max_depth=12,
                            tree_method="hist", n_jobs=-1, random_state=C.SEED)
    rf.fit(tr[feats].to_numpy(), tr[a.target].to_numpy())
    ranks["xgbrf_full"] = importance(xgb_shap(rf, X[samp])[0], feats)
    R = pd.DataFrame(ranks)
    R.to_csv(sd / "stability_importance.csv")
    cor = pd.DataFrame({a_: {b_: spearmanr(R[a_], R[b_]).statistic for b_ in R} for a_ in R})
    cor.to_csv(sd / "stability_rank_spearman.csv")
    (sd / "summary.json").write_text(json.dumps({
        "features": feats, "target": a.target, "importance": imp.round(4).to_dict(),
        "min_rank_rho_vs_full": float(cor["xgb_full"].min()),
        "shap_method": "XGBoost native pred_contribs (path-dependent TreeSHAP)"}, indent=2))
    print("\nRank stability (Spearman rho vs full XGBoost):\n", cor["xgb_full"].round(3).to_string())


if __name__ == "__main__":
    main()
