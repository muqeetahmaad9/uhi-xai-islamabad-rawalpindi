"""Step 10 — paper figures for model performance and SHAP (reads saved outputs only).

Fig 10 model comparison (test metrics with block-bootstrap CIs; spatial vs random CV)
Fig 11 observed vs predicted and residual distribution (test blocks)
Fig 12 observed, predicted and residual LST maps
Fig 13 SHAP global importance + summary
Fig 14 SHAP dependence (top 6) with additive-model partial effects
Fig 15 SHAP maps for the top predictors
Fig 16 Islamabad vs Rawalpindi response comparison
Fig 17 local explanations for six representative cells
Table  outputs/tables/model_performance.csv

Usage:  python scripts/10_figures_results.py features_2023_2025 [--target lst]
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import figstyle as F  # noqa: E402
import modeling as M  # noqa: E402

NAMES = {"lr": "Linear regression", "gam": "Additive spline (GAM-type)", "rf": "Random Forest", "xgb": "XGBoost"}
LAB = {"ndvi": "NDVI", "bld_frac": "Building cover", "dw_built": "Built-up prob. (DW)", "dw_bare": "Bare ground prob. (DW)",
       "dw_crops": "Cropland prob. (DW)", "dist_water": "Distance to water (m)", "elev": "Elevation (m)",
       "slope": "Slope (°)", "cos_incidence": "cos(solar incidence)"}
TAB = C.OUTPUT_DIR / "tables"


def load(period, target):
    rd = M.run_dir(period, target)
    return {
        "rd": rd,
        "test": pd.read_csv(rd / "test_metrics.csv"),
        "cv": pd.read_csv(rd / "cv_folds.csv"),
        "pred": pd.read_parquet(rd / "predictions.parquet"),
        "shap": pd.read_parquet(rd / "shap" / "shap_values_xgb.parquet"),
        "imp": pd.read_csv(rd / "shap" / "importance_xgb.csv", index_col=0)["mean_abs_shap"],
        "local": pd.read_csv(rd / "shap" / "local_cases.csv"),
        "strata": pd.read_csv(rd / "shap" / "strata_dependence.csv"),
        "params": json.loads((rd / "best_params.json").read_text()),
    }


def table(d):
    t = d["test"].copy()
    cv = d["cv"].groupby(["model", "scheme"])["rmse"].mean().unstack()
    rows = []
    for _, r in t.iterrows():
        ci_r = json.loads(r.rmse_ci) if isinstance(r.rmse_ci, str) else r.rmse_ci
        ci_2 = json.loads(r.r2_ci) if isinstance(r.r2_ci, str) else r.r2_ci
        rows.append({"model": NAMES[r.model], "test_R2": r.r2, "R2_CI": f"{ci_2[0]:.2f}–{ci_2[1]:.2f}",
                     "test_RMSE": r.rmse, "RMSE_CI": f"{ci_r[0]:.2f}–{ci_r[1]:.2f}", "test_MAE": r.mae,
                     "bias": r.bias, "spatialCV_RMSE": cv.loc[r.model, "spatial_block"],
                     "randomCV_RMSE": cv.loc[r.model, "random"],
                     "random_understates_%": 100 * (1 - cv.loc[r.model, "random"] / cv.loc[r.model, "spatial_block"]),
                     "residual_MoransI": r.morans_i})
    out = pd.DataFrame(rows)
    out.round(3).to_csv(TAB / "model_performance.csv", index=False)
    return out


def fig10(d, t):
    order = ["lr", "gam", "rf", "xgb"]
    te = d["test"].set_index("model").loc[order]
    cv = d["cv"].groupby(["model", "scheme"])["rmse"].agg(["mean", "std"]).unstack()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.arange(len(order)); cols = ["#8c8c8c", "#dd8452", "#55a868", "#4c72b0"]
    for ax, met, lab in [(axs[0], "rmse", "Test RMSE (°C)"), (axs[1], "r2", "Test R²")]:
        ci = np.array([json.loads(v) if isinstance(v, str) else v for v in te[f"{met}_ci"]])
        ax.bar(x, te[met], color=cols, edgecolor="k", lw=0.4)
        ax.errorbar(x, te[met], yerr=[te[met] - ci[:, 0], ci[:, 1] - te[met]], fmt="none", ecolor="k", capsize=3, lw=0.8)
        for i, v in enumerate(te[met]):
            ax.text(i, v / 2, f"{v:.2f}", ha="center", color="w", fontsize=8, fontweight="bold")
        ax.set_xticks(x, ["Linear\nregression", "Additive\nspline", "Random\nForest", "XGBoost"], fontsize=7)
        ax.set_ylabel(lab); ax.grid(axis="y", lw=0.3)
    axs[0].set_title("(a) Error on held-out test blocks", loc="left"); axs[1].set_title("(b) Explained variance", loc="left")
    axs[1].set_ylim(0, 1)
    ax = axs[2]
    ax.bar(x - 0.2, cv[("mean", "random")].loc[order], 0.4, color="#c7c7c7", edgecolor="k", lw=0.4, label="Random 5-fold CV")
    ax.bar(x + 0.2, cv[("mean", "spatial_block")].loc[order], 0.4, color="#4c72b0", edgecolor="k", lw=0.4,
           yerr=cv[("std", "spatial_block")].loc[order], capsize=3, label="Spatial-block CV (± SD)")
    ax.plot(x, te["rmse"], "D", color="#c44e52", ms=5, label="Test blocks")
    ax.set_xticks(x, [NAMES[m].split(" (")[0].replace(" ", "\n", 1) for m in order], fontsize=7)
    ax.set_ylabel("RMSE (°C)"); ax.legend(fontsize=6, loc="upper right"); ax.grid(axis="y", lw=0.3)
    ax.set_title("(c) Random vs spatial validation", loc="left")
    fig.tight_layout()
    F.footer(fig, f"Test set: {int(te['n'].iloc[0]):,} cells in 28 held-out 5 km blocks; error bars in (a,b) are 95% block-bootstrap "
                  "intervals (1,000 resamples of blocks). CV on 241,591 training cells with tuned hyper-parameters.")
    return F.save(fig, "fig10_model_comparison")


def fig11(d):
    p = d["pred"]; te = p[p.split == "test"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    y = te[C.TARGET].to_numpy(); lim = np.percentile(y, [0.5, 99.5])
    for ax, m in zip(axs[:2], ["rf", "xgb"]):
        yp = te[f"pred_{m}"].to_numpy()
        hb = ax.hexbin(y, yp, gridsize=70, bins="log", mincnt=1, cmap="viridis", extent=(*lim, *lim))
        ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        r = d["test"].set_index("model").loc[m]
        ax.text(0.04, 0.96, f"R² = {r.r2:.2f}\nRMSE = {r.rmse:.2f} °C\nMAE = {r.mae:.2f} °C\nbias = {r.bias:+.2f} °C",
                transform=ax.transAxes, va="top", fontsize=7, bbox=dict(fc="w", ec="0.7"))
        ax.set_xlabel("Observed LST (°C)"); ax.set_ylabel("Predicted LST (°C)")
        ax.set_title(f"({'ab'[m == 'xgb']}) {NAMES[m]}", loc="left")
    fig.colorbar(hb, ax=axs[1], shrink=0.8).set_label("cells (log)")
    ax = axs[2]
    for m, c in [("lr", "#8c8c8c"), ("gam", "#dd8452"), ("xgb", "#4c72b0")]:
        e = te[f"pred_{m}"] - y
        ax.hist(e, bins=np.linspace(-8, 8, 81), histtype="step", lw=1.2, color=c, density=True, label=NAMES[m].split(" (")[0])
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("Residual: predicted − observed (°C)"); ax.set_ylabel("density")
    ax.legend(fontsize=6); ax.set_title("(c) Test residual distribution", loc="left")
    fig.tight_layout()
    F.footer(fig, "Held-out test blocks only (never used for screening, tuning or fitting).")
    return F.save(fig, "fig11_observed_vs_predicted")


def fig12(d, period):
    p = d["pred"].copy(); p["resid"] = p["pred_xgb"] - p[C.TARGET]
    ref = C.EXPORT_DIR / f"{period}.tif"
    obs, transform, _ = M.to_raster(p, C.TARGET, ref); pre, _, _ = M.to_raster(p, "pred_xgb", ref)
    res, _, _ = M.to_raster(p, "resid", ref)
    tst = np.full(obs.shape, np.nan); te = p[p.split == "test"]; tst[te.row, te.col] = 1
    ext = (transform.c, transform.c + transform.a * obs.shape[1], transform.f + transform.e * obs.shape[0], transform.f)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.1))
    lo, hi = np.nanpercentile(obs, [1, 99])
    for k, (ax, v, t) in enumerate(zip(axs[:2], [obs, pre], ["(a) Observed LST", "(b) Predicted LST (XGBoost)"])):
        im = ax.imshow(v, cmap="inferno", vmin=lo, vmax=hi, extent=ext); F.decorate(ax, ext, arrow=(k == 0))
        ax.set_title(t, loc="left")
    fig.colorbar(im, ax=axs[:2], shrink=0.7, pad=0.01).set_label("LST (°C)")
    lim = np.nanpercentile(np.abs(res), 98)
    im = axs[2].imshow(res, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), extent=ext)
    axs[2].contour(np.nan_to_num(tst), levels=[0.5], colors="k", linewidths=0.5, extent=ext, origin="upper")
    F.decorate(axs[2], ext, arrow=False)
    fig.colorbar(im, ax=axs[2], shrink=0.7).set_label("Predicted − observed (°C)")
    axs[2].set_title("(c) Residuals (black outlines: test blocks)", loc="left")
    F.footer(fig, "XGBoost fitted on training blocks; predictions shown for all cells. Residuals in training blocks are in-sample "
                  "and smaller than in test blocks (outlined). May–June 2023–2025. " + F.CRS_NOTE)
    return F.save(fig, "fig12_prediction_maps")


def fig13(d):
    imp = d["imp"]; feats = list(imp.index); s = d["shap"]; p = d["pred"]
    tab = M.load_table(d["period"])
    X = tab[feats].to_numpy(); sv = s[[f"shap_{f}" for f in feats]].to_numpy()
    rng = np.random.default_rng(C.SEED); samp = rng.choice(len(sv), 12000, replace=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    axs[0].barh(range(len(feats))[::-1], imp.values, color="#4c72b0")
    axs[0].set_yticks(range(len(feats))[::-1], [LAB[f] for f in feats], fontsize=7)
    for i, v in enumerate(imp.values):
        axs[0].text(v + 0.02, len(feats) - 1 - i, f"{v:.2f}", va="center", fontsize=6.5)
    axs[0].set_xlabel("Mean |SHAP| (°C)"); axs[0].set_title("(a) Global importance", loc="left")
    ax = axs[1]; cmap = plt.get_cmap("coolwarm")
    for r, f in enumerate(feats[::-1]):
        j = feats.index(f); v = sv[samp, j]; col = pd.Series(X[samp, j]).rank(pct=True).to_numpy()
        h, e = np.histogram(v, 80); dn = h[np.clip(np.digitize(v, e) - 1, 0, len(h) - 1)] / h.max()
        ax.scatter(v, r + (rng.random(len(v)) - 0.5) * 0.8 * dn, c=col, cmap=cmap, s=1.5, lw=0, alpha=0.6, rasterized=True)
    ax.axvline(0, color="0.4", lw=0.6); ax.set_yticks(range(len(feats)), [LAB[f] for f in feats[::-1]], fontsize=7)
    ax.set_xlabel("SHAP value: contribution to predicted LST (°C)")
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap), ax=ax, ticks=[0, 1], pad=0.02)
    cb.ax.set_yticklabels(["low", "high"]); cb.set_label("Predictor value (percentile)")
    ax.set_title("(b) SHAP summary (12,000 random cells)", loc="left")
    fig.tight_layout()
    F.footer(fig, "XGBoost, native path-dependent TreeSHAP (Lundberg et al., 2020) for all 312,130 cells; base value = mean "
                  "prediction. Importance = mean absolute SHAP over all cells.")
    return F.save(fig, "fig13_shap_global")


def gam_partial(gam, X, j, grid):
    Xg = np.repeat(X.mean(0, keepdims=True), len(grid), 0); Xg[:, j] = grid
    Xd = np.repeat(X.mean(0, keepdims=True), len(X), 0); Xd[:, j] = X[:, j]
    return gam.predict(Xg) - gam.predict(Xd).mean()


def fig14(d):
    imp = d["imp"]; feats = d["params"]["features"]; top = list(imp.index[:6])
    tab = M.load_table(d["period"]); X = tab[feats].to_numpy()
    s = d["shap"]; gam = joblib.load(d["rd"] / "models" / "gam.joblib")["model"]
    rng = np.random.default_rng(C.SEED); samp = rng.choice(len(X), 15000, replace=False)
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.4))
    for k, (ax, f) in enumerate(zip(axs.ravel(), top)):
        j = feats.index(f); x = X[samp, j]; v = s[f"shap_{f}"].to_numpy()[samp]
        ax.scatter(x, v, s=1.5, alpha=0.25, color="#4c72b0", lw=0, rasterized=True)
        g = np.linspace(*np.percentile(X[:, j], [1, 99]), 100)
        ax.plot(g, gam_partial(gam, X, j, g), color="#c44e52", lw=1.5)
        ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(LAB[f]); ax.set_ylabel("SHAP (°C)")
        ax.set_title(f"({chr(97 + k)})", loc="left")
    axs[0, 0].plot([], [], "o", color="#4c72b0", ms=3, label="SHAP, XGBoost (cells)")
    axs[0, 0].plot([], [], color="#c44e52", label="Additive spline partial effect")
    axs[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    F.footer(fig, "Blue: SHAP values of 15,000 random cells. Red: partial effect of the same predictor in the additive spline "
                  "model (centred; other predictors at their mean), an independent check on response shape.")
    return F.save(fig, "fig14_shap_dependence")


def fig15(d, period):
    imp = d["imp"]; top = list(imp.index[:4]); s = d["shap"]; ref = C.EXPORT_DIR / f"{period}.tif"
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.7))
    for k, (ax, f) in enumerate(zip(axs, top)):
        arr, transform, _ = M.to_raster(s, f"shap_{f}", ref)
        ext = (transform.c, transform.c + transform.a * arr.shape[1], transform.f + transform.e * arr.shape[0], transform.f)
        lim = np.nanpercentile(np.abs(arr), 99)
        im = ax.imshow(arr, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), extent=ext)
        F.decorate(ax, ext, arrow=(k == 0))
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).set_label("SHAP (°C)")
        ax.set_title(f"({chr(97 + k)}) {LAB[f]}", loc="left")
    F.footer(fig, "Spatial distribution of each predictor's contribution to predicted LST (XGBoost TreeSHAP). Red: raises "
                  "predicted LST relative to the study-area mean prediction; blue: lowers it. " + F.CRS_NOTE)
    return F.save(fig, "fig15_shap_maps")


def fig16(d):
    st = d["strata"]; feats = list(st.feature.unique())
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2))
    for k, (ax, f) in enumerate(zip(axs.ravel(), feats)):
        for s_, c in [("Islamabad", "#55a868"), ("Rawalpindi", "#dd8452")]:
            q = st[(st.feature == f) & (st.stratum == s_)]
            ax.plot(q.bin_center, q.mean_shap, "-o", ms=3, color=c, label=f"{s_} tehsil")
        ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(LAB[f]); ax.set_ylabel("Mean SHAP (°C)")
        ax.set_title(f"({chr(97 + k)})", loc="left")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    F.footer(fig, "Mean SHAP value in 10 equal-width bins spanning the predictor range common to both tehsils "
                  "(5th–95th percentile overlap). Similar curves indicate similar response functions.")
    return F.save(fig, "fig16_strata_comparison")


def fig17(d, period):
    loc = d["local"]; feats = d["params"]["features"]
    B, ext, _ = F.load_stack(period)
    fig = plt.figure(figsize=(13, 8.2))
    axm = fig.add_axes([0.0, 0.08, 0.3, 0.84])
    hs = F.hillshade(B["elev"]); axm.imshow(hs, cmap="gray", extent=ext, vmin=-0.2, vmax=1)
    axm.imshow(B["lst"], cmap="inferno", alpha=0.55, extent=ext); F.decorate(axm, ext)
    for i, r in loc.iterrows():
        axm.plot(r.x, r.y, "o", ms=9, mfc="w", mec="k"); axm.text(r.x, r.y, str(i + 1), ha="center", va="center", fontsize=7)
    axm.set_title("Representative cells (on LST)", loc="left")
    for i, r in loc.iterrows():
        ax = fig.add_axes([0.36 + (i % 3) * 0.215, 0.56 - (i // 3) * 0.47, 0.15, 0.36])
        vals = np.array([r[f] for f in feats]); o = np.argsort(-np.abs(vals)); cur = r.base_value
        path = [cur]
        for k, j in enumerate(o):
            v = vals[j]
            ax.barh(len(o) - 1 - k, v, left=cur, color="#d7191c" if v > 0 else "#2c7bb6", height=0.65)
            cur += v
            path.append(cur)
        pad = 0.08 * (max(path) - min(path) + 1e-6)
        ax.set_xlim(min(path) - pad, max(path) + pad)
        ax.set_yticks(range(len(o)), [LAB[feats[j]].split(" (")[0] for j in o][::-1], fontsize=6)
        ax.axvline(r.base_value, color="0.5", ls="--", lw=0.6); ax.axvline(cur, color="k", lw=0.8)
        ax.tick_params(axis="x", labelsize=6); ax.set_xlabel("LST (°C)", fontsize=6)
        ax.set_title(f"{i + 1}. {r.case}\nobs {r.observed:.1f} · pred {r.predicted:.1f} °C", fontsize=7, loc="left")
    F.footer(fig, "Cells chosen by fixed rules before inspecting SHAP values (median prediction within each group). Bars run "
                  "from the mean prediction (dashed) to the cell's prediction (solid); red raises, blue lowers LST.")
    return F.save(fig, "fig17_local_explanations")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("period"); ap.add_argument("--target", default=C.TARGET)
    a = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    d = load(a.period, a.target); d["period"] = a.period
    print(table(d).round(3).to_string(index=False))
    for f in (fig10(d, None), fig11(d), fig12(d, a.period), fig13(d), fig14(d), fig15(d, a.period), fig16(d), fig17(d, a.period)):
        print("wrote", f)


if __name__ == "__main__":
    main()
