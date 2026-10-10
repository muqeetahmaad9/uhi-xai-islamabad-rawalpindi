"""Step 11 — H4 (multi-source vs spectral indices only) and the building-height check.

All variants use XGBoost with the hyper-parameters tuned for the main model, are
fitted on the training blocks and evaluated on the same held-out test blocks.
Differences in test metrics get 95 % CIs from a PAIRED block bootstrap (identical
resampled blocks for every variant).

Variants
  A_indices        NDVI, NDBI, MNDWI, BSI (Sentinel-2 spectral indices only)
  B_indices_terrain A + elevation, slope, cos(solar incidence)
  C_full           the 9 screened predictors (main model)
  D_full_height    C + mean building height

Building-height diagnostics: Spearman rho(height, building cover) over all cells and
over cells with buildings only; SHAP share of height in built cells (variant D).

Outputs (outputs/models/<period>_<target>/extra/): variants_test.csv,
paired_differences.csv, height_diagnostics.json, fig_extra_h4_height.png

Usage:  python scripts/11_extra_tests.py features_2023_2025
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import figstyle as F  # noqa: E402
import modeling as M  # noqa: E402

BUILT_MIN_FRAC = 0.05   # "cells with buildings" for the height diagnostics


def paired_bootstrap(y, preds, blocks, n_boot=C.N_BOOTSTRAP, seed=C.SEED):
    """RMSE per variant on identical resampled blocks; returns {variant: array of RMSE}."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks); idx = {b: np.flatnonzero(blocks == b) for b in ub}
    out = {k: [] for k in preds}
    for _ in range(n_boot):
        take = np.concatenate([idx[b] for b in rng.choice(ub, len(ub), replace=True)])
        for k, p in preds.items():
            out[k].append(np.sqrt(np.mean((p[take] - y[take]) ** 2)))
    return {k: np.array(v) for k, v in out.items()}


def main():
    period = sys.argv[1]
    rd = M.run_dir(period, C.TARGET); ex = rd / "extra"; ex.mkdir(exist_ok=True)
    main_model = joblib.load(rd / "models" / "xgb.joblib")
    full = main_model["features"]
    variants = {
        "A_indices": ["ndvi", "ndbi", "mndwi", "bsi"],
        "B_indices_terrain": ["ndvi", "ndbi", "mndwi", "bsi", "elev", "slope", "cos_incidence"],
        "C_full": full,
        "D_full_height": full + ["bld_height"],
    }
    df = M.load_table(period)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    y = te[C.TARGET].to_numpy()

    rows, preds, models = [], {}, {}
    for name, feats in variants.items():
        t0 = time.time()
        m = main_model["model"] if name == "C_full" else clone(main_model["model"]).fit(tr[feats].to_numpy(),
                                                                                      tr[C.TARGET].to_numpy())
        p = m.predict(te[feats].to_numpy())
        preds[name], models[name] = p, (m, feats)
        rows.append({"variant": name, "n_predictors": len(feats), "predictors": ", ".join(feats),
                     **M.metrics(y, p), **M.block_bootstrap(y, p, te.block)})
        print(f"{name}: RMSE {rows[-1]['rmse']:.3f}  R2 {rows[-1]['r2']:.3f}  [{time.time() - t0:.0f} s]", flush=True)
    res = pd.DataFrame(rows); res.to_csv(ex / "variants_test.csv", index=False)

    boot = paired_bootstrap(y, preds, te.block.to_numpy())
    comps = [("C_full", "A_indices"), ("C_full", "B_indices_terrain"), ("B_indices_terrain", "A_indices"),
             ("D_full_height", "C_full")]
    diffs = []
    for a_, b_ in comps:
        d = boot[a_] - boot[b_]
        diffs.append({"comparison": f"{a_} minus {b_}", "rmse_diff": float(np.sqrt(np.mean((preds[a_] - y) ** 2))
                                                                         - np.sqrt(np.mean((preds[b_] - y) ** 2))),
                      "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
                      "share_boot_improved": float(np.mean(d < 0))})
    diffs = pd.DataFrame(diffs); diffs.to_csv(ex / "paired_differences.csv", index=False)

    # ---- building-height diagnostics
    built = df.bld_frac >= BUILT_MIN_FRAC
    mD, fD = models["D_full_height"]
    sub = df[built].sample(min(20000, int(built.sum())), random_state=C.SEED)
    contrib = mD.get_booster().predict(xgb.DMatrix(sub[fD].to_numpy()), pred_contribs=True)[:, :-1]
    imp_built = pd.Series(np.abs(contrib).mean(0), index=fD).sort_values(ascending=False)
    diag = {
        "rho_height_cover_all_cells": float(spearmanr(df.bld_height, df.bld_frac).statistic),
        "rho_height_cover_built_cells": float(spearmanr(df.bld_height[built], df.bld_frac[built]).statistic),
        "built_cell_threshold_bld_frac": BUILT_MIN_FRAC, "n_built_cells": int(built.sum()),
        "share_cells_zero_height": float((df.bld_height == 0).mean()),
        "mean_abs_shap_built_cells_variantD": imp_built.round(4).to_dict(),
        "height_rank_in_built_cells": int(list(imp_built.index).index("bld_height") + 1),
    }
    (ex / "height_diagnostics.json").write_text(json.dumps(diag, indent=2))

    # ---- figure
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    lab = {"A_indices": "A. Spectral\nindices", "B_indices_terrain": "B. Indices\n+ terrain",
           "C_full": "C. Full\nmulti-source", "D_full_height": "D. Full\n+ height"}
    cols = ["#c7c7c7", "#8da0cb", "#4c72b0", "#55a868"]
    for ax, met, yl in [(axs[0], "rmse", "Test RMSE (°C)"), (axs[1], "r2", "Test R²")]:
        ci = np.array([r[f"{met}_ci"] for r in rows])
        v = res[met].to_numpy(); x = np.arange(len(v))
        ax.bar(x, v, color=cols, edgecolor="k", lw=0.4)
        ax.errorbar(x, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt="none", ecolor="k", capsize=3, lw=0.8)
        for i, vv in enumerate(v):
            ax.text(i, vv / 2, f"{vv:.2f}", ha="center", color="w", fontsize=8, fontweight="bold")
        ax.set_xticks(x, [lab[k] for k in res.variant], fontsize=7); ax.set_ylabel(yl); ax.grid(axis="y", lw=0.3)
    axs[0].set_title("(a) Test error by predictor set", loc="left"); axs[1].set_title("(b) Explained variance", loc="left")
    axs[1].set_ylim(0, 1)
    ax = axs[2]
    yy = np.arange(len(diffs))[::-1]
    ax.errorbar(diffs.rmse_diff, yy, xerr=[diffs.rmse_diff - diffs.ci_low, diffs.ci_high - diffs.rmse_diff],
                fmt="o", color="k", capsize=3)
    ax.axvline(0, color="0.5", lw=0.7)
    ax.set_yticks(yy, ["C − A", "C − B", "B − A", "D − C"]); ax.set_xlabel("Δ test RMSE (°C), 95% paired CI")
    ax.set_title("(c) Paired differences (negative = better)", loc="left")
    fig.tight_layout()
    F.footer(fig, "XGBoost with the main model's tuned hyper-parameters for every set; same training and held-out test blocks. "
                  "CIs: block bootstrap (1,000 resamples), paired across sets in (c).")
    F.save(fig, "figS2_predictor_sets_height")

    print("\n", res[["variant", "n_predictors", "r2", "rmse", "mae"]].round(3).to_string(index=False))
    print("\n", diffs.round(3).to_string(index=False))
    print("\n", json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
