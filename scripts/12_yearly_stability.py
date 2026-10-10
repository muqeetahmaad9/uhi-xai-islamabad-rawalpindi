"""Step 12 — temporal stability: one XGBoost model per year (2023, 2024, 2025).

Same 9 predictors, same tuned hyper-parameters and same spatial blocks/test set as
the pooled main model (blocks matched by grid row/col). For each year:
test metrics (block-bootstrap CIs) and SHAP importance on a fixed 20,000-cell sample.
Also a transfer matrix: model trained on year i, evaluated on test blocks of year j.

Outputs (outputs/models/<pooled>_<target>/yearly/): yearly_test.csv, yearly_importance.csv,
yearly_rank_spearman.csv, transfer_rmse.csv;  figure outputs/figures/figS3_yearly_stability.png

Usage:  python scripts/12_yearly_stability.py
"""
import sys
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

POOLED = f"features_{C.YEARS[0]}_{C.YEARS[-1]}"
LAB = {"ndvi": "NDVI", "bld_frac": "Building cover", "dw_built": "Built-up (DW)", "dw_bare": "Bare ground (DW)",
       "dw_crops": "Cropland (DW)", "dist_water": "Dist. to water", "elev": "Elevation", "slope": "Slope",
       "cos_incidence": "cos(solar inc.)"}


def main():
    rd = M.run_dir(POOLED, C.TARGET); out = rd / "yearly"; out.mkdir(exist_ok=True)
    main_model = joblib.load(rd / "models" / "xgb.joblib"); feats = main_model["features"]
    splits = pd.read_parquet(C.TABLE_DIR / f"{POOLED}_splits.parquet")

    tabs = {}
    for y in C.YEARS:
        t = pd.read_parquet(C.TABLE_DIR / f"features_{y}.parquet").merge(splits, on=["row", "col"], how="inner")
        tabs[y] = t
    common = set.intersection(*[set(zip(t.row, t.col)) for t in tabs.values()])
    rng = np.random.default_rng(C.SEED)
    samp_rc = pd.DataFrame(list(common), columns=["row", "col"]).sample(20000, random_state=C.SEED)

    models, rows, imps = {}, [], {}
    pooled_tab = M.load_table(POOLED)
    s_pooled = pooled_tab.merge(samp_rc, on=["row", "col"])
    contrib = main_model["model"].get_booster().predict(xgb.DMatrix(s_pooled[feats].to_numpy()), pred_contribs=True)
    imps["pooled 2023–25"] = pd.Series(np.abs(contrib[:, :-1]).mean(0), index=feats)

    for y, t in tabs.items():
        tr, te = t[t.split == "train"], t[t.split == "test"]
        m = clone(main_model["model"]).fit(tr[feats].to_numpy(), tr[C.TARGET].to_numpy())
        models[y] = m
        p = m.predict(te[feats].to_numpy())
        rows.append({"year": y, "n_train": len(tr), "n_test": len(te), **M.metrics(te[C.TARGET], p),
                     **M.block_bootstrap(te[C.TARGET].to_numpy(), p, te.block)})
        s = t.merge(samp_rc, on=["row", "col"])
        c = m.get_booster().predict(xgb.DMatrix(s[feats].to_numpy()), pred_contribs=True)
        imps[str(y)] = pd.Series(np.abs(c[:, :-1]).mean(0), index=feats)
        print(f"{y}: test RMSE {rows[-1]['rmse']:.3f}  R2 {rows[-1]['r2']:.3f}", flush=True)

    res = pd.DataFrame(rows); res.to_csv(out / "yearly_test.csv", index=False)
    I = pd.DataFrame(imps); I.to_csv(out / "yearly_importance.csv")
    R = pd.DataFrame({a: {b: spearmanr(I[a], I[b]).statistic for b in I} for a in I}); R.to_csv(out / "yearly_rank_spearman.csv")
    top3 = {k: list(I[k].sort_values(ascending=False).index[:3]) for k in I}

    T = pd.DataFrame(index=C.YEARS, columns=C.YEARS, dtype=float)
    for yi, m in models.items():
        for yj, t in tabs.items():
            te = t[t.split == "test"]
            T.loc[yi, yj] = np.sqrt(np.mean((m.predict(te[feats].to_numpy()) - te[C.TARGET].to_numpy()) ** 2))
    T.to_csv(out / "transfer_rmse.csv")

    # ---- figure
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.8, 1]})
    order = I["pooled 2023–25"].sort_values(ascending=False).index
    x = np.arange(len(order)); w = 0.2
    cols = {"pooled 2023–25": "#4c72b0", "2023": "#c7e9b4", "2024": "#41b6c4", "2025": "#225ea8"}
    for k, (name, c) in enumerate(cols.items()):
        axs[0].bar(x + (k - 1.5) * w, I.loc[order, name], w, color=c, edgecolor="k", lw=0.3, label=name)
    axs[0].set_xticks(x, [LAB[f] for f in order], rotation=30, ha="right", fontsize=7)
    axs[0].set_ylabel("Mean |SHAP| (°C)"); axs[0].legend(fontsize=7); axs[0].grid(axis="y", lw=0.3)
    rmin = R.values[np.triu_indices(len(R), 1)].min()
    axs[0].set_title(f"(a) Predictor importance by year (rank ρ ≥ {rmin:.2f})", loc="left")
    im = axs[1].imshow(T.values.astype(float), cmap="viridis_r")
    for i in range(3):
        for j in range(3):
            axs[1].text(j, i, f"{T.values[i, j]:.2f}", ha="center", va="center",
                        color="w" if T.values[i, j] > T.values.mean() else "k", fontsize=8)
    axs[1].set_xticks(range(3), [str(y) for y in C.YEARS]); axs[1].set_yticks(range(3), [str(y) for y in C.YEARS])
    axs[1].set_xlabel("Evaluated on test blocks of year"); axs[1].set_ylabel("Trained on year")
    fig.colorbar(im, ax=axs[1], shrink=0.8).set_label("Test RMSE (°C)")
    axs[1].set_title("(b) Temporal transfer", loc="left")
    fig.tight_layout()
    F.footer(fig, "XGBoost per year with the pooled model's predictors, hyper-parameters and spatial test blocks. "
                  "SHAP importance on the same 20,000 cells for every model. Diagonal in (b): within-year test error.")
    F.save(fig, "figS3_yearly_stability")

    print("\n", res[["year", "r2", "rmse", "mae", "bias"]].round(3).to_string(index=False))
    print("\nImportance:\n", I.loc[order].round(3).to_string())
    print("\nRank rho:\n", R.round(3).to_string())
    print("\nTop 3:", top3)
    print("\nTransfer RMSE (rows trained, cols evaluated):\n", T.round(3).to_string())


if __name__ == "__main__":
    main()
