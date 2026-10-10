"""Step 13 — sensitivity: exclude the Margalla Hills / uplands (terrain-confounding test).

Refits XGBoost (main model's predictors and tuned hyper-parameters, same spatial
blocks) on cells below an elevation cut-off, and compares with the main model on the
SAME reduced test cells: test metrics, SHAP importance (same 20,000 plains cells for
both models) and the NDVI / elevation response curves.

Cut-offs: 650 m (primary: above the 95th percentile of urban-cell elevation, 592 m)
and 600 m (stricter).

Outputs: outputs/models/<pooled>_lst/hills/ (tables) and outputs/figures/figS4_hills_excluded.png
Usage:  python scripts/13_sensitivity_hills.py
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
CUTS = [650, 600]
LAB = {"ndvi": "NDVI", "bld_frac": "Building cover", "dw_built": "Built-up (DW)", "dw_bare": "Bare ground (DW)",
       "dw_crops": "Cropland (DW)", "dist_water": "Dist. to water", "elev": "Elevation", "slope": "Slope",
       "cos_incidence": "cos(solar inc.)"}


def shap_imp(model, X):
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    return c, np.abs(c).mean(0)


def binned(x, v, edges):
    b = np.digitize(x, edges[1:-1])
    return pd.Series(v).groupby(b).mean().reindex(range(len(edges) - 1))


def main():
    rd = M.run_dir(POOLED, C.TARGET); out = rd / "hills"; out.mkdir(exist_ok=True)
    mm = joblib.load(rd / "models" / "xgb.joblib"); feats, main_model = mm["features"], mm["model"]
    df = M.load_table(POOLED)

    rows, imps, curves = [], {}, {}
    for cut in CUTS:
        d = df[df.elev < cut]
        tr, te = d[d.split == "train"], d[d.split == "test"]
        m = clone(main_model).fit(tr[feats].to_numpy(), tr[C.TARGET].to_numpy())
        y = te[C.TARGET].to_numpy()
        for name, model in [(f"main model (all cells)", main_model), (f"refit < {cut} m", m)]:
            p = model.predict(te[feats].to_numpy())
            rows.append({"cutoff_m": cut, "model": name, "cells_removed": int((df.elev >= cut).sum()),
                         "share_removed": float((df.elev >= cut).mean()), **M.metrics(y, p),
                         **M.block_bootstrap(y, p, te.block)})
        s = d.sample(20000, random_state=C.SEED)
        Xs = s[feats].to_numpy()
        c_main, i_main = shap_imp(main_model, Xs)
        c_cut, i_cut = shap_imp(m, Xs)
        imps[f"main ({cut} m sample)"] = pd.Series(i_main, index=feats)
        imps[f"refit < {cut} m"] = pd.Series(i_cut, index=feats)
        for f in ("ndvi", "elev", "dw_bare"):
            j = feats.index(f)
            edges = np.linspace(*np.percentile(Xs[:, j], [2, 98]), 13)
            ctr = (edges[:-1] + edges[1:]) / 2
            curves[(cut, f)] = (ctr, binned(Xs[:, j], c_main[:, j], edges), binned(Xs[:, j], c_cut[:, j], edges))
        print(f"cut {cut} m: removed {rows[-1]['cells_removed']:,} cells ({rows[-1]['share_removed']:.1%}); "
              f"refit test RMSE {rows[-1]['rmse']:.3f}, main on same cells {rows[-2]['rmse']:.3f}", flush=True)

    res = pd.DataFrame(rows); res.to_csv(out / "hills_test.csv", index=False)
    I = pd.DataFrame(imps); I.to_csv(out / "hills_importance.csv")
    rho = {cut: spearmanr(I[f"main ({cut} m sample)"], I[f"refit < {cut} m"]).statistic for cut in CUTS}
    pd.Series(rho, name="rank_rho_main_vs_refit").to_csv(out / "hills_rank_rho.csv")

    # ---- figure (primary cut-off)
    cut = CUTS[0]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    order = I[f"refit < {cut} m"].sort_values(ascending=False).index; x = np.arange(len(order))
    axs[0].bar(x - 0.2, I.loc[order, f"main ({cut} m sample)"], 0.4, color="#4c72b0", label="Main model (all cells)")
    axs[0].bar(x + 0.2, I.loc[order, f"refit < {cut} m"], 0.4, color="#dd8452", label=f"Refit without cells ≥ {cut} m")
    axs[0].set_xticks(x, [LAB[f] for f in order], rotation=30, ha="right", fontsize=7)
    axs[0].set_ylabel("Mean |SHAP| (°C), cells < 650 m"); axs[0].legend(fontsize=7); axs[0].grid(axis="y", lw=0.3)
    axs[0].set_title(f"(a) Importance (rank ρ = {rho[cut]:.2f})", loc="left")
    for k, f in enumerate(("ndvi", "elev", "dw_bare")):
        ctr, a, b = curves[(cut, f)]; ax = axs[k + 1]
        ax.plot(ctr, a.values, "-o", ms=3, color="#4c72b0", label="Main model")
        ax.plot(ctr, b.values, "-o", ms=3, color="#dd8452", label=f"Refit < {cut} m")
        ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(LAB[f]); ax.set_ylabel("Mean SHAP (°C)")
        ax.set_title(f"({'bcd'[k]}) {LAB[f]} response", loc="left")
    axs[1].legend(fontsize=6.5)
    fig.tight_layout()
    F.footer(fig, f"Cells at ≥ {cut} m ({res.share_removed.iloc[0]:.1%} of cells; mostly Margalla Hills forest) removed from "
                  "training and testing. Both models explained on the same 20,000 cells below the cut-off; curves are "
                  "binned mean SHAP values.")
    F.save(fig, "figS4_hills_excluded")

    print("\n", res[["cutoff_m", "model", "r2", "rmse", "mae", "bias"]].round(3).to_string(index=False))
    print("\nImportance:\n", I.loc[order].round(3).to_string())
    print("\nRank rho main vs refit:", {k: round(v, 3) for k, v in rho.items()})
    for cut in CUTS:
        ctr, a, b = curves[(cut, "ndvi")]
        print(f"NDVI response span (max-min of binned SHAP), cut {cut}: main {a.max() - a.min():.2f} °C, "
              f"refit {b.max() - b.min():.2f} °C")


if __name__ == "__main__":
    main()
