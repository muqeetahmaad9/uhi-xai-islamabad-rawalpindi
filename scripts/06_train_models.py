"""Step 6 — train, tune (spatial CV), compare and test the models.

Models: linear regression (baseline), additive spline model (GAM-type),
Random Forest, XGBoost. Hyper-parameters are chosen by mean RMSE over the
spatial CV folds on (a block-preserving sample of) the training blocks.
With the chosen settings, each model is then
  * cross-validated on the full training set with spatial folds AND with random
    5-fold CV (the gap quantifies optimism from spatial autocorrelation),
  * refitted on all training blocks and evaluated ONCE on the held-out test blocks
    (block-bootstrap 95 % CIs; Moran's I of test residuals).

Outputs (outputs/models/<period>_<target>/): tuning_<model>.csv, best_params.json,
cv_comparison.csv, test_metrics.csv, predictions.parquet, models/*.joblib, obs_vs_pred.png

Usage:  python scripts/06_train_models.py features_2023_2025 [--target lst_anom] [--models lr gam rf xgb]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import modeling as M  # noqa: E402

N_XGB_ITER = 25


def model_specs():
    return {
        "lr": (M.linear_model(), [{}]),
        "gam": (M.additive_spline_model(), list(ParameterGrid({
            "ridge__alpha": [0.1, 1, 10, 100], "splinetransformer__n_knots": [6, 10]}))),
        "rf": (RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=C.SEED), list(ParameterGrid({
            "max_features": [0.33, 0.6, 1.0], "min_samples_leaf": [1, 5, 20]}))),
        "xgb": (XGBRegressor(tree_method="hist", n_jobs=-1, random_state=C.SEED), list(ParameterSampler({
            "n_estimators": [300, 600, 1000], "learning_rate": [0.02, 0.05, 0.1],
            "max_depth": [3, 4, 6, 8, 10], "min_child_weight": [1, 5, 20],
            "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [1, 5, 10]}, n_iter=N_XGB_ITER, random_state=C.SEED))),
    }


def cv_score(est, X, y, folds):
    """Per-fold metrics for given (train_idx, val_idx) pairs."""
    res = []
    for k, (tr, va) in enumerate(folds):
        m = clone(est).fit(X[tr], y[tr])
        res.append({"fold": k, **M.metrics(y[va], m.predict(X[va]))})
    return pd.DataFrame(res)


def spatial_folds(fold_labels):
    return [(np.flatnonzero(fold_labels != f), np.flatnonzero(fold_labels == f))
            for f in range(C.N_FOLDS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("period")
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--models", nargs="*", default=["lr", "gam", "rf", "xgb"])
    a = ap.parse_args()

    feats = M.selected_predictors(a.period)
    df = M.load_table(a.period)
    out = M.run_dir(a.period, a.target)
    (out / "models").mkdir(exist_ok=True)
    tr, te = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    Xtr, ytr, ftr = tr[feats].to_numpy(), tr[a.target].to_numpy(), tr["fold"].to_numpy()
    Xte, yte = te[feats].to_numpy(), te[a.target].to_numpy()

    # block-preserving tuning sample: uniform cell sample keeps every block and fold represented
    rng = np.random.default_rng(C.SEED)
    tune_idx = np.sort(rng.choice(len(tr), size=min(C.TUNE_MAX_ROWS, len(tr)), replace=False))
    tfolds = spatial_folds(ftr[tune_idx])

    specs = {k: v for k, v in model_specs().items() if k in a.models}
    best, cv_rows, test_rows, preds = {}, [], [], df[["row", "col", "x", "y", "split", "block", "stratum",
                                                       a.target]].copy()
    for name, (base, grid) in specs.items():
        t0 = time.time()
        tuning = []
        for i, params in enumerate(grid):
            s = cv_score(clone(base).set_params(**params), Xtr[tune_idx], ytr[tune_idx], tfolds)
            tuning.append({"grid_idx": i, **params, "cv_rmse_mean": s["rmse"].mean(), "cv_rmse_sd": s["rmse"].std()})
        tuning = pd.DataFrame(tuning).sort_values("cv_rmse_mean")
        tuning.to_csv(out / f"tuning_{name}.csv", index=False)
        params = grid[int(tuning.iloc[0]["grid_idx"])]      # original dict keeps exact types
        est = clone(base).set_params(**params)
        best[name] = params

        sp = cv_score(est, Xtr, ytr, spatial_folds(ftr)).assign(scheme="spatial_block")
        rnd = cv_score(est, Xtr, ytr, list(KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED).split(Xtr)))
        cv = pd.concat([sp, rnd.assign(scheme="random")]).assign(model=name)
        cv_rows.append(cv)

        final = clone(est).fit(Xtr, ytr)
        joblib.dump({"model": final, "features": feats, "target": a.target, "params": params},
                    out / "models" / f"{name}.joblib")
        pte = final.predict(Xte)
        mt = {"model": name, **M.metrics(yte, pte), **M.block_bootstrap(yte, pte, te["block"])}
        mt.update(M.morans_i(pte - yte, te["x"], te["y"]))
        test_rows.append(mt)
        preds[f"pred_{name}"] = final.predict(df[feats].to_numpy())
        print(f"{name}: test RMSE {mt['rmse']:.3f}  R2 {mt['r2']:.3f}  "
              f"(spatial CV RMSE {sp['rmse'].mean():.3f}, random CV RMSE {rnd['rmse'].mean():.3f})  "
              f"[{time.time() - t0:.0f} s]")

    cv_all = pd.concat(cv_rows)
    cv_all.to_csv(out / "cv_folds.csv", index=False)
    summ = cv_all.groupby(["model", "scheme"])[["r2", "rmse", "mae"]].agg(["mean", "std"])
    summ.to_csv(out / "cv_comparison.csv")
    pd.DataFrame(test_rows).to_csv(out / "test_metrics.csv", index=False)
    preds.to_parquet(out / "predictions.parquet", index=False)
    (out / "best_params.json").write_text(json.dumps({"features": feats, "target": a.target,
                                                      "n_train": len(tr), "n_test": len(te),
                                                      "n_tune": len(tune_idx), "params": best},
                                                     indent=2, default=str))
    print("\n", summ.round(3).to_string())

    n = len(specs)
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.3), squeeze=False)
    lim = np.nanpercentile(yte, [0.5, 99.5])
    for ax, name in zip(axs[0], specs):
        p = preds.loc[preds["split"] == "test", f"pred_{name}"]
        ax.hexbin(yte, p, gridsize=60, bins="log", extent=(*lim, *lim), mincnt=1, cmap="viridis")
        ax.plot(lim, lim, "k--", lw=0.8)
        r = next(t for t in test_rows if t["model"] == name)
        ax.set_title(f"{name}: RMSE {r['rmse']:.2f}, R² {r['r2']:.2f}", fontsize=8)
        ax.set_xlabel(f"observed {a.target} (°C)", fontsize=8)
    axs[0][0].set_ylabel("predicted (°C)", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "obs_vs_pred.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
