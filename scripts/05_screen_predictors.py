"""Step 5 — predictor screening on the TRAINING blocks only (test blocks untouched).

1. Spearman and Pearson correlation among candidates and with the target.
2. Greedy collinearity filter in config.CANDIDATE_PREDICTORS priority order:
   a candidate is kept only if |Spearman rho| < CORR_THRESHOLD with every kept one.
3. Iterative VIF: drop the lowest-priority variable among those with VIF > VIF_THRESHOLD.

Outputs (outputs/models/<period>/): correlation_spearman.csv, correlation_pearson.csv,
vif.csv, screening_log.csv, selected_predictors.json, correlation_heatmap.png

Usage:  python scripts/05_screen_predictors.py features_2023_2025
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import modeling as M  # noqa: E402


def vif(X):
    Z = (X - X.mean()) / X.std()
    out = {}
    for c in Z.columns:
        others = Z.drop(columns=c).to_numpy()
        A = np.c_[np.ones(len(Z)), others]
        coef, *_ = np.linalg.lstsq(A, Z[c].to_numpy(), rcond=None)
        r2 = 1 - np.var(Z[c].to_numpy() - A @ coef) / np.var(Z[c].to_numpy())
        out[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out)


def main():
    period = sys.argv[1]
    df = M.load_table(period)
    if "split" not in df:
        raise SystemExit("Run 04_spatial_blocks.py first.")
    tr = df[df["split"] == "train"]
    out = M.run_dir(period)
    cand = C.CANDIDATE_PREDICTORS
    cols = [C.TARGET] + cand

    sp = tr[cols].corr("spearman")
    pe = tr[cols].corr("pearson")
    sp.to_csv(out / "correlation_spearman.csv")
    pe.to_csv(out / "correlation_pearson.csv")

    log, kept = [], []
    for c in cand:
        clash = [(k, sp.loc[c, k]) for k in kept if abs(sp.loc[c, k]) >= C.CORR_THRESHOLD]
        if clash:
            k, r = max(clash, key=lambda t: abs(t[1]))
            log.append({"variable": c, "action": "dropped (collinearity)", "detail": f"rho={r:.2f} with {k}"})
        else:
            kept.append(c)
            log.append({"variable": c, "action": "kept (correlation step)", "detail": ""})

    while True:
        v = vif(tr[kept])
        high = [c for c in kept if v[c] > C.VIF_THRESHOLD]
        if not high:
            break
        drop = max(high, key=kept.index)          # lowest priority among offenders
        log.append({"variable": drop, "action": "dropped (VIF)", "detail": f"VIF={v[drop]:.2f}"})
        kept.remove(drop)
    v = vif(tr[kept])
    v.rename("vif").to_csv(out / "vif.csv")
    pd.DataFrame(log).to_csv(out / "screening_log.csv", index=False)
    (out / "selected_predictors.json").write_text(json.dumps({
        "selected": kept, "target": C.TARGET, "corr_threshold": C.CORR_THRESHOLD,
        "vif_threshold": C.VIF_THRESHOLD, "n_train_cells": int(len(tr)),
        "rho_with_target": sp.loc[kept, C.TARGET].round(3).to_dict(),
        "vif": v.round(2).to_dict()}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sp.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{sp.iat[i, j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, label="Spearman ρ (training blocks)")
    fig.tight_layout(); fig.savefig(out / "correlation_heatmap.png", dpi=200); plt.close(fig)

    print(pd.DataFrame(log).to_string(index=False))
    print("\nSelected:", kept)
    print(v.round(2).to_string())


if __name__ == "__main__":
    main()
