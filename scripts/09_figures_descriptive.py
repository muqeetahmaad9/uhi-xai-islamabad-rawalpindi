"""Step 9 — descriptive statistics, surface UHI and hotspot figures (no model results).

Fig 6  LST by land-cover class and by tehsil
Fig 7  LST vs each selected predictor (hexbin + binned medians, Spearman rho)
Fig 8  LST per year 2023 / 2024 / 2025 + inter-annual consistency
Fig 9  Surface UHI: urban / rural-reference masks, SUHII per year with sensitivity, Gi* hotspots
Fig S1 validation design: variogram + spatial blocks

Tables (outputs/tables/): descriptive_by_landcover.csv, descriptive_by_tehsil.csv,
suhii.csv, suhii_sensitivity.csv, hotspot_summary.csv

Usage:  python scripts/09_figures_descriptive.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import convolve
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import figstyle as F  # noqa: E402
import modeling as M  # noqa: E402

POOLED = f"features_{C.YEARS[0]}_{C.YEARS[-1]}"
YEARS = [f"features_{y}" for y in C.YEARS]
TAB = C.OUTPUT_DIR / "tables"
WC = list(C.WC_CLASSES.values())
WC_NAMES = {"tree": "Tree cover", "shrub": "Shrubland", "grass": "Grassland", "crop": "Cropland",
            "built": "Built-up", "bare": "Bare / sparse", "water": "Water", "wetland": "Wetland"}
WC_COLORS = {"tree": "#006400", "shrub": "#ffbb22", "grass": "#ffff4c", "crop": "#f096ff",
             "built": "#fa0000", "bare": "#b4b4b4", "water": "#0064c8", "wetland": "#0096a0"}
LABEL = {"ndvi": "NDVI", "bld_frac": "Building cover fraction", "dw_built": "Built-up probability (DW)",
         "dw_bare": "Bare-ground probability (DW)", "dw_crops": "Cropland probability (DW)",
         "dist_water": "Distance to water (m)", "elev": "Elevation (m)", "slope": "Slope (°)",
         "cos_incidence": "cos(solar incidence)", "bld_height": "Building height (m)"}


def dominant_class(df):
    return pd.Series(np.array(WC)[np.argmax(df[[f"wc_{c}" for c in WC]].to_numpy(), axis=1)], index=df.index)


# ------------------------------------------------------------------ Fig 6
def fig6_landcover_tehsil(df):
    df = df.assign(lc=dominant_class(df))
    classes = [c for c in WC if (df.lc == c).sum() >= 200]
    desc = df.groupby("lc")["lst"].describe(percentiles=[0.25, 0.5, 0.75]).loc[classes]
    desc.to_csv(TAB / "descriptive_by_landcover.csv")
    dt = df.groupby("stratum_name")["lst"].describe(percentiles=[0.25, 0.5, 0.75])
    dt.to_csv(TAB / "descriptive_by_tehsil.csv")

    order = desc["50%"].sort_values().index.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    data = [df.loc[df.lc == c, "lst"] for c in order]
    bp = axs[0].boxplot(data, showfliers=False, patch_artist=True, widths=0.6, medianprops=dict(color="k"))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(WC_COLORS[c]); patch.set_alpha(0.85)
    axs[0].set_xticks(range(1, len(order) + 1),
                      [f"{WC_NAMES[c]}\n(n={int(desc.loc[c, 'count']):,})" for c in order], fontsize=7)
    axs[0].set_ylabel("LST (°C)"); axs[0].set_title("(a) LST by dominant land-cover class", loc="left")
    axs[0].grid(axis="y", lw=0.3)

    groups = []
    for s in ("Islamabad", "Rawalpindi"):
        d = df[df.stratum_name == s]
        groups += [(f"{s}\nall", d["lst"]), (f"{s}\nbuilt-up", d.loc[d.lc == "built", "lst"])]
    bp = axs[1].boxplot([g[1] for g in groups], showfliers=False, patch_artist=True, widths=0.6,
                        medianprops=dict(color="k"))
    for patch, (lab, _) in zip(bp["boxes"], groups):
        patch.set_facecolor("#55a868" if lab.startswith("Islamabad") else "#dd8452")
    axs[1].set_xticks(range(1, len(groups) + 1), [g[0] for g in groups], fontsize=7)
    axs[1].set_title("(b) LST by tehsil", loc="left"); axs[1].grid(axis="y", lw=0.3)
    fig.tight_layout()
    F.footer(fig, "Boxes: interquartile range and median; whiskers 1.5×IQR; outliers not shown. LST: Landsat 8/9 May–June "
                  "2023–2025 median, 90 m cells (open water excluded). Land cover: dominant ESA WorldCover v200 class per cell.")
    return F.save(fig, "fig06_lst_landcover_tehsil"), desc, dt


# ------------------------------------------------------------------ Fig 7
def fig7_bivariate(df, feats):
    nc = 3; nr = int(np.ceil(len(feats) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(10, 3.0 * nr))
    rows = []
    for ax, f in zip(axs.ravel(), feats):
        x, y = df[f].to_numpy(), df["lst"].to_numpy()
        lo, hi = np.percentile(x, [0.5, 99.5])
        ax.hexbin(x, y, gridsize=55, bins="log", cmap="Greys", mincnt=1, extent=(lo, hi, *np.percentile(y, [0.2, 99.8])))
        edges = np.unique(np.percentile(x[(x >= lo) & (x <= hi)], np.linspace(0, 100, 21)))
        b = np.digitize(x, edges[1:-1])
        med = pd.Series(y).groupby(b).median(); ctr = pd.Series(x).groupby(b).median()
        q1 = pd.Series(y).groupby(b).quantile(0.25); q3 = pd.Series(y).groupby(b).quantile(0.75)
        ax.fill_between(ctr, q1, q3, color="#c44e52", alpha=0.25, lw=0)
        ax.plot(ctr, med, "-", color="#c44e52", lw=1.5)
        rho = spearmanr(x, y).statistic
        rows.append({"feature": f, "spearman_rho_all_cells": rho})
        ax.set_title(f"{LABEL.get(f, f)}   ρ = {rho:+.2f}", fontsize=8, loc="left")
        ax.set_ylabel("LST (°C)", fontsize=7); ax.tick_params(labelsize=6)
    for ax in axs.ravel()[len(feats):]:
        ax.axis("off")
    fig.tight_layout()
    F.footer(fig, "Grey: cell density (log scale). Red line and band: median and interquartile range of LST in 20 "
                  "quantile bins of the predictor. ρ: Spearman rank correlation over all 90 m cells (descriptive, "
                  "spatially autocorrelated; not a significance test). May–June 2023–2025.")
    pd.DataFrame(rows).to_csv(TAB / "bivariate_spearman_all_cells.csv", index=False)
    return F.save(fig, "fig07_bivariate")


# ------------------------------------------------------------------ Fig 8
def fig8_years():
    stacks = [F.load_stack(p) for p in YEARS]
    ext = stacks[0][1]
    L = [s[0]["lst"] for s in stacks]
    allv = np.concatenate([v[~np.isnan(v)] for v in L])
    lo, hi = np.percentile(allv, [1, 99])
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for k, (ax, v, y) in enumerate(zip(axs[:3], L, C.YEARS)):
        im = ax.imshow(v, cmap="inferno", extent=ext, vmin=lo, vmax=hi)
        F.decorate(ax, ext, arrow=(k == 0))
        ax.set_title(f"({chr(97 + k)}) May–June {y} median LST", loc="left")
    fig.colorbar(im, ax=axs[:3], shrink=0.75, pad=0.01).set_label("LST (°C)")
    # inter-annual consistency of the spatial pattern: rank correlation between years (cell level)
    ok = np.all([~np.isnan(v) for v in L], axis=0)
    flat = [v[ok] for v in L]
    pairs = [(0, 1), (0, 2), (1, 2)]
    rhos = {f"{C.YEARS[a]}–{C.YEARS[b]}": spearmanr(flat[a], flat[b]).statistic for a, b in pairs}
    anom = [f - np.median(f) for f in flat]
    ax = axs[3]
    rng = np.random.default_rng(C.SEED); s = rng.choice(len(anom[0]), 20000, replace=False)
    ax.scatter(anom[0][s], anom[2][s], s=1, alpha=0.2, color="k")
    lim = np.percentile(np.r_[anom[0], anom[2]], [0.5, 99.5]); ax.plot(lim, lim, "r--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"{C.YEARS[0]} LST − median (°C)"); ax.set_ylabel(f"{C.YEARS[2]} LST − median (°C)")
    ax.set_title("(d) Pattern consistency", loc="left")
    ax.text(0.03, 0.97, "\n".join(f"ρ {k} = {v:.2f}" for k, v in rhos.items()), transform=ax.transAxes,
            va="top", fontsize=7, bbox=dict(fc="w", ec="0.7"))
    F.footer(fig, "Per-year medians of clear Landsat 8/9 acquisitions (2023: 5, 2024: 7, 2025: 6 dates). (d) cell-level "
                  "deviations from each year's study-area median, random sample of 20,000 cells; ρ = Spearman rank "
                  "correlation over all cells. " + F.CRS_NOTE)
    pd.Series(rhos).rename("spearman_rho").to_csv(TAB / "interannual_rank_correlation.csv")
    return F.save(fig, "fig08_lst_by_year"), rhos


# ------------------------------------------------------------------ SUHI + hotspots
def suhi(df, urban_built, rural_built, elev_tol):
    urban = df["wc_built"] >= urban_built
    zmed = df.loc[urban, "elev"].median()
    rural = (df["wc_built"] < rural_built) & ((df["elev"] - zmed).abs() <= elev_tol)
    return urban, rural, zmed


def suhi_tables():
    rows, sens = [], []
    for p in YEARS + [POOLED]:
        df = M.load_table(p)
        u, r, zmed = suhi(df, C.SUHI_URBAN_BUILT, C.SUHI_RURAL_BUILT, C.SUHI_RURAL_ELEV_TOL_M)
        lc = dominant_class(df)
        row = {"period": p, "urban_cells": int(u.sum()), "rural_cells": int(r.sum()), "urban_median_elev_m": zmed,
               "lst_urban_mean": df.loc[u, "lst"].mean(), "lst_rural_mean": df.loc[r, "lst"].mean()}
        row["suhii_c"] = row["lst_urban_mean"] - row["lst_rural_mean"]
        for c in ("crop", "tree", "shrub", "grass", "bare"):
            m = r & (lc == c)
            row[f"suhii_vs_rural_{c}"] = row["lst_urban_mean"] - df.loc[m, "lst"].mean() if m.sum() >= 100 else np.nan
            row[f"rural_{c}_cells"] = int(m.sum())
        rows.append(row)
        for ub in C.SUHI_SENSITIVITY["urban_built"]:
            for et in C.SUHI_SENSITIVITY["elev_tol_m"]:
                u2, r2, _ = suhi(df, ub, C.SUHI_RURAL_BUILT, et)
                sens.append({"period": p, "urban_built": ub, "elev_tol_m": et,
                             "suhii_c": df.loc[u2, "lst"].mean() - df.loc[r2, "lst"].mean(),
                             "urban_cells": int(u2.sum()), "rural_cells": int(r2.sum())})
    t, s = pd.DataFrame(rows), pd.DataFrame(sens)
    t.to_csv(TAB / "suhii.csv", index=False); s.to_csv(TAB / "suhii_sensitivity.csv", index=False)
    return t, s


def gi_star(v, radius_m=C.HOTSPOT_RADIUS_M, res=C.GRID_RES):
    """Getis-Ord Gi* with a binary circular window (cell included), on a NaN-masked grid."""
    r = int(round(radius_m / res))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = ((xx ** 2 + yy ** 2) * res ** 2 <= radius_m ** 2).astype(float)
    ok = ~np.isnan(v); x = np.where(ok, v, 0.0)
    n = ok.sum(); xbar = x[ok].mean(); s = np.sqrt((x[ok] ** 2).mean() - xbar ** 2)
    wsum = convolve(ok.astype(float), k, mode="constant")          # sum of weights (= w^2 for binary)
    lx = convolve(x, k, mode="constant")
    den = s * np.sqrt((n * wsum - wsum ** 2) / (n - 1))
    z = np.where(ok, (lx - xbar * wsum) / den, np.nan)
    return z


def bh_fdr(p, q):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    thr = q * np.arange(1, m + 1) / m
    passed = p[o] <= thr
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool); out[o[:k]] = True
    return out


def fig9_suhi(t, s):
    B, ext, _ = F.load_stack(POOLED)
    df = M.load_table(POOLED)
    u, r, zmed = suhi(df, C.SUHI_URBAN_BUILT, C.SUHI_RURAL_BUILT, C.SUHI_RURAL_ELEV_TOL_M)
    cls = np.full(B["lst"].shape, np.nan)
    cls[df.row[~u & ~r], df.col[~u & ~r]] = 0
    cls[df.row[r], df.col[r]] = 1
    cls[df.row[u], df.col[u]] = 2

    z = gi_star(B["lst"])
    zz = z[~np.isnan(z)]
    p = 2 * norm.sf(np.abs(zz))
    sig = bh_fdr(p, C.HOTSPOT_FDR_Q)
    hot = np.full(z.shape, np.nan); vals = np.zeros(zz.shape)
    vals[sig & (zz > 0)] = 1; vals[sig & (zz < 0)] = -1
    hot[~np.isnan(z)] = vals
    # hotspot composition by land cover / tehsil
    df["gi_class"] = hot[df.row, df.col]
    df["lc"] = dominant_class(df)
    comp = (df.groupby("gi_class")["lc"].value_counts(normalize=True).unstack(fill_value=0) * 100).round(1)
    comp["cells"] = df.groupby("gi_class").size()
    comp.index = comp.index.map({-1.0: "cold spot", 0.0: "not significant", 1.0: "hot spot"})
    comp.to_csv(TAB / "hotspot_summary.csv")
    tehsil = (df.groupby("stratum_name")["gi_class"].value_counts(normalize=True).unstack(fill_value=0) * 100).round(1)
    tehsil.to_csv(TAB / "hotspot_by_tehsil.csv")

    fig = plt.figure(figsize=(14, 4.6))
    a1 = fig.add_axes([0.02, 0.16, 0.27, 0.78]); a2 = fig.add_axes([0.35, 0.2, 0.27, 0.68]); a3 = fig.add_axes([0.66, 0.16, 0.30, 0.78])
    a1.imshow(cls, cmap=ListedColormap(["#e6e6e6", "#4c72b0", "#c44e52"]), vmin=-0.5, vmax=2.5, extent=ext,
              interpolation="nearest")
    F.decorate(a1, ext)
    a1.legend(handles=[Patch(color="#c44e52", label=f"Urban (built ≥ {C.SUHI_URBAN_BUILT:.0%}; n={int(u.sum()):,})"),
                       Patch(color="#4c72b0", label=f"Rural reference (built < {C.SUHI_RURAL_BUILT:.0%}, "
                                                    f"elev. {zmed:.0f}±{C.SUHI_RURAL_ELEV_TOL_M} m; n={int(r.sum()):,})"),
                       Patch(color="#e6e6e6", label="Other")], fontsize=5.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    a1.set_title("(a) SUHI masks", loc="left")

    lab = [p.replace("features_", "").replace("_", "–") for p in t.period]
    xs = np.arange(len(t))
    a2.bar(xs - 0.2, t.suhii_c, 0.4, color="#c44e52", label="vs all rural reference")
    a2.bar(xs + 0.2, t.suhii_vs_rural_crop, 0.4, color="#f096ff", edgecolor="k", lw=0.3, label="vs rural cropland")
    for i, pp in enumerate(t.period):
        ss = s[s.period == pp].suhii_c
        a2.plot([i - 0.2, i - 0.2], [ss.min(), ss.max()], color="k", lw=1)
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xticks(xs, lab); a2.set_ylabel("SUHII = LST urban − LST rural (°C)")
    a2.set_title("(b) Daytime surface UHI intensity", loc="left"); a2.legend(fontsize=6); a2.grid(axis="y", lw=0.3)

    a3.imshow(hot, cmap=ListedColormap(["#2c7bb6", "#f0f0f0", "#d7191c"]), vmin=-1.5, vmax=1.5, extent=ext,
              interpolation="nearest")
    F.decorate(a3, ext)
    a3.legend(handles=[Patch(color="#d7191c", label="Hot spot"), Patch(color="#2c7bb6", label="Cold spot"),
                       Patch(color="#f0f0f0", label="Not significant")], fontsize=6, loc="upper center",
               bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
    a3.set_title(f"(c) LST hot/cold spots (Gi*, {C.HOTSPOT_RADIUS_M / 1000:.0f} km, FDR q={C.HOTSPOT_FDR_Q})", loc="left")
    F.footer(fig, "SUHII uses LST (surface UHI), not air temperature. Urban/rural from ESA WorldCover v200 built-up "
                  "fraction per 90 m cell; rural reference restricted to elevations near the urban median to limit "
                  "lapse-rate effects. Error bars in (b): range over urban threshold "
                  f"{C.SUHI_SENSITIVITY['urban_built']} × elevation tolerance {C.SUHI_SENSITIVITY['elev_tol_m']} m. "
                  "(c) Getis-Ord Gi* on 2023–2025 median LST, binary circular window; significance is approximate "
                  "because LST is spatially autocorrelated. " + F.CRS_NOTE)
    return F.save(fig, "fig09_suhi_hotspots"), comp, tehsil


# ------------------------------------------------------------------ Fig S1
def figS1_design():
    info = json.loads((C.OUTPUT_DIR / "models" / POOLED / "blocks.json").read_text())
    df = M.load_table(POOLED)
    B, ext, _ = F.load_stack(POOLED)
    fig = plt.figure(figsize=(11, 4.6))
    a1 = fig.add_axes([0.06, 0.14, 0.38, 0.76]); a2 = fig.add_axes([0.52, 0.06, 0.44, 0.88])
    img = plt.imread(C.OUTPUT_DIR / "models" / POOLED / "variogram.png")
    a1.imshow(img); a1.axis("off")
    a1.set_title(f"(a) Empirical variograms; block size {info['block_size_m'] / 1000:.0f} km", loc="left")
    g = np.full(B["lst"].shape, np.nan)
    g[df.row, df.col] = np.where(df.split == "test", -1, df.fold)
    cmap = ListedColormap(["#000000", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"])
    a2.imshow(g, cmap=cmap, vmin=-1.5, vmax=4.5, extent=ext, interpolation="nearest")
    F.decorate(a2, ext)
    n = info["cells_by_split"]
    a2.legend(handles=[Patch(color="#000000", label=f"Test blocks (n={n['test']:,} cells)")] +
                      [Patch(color=cmap.colors[f + 1], label=f"CV fold {f + 1}") for f in range(C.N_FOLDS)],
              fontsize=6, loc="lower right")
    a2.set_title(f"(b) {info['n_blocks']} spatial blocks: {info['n_test_blocks']} test, rest in {C.N_FOLDS} CV folds",
                 loc="left")
    F.footer(fig, "Block size from the practical range of linear-model residuals (rounded up to 500 m, capped at "
                  f"{C.BLOCK_SIZE_LIMITS_M[1] / 1000:.0f} km). Test blocks sampled within jurisdiction × urban/non-urban "
                  "strata and never used for screening or tuning. " + F.CRS_NOTE)
    return F.save(fig, "figS1_validation_design")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    df = M.load_table(POOLED)
    feats = json.loads((C.OUTPUT_DIR / "models" / POOLED / "selected_predictors.json").read_text())["selected"]
    p, desc, dt = fig6_landcover_tehsil(df); print("wrote", p); print(desc[["count", "mean", "50%"]].round(2).to_string())
    print(dt[["count", "mean", "50%"]].round(2).to_string())
    print("wrote", fig7_bivariate(df, feats))
    p, rhos = fig8_years(); print("wrote", p, {k: round(v, 3) for k, v in rhos.items()})
    t, s = suhi_tables()
    print(t[["period", "urban_cells", "rural_cells", "lst_urban_mean", "lst_rural_mean", "suhii_c",
             "suhii_vs_rural_crop", "suhii_vs_rural_tree"]].round(2).to_string(index=False))
    print("sensitivity range per period:\n", s.groupby("period").suhii_c.agg(["min", "max"]).round(2).to_string())
    p, comp, tehsil = fig9_suhi(t, s); print("wrote", p); print(comp.to_string()); print(tehsil.to_string())
    print("wrote", figS1_design())


if __name__ == "__main__":
    main()
