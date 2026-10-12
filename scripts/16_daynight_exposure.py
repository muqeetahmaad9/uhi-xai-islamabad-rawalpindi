"""Step 16 — day vs night surface UHI (MODIS) and population exposure (GHS-POP / WorldPop).

Needs data/gee_exports/context_<years>.tif from scripts/15_export_context.py.

1. SUHII for Landsat (daytime, ~10:40 PKT) and MODIS Terra/Aqua day and night (1 km),
   pre-monsoon and all-year, under the paper's urban/rural definition and three
   alternatives (1 km-neighbourhood masks; no elevation matching; 5 km rural ring),
   with 95% block-bootstrap CIs over the 5 km validation blocks.
2. Exposure: population-weighted vs area-mean LST, population share in the hottest
   LST quintile, population in Landsat Gi* hot/cold spots, and LST by population density.

Fig 18 and tables daynight_suhii.csv, exposure_summary.csv, exposure_density_bins.csv.

Usage:  python scripts/16_daynight_exposure.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import figstyle as F  # noqa: E402
import modeling as M  # noqa: E402

POOLED = f"features_{C.YEARS[0]}_{C.YEARS[-1]}"
CTX = f"context_{C.YEARS[0]}_{C.YEARS[-1]}"
TAB = C.OUTPUT_DIR / "tables"
MIN_MODIS_OBS = 10
NEIGH_CELLS = 11                     # 11 x 90 m ~ 1 km window, matches the MODIS footprint
RING_KM = 5
SOURCES = {"lst": "Landsat 8/9 day (~10:40)", "mod_day": "MODIS Terra day (~10:30)",
           "myd_day": "MODIS Aqua day (~13:30)", "mod_night": "MODIS Terra night (~22:30)",
           "myd_night": "MODIS Aqua night (~01:30)"}

_spec = importlib.util.spec_from_file_location("s09", Path(__file__).with_name("09_figures_descriptive.py"))
S09 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S09)


def load_context(df):
    man = json.loads((C.OUTPUT_DIR / f"exports_manifest_{CTX}.json").read_text())
    with rasterio.open(C.EXPORT_DIR / f"{CTX}.tif") as s:
        a = s.read(masked=True).astype("float64").filled(np.nan)
    with rasterio.open(C.EXPORT_DIR / f"{POOLED}.tif") as s:
        if (s.height, s.width) != a.shape[1:]:
            raise SystemExit(f"grid mismatch: {a.shape[1:]} vs {(s.height, s.width)}")
    for i, b in enumerate(man["bands"]):
        df[b] = a[i][df.row, df.col]
    for k in [c for c in man["bands"] if c.startswith(("mod_", "myd_"))]:
        df.loc[df[f"n_{k}"] < MIN_MODIS_OBS, k] = np.nan
    return df, a.shape[1:]


def grid_of(df, col, shape):
    g = np.full(shape, np.nan); g[df.row, df.col] = df[col].to_numpy(); return g


def masks(df, shape):
    """Urban / rural boolean Series for each definition."""
    out = {}
    u, r, zmed = S09.suhi(df, C.SUHI_URBAN_BUILT, C.SUHI_RURAL_BUILT, C.SUHI_RURAL_ELEV_TOL_M)
    out["paper (90 m, elev-matched)"] = (u, r)
    b = np.nan_to_num(grid_of(df, "wc_built", shape)); ok = (~np.isnan(grid_of(df, "wc_built", shape))).astype(float)
    nb = uniform_filter(b, NEIGH_CELLS, mode="constant") / np.maximum(uniform_filter(ok, NEIGH_CELLS, mode="constant"), 1e-9)
    nbs = pd.Series(nb[df.row, df.col], index=df.index)
    u1 = nbs >= C.SUHI_URBAN_BUILT
    z1 = df.loc[u1, "elev"].median()
    out["1 km neighbourhood, elev-matched"] = (u1, (nbs < C.SUHI_RURAL_BUILT) & ((df.elev - z1).abs() <= C.SUHI_RURAL_ELEV_TOL_M))
    out["no elevation matching"] = (u, df.wc_built < C.SUHI_RURAL_BUILT)
    ug = np.zeros(shape, bool); ug[df.row[u], df.col[u]] = True
    dist_km = distance_transform_edt(~ug) * C.GRID_RES / 1000
    ring = pd.Series(dist_km[df.row, df.col], index=df.index) <= RING_KM
    out[f"{RING_KM} km rural ring"] = (u, ring & (df.wc_built < C.SUHI_RURAL_BUILT))
    return out


def suhii_ci(df, col, u, r, n_boot=C.N_BOOTSTRAP, seed=C.SEED):
    v = df[col]; okv = v.notna()
    g = pd.DataFrame({"block": df.block, "us": v.where(u & okv, 0), "un": (u & okv).astype(int),
                      "rs": v.where(r & okv, 0), "rn": (r & okv).astype(int)}).groupby("block").sum()
    est = g.us.sum() / g.un.sum() - g.rs.sum() / g.rn.sum()
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(g), (n_boot, len(g)))
    a = g.to_numpy()
    s = a[idx].sum(axis=1)
    boot = s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return est, lo, hi, int((u & okv).sum()), int((r & okv).sum())


def suhii_table(df, M_):
    rows = []
    for mname, (u, r) in M_.items():
        for period in ("pm", "yr"):
            for src, label in SOURCES.items():
                col = src if src == "lst" else f"{src}_{period}"
                if src == "lst" and period == "yr":
                    continue                                   # Landsat stack is pre-monsoon only
                est, lo, hi, nu, nr = suhii_ci(df, col, u, r)
                rows.append({"definition": mname, "period": {"pm": "May–Jun", "yr": "all year"}[period],
                             "source": label, "suhii_c": est, "ci_lo": lo, "ci_hi": hi,
                             "urban_cells": nu, "rural_cells": nr})
    t = pd.DataFrame(rows); t.to_csv(TAB / "daynight_suhii.csv", index=False)
    return t


def gi_classes(df, shape):
    z = S09.gi_star(grid_of(df, "lst", shape)); zz = z[~np.isnan(z)]
    sig = S09.bh_fdr(2 * norm.sf(np.abs(zz)), C.HOTSPOT_FDR_Q)
    vals = np.zeros(zz.shape); vals[sig & (zz > 0)] = 1; vals[sig & (zz < 0)] = -1
    hot = np.full(shape, np.nan); hot[~np.isnan(z)] = vals
    return pd.Series(hot[df.row, df.col], index=df.index)


def exposure(df, shape):
    rows = []
    cols = {"lst": "lst", "mod_day": "mod_day_pm", "myd_day": "myd_day_pm",
            "mod_night": "mod_night_pm", "myd_night": "myd_night_pm"}
    for pop in ("pop_ghs2025", "pop_wp2020"):
        P = df[pop].clip(lower=0)
        for src, col in cols.items():
            ok = df[col].notna(); v = df.loc[ok, col]; w = P[ok]
            q80 = v.quantile(0.8)
            rows.append({"population": pop, "source": SOURCES[src], "area_mean_c": v.mean(),
                         "pop_weighted_mean_c": np.average(v, weights=w),
                         "pop_share_hottest_quintile_pct": 100 * w[v >= q80].sum() / w.sum(),
                         "pop_share_coolest_quintile_pct": 100 * w[v <= v.quantile(0.2)].sum() / w.sum(),
                         "spearman_rho_vs_density": spearmanr(v, w).statistic})
        gi = gi_classes(df, shape)
        for k, name in ((1, "Landsat Gi* hot spots"), (-1, "Landsat Gi* cold spots")):
            m = gi == k
            rows.append({"population": pop, "source": name, "persons": P[m].sum(),
                         "pop_share_pct": 100 * P[m].sum() / P.sum(), "area_share_pct": 100 * m.mean()})
        rows.append({"population": pop, "source": "total in analysed cells", "persons": P.sum()})
    t = pd.DataFrame(rows); t.to_csv(TAB / "exposure_summary.csv", index=False)

    dens = df.pop_ghs2025.clip(lower=0) / (C.GRID_RES ** 2 / 1e4)           # persons per ha
    edges = [-0.1, 1, 10, 25, 50, 100, 200, 400, np.inf]
    lab = ["<1", "1–10", "10–25", "25–50", "50–100", "100–200", "200–400", ">400"]
    b = pd.cut(dens, edges, labels=lab)
    d = pd.DataFrame({"bin": b})
    for src, col in cols.items():
        d[src] = df[col] - df[col].mean()                                     # anomaly vs AOI mean
    db = d.groupby("bin", observed=False).agg(["mean", "count"])
    db.columns = [f"{a}_{s}" for a, s in db.columns]
    db.to_csv(TAB / "exposure_density_bins.csv")
    return t, db, lab


def figure(df, shape, st, db, lab):
    B, ext, _ = F.load_stack(POOLED)
    fig = plt.figure(figsize=(13, 9.6))
    a1 = fig.add_axes([0.03, 0.53, 0.40, 0.42]); a2 = fig.add_axes([0.52, 0.56, 0.45, 0.38])
    a3 = fig.add_axes([0.06, 0.07, 0.40, 0.36]); a4 = fig.add_axes([0.56, 0.07, 0.41, 0.36])

    night = grid_of(df, "myd_night_pm", shape)
    lo, hi = np.nanpercentile(night, [1, 99])
    im = a1.imshow(night, cmap="inferno", vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
    F.decorate(a1, ext)
    fig.colorbar(im, ax=a1, fraction=0.04, pad=0.01).set_label("Night LST (°C)")
    a1.set_title("(a) MODIS Aqua night LST (~01:30), May–Jun 2023–2025, 1 km", loc="left")

    s = st[st.period == "May–Jun"]
    defs = [d for d in dict.fromkeys(s.definition) if "ring" not in d]   # ring == no-elev here (table)
    srcs = list(SOURCES.values())
    w = 0.8 / len(defs); x = np.arange(len(srcs))
    for i, dname in enumerate(defs):
        q = s[s.definition == dname].set_index("source").reindex(srcs)
        a2.bar(x + (i - (len(defs) - 1) / 2) * w, q.suhii_c, w, yerr=[q.suhii_c - q.ci_lo, q.ci_hi - q.suhii_c],
               capsize=2, label=dname, error_kw=dict(lw=0.6))
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xticks(x, [t.replace(" (", "\n(") for t in srcs], fontsize=7)
    a2.set_ylabel("SUHII (°C), urban − rural"); a2.legend(fontsize=6.5, frameon=False)
    a2.set_title("(b) Surface UHI by time of day and urban/rural definition (95% block-bootstrap CI)", loc="left")

    for src, mk in zip(SOURCES, "osD^v"):
        a3.plot(range(len(lab)), db[f"{src}_mean"], marker=mk, ms=4, lw=1, label=SOURCES[src])
    a3.axhline(0, color="k", lw=0.6)
    a3.set_xticks(range(len(lab)), lab); a3.set_xlabel("Population density, GHS-POP 2025 (persons ha⁻¹)")
    a3.set_ylabel("LST anomaly vs study-area mean (°C)"); a3.legend(fontsize=6.5, frameon=False)
    a3.set_title("(c) LST by population density", loc="left")

    e = df.pop_ghs2025.clip(lower=0)
    shares = {}
    for src in ("lst", "myd_night"):
        col = "lst" if src == "lst" else "myd_night_pm"; ok = df[col].notna()
        qn = pd.qcut(df.loc[ok, col], 5, labels=["Q1\ncoolest", "Q2", "Q3", "Q4", "Q5\nhottest"])
        shares[SOURCES[src]] = 100 * e[ok].groupby(qn, observed=False).sum() / e[ok].sum()
    sh = pd.DataFrame(shares)
    sh.plot.bar(ax=a4, rot=0, color=["#d95f02", "#3b3b8f"], width=0.75)
    a4.axhline(20, color="0.4", ls="--", lw=0.8); a4.text(2.5, 21, "equal share (20%)", fontsize=6.5, ha="center", color="0.3")
    a4.set_ylabel("Share of population (%)"); a4.set_xlabel("Area-based LST quintile"); a4.legend(fontsize=6.5, frameon=False)
    a4.set_title("(d) Where people live: share of population per LST quintile, day vs night", loc="left")

    F.footer(fig, "MODIS LST v061 (MOD11A1/MYD11A1), QA ≤ 1 and error ≤ 2 K, ≥ 10 valid observations per pixel, sampled "
                  "to the 90 m grid by nearest neighbour (1 km support). Population: GHS-POP R2023A epoch 2025. "
                  "Urban/rural definitions as in Section 2; '1 km neighbourhood' uses the 11 × 11-cell mean built-up fraction. "
                  + F.CRS_NOTE)
    return F.save(fig, "fig18_daynight_exposure")


def main():
    df = M.load_table(POOLED)
    df, shape = load_context(df)
    M_ = masks(df, shape)
    st = suhii_table(df, M_)
    pd.set_option("display.width", 200)
    print(st.round(2).to_string(index=False))
    t, db, lab = exposure(df, shape)
    print(t.round(2).to_string(index=False)); print(db.round(2).to_string())
    print("wrote", figure(df, shape, st, db, lab))


if __name__ == "__main__":
    main()
