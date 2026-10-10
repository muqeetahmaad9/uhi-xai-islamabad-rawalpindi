"""Step 8 — paper figures describing the study area and input data (no model results).

Fig 1 study area · Fig 2 Landsat availability · Fig 3 predictor maps
Fig 4 land cover (ESA WorldCover dominant class) · Fig 5 pre-monsoon LST

Usage:  python scripts/08_figures_inputs.py features_2023_2025
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import figstyle as F  # noqa: E402

PERIOD_TXT = "May–June 2023–2025"


def fig1_study_area(B, ext):
    fig, ax = plt.subplots(figsize=(6.6, 5.9))
    hs = F.hillshade(B["elev"])
    ax.imshow(hs, cmap="gray", extent=ext, vmin=-0.2, vmax=1, zorder=1)
    im = ax.imshow(B["elev"], cmap="terrain", extent=ext, alpha=0.55, zorder=2,
                   vmin=np.nanpercentile(B["elev"], 1), vmax=np.nanpercentile(B["elev"], 99.5))
    F.decorate(ax, ext)
    g = F.aoi_utm()
    for _, r in g.iterrows():
        p = r.geometry.representative_point()
        ax.text(p.x, p.y, f"{C.STRATUM_NAMES[r.stratum]} tehsil", ha="center", fontsize=9, fontweight="bold",
                color="k", zorder=11, bbox=dict(fc="w", ec="none", alpha=0.6, pad=1.5))
    # Rawalpindi city: densest building cover (3 km moving mean) inside the Rawalpindi tehsil
    dens = uniform_filter(np.nan_to_num(B["bld_frac"]), size=33)
    dens[B["stratum"] != 2] = -1
    i, j = np.unravel_index(np.argmax(dens), dens.shape)
    cx, cy = ext[0] + (j + 0.5) * C.GRID_RES, ext[3] - (i + 0.5) * C.GRID_RES
    ax.plot(cx, cy, "o", ms=4, mfc="w", mec="k", zorder=11)
    ax.annotate("Rawalpindi city", (cx, cy), xytext=(cx - 8000, cy - 6000), textcoords="data",
                fontsize=7, style="italic", zorder=11, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5), bbox=dict(fc="w", ec="none", alpha=0.6, pad=1))
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02); cb.set_label("Elevation (m a.s.l.)")
    ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")

    # Journal policy: no location map larger than the study area; give coordinates instead.
    c = gpd.read_file(C.AOI_FILE).to_crs(C.CRS).dissolve().centroid.to_crs(4326).iloc[0]
    ax.text(0.98, 0.02, f"Centre: {c.y:.2f}° N, {c.x:.2f}° E\nIslamabad Capital Territory and\nRawalpindi District, Pakistan",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, zorder=11,
            bbox=dict(fc="w", ec="0.6", lw=0.5, pad=2))
    F.footer(fig, "Study area: Islamabad and Rawalpindi tehsils (geoBoundaries PAK ADM3, commit 9469f09; ODbL 1.0), "
                  "2,522 km². Elevation: Copernicus DEM GLO-30 (2024_1) aggregated to 90 m; hillshade az 315°, alt 45°. "
                  "Map lines delineate study areas and do not necessarily depict accepted national boundaries. "
                  + F.CRS_NOTE)
    return F.save(fig, "fig01_study_area")


def fig2_availability(B, ext):
    d = pd.read_csv(C.OUTPUT_DIR / "availability" / "dates_all_seasons.csv")
    d = d[d.season == "premonsoon"].sort_values("date").reset_index(drop=True)
    d["used"] = d.aoi_clear_fraction >= C.MIN_AOI_CLEAR_FRACTION
    fig = plt.figure(figsize=(9, 3.6))
    ax = fig.add_axes([0.06, 0.2, 0.52, 0.7])
    x = np.arange(len(d))
    ax.bar(x, d.aoi_clear_fraction * 100, color=np.where(d.used, "#4c72b0", "#c7c7c7"), edgecolor="k", lw=0.3)
    for i, r in d.iterrows():
        ax.text(i, r.aoi_clear_fraction * 100 + 2, "L8" if r.spacecraft == "LANDSAT_8" else "L9",
                ha="center", fontsize=5.5)
    ax.axhline(C.MIN_AOI_CLEAR_FRACTION * 100, color="k", ls="--", lw=0.6)
    ax.set_xticks(x, d.date, rotation=90, fontsize=5.5); ax.set_ylim(0, 112)
    ax.set_ylabel("Clear-sky share of study area (%)")
    ax.set_title("(a) Landsat 8/9 acquisitions, May–June (WRS-2 path 150)", loc="left")
    ax.legend(handles=[Patch(color="#4c72b0", label=f"used (≥{C.MIN_AOI_CLEAR_FRACTION:.0%} clear)"),
                       Patch(color="#c7c7c7", label="excluded")], fontsize=6, loc="lower right")
    for yr in d.date.str[:4].unique()[1:]:
        ax.axvline(d.index[d.date.str.startswith(yr)][0] - 0.5, color="0.5", lw=0.5)

    ax2 = fig.add_axes([0.62, 0.08, 0.36, 0.86])
    n = B["lst_n_obs"]
    vals = np.arange(np.nanmin(n).round(), np.nanmax(n).round() + 1)
    cmap = plt.get_cmap("viridis", len(vals))
    im = ax2.imshow(np.round(n), cmap=cmap, extent=ext, vmin=vals[0] - 0.5, vmax=vals[-1] + 0.5)
    F.decorate(ax2, ext, bar_km=10)
    cb = fig.colorbar(im, ax=ax2, shrink=0.7, ticks=vals); cb.set_label("Clear observations per cell")
    ax2.set_title("(b) Clear Landsat observations, 2023–2025", loc="left")
    F.footer(fig, f"Landsat Collection 2 Level-2 ST (USGS); QA_PIXEL fill/cloud/dilated cloud/cirrus/shadow/snow masked; "
                  f"ST_QA uncertainty > {C.MAX_ST_UNCERTAINTY_K} K masked. Same-day tiles mosaicked. " + F.CRS_NOTE)
    return F.save(fig, "fig02_landsat_availability")


PRED = [  # band, label, cmap, percentile clip
    ("ndvi", "NDVI", "YlGn", (2, 98)),
    ("bld_frac", "Building cover fraction", "magma_r", (0, 99)),
    ("dw_built", "Built-up probability (DW)", "Reds", (2, 98)),
    ("dw_bare", "Bare-ground probability (DW)", "YlOrBr", (2, 98)),
    ("dw_crops", "Cropland probability (DW)", "PuRd", (2, 98)),
    ("dist_water", "Distance to permanent water (m)", "Blues_r", (0, 100)),
    ("elev", "Elevation (m)", "terrain", (1, 99.5)),
    ("slope", "Slope (°)", "copper_r", (1, 99)),
    ("cos_incidence", "cos(solar incidence angle)", "gray", (1, 99)),
]


def fig3_predictors(B, ext, selected):
    rows = [p for p in PRED if p[0] in selected]
    nc = 3; nr = int(np.ceil(len(rows) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(9.5, 3.3 * nr))
    for k, (ax, (b, lab, cm, pc)) in enumerate(zip(axs.ravel(), rows)):
        v = B[b]; lo, hi = np.nanpercentile(v, pc)
        im = ax.imshow(v, cmap=cm, extent=ext, vmin=lo, vmax=hi)
        F.km_axes(ax, ext); F.outline(ax, lw=0.4)
        if k == 0:
            F.scalebar(ax, ext, 10000); F.north_arrow(ax)
        cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02); cb.ax.tick_params(labelsize=6)
        ax.set_title(f"({chr(97 + k)}) {lab}", loc="left", fontsize=8)
    for ax in axs.ravel()[len(rows):]:
        ax.axis("off")
    fig.tight_layout()
    F.footer(fig, f"Sentinel-2 L2A (Cloud Score+ cs_cdf ≥ {C.CS_THRESHOLD}) median NDVI, {PERIOD_TXT}; Dynamic World V1 "
                  f"mean class probabilities, {PERIOD_TXT}; Google Open Buildings 2.5D Temporal ({C.OB_YEAR}, presence > "
                  f"{C.OB_PRESENCE_THRESHOLD}); JRC Global Surface Water v1.4 (occurrence ≥ {C.GSW_OCCURRENCE_THRESHOLD}%, "
                  f"capped at {C.WATER_DISTANCE_CAP_M / 1000:.0f} km); Copernicus DEM GLO-30. Solar incidence at mean "
                  "overpass sun geometry. All aggregated to 90 m. " + F.CRS_NOTE)
    return F.save(fig, "fig03_predictors")


WC_COLORS = {"tree": "#006400", "shrub": "#ffbb22", "grass": "#ffff4c", "crop": "#f096ff",
             "built": "#fa0000", "bare": "#b4b4b4", "water": "#0064c8", "wetland": "#0096a0"}
WC_NAMES = {"tree": "Tree cover", "shrub": "Shrubland", "grass": "Grassland", "crop": "Cropland",
            "built": "Built-up", "bare": "Bare / sparse", "water": "Permanent water", "wetland": "Herbaceous wetland"}


def fig4_landcover(B, ext):
    cls = list(C.WC_CLASSES.values())
    stack = np.stack([B[f"wc_{c}"] for c in cls])
    inside = ~np.isnan(stack[0])
    dom = np.full(stack.shape[1:], np.nan)
    dom[inside] = np.argmax(np.nan_to_num(stack[:, inside]), axis=0)
    share = {c: float(np.nansum(B[f"wc_{c}"]) / inside.sum() * 100) for c in cls}
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = ListedColormap([WC_COLORS[c] for c in cls])
    ax.imshow(dom, cmap=cmap, norm=BoundaryNorm(np.arange(len(cls) + 1) - 0.5, len(cls)), extent=ext,
              interpolation="nearest")
    F.decorate(ax, ext)
    ax.legend(handles=[Patch(color=WC_COLORS[c], label=f"{WC_NAMES[c]} ({share[c]:.1f}%)") for c in cls
                       if share[c] >= 0.05], fontsize=6, loc="lower right", title="Area share", title_fontsize=6)
    ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
    F.footer(fig, "Dominant ESA WorldCover v200 (2021, 10 m) class within each 90 m cell; area shares computed from "
                  "10 m class fractions. " + F.CRS_NOTE)
    return F.save(fig, "fig04_landcover"), share


def fig5_lst(B, ext):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
    lo, hi = np.nanpercentile(B["lst"], [1, 99])
    im = axs[0].imshow(B["lst"], cmap="inferno", extent=ext, vmin=lo, vmax=hi)
    F.decorate(axs[0], ext); fig.colorbar(im, ax=axs[0], shrink=0.75).set_label("LST (°C)")
    axs[0].set_title(f"(a) Median land surface temperature, {PERIOD_TXT}", loc="left")
    lo, hi = np.nanpercentile(B["lst_sd"], [1, 99])
    im = axs[1].imshow(B["lst_sd"], cmap="viridis", extent=ext, vmin=lo, vmax=hi)
    F.decorate(axs[1], ext, arrow=False); fig.colorbar(im, ax=axs[1], shrink=0.75).set_label("SD across dates (°C)")
    axs[1].set_title("(b) Standard deviation across acquisition dates", loc="left")
    fig.tight_layout()
    F.footer(fig, "Landsat 8/9 Collection 2 Level-2 surface temperature (ST_B10; K = DN × 0.00341802 + 149.0), per-pixel "
                  "median of 18 clear acquisition dates (mid-morning Landsat overpass), aggregated to 90 m. LST is a radiometric "
                  "skin temperature, not air temperature. " + F.CRS_NOTE)
    return F.save(fig, "fig05_lst")


def main():
    period = sys.argv[1]
    B, ext, _ = F.load_stack(period)
    sel = json.loads((C.OUTPUT_DIR / "models" / period / "selected_predictors.json").read_text())["selected"]
    for p in [fig1_study_area(B, ext), fig2_availability(B, ext), fig3_predictors(B, ext, sel)]:
        print("wrote", p)
    p, share = fig4_landcover(B, ext)
    print("wrote", p, {k: round(v, 1) for k, v in share.items()})
    print("wrote", fig5_lst(B, ext))


if __name__ == "__main__":
    main()
