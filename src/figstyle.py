"""Shared map styling for paper figures: grid in km, scale bar, north arrow, CRS/source notes."""
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import FancyArrow, Rectangle

import config as C

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.linewidth": 0.6, "savefig.dpi": 300, "savefig.bbox": "tight"})

FIG_DIR = C.OUTPUT_DIR / "figures"


def load_stack(period):
    """Bands as a dict of 2-D float arrays (NaN outside AOI), plus extent and transform."""
    with rasterio.open(C.EXPORT_DIR / f"{period}.tif") as s:
        a = s.read(masked=True).astype("float64").filled(np.nan)
        ext = (s.bounds.left, s.bounds.right, s.bounds.bottom, s.bounds.top)
        transform = s.transform
    inside = ~np.isnan(a[C.FEATURE_BANDS.index("stratum")])
    bands = {b: np.where(inside, a[i], np.nan) for i, b in enumerate(C.FEATURE_BANDS)}
    return bands, ext, transform


def aoi_utm():
    return gpd.read_file(C.AOI_FILE).to_crs(C.CRS)


def hillshade(elev, res=C.GRID_RES, az=315, alt=45):
    z = np.where(np.isnan(elev), np.nanmean(elev), elev)
    dy, dx = np.gradient(z, res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    a, al = np.radians(az), np.radians(alt)
    hs = np.sin(al) * np.cos(slope) + np.cos(al) * np.sin(slope) * np.cos(a - aspect)
    return np.where(np.isnan(elev), np.nan, hs)


def km_axes(ax, ext):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    xt = np.arange(np.ceil(ext[0] / 20000) * 20000, ext[1], 20000)
    yt = np.arange(np.ceil(ext[2] / 20000) * 20000, ext[3], 20000)
    ax.set_xticks(xt, [f"{v / 1000:.0f}" for v in xt]); ax.set_yticks(yt, [f"{v / 1000:.0f}" for v in yt])
    ax.tick_params(labelsize=6, length=2)
    ax.yaxis.set_tick_params(rotation=90)


def scalebar(ax, ext, length_m=10000, loc=(0.06, 0.05)):
    x0 = ext[0] + loc[0] * (ext[1] - ext[0]); y0 = ext[2] + loc[1] * (ext[3] - ext[2])
    h = 0.012 * (ext[3] - ext[2])
    for i in range(2):
        ax.add_patch(Rectangle((x0 + i * length_m / 2, y0), length_m / 2, h,
                               facecolor="k" if i == 0 else "w", edgecolor="k", lw=0.5, zorder=10))
    for v in (0, length_m / 2, length_m):
        ax.text(x0 + v, y0 + 1.6 * h, f"{v / 1000:.0f}", ha="center", va="bottom", fontsize=5.5, zorder=10)
    ax.text(x0 + length_m + 0.01 * (ext[1] - ext[0]), y0, "km", fontsize=5.5, va="bottom", zorder=10)


def north_arrow(ax, loc=(0.93, 0.86)):
    ax.annotate("N", xy=(loc[0], loc[1] + 0.08), xytext=(loc[0], loc[1]), xycoords="axes fraction",
                ha="center", va="center", fontsize=8, fontweight="bold",
                arrowprops=dict(facecolor="k", width=2.5, headwidth=7, headlength=6, lw=0), zorder=10)


def outline(ax, lw=0.6, strata=True):
    g = aoi_utm()
    (g if strata else g.dissolve()).boundary.plot(ax=ax, color="k", lw=lw, zorder=5)


def decorate(ax, ext, bar_km=10, arrow=True):
    km_axes(ax, ext); scalebar(ax, ext, bar_km * 1000); outline(ax)
    if arrow:
        north_arrow(ax)


def footer(fig, text):
    fig.text(0.01, -0.01, text, fontsize=5.5, ha="left", va="top", color="0.25", wrap=True)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    return FIG_DIR / f"{name}.png"


CRS_NOTE = "WGS 84 / UTM zone 43N (EPSG:32643); axes in km. Grid north."
