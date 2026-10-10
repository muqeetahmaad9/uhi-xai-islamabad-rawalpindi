"""Earth Engine building blocks shared by the acquisition scripts.

Design notes (also the basis of the Methods text):
* LST = USGS Landsat Collection 2 Level-2 ST_B10, Landsat 8 + 9, Tier 1.
* Same-day tiles (same WRS path, adjacent rows) are mosaicked into one
  "acquisition date" so that per-date AOI statistics are not biased by
  partial scene footprints.
* All layers are aggregated (area-weighted mean) to a 90 m grid aligned to the
  Landsat 30 m grid in EPSG:32643.
"""
import json
import math

import ee

import config as C


# ------------------------------------------------------------------ setup
def init():
    if not C.GEE_PROJECT:
        raise SystemExit("Set the EE_PROJECT environment variable to your Earth Engine Cloud project id.")
    ee.Initialize(project=C.GEE_PROJECT)


def load_aoi():
    """Return (FeatureCollection with 'stratum', dissolved Geometry)."""
    gj = json.loads(C.AOI_FILE.read_text())
    feats = [
        ee.Feature(ee.Geometry(f["geometry"], None, False),
                   {"stratum": f["properties"]["stratum"], "name": f["properties"]["shapeName"]})
        for f in gj["features"]
    ]
    fc = ee.FeatureCollection(feats)
    return fc, fc.geometry()


def grid_projection(aoi_bounds_utm, ref_transform):
    """90 m projection whose origin lies on the Landsat 30 m grid.

    ref_transform: crsTransform [30,0,tx,0,-30,ty] of a Landsat ST_B10 band in C.CRS.
    """
    _, _, tx, _, _, ty = ref_transform
    minx, _, _, maxy = aoi_bounds_utm
    x0 = tx + 30 * math.floor((minx - tx) / 30)
    y0 = ty + 30 * math.ceil((maxy - ty) / 30)
    transform = [C.GRID_RES, 0, x0, 0, -C.GRID_RES, y0]
    return ee.Projection(C.CRS, transform), transform


def to_grid(img, native_proj, grid, max_pixels=1024):
    """Area-weighted mean aggregation from a fine native grid to the analysis grid."""
    return (img.setDefaultProjection(native_proj)
               .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=max_pixels)
               .reproject(grid))


def season_filter(months):
    m0, m1 = months
    return ee.Filter.calendarRange(m0, m1, "month")  # wraps when m1 < m0 (e.g. Dec–Feb)


# ------------------------------------------------------------------ Landsat LST
def _mask_landsat(img):
    qa = img.select("QA_PIXEL")
    bits = sum(1 << b for b in C.QA_MASK_BITS)
    clear = qa.bitwiseAnd(bits).eq(0)
    st = img.select("ST_B10")
    unc = img.select("ST_QA").multiply(C.ST_QA_SCALE).rename("lst_unc")
    good = clear.And(st.gt(0)).And(unc.gte(0))
    if C.MAX_ST_UNCERTAINTY_K is not None:
        good = good.And(unc.lte(C.MAX_ST_UNCERTAINTY_K))
    lst = st.multiply(C.ST_SCALE).add(C.ST_OFFSET).subtract(273.15).rename("lst")
    out = lst.addBands(unc).updateMask(good).toFloat()   # uniform type across dates
    return out.copyProperties(img, ["system:time_start", "SPACECRAFT_ID", "WRS_PATH", "WRS_ROW",
                                    "SUN_AZIMUTH", "SUN_ELEVATION", "CLOUD_COVER"]) \
              .set("date", img.date().format("YYYY-MM-dd"))


def landsat_raw(aoi, start, end, months=C.SEASON_MONTHS):
    col = ee.ImageCollection(C.LANDSAT_COLLECTIONS[0])
    for c in C.LANDSAT_COLLECTIONS[1:]:
        col = col.merge(ee.ImageCollection(c))
    return col.filterBounds(aoi).filterDate(start, end).filter(season_filter(months))


def landsat_dates(aoi, start, end, months=C.SEASON_MONTHS, with_stats=True):
    """One masked mosaic per acquisition date, with AOI clear fraction and AOI median LST."""
    masked = landsat_raw(aoi, start, end, months).map(_mask_landsat)
    dates = masked.aggregate_array("date").distinct().sort()

    def per_date(d):
        sub = masked.filter(ee.Filter.eq("date", d))
        first = ee.Image(sub.first())
        mos = sub.mosaic().setDefaultProjection(first.select("lst").projection())
        props = {
            "date": d,
            "system:time_start": first.get("system:time_start"),
            "spacecraft": first.get("SPACECRAFT_ID"),
            "wrs_path": first.get("WRS_PATH"),
            "wrs_rows": sub.aggregate_array("WRS_ROW"),
            "n_tiles": sub.size(),
            "sun_azimuth": sub.aggregate_mean("SUN_AZIMUTH"),
            "sun_elevation": sub.aggregate_mean("SUN_ELEVATION"),
            "scene_cloud_cover_mean": sub.aggregate_mean("CLOUD_COVER"),
        }
        return mos.set(props)

    daily = ee.ImageCollection.fromImages(dates.map(per_date))
    if not with_stats:
        return daily

    def add_stats(img):
        valid = img.select("lst").mask().gt(0).unmask(0)
        frac = valid.reduceRegion(ee.Reducer.mean(), aoi, scale=C.GRID_RES, crs=C.CRS, maxPixels=1e9).get("lst")
        med = img.select("lst").reduceRegion(ee.Reducer.median(), aoi, scale=C.GRID_RES, crs=C.CRS,
                                             maxPixels=1e9).get("lst")
        return img.set({"aoi_clear_fraction": frac, "aoi_median_lst": med})

    return daily.map(add_stats)


def lst_composite(daily):
    """Median LST, median scene-normalised LST, clear-obs count, inter-date SD, median uncertainty.

    `daily` must already be filtered to the dates to use and carry 'aoi_median_lst'.
    """
    def anom(img):
        return img.addBands(img.select("lst").subtract(ee.Number(img.get("aoi_median_lst")))
                            .rename("lst_anom").toFloat())

    d = daily.map(anom)
    return ee.Image.cat([
        d.select("lst").median().rename("lst"),
        d.select("lst_anom").median().rename("lst_anom"),
        d.select("lst").count().rename("lst_n_obs"),
        d.select("lst").reduce(ee.Reducer.stdDev()).rename("lst_sd"),
        d.select("lst_unc").median().rename("lst_unc"),
    ]).toFloat()


# ------------------------------------------------------------------ Sentinel-2
def s2_indices(aoi, start, end, months=C.SEASON_MONTHS):
    """Median per-image NDVI, MNDWI, NDBI, BSI from Cloud Score+-masked S2 L2A."""
    cs = ee.ImageCollection(C.CS_PLUS_COLLECTION)
    col = (ee.ImageCollection(C.S2_COLLECTION)
           .filterBounds(aoi).filterDate(start, end).filter(season_filter(months))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", C.S2_MAX_SCENE_CLOUD))
           .linkCollection(cs, [C.CS_BAND]))

    def prep(img):
        r = img.select(["B2", "B3", "B4", "B8", "B11"]).multiply(C.S2_SCALE)
        b2, b3, b4, b8, b11 = (r.select(b) for b in ["B2", "B3", "B4", "B8", "B11"])
        ndvi = r.normalizedDifference(["B8", "B4"]).rename("ndvi")
        mndwi = r.normalizedDifference(["B3", "B11"]).rename("mndwi")      # Xu (2006)
        ndbi = r.normalizedDifference(["B11", "B8"]).rename("ndbi")        # Zha et al. (2003)
        bsi = (b11.add(b4).subtract(b8.add(b2))).divide(b11.add(b4).add(b8).add(b2)).rename("bsi")
        return (ee.Image.cat([ndvi, mndwi, ndbi, bsi])
                .updateMask(img.select(C.CS_BAND).gte(C.CS_THRESHOLD)))

    native = ee.Image(col.first()).select("B4").projection()
    comp = col.map(prep).median().toFloat()
    return comp, native, col.size()


def dynamic_world(aoi, start, end, months=C.SEASON_MONTHS):
    col = (ee.ImageCollection(C.DW_COLLECTION).filterBounds(aoi)
           .filterDate(start, end).filter(season_filter(months)))
    native = ee.Image(col.first()).select("built").projection()
    comp = col.select(C.DW_BANDS).mean().rename([f"dw_{b}" for b in C.DW_BANDS]).toFloat()
    return comp, native


def worldcover_fractions():
    wc = ee.ImageCollection(C.WORLDCOVER).first().select("Map")
    fr = ee.Image.cat([wc.eq(k).rename(f"wc_{v}") for k, v in C.WC_CLASSES.items()]).toFloat()
    return fr, wc.projection()


def open_buildings(aoi):
    """Building cover fraction and mean building height (m) of building pixels, year C.OB_YEAR.

    The rasters are served at 0.5 m but their documented effective resolution is
    ~4 m, so they are read at 4 m (64x fewer pixels; tested: mean building cover
    over an 8 km box 0.2310 vs 0.2304 at 0.5 m, 2 s vs 159 s).
    """
    col = (ee.ImageCollection(C.OPEN_BUILDINGS).filterBounds(aoi)
           .filter(ee.Filter.calendarRange(C.OB_YEAR, C.OB_YEAR, "year")))
    native = ee.Image(col.first()).select("building_presence").projection().atScale(4)
    mos = col.mosaic()
    bld = mos.select("building_presence").gt(C.OB_PRESENCE_THRESHOLD).unmask(0)
    h = mos.select("building_height").unmask(0).multiply(bld)
    return ee.Image.cat([bld.rename("bld_frac"), h.rename("bld_h_x_frac")]).toFloat(), native, col.size()


# ------------------------------------------------------------------ terrain / water
def dem(aoi):
    col = ee.ImageCollection(C.DEM_COLLECTION).filterBounds(aoi).select("DEM")
    native = ee.Image(col.first()).projection()
    return col.mosaic().setDefaultProjection(native).rename("elev"), native


def terrain_features(elev90, sun_azimuth, sun_elevation):
    """Slope (deg) and cosine of solar incidence angle at mean overpass sun geometry.

    Computed on the 90 m-aggregated DEM so that building/tree edges in the DSM do
    not dominate slope.
    """
    slope = ee.Terrain.slope(elev90).rename("slope")
    aspect = ee.Terrain.aspect(elev90)
    deg = math.pi / 180
    zen = ee.Number(90).subtract(sun_elevation).multiply(deg)
    az = ee.Number(sun_azimuth).multiply(deg)
    s = slope.multiply(deg)
    a = aspect.multiply(deg)
    cos_i = (s.cos().multiply(zen.cos())
             .add(s.sin().multiply(zen.sin()).multiply(a.multiply(-1).add(az).cos()))
             .rename("cos_incidence"))
    return slope.toFloat(), cos_i.toFloat()


def water_features(aoi, grid):
    """Cell-mean GSW water occurrence (%) and distance (m) to permanent water, capped."""
    gsw = ee.Image(C.GSW_IMAGE).select("occurrence")
    native = gsw.projection()
    occ = gsw.unmask(0).rename("gsw_occurrence")
    water = gsw.gte(C.GSW_OCCURRENCE_THRESHOLD).unmask(0)
    region = aoi.buffer(C.WATER_SEARCH_BUFFER_M)
    w30 = water.clip(region).reproject(crs=C.CRS, scale=30)
    d2 = w30.fastDistanceTransform(neighborhood=256, units="pixels", metric="squared_euclidean")
    dist = (d2.sqrt().multiply(30).unmask(C.WATER_DISTANCE_CAP_M)
            .min(C.WATER_DISTANCE_CAP_M).rename("dist_water"))
    occ90 = to_grid(occ.toFloat(), native, grid)
    dist90 = to_grid(dist.toFloat(), ee.Projection(C.CRS).atScale(30), grid)
    return occ90, dist90


def strata_image(aoi_fc, grid):
    return (aoi_fc.reduceToImage(["stratum"], ee.Reducer.first())
            .rename("stratum").reproject(grid).toFloat())


# ------------------------------------------------------------------ context
def era5_at(img, aoi):
    """AOI-mean ERA5-Land instantaneous values at the hour nearest the overpass (context only)."""
    t = ee.Date(img.get("system:time_start")).advance(30, "minute")
    hour = ee.Date.fromYMD(t.get("year"), t.get("month"), t.get("day")).advance(t.get("hour"), "hour")
    col = ee.ImageCollection(C.ERA5_LAND).filterDate(hour, hour.advance(1, "hour"))
    vals = ee.Algorithms.If(
        col.size().gt(0),
        ee.Image(col.first()).select(C.ERA5_BANDS)
          .reduceRegion(ee.Reducer.mean(), aoi, scale=11132, maxPixels=1e8),
        ee.Dictionary({}),
    )
    return ee.Dictionary(vals).set("era5_hour_utc", hour.format("YYYY-MM-dd HH:mm"))
