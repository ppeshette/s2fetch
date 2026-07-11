"""fetch(): AOI + date window + bands + cloud filter -> lazy cloud-masked xarray.

Domain-agnostic. L2A (default) is bottom-of-atmosphere surface reflectance, corrected
via Sen2Cor. L1C is top-of-atmosphere reflectance -- radiometrically calibrated but
*not* atmospherically corrected. No spectral indices here.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence, Union

import odc.stac
import pystac_client
import xarray as xr
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from .bands import BANDS, DEFAULT_BANDS, SCL, asset_to_canonical, resolve_assets
from .cloudmask import DEFAULT_MASK_CLASSES, apply_scl_mask
from .providers import get_provider, resolve_collection

BBox = tuple[float, float, float, float]
AOI = Union[BBox, BaseGeometry]


def _bbox_of(aoi: AOI) -> BBox:
    if isinstance(aoi, BaseGeometry):
        return tuple(aoi.bounds)  # (minx, miny, maxx, maxy)
    if len(aoi) != 4:
        raise ValueError(
            f"aoi bbox must be (minx, miny, maxx, maxy); got {aoi!r}"
        )
    return tuple(float(v) for v in aoi)


_MASK_METHODS = (None, "scl")


def fetch(
    aoi: AOI,
    start: str,
    end: str,
    bands: Sequence[str] = DEFAULT_BANDS,
    cloud_max: float = 20,
    provider: str = "planetary_computer",
    level: str = "L2A",
    resolution: float = 20,
    mask_method: str | None = None,
    allow_resample: bool = False,
    crs=None,
    groupby: str = "solar_day",
    mask_classes: Iterable[int] = DEFAULT_MASK_CLASSES,
    drop_scl: bool = True,
) -> xr.Dataset:
    """Fetch a lazy, dask-backed Sentinel-2 Dataset for an AOI and date window.

    Parameters
    ----------
    aoi : (minx, miny, maxx, maxy) lon/lat bbox, or a shapely geometry.
    start, end : ISO date strings "YYYY-MM-DD" (inclusive window "start/end").
    bands : canonical band ids (see bands.BANDS). Default is the 6-band subset.
        Only these bands (plus SCL, if `mask_method="scl"` implicitly adds it) are
        ever requested from the STAC/COG source -- there's no fetch-then-filter.
    cloud_max : scene-level eo:cloud_cover upper bound, percent.
    provider : "planetary_computer" (default) or "earth_search".
    level : processing level, "L2A" (default, surface reflectance) or "L1C"
        (top-of-atmosphere reflectance, not atmospherically corrected). Not every
        provider serves every level -- e.g. Planetary Computer is L2A-only.
    resolution : output resolution in metres.
    mask_method : None (default; no masking, nothing pulled in beyond `bands`) or
        "scl" (adds SCL to the request if not already present, applies the SCL mask,
        then drops SCL per `drop_scl`). SCL only exists at L2A; raises if set with
        level="L1C" -- fetch L1C bands and mask them yourself downstream instead.
    allow_resample : every requested band (including SCL, when `mask_method="scl"`)
        has a fixed native Sentinel-2 ground sample distance (bands.BANDS). If any
        requested band's native GSD differs from `resolution`, fetch() raises rather
        than let odc-stac silently resample it -- set this to opt in explicitly. Even
        with this set, a UserWarning is still emitted naming exactly which bands are
        being resampled and their native GSD -- opting in silences the error, not the
        visibility.
    crs : output CRS; None lets odc-stac pick native UTM.
    groupby : odc-stac grouping; "solar_day" mosaics same-day tiles.
    mask_classes : SCL classes to mask out (see cloudmask). Only used if
        mask_method="scl".
    drop_scl : drop the SCL variable from the result after masking. Only used if
        mask_method="scl".

    Returns a lazy xarray.Dataset with canonical band variables and a time axis.
    Does not compute; the caller computes.
    """
    level = level.upper()
    if mask_method not in _MASK_METHODS:
        raise ValueError(
            f"unknown mask_method {mask_method!r}; supported: {_MASK_METHODS}"
        )
    if mask_method is not None and level != "L2A":
        raise ValueError(
            f"mask_method={mask_method!r} requires level='L2A' (SCL doesn't exist at "
            f"{level!r}); pass mask_method=None and mask it yourself downstream."
        )

    band_ids = list(bands)
    if mask_method == "scl" and SCL not in band_ids:
        band_ids.append(SCL)

    mismatches = []
    for bid in band_ids:
        band = BANDS.get(bid)
        if band is None:
            raise KeyError(f"unknown band id {bid!r}; known ids: {sorted(BANDS)}")
        if band.native_resolution_m != resolution:
            mismatches.append((bid, band.native_resolution_m))
    if mismatches:
        detail = ", ".join(f"{bid} (native {native}m)" for bid, native in mismatches)
        if not allow_resample:
            raise ValueError(
                f"resolution={resolution}m would resample: {detail}. Pass "
                "allow_resample=True to proceed, or choose a resolution matching "
                "your bands' native GSD."
            )
        warnings.warn(
            f"resolution={resolution}m is resampling: {detail} (allow_resample=True)",
            stacklevel=2,
        )

    prov = get_provider(provider)
    collection = resolve_collection(provider, level)
    bbox = _bbox_of(aoi)

    client = pystac_client.Client.open(prov.stac_url, modifier=prov.modifier)

    search_kwargs = dict(
        collections=[collection],
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_max}},
    )
    if isinstance(aoi, BaseGeometry):
        search_kwargs["intersects"] = mapping(aoi)
    else:
        search_kwargs["bbox"] = bbox

    items = list(client.search(**search_kwargs).items())
    if not items:
        raise RuntimeError(
            f"no {collection} items for bbox={bbox} datetime={start}/{end} "
            f"eo:cloud_cover<{cloud_max} on provider {provider!r}"
        )

    asset_keys = resolve_assets(band_ids, provider, level=level)

    ds = odc.stac.load(
        items,
        bands=asset_keys,
        bbox=bbox,
        resolution=resolution,
        crs=crs,
        groupby=groupby,
        chunks={},  # keep it lazy/dask-backed
    )

    # Rename provider asset keys -> canonical ids so downstream is provider-independent.
    rename = {k: v for k, v in asset_to_canonical(provider, level=level).items() if k in ds}
    ds = ds.rename(rename)

    if mask_method == "scl":
        ds = apply_scl_mask(
            ds, scl_var=SCL, mask_classes=mask_classes, drop_scl=drop_scl
        )

    return ds
