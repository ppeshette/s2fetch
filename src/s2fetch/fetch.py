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

from .bands import BANDS, DEFAULT_BANDS, SCL, asset_to_canonical, available_bands, resolve_assets
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
_PICK_DAY_METHODS = (None, "first", "last")


def _resolve_bands(bands: Sequence[str] | str, provider: str, level: str) -> list[str]:
    """Normalize `bands` into a list of canonical ids, catching silent-failure traps
    before they reach BANDS lookups: a bare string (str is itself a Sequence[str], so
    bands="B02" would iterate as ['B','0','2']), an unrecognized provider (which would
    make bands="all" silently expand to [] instead of naming the bad provider), and
    empty input (which raises with a hint listing what's actually available for this
    (provider, level) rather than just pointing at the whole BANDS registry)."""
    if isinstance(bands, str) and bands not in ("all", ""):
        raise TypeError(
            f"bands={bands!r} is a bare string, which iterates character-by-character; "
            f"pass a list/tuple, e.g. bands=[{bands!r}]."
        )
    if bands == "all":
        get_provider(provider)
        bands = available_bands(provider, level)
    bands = list(bands)
    if not bands:
        get_provider(provider)
        raise ValueError(
            "bands must not be empty; pass at least one canonical band id. Available "
            f"for provider={provider!r} level={level!r}: {available_bands(provider, level)}"
        )
    return bands


def fetch(
    aoi: AOI,
    start: str,
    end: str,
    bands: Sequence[str] | str = DEFAULT_BANDS,
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
    pick_day: str | None = None,
) -> xr.Dataset:
    """Fetch a lazy, dask-backed Sentinel-2 Dataset for an AOI and date window.

    Parameters
    ----------
    aoi : (minx, miny, maxx, maxy) lon/lat bbox, or a shapely geometry.
    start, end : ISO date strings "YYYY-MM-DD" (inclusive window "start/end"). Every
        scene in this window matching `cloud_max` is returned -- not just the first
        available -- each as its own step on the result's `time` axis (see `groupby`).
        Narrow the window to a single day to get one scene.
    bands : canonical band ids (see bands.BANDS), or "all" for every band available
        at this (provider, level) -- e.g. Planetary Computer has no B10, L1C has no
        SCL, so "all" is provider/level-aware, not just every key in BANDS. Default is
        the 6-band subset. Must not be empty -- raises rather than silently returning
        a Dataset with no data variables. Only these bands (plus SCL, if
        `mask_method="scl"` implicitly adds it) are ever requested from the STAC/COG
        source -- there's no fetch-then-filter. "all" mixes native GSDs, so it needs
        `allow_resample=True` at a single `resolution` -- see `fetch_native()` for
        every band split by native resolution instead, with no resampling.
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
    groupby : odc-stac grouping; "solar_day" (default) merges same-day tiles/orbits
        into one time step. Distinct dates within `start`/`end` remain separate time
        steps regardless -- this does not collapse the window to a single scene.
    mask_classes : SCL classes to mask out (see cloudmask). Only used if
        mask_method="scl".
    drop_scl : drop the SCL variable from the result after masking. Only used if
        mask_method="scl".
    pick_day : None (default; every matching scene in `start`/`end` as its own time
        step) or "first"/"last" -- narrows the result to just the earliest/latest
        matching date, mosaicked per `groupby` if the AOI spans multiple tiles that
        day (not just a single STAC item), instead of the whole window.

    Returns a lazy xarray.Dataset with canonical band variables and a time axis.
    Does not compute; the caller computes.
    """
    level = level.upper()
    bands = _resolve_bands(bands, provider, level)
    if mask_method not in _MASK_METHODS:
        raise ValueError(
            f"unknown mask_method {mask_method!r}; supported: {_MASK_METHODS}"
        )
    if mask_method is not None and level != "L2A":
        raise ValueError(
            f"mask_method={mask_method!r} requires level='L2A' (SCL doesn't exist at "
            f"{level!r}); pass mask_method=None and mask it yourself downstream."
        )
    if pick_day not in _PICK_DAY_METHODS:
        raise ValueError(
            f"unknown pick_day {pick_day!r}; supported: {_PICK_DAY_METHODS}"
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
                "allow_resample=True to proceed, choose a resolution matching your "
                "bands' native GSD, or call fetch_native(...) to get each native-"
                "resolution group as its own Dataset with no resampling."
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

    if pick_day is not None:
        items = sorted(items, key=lambda it: it.datetime)
        target = items[0].datetime.date() if pick_day == "first" else items[-1].datetime.date()
        items = [it for it in items if it.datetime.date() == target]

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


def fetch_native(
    aoi: AOI,
    start: str,
    end: str,
    bands: Sequence[str] | str = "all",
    cloud_max: float = 20,
    provider: str = "planetary_computer",
    level: str = "L2A",
    crs=None,
    groupby: str = "solar_day",
    pick_day: str | None = None,
) -> dict[int, xr.Dataset]:
    """Fetch bands grouped by native Sentinel-2 GSD -- one Dataset per resolution,
    never resampled.

    A single xarray.Dataset can't hold bands on different pixel grids (same reason
    fetch()'s `allow_resample` guard exists), so requesting every band at its native
    resolution can't come back as one Dataset. This groups the requested bands by
    `native_resolution_m` and calls `fetch()` once per group, each at that group's own
    native resolution -- `allow_resample` never needs to be set, because each group is
    single-GSD by construction.

    Parameters mirror `fetch()` minus `resolution`/`allow_resample` (implied by the
    grouping) and `mask_method`/`mask_classes`/`drop_scl`: masking isn't offered here.
    SCL is native 20m, so masking a 10m or 60m group with it would itself require
    resampling -- exactly what this function exists to avoid. Request bands="all"
    (default) or include "SCL" explicitly to get it back as a plain, unmasked band in
    the 20m group, and call `apply_scl_mask()` yourself if you want it applied.

    bands : canonical band ids, or "all" (default) for every band available at this
        (provider, level).

    Returns a dict keyed by native resolution in metres (e.g. {10: ds, 20: ds, 60: ds}),
    containing only the groups actually present among the requested bands.
    """
    level = level.upper()
    bands = _resolve_bands(bands, provider, level)

    groups: dict[int, list[str]] = {}
    for bid in bands:
        band = BANDS.get(bid)
        if band is None:
            raise KeyError(f"unknown band id {bid!r}; known ids: {sorted(BANDS)}")
        groups.setdefault(band.native_resolution_m, []).append(bid)

    return {
        res: fetch(
            aoi, start, end,
            bands=group_bands,
            cloud_max=cloud_max,
            provider=provider,
            level=level,
            resolution=res,
            crs=crs,
            groupby=groupby,
            pick_day=pick_day,
        )
        for res, group_bands in groups.items()
    }
