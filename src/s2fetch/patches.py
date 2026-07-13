"""Write a (computed) Dataset to GeoTIFF, tiled into size x size patches for ML
dataloaders (``to_patches``) or as one full-extent file per time step (``to_geotiff``).

Multi-band per file: each GeoTIFF holds all band variables as bands, preserving CRS
and geotransform via rioxarray. Domain-agnostic -- just raster I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr


def _write_raster(frame: xr.Dataset, var_names: list[str], path: Path, crs) -> None:
    da = frame[var_names].to_array(dim="band")
    da.attrs["long_name"] = tuple(var_names)
    da = da.rio.write_crs(crs)
    da.rio.to_raster(path)


def to_patches(
    ds: xr.Dataset,
    size: int = 256,
    stride: Optional[int] = None,
    out_dir: str | Path = "patches",
    prefix: str = "patch",
    bands: Optional[Sequence[str]] = None,
    time_index: Optional[int] = None,
) -> list[Path]:
    """Tile ``ds`` into ``size`` x ``size`` GeoTIFFs and write them to ``out_dir``.

    Parameters
    ----------
    ds : Dataset with 2D spatial dims (y, x) and optionally a time dim. If a time dim
        is present, pass ``time_index`` to select one step, else each step is tiled and
        the timestamp is included in the filename.
    size : patch edge length in pixels.
    stride : step between patch origins; defaults to ``size`` (non-overlapping).
    out_dir : directory to write into (created if absent).
    prefix : filename prefix.
    bands : band variables to stack into the GeoTIFF; default all data_vars.
    time_index : if set, select this time step before tiling.

    Returns the list of written GeoTIFF paths. Edge tiles smaller than ``size`` are
    skipped. Expects an already-computed Dataset; call ``.compute()`` first for large
    lazy inputs to avoid re-reading COGs per tile.
    """
    stride = stride or size
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "time" in ds.dims and time_index is not None:
        ds = ds.isel(time=time_index)

    var_names = list(bands) if bands is not None else list(ds.data_vars)

    # Determine spatial dim names (odc-stac / rioxarray default to y, x).
    ydim = "y" if "y" in ds.dims else ("latitude" if "latitude" in ds.dims else None)
    xdim = "x" if "x" in ds.dims else ("longitude" if "longitude" in ds.dims else None)
    if ydim is None or xdim is None:
        raise ValueError(
            f"could not find spatial dims (y/x) in {list(ds.dims)}"
        )

    written: list[Path] = []
    time_steps = ds.sizes.get("time", None)
    iterate = range(time_steps) if time_steps else [None]

    for t in iterate:
        frame = ds.isel(time=t) if t is not None else ds
        stamp = ""
        if t is not None:
            stamp = "_" + str(frame["time"].values)[:10]  # YYYY-MM-DD

        ny = frame.sizes[ydim]
        nx = frame.sizes[xdim]
        for iy in range(0, ny - size + 1, stride):
            for ix in range(0, nx - size + 1, stride):
                tile = frame.isel(
                    {ydim: slice(iy, iy + size), xdim: slice(ix, ix + size)}
                )
                path = out / f"{prefix}{stamp}_y{iy}_x{ix}.tif"
                _write_raster(tile, var_names, path, frame.rio.crs or ds.rio.crs)
                written.append(path)

    return written


def to_geotiff(
    ds: xr.Dataset,
    out_dir: str | Path = ".",
    prefix: str = "scene",
    bands: Optional[Sequence[str]] = None,
    time_index: Optional[int] = None,
) -> list[Path]:
    """Write ``ds`` to one full-extent GeoTIFF per time step, no tiling.

    Parameters
    ----------
    ds : Dataset with 2D spatial dims (y, x) and optionally a time dim. If a time dim
        is present, pass ``time_index`` to select one step (one file written), else
        each step is written separately with the timestamp in the filename.
    out_dir : directory to write into (created if absent).
    prefix : filename prefix.
    bands : band variables to stack into the GeoTIFF; default all data_vars.
    time_index : if set, select this time step before writing.

    Returns the list of written GeoTIFF paths. Every band in ``ds`` (or ``bands``)
    must already share one pixel grid -- a GeoTIFF's bands all share one
    width/height/geotransform, so a Dataset assembled from mixed native GSDs (e.g. via
    ``allow_resample=False`` calls at different resolutions) can't be written as a
    single file; write each resolution group separately instead. Expects an
    already-computed Dataset; call ``.compute()`` first for large lazy inputs to avoid
    re-reading COGs per file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "time" in ds.dims and time_index is not None:
        ds = ds.isel(time=time_index)

    var_names = list(bands) if bands is not None else list(ds.data_vars)

    written: list[Path] = []
    time_steps = ds.sizes.get("time", None)
    iterate = range(time_steps) if time_steps else [None]

    for t in iterate:
        frame = ds.isel(time=t) if t is not None else ds
        stamp = ""
        if t is not None:
            stamp = "_" + str(frame["time"].values)[:10]  # YYYY-MM-DD

        path = out / f"{prefix}{stamp}.tif"
        _write_raster(frame, var_names, path, frame.rio.crs or ds.rio.crs)
        written.append(path)

    return written
