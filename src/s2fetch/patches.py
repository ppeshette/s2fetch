"""Tile a (computed) Dataset into size x size GeoTIFF patches for ML dataloaders.

Multi-band per patch: each GeoTIFF holds all band variables as bands, preserving CRS
and geotransform via rioxarray. Domain-agnostic -- just windowed raster tiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr


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
                da = tile[var_names].to_array(dim="band")
                da = da.rio.write_crs(frame.rio.crs or ds.rio.crs)
                path = out / f"{prefix}{stamp}_y{iy}_x{ix}.tif"
                da.rio.to_raster(path)
                written.append(path)

    return written
