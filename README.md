# s2fetch

Standalone Sentinel-2 acquisition utility:

    AOI + date window + bands + cloud filter  ->  cloud-masked xarray

s2fetch queries a STAC catalog, loads the matching scenes as a lazy, dask-backed xarray
via windowed COG reads, optionally masks clouds from the Scene Classification Layer, and
optionally tiles the result into GeoTIFF patches for ML dataloaders. Defaults to L2A —
atmospherically corrected surface reflectance (Sen2Cor, bottom-of-atmosphere), not raw;
L1C (top-of-atmosphere) is also supported for building your own cloud detection instead
of relying on SCL.

It is deliberately domain-agnostic: it returns generic reflectance arrays and contains
no spectral indices or ML dependencies. Domain logic (dNBR, water indices, mineral
absorption) lives in the consumer, which imports s2fetch.

## Install

```bash
conda env create -f environment.yml
conda activate s2fetch
python -c "import s2fetch; print('ok')"
```

The conda env pins the compiled geo libs (gdal/rasterio/pyproj/shapely/geopandas) from
conda-forge, then `pip install -e .`. `pyproject.toml` is the canonical dependency list.

## Use

```python
from s2fetch import fetch, to_geotiff, to_patches

ds = fetch(
    aoi=(-119.75, 34.40, -119.65, 34.48),   # (minx, miny, maxx, maxy) lon/lat, or a shapely geom
    start="2023-07-01", end="2023-09-30",
    bands=("B02", "B03", "B04"),               # native 10m; see "Resolution" below for mixed sets
    cloud_max=10,                               # scene-level eo:cloud_cover upper bound (%)
    provider="planetary_computer",              # or "earth_search" / "cdse"
    resolution=10,                              # must match requested bands' native GSD, or...
    mask_method="scl",                          # None (default) or "scl" -> SCL mask -> NaN, SCL dropped
    allow_resample=True,                        # ...set this to let odc-stac resample instead
)

ds = ds.compute()                             # fetch() is lazy; caller computes
paths = to_patches(ds, size=256, time_index=0, out_dir="patches")

# or, for a whole scene as one file instead of tiles:
paths = to_geotiff(ds, time_index=0, out_dir="scenes")
```

`to_geotiff` writes one full-extent GeoTIFF per time step (or one, with `time_index`)
instead of tiling -- useful when a consumer wants a whole scene rather than an ML
patch grid (e.g. a baseline scene for visual comparison). Every band written together
must already share one pixel grid, same as `to_patches` -- a Dataset assembled from
mixed native-resolution `fetch()` calls (see "Resolution" below) can't be written as a
single file; call `to_geotiff` once per resolution group instead.

### Date range

`fetch()` returns every scene between `start` and `end` matching `cloud_max` -- not
just the first available -- each as its own step on the returned Dataset's `time` axis.
`groupby="solar_day"` (default) merges same-day tiles/orbits into one time step; it
does not collapse the whole window down to a single scene. For one scene, either narrow
`start`/`end` to a single day, or fetch the window and pick a step yourself (e.g.
`ds.isel(time=0)` takes whichever scene sorts first chronologically, not necessarily
the lowest cloud cover -- inspect `ds.time` alongside item-level cloud metadata if that
distinction matters).

Pass `pick_day="first"` or `pick_day="last"` to search the whole `start`/`end` window
but only load the earliest or latest matching date -- mosaicked per `groupby` if the
AOI spans multiple tiles that day, not truncated to a single STAC item. Useful when you
want "the most recent low-cloud scene in this range" without knowing its date up front.

### Resolution

Every band has a fixed native Sentinel-2 ground sample distance (10m: B02/B03/B04/B08;
20m: B05/B06/B07/B8A/B11/B12/SCL; 60m: B01/B09/B10) — this is real sensor physics, not
a provider quirk, and s2fetch tracks it per band (`bands.BANDS[id].native_resolution_m`).

`fetch()` refuses to resample silently: if any requested band's native GSD doesn't
match `resolution`, it raises listing exactly which bands and their native resolutions,
rather than letting `odc-stac` reproject them without you noticing. Pass
`allow_resample=True` to opt in explicitly — this silences the *error*, not the
*visibility*: a `UserWarning` naming the same mismatched bands still fires every time,
so resampling is never invisible even once you've opted in. This applies to SCL too
when `mask_method="scl"` implicitly adds it — a 10m band set + SCL (native 20m) always
needs `allow_resample=True`, since SCL can't join them on a 10m grid without resampling.

`DEFAULT_BANDS` (`B02, B03, B04, B8A, B11, B12`) spans both 10m and 20m natively, so
fetching it at any single resolution requires `allow_resample=True`.

Pass `bands="all"` to request every band available for the given `(provider, level)`
-- e.g. Planetary Computer has no B10, L1C has no SCL, so this is provider/level-aware,
not just every key in `bands.BANDS`. Since "all" always spans multiple native GSDs, it
needs `allow_resample=True` at a single `resolution` like any other mixed-GSD request.

For every band at its own native resolution with no resampling at all, use
`fetch_native()` instead of `fetch()`:

```python
from s2fetch import fetch_native

by_res = fetch_native(aoi=AOI, start="2023-07-01", end="2023-09-30")  # bands="all" by default
ds_10m = by_res[10]   # B02, B03, B04, B08
ds_20m = by_res[20]   # B05, B06, B07, B8A, B11, B12, SCL
ds_60m = by_res[60]   # B01, B09, B10 (B10 only at level="L1C")
```

`fetch_native()` groups the requested bands by `native_resolution_m` and calls
`fetch()` once per group at that group's own resolution, so `allow_resample` never
needs to be set -- each call is single-GSD by construction. A single GeoTIFF or
xarray.Dataset can't hold bands on different pixel grids, so this can only ever return
one Dataset per resolution, never one merged Dataset -- same constraint `to_geotiff()`
has (see above).

`fetch_native()` doesn't offer `mask_method`: SCL is native 20m, so masking a 10m or
60m group with it would itself require resampling -- exactly what this function exists
to avoid. `bands="all"` (the default) or an explicit `"SCL"` still gets you SCL back as
a plain, unmasked band in the 20m group; call `apply_scl_mask()` on it yourself.

### Cloud masking

`mask_method` defaults to `None`: `fetch()` returns exactly the bands you asked for,
nothing else pulled in, no masking applied. Pass `mask_method="scl"` to opt in to
SCL-based masking (adds SCL to the request if not already present, sets masked-class
pixels to NaN, then drops SCL per `drop_scl`). SCL is an L2A-only product — `mask_method`
must be `None` at `level="L1C"`.

If you'd rather build your own cloud mask (e.g. at each band's native GSD, or derived
from L1C bands — see below), just request `bands=(..., "SCL")` yourself and call
`s2fetch.apply_scl_mask()` on the result, or ignore SCL/masking entirely and do your own
thing downstream. `mask_method` is a convenience for the common case, not the only path.

## Providers

| Provider | `provider=` | Levels | Auth |
|---|---|---|---|
| Planetary Computer (default) | `planetary_computer` | L2A only | Anonymous; download URLs signed automatically |
| Earth Search (Element84/AWS) | `earth_search` | L2A, L1C | L2A is fully anonymous (public COG bucket). **L1C is not** — its STAC items point at the legacy requester-pays JP2 bucket, so real reads need a **paid, billing-attached** AWS account even though the catalog itself is browsable anonymously. |
| CDSE (Copernicus) | `cdse` | L2A, L1C | Catalog search is anonymous. Real asset reads need CDSE's own S3-compatible credentials — free to obtain (unlike Earth Search's L1C AWS requirement), but still real credentials, not anonymous access. See below. |

Bands are keyed by canonical id + wavelength and mapped per-provider to that provider's
asset keys, so a `B12` is a `B12` regardless of source. See `src/s2fetch/bands.py`.

`fetch(..., level="L1C")` gets top-of-atmosphere reflectance instead of L2A's
atmospherically corrected surface reflectance — useful if you want to run your own
cloud detection rather than rely on the L2A-only SCL mask.

To combine L2A reflectance with an L1C-derived mask, call `fetch()` twice (once per
level, `mask_method=None` on both — the default — identical `aoi`/`resolution`/`crs` so
the grids align) and apply your own mask downstream — this is domain logic, so it isn't
fused into a single `fetch()` call.

### CDSE credentials

1. Create a free account at https://dataspace.copernicus.eu/.
2. Generate S3 access key/secret at https://eodata-s3keysmanager.dataspace.copernicus.eu/.
3. Set `CDSE_S3_ACCESS_KEY_ID` and `CDSE_S3_SECRET_ACCESS_KEY` in your environment.

`fetch(provider="cdse", ...)` presigns each asset's `s3://eodata/...` href into a
temporary HTTPS URL at search time (like Planetary Computer's signing, just against
CDSE's own S3-compatible endpoint), so the later lazy dask read needs no S3 config of
its own. CDSE's L2A collection stores each band at multiple resolutions
(`B02_10m`/`B02_20m`/`B02_60m`); s2fetch always requests the native-resolution asset
for each canonical band to avoid double-resampling.

## Tests

```bash
pytest -m "not network"   # offline unit checks (band remap, provider registry)
pytest                    # + live PC / Earth Search fetches
```
