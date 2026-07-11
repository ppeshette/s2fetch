# s2fetch — Build Spec

Everything a cold session needs to build s2fetch v1. Read `CLAUDE.md` first for the why and the
design principles; this file is the how. Build order is at the bottom.

Scope of v1: one working provider (Planetary Computer) proven end-to-end, provider-swappable
architecture in place, Earth Search and CDSE wired as additional providers. Cloud masking via
SCL. Patch tiling. `level=` parameter for L1C/L2A. No domain logic.

---

## Verified provider facts (re-verified live 2026-07-10 — re-verify if access behavior looks off)

| Provider | STAC endpoint | Levels | Auth | Asset format | Notes |
|---|---|---|---|---|---|
| Planetary Computer (**default**) | `https://planetarycomputer.microsoft.com/api/stac/v1` | L2A only (no public L1C collection) | Anonymous browse; download URLs signed by `planetary-computer` pkg (`sign_inplace`), no account | Per-band COG, windowed reads | Confirmed free public STAC. "PC Pro" is a separate private-data product, not a paywall of the public catalog. |
| Earth Search (Element84/AWS) | `https://earth-search.aws.element84.com/v1` | L2A, L1C | L2A: fully anonymous, no account, no signing. **L1C: not anonymous** — its items point at the legacy requester-pays JP2 bucket; real reads need a paid, billing-attached AWS account despite the catalog itself being anonymously browsable. | Per-band COG on public S3 (L2A only; L1C is JP2 on the requester-pays bucket) | L2A confirmed free/anonymous, simplest, third-party mirror. Same asset-key naming (blue/green/red/...) across both levels, except L1C has no `scl` and adds `cirrus` (B10, dropped by L2A processing). |
| CDSE (Copernicus) | `https://stac.dataspace.copernicus.eu/v1` (the `/stac` path on `catalogue.dataspace.copernicus.eu` redirects here; a separate small "asset-level" catalog under the same root has no Sentinel-2 collections — don't confuse the two) | L2A, L1C | Search is fully anonymous. Real asset reads need CDSE's own S3-compatible credentials (endpoint `https://eodata.dataspace.copernicus.eu`, region `default`) — free to obtain via https://eodata-s3keysmanager.dataspace.copernicus.eu/ (requires a free CDSE account), but real credentials, not anonymous access. No bearer-token alternative for S3 reads. | Per-band JP2 via authenticated S3 (`s3://eodata/...`), confirmed live | Authoritative ESA source of record. Wired via a presigning `modifier` (like PC's `sign_inplace`) so lazy dask reads need no GDAL/AWS env config. **L2A asset keys are per-resolution** (`B02_10m`/`B02_20m`/`B02_60m`, ...) — use each band's native resolution to avoid double-resampling. L1C asset keys are plain (`B02`). Also exposes openEO for server-side processing (not used here). |

Collection id is `sentinel-2-l2a` on all three providers, `sentinel-2-l1c` on Earth Search and
CDSE. (Earth Search also has a Collection-1 variant `sentinel-2-c1-l2a`.)

**pystac-client modifier gotcha:** the `modifier` callable is invoked on the *raw JSON search
response* (a `{"type": "FeatureCollection", "features": [...]}` dict), not on constructed
`pystac.Item` objects. A modifier that only checks `.assets`/`.href` attributes silently no-ops
on every real search. `planetary_computer.sign_mapping` handles this by mutating
`feature["assets"][k]["href"]` dict-style; `cdse_sign_inplace` in `providers.py` mirrors that.

---

## Package layout (src-layout)

```
s2fetch/
  CLAUDE.md
  BUILD.md
  README.md              # short public description; write last
  pyproject.toml         # canonical deps (below)
  environment.yml        # minimal conda bootstrap (below)
  src/s2fetch/
    __init__.py          # re-export public API: fetch, to_patches, BANDS, PROVIDERS
    providers.py         # provider registry: STAC url + item modifier per provider
    bands.py             # canonical band <-> wavelength <-> per-provider asset-key registry
    fetch.py             # fetch(): AOI + date + bands + cloud_max -> xarray
    cloudmask.py         # SCL-based mask
    patches.py           # xarray -> NxN GeoTIFF tiles
  tests/
    test_smoke.py        # the live PC fetch (below), marked network
```

Public API surface (keep small):

```python
def fetch(
    aoi,                 # (minx, miny, maxx, maxy) lon/lat bbox, or a shapely geometry
    start, end,          # ISO date strings "YYYY-MM-DD"
    bands=("B02","B03","B04","B8A","B11","B12"),  # canonical ids (benchmark 6-band default)
    cloud_max=20,        # scene-level eo:cloud_cover upper bound (percent)
    provider="planetary_computer",   # or "earth_search" / "cdse"
    level="L2A",         # or "L1C"; not every provider serves every level
    resolution=20,       # metres; must match requested bands' native GSD unless allow_resample
    mask_method=None,    # None (no masking) or "scl" (SCL mask -> NaN, SCL dropped)
    allow_resample=False,  # explicit opt-in required if any band's native GSD != resolution
    crs=None,            # default: let odc-stac pick native UTM
    groupby="solar_day", # mosaic same-day tiles
    mask_classes=...,    # SCL classes to mask out, only used if mask_method="scl"
    drop_scl=True,       # only used if mask_method="scl"
) -> "xarray.Dataset":   # lazy, dask-backed; caller computes
    ...

def to_patches(ds, size=256, stride=None, out_dir=..., prefix=...) -> list[Path]:
    # tile an (already-computed) Dataset into size x size GeoTIFFs
    ...
```

**No silent resampling.** Every band has a fixed native Sentinel-2 GSD (bands.py's
`native_resolution_m`), including SCL (always 20m). If any requested band's native GSD
differs from `resolution`, `fetch()` raises rather than let odc-stac quietly reproject
it — the caller must pass `allow_resample=True`. This check runs before any STAC search
(offline-testable). Consequence: `DEFAULT_BANDS` mixes 10m and 20m bands, so fetching it
at any single resolution needs `allow_resample=True`; masking via `mask_method="scl"`
similarly always needs it unless every requested band is already native 20m.

**Masking is opt-in, not automatic.** `mask_method=None` is the default: `fetch()`
returns exactly the requested bands, nothing more pulled in, no NaN-ing applied. This
was a deliberate reversal from an earlier `mask_clouds: bool = True` design — SCL-based
masking forces a resolution compromise (resample your finer bands down to SCL's 20m, or
upsample SCL's categorical classification into fake fine detail), which is exactly the
kind of implicit behavior this project's "no silent resampling" principle rules out.
`mask_method="scl"` is a convenience for callers who do want it; building your own mask
(e.g. at native GSD, or derived from L1C bands per the level docs below) means just
requesting `bands=(..., "SCL")` yourself, or skipping SCL/masking entirely.

---

## providers.py

Registry mapping a provider string to its STAC endpoint and the pystac-client `modifier`
(the item-signing hook). Keep it a plain dict/dataclass so adding a provider is one entry.

```python
import planetary_computer

PROVIDERS = {
    "planetary_computer": Provider(
        stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="sentinel-2-l2a",
        modifier=planetary_computer.sign_inplace,   # signs asset hrefs on search
    ),
    "earth_search": Provider(
        stac_url="https://earth-search.aws.element84.com/v1",
        collection="sentinel-2-l2a",
        modifier=None,                               # anonymous, no signing
    ),
    # "cdse": stub — raise NotImplementedError with a pointer to CDSE auth setup.
}
```

`pystac_client.Client.open(url, modifier=provider.modifier)` applies signing at search time for
PC. Earth Search passes `modifier=None`.

---

## bands.py

Canonical band registry. Each band: canonical id, center wavelength (nm), native ground sample
distance (metres), and a per-provider asset key. This is the load-bearing abstraction — fetch.py
resolves canonical ids to provider asset keys through this table, never inline. Native resolution
is real sensor physics (same category as wavelength), tracked so fetch.py can refuse silent
resampling.

| Canonical | λ nm | native GSD | PC asset | Earth Search asset | CDSE L2A asset | CDSE L1C asset |
|---|---|---|---|---|---|---|
| B01 | 443 | 60m | B01 | coastal | B01_60m | B01 |
| B02 | 490 | 10m | B02 | blue | B02_10m | B02 |
| B03 | 560 | 10m | B03 | green | B03_10m | B03 |
| B04 | 665 | 10m | B04 | red | B04_10m | B04 |
| B05 | 705 | 20m | B05 | rededge1 | B05_20m | B05 |
| B06 | 740 | 20m | B06 | rededge2 | B06_20m | B06 |
| B07 | 783 | 20m | B07 | rededge3 | B07_20m | B07 |
| B08 | 842 | 10m | B08 | nir | B08_10m | B08 |
| B8A | 865 | 20m | B8A | nir08 | B8A_20m | B8A |
| B09 | 945 | 60m | B09 | nir09 | B09_60m | B09 |
| B10 | 1375 | 60m | — (dropped by L2A) | cirrus (L1C only) | — (dropped by L2A) | B10 |
| B11 | 1610 | 20m | B11 | swir16 | B11_20m | B11 |
| B12 | 2190 | 20m | B12 | swir22 | B12_20m | B12 |
| SCL | — | 20m | SCL | scl | SCL_20m | — (L1C has no SCL) |

Benchmark 6-band subset (default): `B02, B03, B04, B8A, B11, B12` (mixes 10m and 20m natives —
fetching it at any single resolution needs `allow_resample=True`). Shape:
`Band(id, wavelength_nm, native_resolution_m, assets={"planetary_computer": "...", ...},
level_assets={"cdse": {"L1C": "...", "L2A": "..."}})` — CDSE's asset keys vary by level (its L2A
collection stores each band at multiple resampled resolutions as separate assets; always select
the native-resolution one), everyone else's don't. Provide a helper to resolve a list of
canonical ids + provider + level -> list of asset keys.

---

## fetch.py

1. Validate `mask_method` (`None` or `"scl"`) and that it isn't combined with `level="L1C"`
   (SCL doesn't exist at L1C).
2. Build the requested band id list; append `"SCL"` if `mask_method="scl"` and not already
   present.
3. **Before opening any STAC client**, check every requested band's `native_resolution_m`
   against `resolution`. If any differ and `allow_resample` isn't set, raise listing the
   mismatches. This makes the check offline-testable and fails fast.
4. Open the provider's STAC client with its modifier.
5. `search(collections=[collection], bbox=aoi_bbox, datetime=f"{start}/{end}",
   query={"eo:cloud_cover": {"lt": cloud_max}})`. If `aoi` is a geometry, pass `intersects=`.
6. Resolve canonical bands -> provider asset keys via bands.py (level-aware).
7. `odc.stac.load(items, bands=asset_keys, bbox=aoi_bbox, resolution=resolution, crs=crs,
   groupby=groupby, chunks={})` -> lazy xarray Dataset. `chunks={}` keeps it dask-lazy.
8. Rename provider asset keys back to canonical ids on the returned Dataset so downstream code
   is provider-independent (a B12 is a B12 regardless of source).
9. If `mask_method == "scl"`, apply cloudmask.py, then optionally drop the SCL variable.

Return the lazy Dataset. Do not `.compute()` inside fetch.

See the pystac-client modifier gotcha noted near the top of this file if wiring a new
signing/presigning provider.

---

## cloudmask.py

Sentinel-2 Scene Classification Layer (SCL) values:

| SCL | class | mask out by default |
|---|---|---|
| 0 | no data | yes |
| 1 | saturated / defective | yes |
| 2 | dark area pixels | no |
| 3 | cloud shadows | yes |
| 4 | vegetation | no |
| 5 | not vegetated | no |
| 6 | water | no |
| 7 | unclassified | no |
| 8 | cloud medium probability | yes |
| 9 | cloud high probability | yes |
| 10 | thin cirrus | yes |
| 11 | snow / ice | configurable (default no; maritime keep, alpine may drop) |

Default mask set: `{0, 1, 3, 8, 9, 10}`. Set masked reflectance to NaN. Make the mask-out set a
parameter so a maritime consumer can keep water (6) and a snow-region consumer can add 11.

---

## Environment files (ready to paste)

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "s2fetch"
version = "0.1.0"
description = "Standalone Sentinel-2 L2A acquisition utility: AOI + date window -> cloud-masked xarray, provider-swappable STAC backend."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Paul Peshette" }]
keywords = ["sentinel-2", "stac", "remote-sensing", "earth-observation", "odc-stac"]
dependencies = [
    "numpy",
    "rasterio",
    "rioxarray",
    "shapely",
    "pyproj",
    "geopandas",
    "xarray",
    "dask",
    "pystac-client>=0.7",
    "odc-stac>=0.3.9",
    "planetary-computer>=1.0",
    "tqdm",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[tool.setuptools.packages.find]
where = ["src"]
```

`environment.yml` (minimal conda bootstrap — compiled geo libs from conda-forge, rest via pip).
Trimmed from the original draft: `pyproj`/`shapely`/`numpy` and the explicit `conda-forge`
channels block were dropped as redundant — they're pulled in transitively by
`gdal`/`rasterio`/`geopandas`, and channel selection is left to the user's conda config:

```yaml
name: s2fetch
dependencies:
  - python=3.11
  - gdal
  - rasterio
  - geopandas
  - pip:
      - -e .
```

Build the env:
```
conda env create -f environment.yml
conda activate s2fetch
python -c "import s2fetch; print('ok')"
```

If that one-shot solve hangs (classic solver + a large/old base env can make this take
hours or never finish — see "Environment build notes" near the Build order below), fall back
to incremental installs instead, which give the solver a much smaller problem per step:
```
conda create --name s2fetch python=3.11
conda activate s2fetch
conda install geopandas
conda install gdal
conda install rasterio
pip install -e .
```

---

## Verification — the one live fetch (do this before building patches/masking out)

Smoke test against Planetary Computer with a small AOI and a low-cloud window. Santa Barbara
coast, summer 2023:

```python
from s2fetch import fetch

ds = fetch(
    aoi=(-119.75, 34.40, -119.65, 34.48),
    start="2023-07-01", end="2023-09-30",
    bands=("B04", "B03", "B02"),
    cloud_max=10,
    provider="planetary_computer",
)
print(ds)                 # dims, data_vars B04/B03/B02, a time axis
sub = ds.isel(time=0)
arr = sub[["B04", "B03", "B02"]].to_array().compute()
print(arr.shape, float(arr.min()), float(arr.max()))
```

Pass criteria: search returns >=1 item, the Dataset has the three canonical band variables and a
time coordinate, and `.compute()` on one timestep pulls real reflectance without auth errors.
That proves pystac-client + odc-stac + PC signing work end-to-end. Only then build out
cloudmask.py and patches.py.

Then repeat the same call with `provider="earth_search"` to confirm the band-key remapping works
across providers (it exercises the `blue/green/red` -> `B02/B03/B04` translation).

---

## Build order

0. ✅ create the src-layout skeleton; write `pyproject.toml` and `environment.yml`.
1. ✅ `bands.py` — the registry table above, plus `native_resolution_m` per band.
2. ✅ `providers.py` — PC, Earth Search, and CDSE all fully wired (not stubbed).
3. ✅ `fetch.py` — search + odc-stac load + canonical rename + `level=` + `mask_method=` +
   `allow_resample=` guard.
4. ✅ Conda env built, `pip install -e .`, PC smoke test passing (env: `s2fetch`, built via a
   plain `conda create` + incremental installs, not the one-shot `conda env create` -- see
   "Environment build notes" below).
5. ✅ Earth Search confirmed (provider swap + band remap), both L2A and L1C.
6. ✅ `cloudmask.py` — SCL masking, opt-in via `mask_method="scl"` (see below).
7. ✅ `patches.py` — tile to GeoTIFFs, live-tested.
8. ✅ `README.md`, `tests/test_smoke.py` (network-marked), `.gitignore`.
9. ✅ CDSE provider — implemented (STAC search, band/collection resolution, S3 presigning
   modifier), all live-testable-without-credentials paths verified (search, collection
   resolution, band mapping, presigning shape, credential-missing error). **Not yet verified
   with real CDSE S3 credentials** — `test_cdse_fetch_l2a`/`test_cdse_fetch_l1c` are written
   and gated on `CDSE_S3_ACCESS_KEY_ID`/`CDSE_S3_SECRET_ACCESS_KEY` env vars, currently skip.
   User has a free CDSE account and S3 keys generated, hasn't run the credentialed tests yet
   (2026-07-10) — pick this up by setting those two env vars and running
   `pytest -v -k cdse`. No AWS credentials for Earth Search's L1C requester-pays path are
   planned (out of scope by choice, not a blocker) — `test_earth_search_fetch_l1c` will
   continue to skip its final read assertion indefinitely.

### Environment build notes

The straightforward `conda env create -f environment.yml` path hit a very slow classic-solver
resolve (base conda was old, pre-libmamba default) that never usefully completed. What actually
worked: a plain `conda create --name s2fetch python=3.11`, then incremental
`conda install geopandas` / `gdal` / `rasterio` (pulls in pyproj/shapely/numpy/pip
transitively) as separate steps rather than one combined solve, then `pip install -e .`.
`environment.yml`'s dependency list is still accurate/authoritative for what the env needs;
it's the *one-shot solve* that's fragile on an old-conda machine, not the package list. If
`conda env create` hangs, fall back to the incremental sequence above.

## Consuming from burn_scar_fm_bench

The benchmark runs in its `terratorch` conda env (separate). It gets s2fetch by
`pip install -e ../s2fetch` into that env, or by calling it as a subprocess. Burn-specific label
logic (dNBR from pre/post pairs, MTBS-boundary rasterization) lives in the benchmark and imports
s2fetch — not here.
