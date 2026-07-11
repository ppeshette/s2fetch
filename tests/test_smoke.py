"""Smoke tests. The live fetches hit real STAC APIs and are marked ``network``.

Run only the offline tests:      pytest -m "not network"
Run everything (needs network):  pytest
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

# Earth Search's L1C collection's assets sit on a requester-pays JP2 bucket (unlike
# its anonymous L2A COGs) -- reading them needs real AWS billing credentials.
_HAS_AWS_CREDS = bool(
    os.environ.get("AWS_ACCESS_KEY_ID") or (Path.home() / ".aws" / "credentials").exists()
)

# CDSE's asset reads need its own S3-compatible credentials (free to obtain, but real
# credentials -- see providers.py). Generate at the CDSE S3 keys manager portal.
_HAS_CDSE_CREDS = bool(
    os.environ.get("CDSE_S3_ACCESS_KEY_ID") and os.environ.get("CDSE_S3_SECRET_ACCESS_KEY")
)

from s2fetch import BANDS, DEFAULT_BANDS, fetch, to_patches
from s2fetch.bands import asset_to_canonical, resolve_assets
from s2fetch.providers import cdse_sign_inplace, get_provider, resolve_collection

# Santa Barbara coast, small AOI, summer 2023 low-cloud window.
AOI = (-119.75, 34.40, -119.65, 34.48)
RGB = ("B04", "B03", "B02")


# ----- offline unit checks -------------------------------------------------

def test_resolve_assets_pc():
    assert resolve_assets(RGB, "planetary_computer") == ["B04", "B03", "B02"]


def test_resolve_assets_earth_search_remap():
    assert resolve_assets(RGB, "earth_search") == ["red", "green", "blue"]


def test_asset_to_canonical_roundtrip():
    rev = asset_to_canonical("earth_search")
    assert rev["blue"] == "B02" and rev["swir22"] == "B12" and rev["scl"] == "SCL"


def test_default_bands_all_known():
    assert all(b in BANDS for b in DEFAULT_BANDS)


def test_cdse_provider_registered():
    prov = get_provider("cdse")
    assert prov.collections == {"L2A": "sentinel-2-l2a", "L1C": "sentinel-2-l1c"}


def test_resolve_collection_pc_l2a():
    assert resolve_collection("planetary_computer", "L2A") == "sentinel-2-l2a"


def test_resolve_collection_earth_search_l1c_case_insensitive():
    assert resolve_collection("earth_search", "l1c") == "sentinel-2-l1c"


def test_resolve_collection_pc_l1c_unsupported():
    with pytest.raises(KeyError):
        resolve_collection("planetary_computer", "L1C")


def test_resolve_collection_cdse_both_levels():
    assert resolve_collection("cdse", "L2A") == "sentinel-2-l2a"
    assert resolve_collection("cdse", "L1C") == "sentinel-2-l1c"


def test_resolve_assets_cdse_l2a_is_resolution_suffixed():
    assert resolve_assets(RGB, "cdse", level="L2A") == ["B04_10m", "B03_10m", "B02_10m"]


def test_resolve_assets_cdse_l1c_is_plain():
    assert resolve_assets(RGB, "cdse", level="L1C") == ["B04", "B03", "B02"]


def test_asset_to_canonical_cdse_is_level_scoped():
    l2a = asset_to_canonical("cdse", level="L2A")
    l1c = asset_to_canonical("cdse", level="L1C")
    assert l2a["B02_10m"] == "B02" and l2a["SCL_20m"] == "SCL"
    assert l1c["B02"] == "B02" and "SCL" not in l1c.values()


def test_cdse_sign_inplace_requires_credentials(monkeypatch):
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("CDSE_S3_SECRET_ACCESS_KEY", raising=False)
    from s2fetch.providers import _cdse_s3_client
    _cdse_s3_client.cache_clear()

    fake_item = SimpleNamespace(assets={"blue": SimpleNamespace(href="s3://eodata/x/B02_10m.jp2")})
    with pytest.raises(RuntimeError):
        cdse_sign_inplace(fake_item)


def test_cdse_sign_inplace_handles_raw_featurecollection_dict(monkeypatch):
    # pystac-client actually invokes the modifier on the raw JSON search response (a
    # FeatureCollection dict with "features"), not on constructed pystac.Item objects
    # -- confirmed by reading planetary_computer.sign_mapping's own handling of this
    # shape. A modifier that only checks `.assets`/`.href` attributes silently no-ops
    # on every real search; this pins the dict-shaped code path so that regresses loudly.
    monkeypatch.setenv("CDSE_S3_ACCESS_KEY_ID", "fake-access-key")
    monkeypatch.setenv("CDSE_S3_SECRET_ACCESS_KEY", "fake-secret-key")
    from s2fetch.providers import _cdse_s3_client
    _cdse_s3_client.cache_clear()

    page = {
        "type": "FeatureCollection",
        "features": [
            {"id": "item1", "assets": {"B02_10m": {"href": "s3://eodata/x/B02_10m.jp2"}}}
        ],
    }
    cdse_sign_inplace(page)
    href = page["features"][0]["assets"]["B02_10m"]["href"]
    assert href.startswith("https://eodata.dataspace.copernicus.eu/")
    assert "X-Amz-Signature" in href


def test_fetch_rejects_masking_at_l1c():
    with pytest.raises(ValueError):
        fetch(
            aoi=AOI,
            start="2023-07-01",
            end="2023-09-30",
            provider="earth_search",
            level="L1C",
            mask_method="scl",
        )


def test_fetch_rejects_unknown_mask_method():
    with pytest.raises(ValueError):
        fetch(aoi=AOI, start="2023-07-01", end="2023-09-30", mask_method="fmask")


def test_fetch_rejects_resolution_mismatch_without_allow_resample():
    # RGB is all native 10m; default resolution=20 would resample it. No network
    # needed -- this check happens before any STAC search.
    with pytest.raises(ValueError):
        fetch(aoi=AOI, start="2023-07-01", end="2023-09-30", bands=RGB, resolution=20)


def test_fetch_allows_matching_resolution_without_allow_resample():
    # Shouldn't raise on the resolution check; will fail later trying to reach the
    # network with a bogus provider, proving it got *past* the resolution guard.
    with pytest.raises(KeyError):
        fetch(
            aoi=AOI, start="2023-07-01", end="2023-09-30",
            bands=RGB, resolution=10, provider="not-a-real-provider",
        )


def test_fetch_still_warns_when_resample_allowed():
    # allow_resample=True silences the error, not the visibility -- a UserWarning
    # naming the mismatched bands should still fire.
    with pytest.warns(UserWarning, match="B04.*native 10m"):
        with pytest.raises(KeyError):  # bogus provider stops it before any network call
            fetch(
                aoi=AOI, start="2023-07-01", end="2023-09-30",
                bands=RGB, resolution=20, allow_resample=True,
                provider="not-a-real-provider",
            )


# ----- live fetches --------------------------------------------------------

@pytest.mark.network
def test_pc_fetch_rgb():
    ds = fetch(
        aoi=AOI,
        start="2023-07-01",
        end="2023-09-30",
        bands=RGB,
        cloud_max=10,
        provider="planetary_computer",
        resolution=10,  # RGB is all native 10m; no resampling needed
    )
    for b in RGB:
        assert b in ds.data_vars
    assert "time" in ds.dims
    arr = ds[list(RGB)].isel(time=0).to_array().compute()
    assert arr.shape[0] == 3
    assert float(arr.max()) > 0


@pytest.mark.network
def test_earth_search_fetch_rgb():
    ds = fetch(
        aoi=AOI,
        start="2023-07-01",
        end="2023-09-30",
        bands=RGB,
        cloud_max=10,
        provider="earth_search",
        resolution=10,
    )
    # band-key remap: earth_search returns red/green/blue, renamed to canonical.
    for b in RGB:
        assert b in ds.data_vars


@pytest.mark.network
def test_pc_fetch_masked():
    ds = fetch(
        aoi=AOI,
        start="2023-07-01",
        end="2023-09-30",
        bands=RGB,
        cloud_max=10,
        provider="planetary_computer",
        mask_method="scl",
        resolution=20,  # SCL is native 20m; RGB (native 10m) gets resampled to join it
        allow_resample=True,
    )
    assert "SCL" not in ds.data_vars  # dropped after masking
    for b in RGB:
        assert b in ds.data_vars


@pytest.mark.network
def test_earth_search_fetch_l1c():
    ds = fetch(
        aoi=AOI,
        start="2023-07-01",
        end="2023-09-30",
        bands=RGB,
        cloud_max=10,
        provider="earth_search",
        level="L1C",
        resolution=10,
    )
    # Item/asset resolution and band remap work without credentials -- only the
    # actual pixel read needs AWS billing creds for this requester-pays bucket.
    for b in RGB:
        assert b in ds.data_vars

    if not _HAS_AWS_CREDS:
        pytest.skip(
            "no AWS credentials configured; Earth Search L1C assets sit on a "
            "requester-pays bucket (see providers.py). Use CDSE, or set "
            "AWS_ACCESS_KEY_ID / ~/.aws/credentials to exercise the real read."
        )

    arr = ds[list(RGB)].isel(time=0).to_array().compute()
    assert float(arr.max()) > 0


@pytest.mark.network
@pytest.mark.skipif(not _HAS_CDSE_CREDS, reason=(
    "no CDSE S3 credentials configured; set CDSE_S3_ACCESS_KEY_ID and "
    "CDSE_S3_SECRET_ACCESS_KEY (generate free at "
    "https://eodata-s3keysmanager.dataspace.copernicus.eu/)"
))
def test_cdse_fetch_l2a():
    ds = fetch(
        aoi=AOI,
        start="2023-07-01",
        end="2023-09-30",
        bands=RGB,
        cloud_max=10,
        provider="cdse",
        level="L2A",
        mask_method="scl",
        resolution=20,
        allow_resample=True,
    )
    assert "SCL" not in ds.data_vars
    for b in RGB:
        assert b in ds.data_vars
    arr = ds[list(RGB)].isel(time=0).to_array().compute()
    assert float(arr.max()) > 0


@pytest.mark.network
@pytest.mark.skipif(not _HAS_CDSE_CREDS, reason=(
    "no CDSE S3 credentials configured; set CDSE_S3_ACCESS_KEY_ID and "
    "CDSE_S3_SECRET_ACCESS_KEY (generate free at "
    "https://eodata-s3keysmanager.dataspace.copernicus.eu/)"
))
def test_cdse_fetch_l1c():
    ds = fetch(
        aoi=AOI,
        start="2023-07-01",
        end="2023-09-30",
        bands=RGB,
        cloud_max=10,
        provider="cdse",
        level="L1C",
        resolution=10,
    )
    for b in RGB:
        assert b in ds.data_vars
    arr = ds[list(RGB)].isel(time=0).to_array().compute()
    assert float(arr.max()) > 0


@pytest.mark.network
def test_to_patches_writes_tiles(tmp_path):
    ds = fetch(
        aoi=AOI,
        start="2023-07-01",
        end="2023-09-30",
        bands=RGB,
        cloud_max=10,
        provider="planetary_computer",
        resolution=10,
    ).isel(time=0).compute()

    paths = to_patches(ds, size=256, out_dir=tmp_path, prefix="sb")
    assert len(paths) >= 1
    assert all(p.exists() and p.suffix == ".tif" for p in paths)

    import rioxarray

    tile = rioxarray.open_rasterio(paths[0])
    assert tile.shape == (len(RGB), 256, 256)
    assert tile.rio.crs is not None
