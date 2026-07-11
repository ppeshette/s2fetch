"""Provider registry: STAC endpoint + per-level collection + item modifier per provider.

A provider is one dict entry. ``collections`` maps processing level ("L2A", "L1C")
to that provider's STAC collection id -- a provider simply has no entry for a level
it doesn't serve (e.g. Planetary Computer has no public L1C collection). The
``modifier`` is the pystac-client item-signing hook applied at search time
(Planetary Computer and CDSE both sign/presign asset hrefs; Earth Search is anonymous
for L2A and needs none).

Confirmed live (see CLAUDE.md working preferences -- re-verify if this looks stale):
Planetary Computer serves L2A only. Earth Search serves both L2A and L1C, with the
same per-band asset-key naming across levels (blue/green/red/... ), except L1C has no
`scl` asset (SCL is an L2A/Sen2Cor-only product) and adds `cirrus` (B10, dropped by L2A
processing).

Earth Search's "fully anonymous" access only holds for L2A (public COG bucket). Its
L1C collection's items point at the legacy requester-pays JP2 bucket -- the STAC
catalog is anonymously browsable, but actual asset reads fail with an AWS credentials
error unless the caller has AWS billing credentials configured.

CDSE (stac.dataspace.copernicus.eu/v1) serves both L2A (`sentinel-2-l2a`) and L1C
(`sentinel-2-l1c`), fully anonymous to *search*. Its asset hrefs are `s3://eodata/...`
JP2s on CDSE's own S3-compatible object store, not a public bucket or AWS -- reading
them needs S3 credentials (free to obtain, but real credentials: see
`cdse_sign_inplace`). L2A asset keys are per-resolution (`B02_10m`, `B02_60m`, ...);
L1C asset keys are plain (`B02`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import boto3
import planetary_computer

_CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
_CDSE_S3_REGION = "default"
_CDSE_PRESIGN_EXPIRES_SECONDS = 3600


@dataclass(frozen=True)
class Provider:
    stac_url: str
    collections: dict[str, str]    # processing level -> STAC collection id
    modifier: Optional[Callable]   # pystac-client `modifier`; None = no signing


@lru_cache(maxsize=1)
def _cdse_s3_client():
    access_key = os.environ.get("CDSE_S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("CDSE_S3_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(
            "provider='cdse' requires S3 credentials: set CDSE_S3_ACCESS_KEY_ID and "
            "CDSE_S3_SECRET_ACCESS_KEY. Generate free credentials (requires a free "
            "Copernicus Data Space Ecosystem account) at "
            "https://eodata-s3keysmanager.dataspace.copernicus.eu/"
        )
    return boto3.client(
        "s3",
        endpoint_url=_CDSE_S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=_CDSE_S3_REGION,
    )


def _sign_asset_href(asset_like) -> None:
    """Presign one asset's href in place. ``asset_like`` is either a raw dict
    (``{"href": ...}``, as seen in a search response payload) or a pystac Asset-like
    object with an ``.href`` attribute."""
    is_dict = isinstance(asset_like, dict)
    href = asset_like.get("href") if is_dict else getattr(asset_like, "href", None)
    if not href or not href.startswith("s3://"):
        return

    bucket, _, key = href.removeprefix("s3://").partition("/")
    signed = _cdse_s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=_CDSE_PRESIGN_EXPIRES_SECONDS,
    )
    if is_dict:
        asset_like["href"] = signed
    else:
        asset_like.href = signed


def _iter_asset_containers(stac_object):
    """Yield each asset mapping found in ``stac_object``, whatever shape pystac-client
    hands the modifier: a raw search-response dict (``{"type": "FeatureCollection",
    "features": [...]}``), a raw single-Item dict (has an "assets" key), or a
    constructed pystac object exposing an ``.assets`` mapping."""
    if isinstance(stac_object, dict):
        if stac_object.get("type") == "FeatureCollection" and "features" in stac_object:
            for feature in stac_object["features"]:
                assets = feature.get("assets")
                if assets:
                    yield assets
        elif "assets" in stac_object:
            yield stac_object["assets"]
    else:
        assets = getattr(stac_object, "assets", None)
        if assets:
            yield assets


def cdse_sign_inplace(stac_object):
    """pystac-client `modifier`: rewrite `s3://eodata/...` asset hrefs into presigned
    HTTPS GET URLs (~1hr expiry). Mirrors planetary_computer.sign_inplace's approach so
    the auth is baked into the URL at search time -- a later lazy dask read (at
    .compute(), possibly in a worker thread with no access to this process's env/GDAL
    config) needs no S3 credentials of its own, just a plain HTTPS GET.

    pystac-client calls this with the *raw JSON search response* (a FeatureCollection
    dict), not a constructed pystac.Item -- confirmed by reading how
    planetary_computer.sign_mapping handles the same shape. Handling only
    object-attribute access here would silently no-op on every real search.
    """
    for assets in _iter_asset_containers(stac_object):
        for asset in assets.values():
            _sign_asset_href(asset)
    return stac_object


PROVIDERS: dict[str, Provider] = {
    "planetary_computer": Provider(
        stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collections={"L2A": "sentinel-2-l2a"},  # no public L1C collection
        modifier=planetary_computer.sign_inplace,  # signs asset hrefs on search
    ),
    "earth_search": Provider(
        stac_url="https://earth-search.aws.element84.com/v1",
        collections={"L2A": "sentinel-2-l2a", "L1C": "sentinel-2-l1c"},
        modifier=None,  # fully anonymous for L2A; L1C needs AWS requester-pays creds
    ),
    "cdse": Provider(
        stac_url="https://stac.dataspace.copernicus.eu/v1",
        collections={"L2A": "sentinel-2-l2a", "L1C": "sentinel-2-l1c"},
        modifier=cdse_sign_inplace,  # presigns s3://eodata hrefs; needs CDSE S3 creds
    ),
}


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError as e:
        raise KeyError(
            f"unknown provider {name!r}; available: {sorted(PROVIDERS)}"
        ) from e


def resolve_collection(provider: str, level: str) -> str:
    """Map (provider, processing level) -> that provider's STAC collection id.

    ``level`` is case-insensitive ("l1c"/"L1C" both work). Raises clearly if the
    provider has no collection for that level (e.g. Planetary Computer + "L1C").
    """
    prov = get_provider(provider)
    level = level.upper()
    try:
        return prov.collections[level]
    except KeyError as e:
        raise KeyError(
            f"provider {provider!r} has no {level!r} collection; "
            f"available levels: {sorted(prov.collections)}"
        ) from e
