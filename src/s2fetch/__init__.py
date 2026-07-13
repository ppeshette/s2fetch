"""s2fetch — standalone Sentinel-2 L2A acquisition.

AOI + date window + bands + cloud filter -> cloud-masked xarray, provider-swappable
STAC backend. Domain-agnostic: returns generic surface reflectance, no spectral
indices, no ML deps.
"""

from importlib.metadata import version as _version

from .bands import BANDS, DEFAULT_BANDS, Band, available_bands
from .cloudmask import DEFAULT_MASK_CLASSES, apply_scl_mask
from .fetch import fetch, fetch_native
from .patches import to_geotiff, to_patches
from .providers import PROVIDERS, Provider, get_provider, resolve_collection

__version__ = _version("s2fetch")

__all__ = [
    "fetch",
    "fetch_native",
    "to_patches",
    "to_geotiff",
    "apply_scl_mask",
    "BANDS",
    "DEFAULT_BANDS",
    "Band",
    "available_bands",
    "DEFAULT_MASK_CLASSES",
    "PROVIDERS",
    "Provider",
    "get_provider",
    "resolve_collection",
    "__version__",
]
