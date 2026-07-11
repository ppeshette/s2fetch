"""s2fetch — standalone Sentinel-2 L2A acquisition.

AOI + date window + bands + cloud filter -> cloud-masked xarray, provider-swappable
STAC backend. Domain-agnostic: returns generic surface reflectance, no spectral
indices, no ML deps.
"""

from importlib.metadata import version as _version

from .bands import BANDS, DEFAULT_BANDS, Band
from .cloudmask import DEFAULT_MASK_CLASSES, apply_scl_mask
from .fetch import fetch
from .patches import to_patches
from .providers import PROVIDERS, Provider, get_provider, resolve_collection

__version__ = _version("s2fetch")

__all__ = [
    "fetch",
    "to_patches",
    "apply_scl_mask",
    "BANDS",
    "DEFAULT_BANDS",
    "Band",
    "DEFAULT_MASK_CLASSES",
    "PROVIDERS",
    "Provider",
    "get_provider",
    "resolve_collection",
    "__version__",
]
