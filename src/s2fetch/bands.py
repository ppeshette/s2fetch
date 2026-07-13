"""Canonical band registry.

Each band has a canonical id, a center wavelength (nm), its native ground sample
distance (metres), and a per-provider asset key. fetch.py resolves canonical ids to
provider asset keys *only* through this table -- never inline. This indirection is
what makes the backend provider-swappable and is the hook for wavelength-conditional
models and future sensor extensibility. ``native_resolution_m`` is real sensor physics
(same category as wavelength, not a provider quirk) -- fetch.py uses it to refuse
silent resampling: requesting a band at a resolution other than its native GSD is only
allowed when the caller explicitly opts in.

Most providers use one asset key per band regardless of processing level (``assets``).
CDSE is the exception: its L2A collection stores each band at multiple resampled
resolutions as separate assets (``B02_10m``/``B02_20m``/``B02_60m``, ...), so its keys
are level-dependent -- those live in ``level_assets`` and take priority over ``assets``
when present. s2fetch always selects the asset matching the band's native resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Band:
    id: str                        # canonical id, e.g. "B12"
    wavelength_nm: float | None     # center wavelength; None for non-spectral (SCL)
    native_resolution_m: int        # native Sentinel-2 ground sample distance
    assets: dict[str, str]          # provider -> asset key (level-invariant providers)
    level_assets: dict[str, dict[str, str]] = field(default_factory=dict)
    # provider -> {level: asset key}, overrides `assets` for that (provider, level)


# Canonical registry. Wavelengths are Sentinel-2 band centers (nm); native resolutions
# are Sentinel-2's actual per-band GSD (10m: B02/B03/B04/B08; 20m: B05/B06/B07/B8A/
# B11/B12/SCL; 60m: B01/B09/B10). CDSE L2A keys use each band's native-resolution asset
# (not a resampled duplicate) to avoid double-resampling before odc-stac reprojects to
# the caller's requested `resolution`. CDSE L1C keys are plain (no per-resolution
# duplicates exist there).
BANDS: dict[str, Band] = {
    b.id: b
    for b in [
        Band("B01", 443, 60, {"planetary_computer": "B01", "earth_search": "coastal"},
             level_assets={"cdse": {"L1C": "B01", "L2A": "B01_60m"}}),
        Band("B02", 490, 10, {"planetary_computer": "B02", "earth_search": "blue"},
             level_assets={"cdse": {"L1C": "B02", "L2A": "B02_10m"}}),
        Band("B03", 560, 10, {"planetary_computer": "B03", "earth_search": "green"},
             level_assets={"cdse": {"L1C": "B03", "L2A": "B03_10m"}}),
        Band("B04", 665, 10, {"planetary_computer": "B04", "earth_search": "red"},
             level_assets={"cdse": {"L1C": "B04", "L2A": "B04_10m"}}),
        Band("B05", 705, 20, {"planetary_computer": "B05", "earth_search": "rededge1"},
             level_assets={"cdse": {"L1C": "B05", "L2A": "B05_20m"}}),
        Band("B06", 740, 20, {"planetary_computer": "B06", "earth_search": "rededge2"},
             level_assets={"cdse": {"L1C": "B06", "L2A": "B06_20m"}}),
        Band("B07", 783, 20, {"planetary_computer": "B07", "earth_search": "rededge3"},
             level_assets={"cdse": {"L1C": "B07", "L2A": "B07_20m"}}),
        Band("B08", 842, 10, {"planetary_computer": "B08", "earth_search": "nir"},
             level_assets={"cdse": {"L1C": "B08", "L2A": "B08_10m"}}),
        Band("B8A", 865, 20, {"planetary_computer": "B8A", "earth_search": "nir08"},
             level_assets={"cdse": {"L1C": "B8A", "L2A": "B8A_20m"}}),
        Band("B09", 945, 60, {"planetary_computer": "B09", "earth_search": "nir09"},
             level_assets={"cdse": {"L1C": "B09", "L2A": "B09_60m"}}),
        # cirrus band; dropped by L2A/Sen2Cor processing, so only available at L1C.
        Band("B10", 1375, 60, {"earth_search": "cirrus"},
             level_assets={"cdse": {"L1C": "B10"}}),
        Band("B11", 1610, 20, {"planetary_computer": "B11", "earth_search": "swir16"},
             level_assets={"cdse": {"L1C": "B11", "L2A": "B11_20m"}}),
        Band("B12", 2190, 20, {"planetary_computer": "B12", "earth_search": "swir22"},
             level_assets={"cdse": {"L1C": "B12", "L2A": "B12_20m"}}),
        # SCL is an L2A-only product for every provider, not just CDSE -- all three
        # mappings go through level_assets (not the level-invariant `assets` dict) so
        # requesting it at level="L1C" raises instead of resolving a nonexistent asset.
        Band("SCL", None, 20, {},
             level_assets={
                 "planetary_computer": {"L2A": "SCL"},
                 "earth_search": {"L2A": "scl"},
                 "cdse": {"L2A": "SCL_20m"},
             }),
    ]
}

# Benchmark 6-band default subset.
DEFAULT_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B8A", "B11", "B12")

SCL = "SCL"


def _asset_key(band: Band, provider: str, level: str) -> str:
    by_level = band.level_assets.get(provider)
    if by_level is not None and level in by_level:
        return by_level[level]
    try:
        return band.assets[provider]
    except KeyError as e:
        available = sorted(by_level) if by_level else []
        hint = f"; available at levels {available} for this provider" if available else ""
        raise KeyError(
            f"band {band.id!r} has no asset mapping for provider {provider!r} "
            f"at level {level!r}{hint}"
        ) from e


def available_bands(provider: str, level: str = "L2A") -> list[str]:
    """Canonical band ids with a valid asset mapping for this (provider, level).

    Preserves BANDS' registration order. Backs fetch()'s bands="all" -- e.g. PC has no
    B10 and L1C has no SCL, so this is not just every key in BANDS.
    """
    level = level.upper()
    out = []
    for band in BANDS.values():
        try:
            _asset_key(band, provider, level)
        except KeyError:
            continue
        out.append(band.id)
    return out


def resolve_assets(
    band_ids: list[str] | tuple[str, ...],
    provider: str,
    level: str = "L2A",
) -> list[str]:
    """Map canonical band ids -> this provider's asset keys for the given level.

    Preserves input order and de-duplicates. Raises on unknown band ids or a band with
    no mapping for ``(provider, level)``. ``level`` only matters for providers whose
    asset keys vary by processing level (currently just CDSE); others ignore it.
    """
    assets: list[str] = []
    seen: set[str] = set()
    for bid in band_ids:
        band = BANDS.get(bid)
        if band is None:
            raise KeyError(
                f"unknown band id {bid!r}; known ids: {sorted(BANDS)}"
            )
        key = _asset_key(band, provider, level)
        if key not in seen:
            seen.add(key)
            assets.append(key)
    return assets


def asset_to_canonical(provider: str, level: str = "L2A") -> dict[str, str]:
    """Reverse map: this provider's asset key -> canonical id, for renaming a loaded
    Dataset back to provider-independent canonical band names. Level-scoped for
    providers (CDSE) whose asset keys vary by processing level."""
    out: dict[str, str] = {}
    for band in BANDS.values():
        by_level = band.level_assets.get(provider)
        if by_level is not None and level in by_level:
            out[by_level[level]] = band.id
        elif provider in band.assets:
            out[band.assets[provider]] = band.id
    return out
