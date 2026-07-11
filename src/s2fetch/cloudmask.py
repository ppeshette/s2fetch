"""SCL-based cloud masking.

Sentinel-2 Scene Classification Layer (SCL) integer classes:

    0  no data                     8  cloud medium probability
    1  saturated / defective       9  cloud high probability
    2  dark area pixels           10  thin cirrus
    3  cloud shadows              11  snow / ice
    4  vegetation
    5  not vegetated
    6  water
    7  unclassified

Default mask-out set is {0, 1, 3, 8, 9, 10}: no-data, defective, cloud shadow,
clouds, and cirrus. Water (6) and snow (11) are kept by default -- a maritime
consumer keeps water; an alpine consumer can add 11. The mask-out set is a parameter
so those choices live with the caller, not here.
"""

from __future__ import annotations

from typing import Iterable

import xarray as xr

# no-data, saturated/defective, cloud shadow, cloud med, cloud high, thin cirrus
DEFAULT_MASK_CLASSES: frozenset[int] = frozenset({0, 1, 3, 8, 9, 10})


def apply_scl_mask(
    ds: xr.Dataset,
    scl_var: str = "SCL",
    mask_classes: Iterable[int] = DEFAULT_MASK_CLASSES,
    drop_scl: bool = True,
) -> xr.Dataset:
    """Set reflectance to NaN wherever the SCL band is in ``mask_classes``.

    Operates lazily (dask-friendly): builds a boolean mask from the SCL variable and
    ``xr.where``s every non-SCL data variable. Masked bands are promoted to float so
    NaN is representable. When ``drop_scl`` is set, the SCL variable is dropped from
    the returned Dataset.
    """
    if scl_var not in ds:
        raise KeyError(
            f"cannot mask: SCL variable {scl_var!r} not in Dataset "
            f"(data_vars: {list(ds.data_vars)}). Fetch with the SCL band included."
        )

    classes = list(mask_classes)
    scl = ds[scl_var]
    keep = ~scl.isin(classes)  # True where the pixel should be kept

    out = ds.copy()
    for name in ds.data_vars:
        if name == scl_var:
            continue
        # xr.where promotes to float and inserts NaN where keep is False
        out[name] = xr.where(keep, ds[name], float("nan"))

    if drop_scl:
        out = out.drop_vars(scl_var)
    return out
