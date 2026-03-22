"""Film measurement model: raw scanner values → transmittance → density.

For flatbed scanners (Epson V600 / SilverFast), the raw 16-bit linear scan
is already flat-fielded internally, so transmittance ≈ raw / max_value.
Dark subtraction accounts for scanner black level.
"""

import numpy as np

EPS = 1e-8


def normalize_transmittance(
    raw_rgb: np.ndarray,
    dark_rgb: np.ndarray | None = None,
    light_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Convert raw scanner values to per-channel transmittance.

    Parameters
    ----------
    raw_rgb : HxWx3 uint16 or float
        Raw linear scan of the negative.
    dark_rgb : 1x3 or HxWx3, optional
        Dark frame / scanner black level. Defaults to 0.
    light_rgb : 1x3 or HxWx3, optional
        Blank light field (no film in path). For flatbed scanners this is
        approximately the max sensor value. Defaults to 65535 for uint16.

    Returns
    -------
    T : HxWx3 float64
        Per-channel transmittance, nominally in [0, 1+].
    """
    img = raw_rgb.astype(np.float64)

    if dark_rgb is None:
        dark = np.zeros(3, dtype=np.float64)
    else:
        dark = np.asarray(dark_rgb, dtype=np.float64)

    if light_rgb is None:
        if raw_rgb.dtype == np.uint16:
            light = np.full(3, 65535.0, dtype=np.float64)
        else:
            light = np.full(3, 1.0, dtype=np.float64)
    else:
        light = np.asarray(light_rgb, dtype=np.float64)

    num = img - dark
    den = np.maximum(light - dark, EPS)
    T = num / den

    return np.clip(T, EPS, None)


def transmittance_to_density(T: np.ndarray) -> np.ndarray:
    """Convert transmittance to optical density: D = -log10(T)."""
    return -np.log10(np.clip(T, EPS, None))


def estimate_dmin(
    density: np.ndarray,
    rebate_mask: np.ndarray | None = None,
    percentile: float = 1.0,
) -> np.ndarray:
    """Estimate D-min (film base + fog + orange mask) per channel.

    Parameters
    ----------
    density : HxWx3 float64
        Full-frame density image.
    rebate_mask : HxW bool, optional
        Mask of unexposed rebate (film edge) pixels. This is the best
        source for Dmin since rebate has no image content.
    percentile : float
        If no rebate mask, use this percentile of the density image.
        Low percentile approximates the least-exposed areas.

    Returns
    -------
    dmin : (3,) float64
        Per-channel D-min values.
    """
    if rebate_mask is not None:
        pixels = density[rebate_mask]
        if len(pixels) > 0:
            return np.median(pixels, axis=0)

    # Fallback: low percentile of the whole frame
    flat = density.reshape(-1, 3)
    return np.percentile(flat, percentile, axis=0)


def subtract_dmin(
    density: np.ndarray,
    dmin: np.ndarray,
) -> np.ndarray:
    """Remove film base/orange mask baseline, yielding net image density."""
    return np.maximum(density - dmin[None, None, :], 0.0)
