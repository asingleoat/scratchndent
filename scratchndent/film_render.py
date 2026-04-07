"""Display rendering: scene-linear → display RGB.

Separate from inversion — "make it positive" and "make it look nice" are
different problems. This module handles tone mapping, exposure, contrast,
and gamut mapping for display output.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _sigmoid_tonemap_kernel(flat, out, n_pixels,
                            mid_grey, white_point, black_point, contrast):
    """Per-pixel filmic sigmoid tone mapping.

    Maps scene-linear [0, ∞) to display [black_point, white_point].
    Uses a simple sigmoid: x^p / (x^p + mid^p) scaled to output range.
    """
    for i in prange(n_pixels):
        for c in range(3):
            x = flat[i, c]
            if x <= 0.0:
                out[i, c] = black_point
                continue

            # Filmic sigmoid: x^contrast / (x^contrast + mid^contrast)
            xp = x ** contrast
            mp = mid_grey ** contrast
            sigmoid = xp / (xp + mp)

            # Scale to display range
            out[i, c] = black_point + (white_point - black_point) * sigmoid


@njit(parallel=True, cache=True)
def _linear_to_srgb_kernel(flat, out, n_pixels):
    """Linear to sRGB gamma."""
    for i in prange(n_pixels):
        for c in range(3):
            v = flat[i, c]
            if v < 0.0:
                v = 0.0
            if v <= 0.0031308:
                out[i, c] = v * 12.92
            else:
                out[i, c] = 1.055 * v ** (1.0 / 2.4) - 0.055


def sigmoid_tonemap(
    scene_linear: np.ndarray,
    mid_grey: float = 0.18,
    white_point: float = 1.0,
    black_point: float = 0.0,
    contrast: float = 1.4,
) -> np.ndarray:
    """Apply filmic sigmoid tone mapping.

    Parameters
    ----------
    scene_linear : HxWx3 float64
        Scene-linear RGB (positive, may exceed 1.0).
    mid_grey : float
        Scene-linear value that maps to 50% display output.
    white_point : float
        Maximum display output.
    black_point : float
        Minimum display output (lift).
    contrast : float
        Sigmoid steepness. 1.0 = soft, 1.5 = punchy.
    """
    h, w, _ = scene_linear.shape
    flat = scene_linear.reshape(-1, 3).astype(np.float64)
    out = np.empty_like(flat)
    _sigmoid_tonemap_kernel(flat, out, flat.shape[0],
                            mid_grey, white_point, black_point, contrast)
    return out.reshape(h, w, 3)


def apply_srgb_gamma(linear: np.ndarray) -> np.ndarray:
    """Apply sRGB transfer function."""
    h, w, c = linear.shape
    flat = linear.reshape(-1, 3).astype(np.float64)
    out = np.empty_like(flat)
    _linear_to_srgb_kernel(flat, out, flat.shape[0])
    return out.reshape(h, w, c)



def render_to_display(
    scene_linear: np.ndarray,
    *,
    contrast: float = 1.4,
    black_point: float = 0.0,
    curve_k: float = 5.0,
    percentile_lo: float = 0.5,
    percentile_hi: float = 99.5,
    exposure_compensation: float = 0.0,
) -> np.ndarray:
    """Render corrected density values to display-ready sRGB uint16.

    Density values are already in a perceptual/log space (similar to
    gamma-encoded), so we do NOT apply sRGB gamma — that would double-
    encode and push everything too bright.

    The pipeline:
    1. Normalize density range to [0, 1] using robust percentiles
    2. Apply an S-curve for contrast (expands midtones, compresses extremes)
    3. Scale to uint16

    Parameters
    ----------
    scene_linear : HxWx3 float64
        Corrected net density from film inversion. Higher = brighter scene.
    contrast : float
        S-curve strength. 1.0 = linear (no contrast boost), 1.5 = moderate,
        2.0 = high contrast. Default 1.4.
    black_point : float
        Display black level (0.0 = full black).

    Returns
    -------
    display_rgb : HxWx3 uint16
        Display-ready 16-bit image.
    """
    img = scene_linear.copy()

    # Map the actual data range to [0, 1] using robust percentiles
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    pos = luminance[luminance > 0.001]
    if len(pos) > 0:
        lo = float(np.percentile(pos, percentile_lo))
        hi = float(np.percentile(pos, percentile_hi))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    print(f"  Data range: {lo:.4f} - {hi:.4f}")

    # Normalize to [0, 1]
    display = (img - lo) / (hi - lo)
    display = np.clip(display, 0.0, 1.0)

    # Exposure compensation: applied after normalization as a power curve.
    # Positive values brighten (gamma < 1), negative darken (gamma > 1).
    # This preserves channel ratios (color-neutral) because it's applied
    # equally to all channels in the normalized domain.
    if abs(exposure_compensation) > 0.001:
        gamma = 1.0 / (1.0 + exposure_compensation)
        print(f"  Exposure compensation: {exposure_compensation:+.2f} (gamma={gamma:.3f})")
        display = np.power(display, gamma)

    # S-curve for contrast: centered at 0.5, symmetric, adjustable strength.
    # Maps [0,1] → [0,1] with midtones expanded and extremes compressed.
    # contrast=1.0 is identity (no curve). The effective steepness scales
    # as (contrast - 1) so small adjustments above 1.0 are gentle.
    if contrast > 1.001:
        k = (contrast - 1.0) * curve_k  # 0 at contrast=1.0, ramps up smoothly
        if k > 0.1:
            raw = 1.0 / (1.0 + np.exp(-k * (display - 0.5)))
            raw_lo = 1.0 / (1.0 + np.exp(-k * (0.0 - 0.5)))
            raw_hi = 1.0 / (1.0 + np.exp(-k * (1.0 - 0.5)))
            display = (raw - raw_lo) / (raw_hi - raw_lo)
            print(f"  S-curve contrast (k={k:.1f})")

    # No sRGB gamma — density values are already perceptually spaced
    display = np.clip(display, 0.0, 1.0)
    return np.clip(display * 65535.0, 0, 65535).astype(np.uint16)
