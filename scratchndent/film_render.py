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
    contrast: float = 1.2,
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
    contrast: float = 1.6,
    black_point: float = 0.0,
) -> np.ndarray:
    """Full display rendering pipeline: scene-linear → sRGB uint16.

    The sigmoid tone mapper's midpoint is set to the image's median
    luminance so the S-curve is centered on the actual content,
    spreading tones evenly across the display range.

    Parameters
    ----------
    scene_linear : HxWx3 float64
        Scene-linear RGB from film inversion (net density values).
    contrast : float
        Sigmoid contrast. 1.0 = flat, 1.5 = punchy, 2.0 = high contrast.
    black_point : float
        Display black level (0.0 = full black).

    Returns
    -------
    display_rgb : HxWx3 uint16
        sRGB-encoded 16-bit display image.
    """
    img = scene_linear.copy()

    # Map the actual data range to [0, 1] using robust percentiles,
    # then apply a gamma curve for contrast. This is content-independent —
    # it uses the full display range regardless of scene content.
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    pos = luminance[luminance > 0.001]
    if len(pos) > 0:
        lo = float(np.percentile(pos, 0.5))
        hi = float(np.percentile(pos, 99.5))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    print(f"  Data range: {lo:.4f} - {hi:.4f}")

    # Normalize to [0, 1] per-channel using the luminance-derived range
    display = (img - lo) / (hi - lo)
    display = np.clip(display, 0.0, 1.0)

    # Apply contrast curve: gamma < 1 pushes midtones darker (more contrast),
    # gamma > 1 lifts shadows. The 'contrast' parameter maps to gamma as
    # 1/contrast so that higher contrast = darker midtones = more punch.
    gamma = 1.0 / contrast
    print(f"  Applying contrast curve (gamma={gamma:.3f})...")
    display = np.power(display, gamma)

    print("  Applying sRGB gamma...")
    display = apply_srgb_gamma(display)

    return np.clip(display * 65535.0, 0, 65535).astype(np.uint16)
