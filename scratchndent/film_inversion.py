"""Film inversion pipeline: density-domain processing and neutral balancing.

This is the core inversion module that ties measurement and calibration
together. It takes a raw scanner image through the physically-grounded
pipeline: transmittance → density → Dmin removal → calibrated transform.

Neutral balancing is done in density space (before inversion), which is
more correct than balancing in RGB after inversion.
"""

import numpy as np

from scratchndent.film_measurement import (
    normalize_transmittance,
    transmittance_to_density,
    estimate_dmin,
    subtract_dmin,
)
from scratchndent.film_calibration import (
    apply_density_transform,
    default_identity_coeffs,
    default_kodak_gold_coeffs,
    default_kodak_portra_coeffs,
    load_profile,
)


STOCK_DEFAULTS = {
    "kodak_gold": default_identity_coeffs,
    "kodak_portra": default_identity_coeffs,
}




def compute_dmin(
    raw_rgb: np.ndarray,
    *,
    rebate_mask: np.ndarray | None = None,
    dark_rgb: np.ndarray | None = None,
    light_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Dmin from a full image, using rebate mask if available.

    This should be called once on the full strip, and the result passed
    to invert_negative for each cropped frame.

    Returns
    -------
    dmin : (3,) float64
        Per-channel Dmin values.
    """
    T = normalize_transmittance(raw_rgb, dark_rgb, light_rgb)
    D = transmittance_to_density(T)
    dmin = estimate_dmin(D, rebate_mask=rebate_mask)
    print(f"    Dmin: R={dmin[0]:.4f} G={dmin[1]:.4f} B={dmin[2]:.4f}")
    return dmin


def invert_negative(
    raw_rgb: np.ndarray,
    *,
    dmin: np.ndarray | None = None,
    stock: str = "kodak_gold",
    profile_path: str | None = None,
    dark_rgb: np.ndarray | None = None,
    light_rgb: np.ndarray | None = None,
    rebate_mask: np.ndarray | None = None,
    exposure_compensation: float = 0.0,
) -> np.ndarray:
    """Full film inversion: raw scan → scene-linear RGB.

    Parameters
    ----------
    raw_rgb : HxWx3 uint16
        Raw linear scan from SilverFast / scanner.
    dmin : (3,) float64, optional
        Pre-computed Dmin values (from compute_dmin on the full strip).
        If provided, rebate_mask is ignored for Dmin estimation.
    stock : str
        Film stock name for default coefficients ("kodak_gold", "kodak_portra").
    profile_path : str, optional
        Path to a JSON calibration profile. Overrides stock defaults.
    dark_rgb : (3,), optional
        Scanner dark level per channel. Defaults to 0.
    light_rgb : (3,), optional
        Scanner light field per channel. Defaults to 65535 for uint16.
    rebate_mask : HxW bool, optional
        Mask of unexposed rebate pixels for Dmin estimation. Only used
        if dmin is not provided.
    exposure_compensation : float
        Density-domain exposure shift (positive = brighter output).

    Returns
    -------
    scene_linear : HxWx3 float64
        Scene-linear RGB values, suitable for tone mapping / display rendering.
    """
    # Load calibration coefficients
    if profile_path is not None:
        coeffs, _meta = load_profile(profile_path)
        print(f"  Using calibration profile: {profile_path}")
    elif stock in STOCK_DEFAULTS:
        coeffs = STOCK_DEFAULTS[stock]()
        print(f"  Using default profile for {stock}")
    else:
        raise ValueError(f"Unknown film stock '{stock}' and no profile_path given. "
                         f"Available stocks: {list(STOCK_DEFAULTS.keys())}")

    # Stage 1: Measurement model
    print("  Converting to transmittance...")
    T = normalize_transmittance(raw_rgb, dark_rgb, light_rgb)

    print("  Computing density...")
    D = transmittance_to_density(T)

    # Stage 2: Film model — remove base+fog/orange mask
    if dmin is None:
        print("  Estimating Dmin...")
        dmin = estimate_dmin(D, rebate_mask=rebate_mask)
        print(f"    Dmin: R={dmin[0]:.4f} G={dmin[1]:.4f} B={dmin[2]:.4f}")

    net_D = subtract_dmin(D, dmin)

    # Optional exposure compensation (in density: positive = brighter)
    if exposure_compensation != 0.0:
        print(f"  Exposure compensation: {exposure_compensation:+.2f}")
        net_D = net_D - exposure_compensation

    # Stage 3: Calibrated stock+scanner transform
    # The polynomial maps scanner net-density → corrected density,
    # accounting for cross-channel dye coupling.
    print("  Applying stock+scanner density transform...")
    corrected_D = apply_density_transform(net_D, coeffs)

    # Stage 4: Net density IS the scene-linear signal.
    # After Dmin subtraction, higher density = more exposure = brighter scene.
    # The polynomial transform has already corrected for cross-channel coupling.
    # Values are non-negative (clipped at Dmin subtraction).
    scene_linear = np.maximum(corrected_D, 0.0)

    return scene_linear
