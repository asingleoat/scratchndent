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
    default_kodak_gold_coeffs,
    default_kodak_portra_coeffs,
    load_profile,
)


STOCK_DEFAULTS = {
    "kodak_gold": default_kodak_gold_coeffs,
    "kodak_portra": default_kodak_portra_coeffs,
}


def exposure_anchor(
    net_density: np.ndarray,
    neutral_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-channel exposure anchor point.

    Used for optional global exposure normalization. A neutral mask
    (known grey area) gives the most accurate anchor; otherwise we
    use the median density as a scene heuristic.
    """
    if neutral_mask is not None:
        pixels = net_density[neutral_mask]
        if len(pixels) > 0:
            return np.median(pixels, axis=0)

    flat = net_density.reshape(-1, 3)
    return np.median(flat, axis=0)


def apply_exposure_normalization(
    net_density: np.ndarray,
    anchor: np.ndarray,
    target_anchor: np.ndarray,
) -> np.ndarray:
    """Affine shift in density space to normalize exposure.

    This is equivalent to adjusting the exposure/white-balance in the
    density domain, which is more physically correct than doing it
    after RGB conversion.
    """
    delta = target_anchor - anchor
    return net_density + delta[None, None, :]


def apply_neutral_balance(
    net_density: np.ndarray,
    neutral_mask: np.ndarray | None = None,
    target_neutral: np.ndarray | None = None,
) -> np.ndarray:
    """Balance channels so neutral areas have equal density.

    In density space, a neutral (grey) subject should have roughly equal
    density across channels after Dmin subtraction. Any residual per-channel
    offset is from scene illuminant color cast or film response differences.

    Parameters
    ----------
    net_density : HxWx3
        Dmin-subtracted density.
    neutral_mask : HxW bool, optional
        Mask of known neutral areas. If None, uses median of the image.
    target_neutral : (3,), optional
        Target density for neutrals. Defaults to the mean across channels
        of the measured neutral.
    """
    anchor = exposure_anchor(net_density, neutral_mask)

    if target_neutral is None:
        # Make neutrals equal by shifting to the channel mean
        target_neutral = np.full(3, np.mean(anchor))

    return apply_exposure_normalization(net_density, anchor, target_neutral)


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
    neutral_balance: bool = True,
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
    neutral_balance : bool
        If True, balance channels in density space for neutral rendition.
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

    # Optional neutral balance in density domain
    if neutral_balance:
        print("  Applying neutral balance in density space...")
        net_D = apply_neutral_balance(net_D)

    # Optional exposure compensation (in density: negative = brighter)
    if exposure_compensation != 0.0:
        print(f"  Exposure compensation: {exposure_compensation:+.2f}")
        net_D = net_D - exposure_compensation

    # Stage 3: Calibrated stock+scanner transform
    # The polynomial maps scanner net-density → corrected density,
    # accounting for cross-channel dye coupling.
    print("  Applying stock+scanner density transform...")
    corrected_D = apply_density_transform(net_D, coeffs)

    # Stage 4: Density → scene-linear
    # D = -log10(T), so T = 10^(-D). Higher density = less light = darker.
    scene_linear = np.power(10.0, -corrected_D)
    scene_linear = np.maximum(scene_linear, 0.0)

    return scene_linear
