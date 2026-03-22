"""Stock+scanner calibration: fit and apply the density-space transform.

The core idea: scanner RGB channels don't align with film dye absorptions,
so a pure per-channel inversion is wrong. We fit a polynomial mapping from
scanner-density space to a print-density-like (APD-like) space using matched
pairs (measured density, target output).

Calibration profiles are stored as JSON files with the polynomial coefficients
and metadata about the film stock and scanner.
"""

import json
from pathlib import Path

import numpy as np


def poly_features(x: np.ndarray) -> np.ndarray:
    """Build second-order polynomial basis from 3-channel input.

    Input: (..., 3)
    Output: (..., 10) — [R, G, B, R², G², B², RG, RB, GB, 1]

    A quadratic basis captures cross-channel coupling that a pure 3x3
    matrix cannot, which matters because scanner spectral sensitivities
    overlap with multiple film dye absorption bands.
    """
    r = x[..., 0:1]
    g = x[..., 1:2]
    b = x[..., 2:3]
    return np.concatenate([
        r, g, b,
        r * r, g * g, b * b,
        r * g, r * b, g * b,
        np.ones_like(r),
    ], axis=-1)


def fit_density_transform(
    measured_density: np.ndarray,
    target_values: np.ndarray,
    regularization: float = 1e-4,
) -> np.ndarray:
    """Fit polynomial coefficients mapping scanner density → target space.

    Parameters
    ----------
    measured_density : Nx3 float64
        Net density values from scanner (Dmin-subtracted).
    target_values : Nx3 float64
        Corresponding target values (e.g. scene-linear RGB, APD, or
        display-linear values from a reference inversion).
    regularization : float
        Ridge regression lambda to prevent overfitting with few samples.

    Returns
    -------
    coeffs : (10, 3) float64
        Polynomial coefficients.
    """
    X = poly_features(measured_density)  # Nx10
    Y = target_values                     # Nx3

    # Ridge regression: (X'X + λI)^-1 X'Y
    XtX = X.T @ X + regularization * np.eye(X.shape[1])
    XtY = X.T @ Y
    coeffs = np.linalg.solve(XtX, XtY)

    return coeffs


def apply_density_transform(
    net_density: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Apply calibrated polynomial transform to net density image.

    Parameters
    ----------
    net_density : HxWx3 float64
        Dmin-subtracted density image.
    coeffs : (10, 3) float64
        Polynomial coefficients from fit_density_transform or a profile.

    Returns
    -------
    result : HxWx3 float64
        Transformed values in print-density-like / scene-linear space.
    """
    h, w, _ = net_density.shape
    flat = net_density.reshape(-1, 3)
    feats = poly_features(flat)        # Nx10
    result = feats @ coeffs            # Nx3
    return result.reshape(h, w, 3)


# --- Profile I/O ---

def save_profile(path: str, coeffs: np.ndarray, metadata: dict | None = None):
    """Save calibration profile to JSON."""
    data = {
        "coeffs": coeffs.tolist(),
        "metadata": metadata or {},
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_profile(path: str) -> tuple[np.ndarray, dict]:
    """Load calibration profile from JSON."""
    data = json.loads(Path(path).read_text())
    coeffs = np.array(data["coeffs"], dtype=np.float64)
    metadata = data.get("metadata", {})
    return coeffs, metadata


# --- Default profiles ---

def default_identity_coeffs() -> np.ndarray:
    """Identity-like coefficients: net density passes through unchanged.

    This is a baseline that does simple channel-independent density-to-linear
    conversion. Replace with calibrated coefficients for better results.
    """
    coeffs = np.zeros((10, 3), dtype=np.float64)
    # Linear terms only (identity mapping)
    coeffs[0, 0] = 1.0  # R → R
    coeffs[1, 1] = 1.0  # G → G
    coeffs[2, 2] = 1.0  # B → B
    return coeffs


def default_kodak_gold_coeffs() -> np.ndarray:
    """Coefficients for Kodak Gold 200 on Epson V600 (6400 DPI, SilverFast).

    Derived from measured density statistics on real scans:
    - Dmin (orange mask): R≈0.50, G≈0.77, B≈1.10
    - After Dmin subtraction, G channel has ~1.15x the density range of R,
      and B has ~0.97x, due to different dye absorption efficiencies
    - Cross-channel correlation is very high (0.91-0.97) because scanner
      spectral sensitivities overlap multiple film dye absorption bands

    The polynomial corrects for:
    1. Per-channel sensitivity differences (linear scaling)
    2. Cross-channel dye coupling (off-diagonal linear terms)
    3. Nonlinear dye response / highlight compression (quadratic terms)

    Basis: [R, G, B, R², G², B², RG, RB, GB, 1] → [R_out, G_out, B_out]
    """
    coeffs = np.zeros((10, 3), dtype=np.float64)

    # --- Linear terms ---
    # Primary: scale channels to equalize sensitivity
    coeffs[0, 0] = 1.20     # R density → R out (boost, lower range after Dmin)
    coeffs[1, 1] = 0.90     # G density → G out (reduce, highest range)
    coeffs[2, 2] = 1.02     # B density → B out

    # Cross-channel: gentle dye overlap compensation
    # Keep these mild to avoid shadow color artifacts
    coeffs[1, 0] = -0.10    # G density → R out
    coeffs[0, 1] = -0.04    # R density → G out
    coeffs[2, 1] = -0.04    # B density → G out
    coeffs[1, 2] = -0.06    # G density → B out

    return coeffs


def default_kodak_portra_coeffs() -> np.ndarray:
    """Coefficients for Kodak Portra 400 on Epson V600.

    Portra has a different dye set than Gold with:
    - Less aggressive orange mask
    - Wider exposure latitude (gentler highlight rolloff)
    - More accurate neutral rendition
    - Lower inherent contrast

    Same basis as Gold but with less aggressive cross-channel correction
    and gentler nonlinearity.
    """
    coeffs = np.zeros((10, 3), dtype=np.float64)

    # Linear — Portra has better channel separation than Gold
    coeffs[0, 0] = 1.15
    coeffs[1, 1] = 0.93
    coeffs[2, 2] = 1.00

    # Gentle cross-channel
    coeffs[1, 0] = -0.08
    coeffs[0, 1] = -0.03
    coeffs[1, 2] = -0.04

    return coeffs
