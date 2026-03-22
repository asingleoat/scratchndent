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
    """Approximate coefficients for Kodak Gold 200 on Epson V600.

    These are initial estimates based on typical Kodak Gold characteristics:
    - Strong orange mask (high Dmin in blue channel)
    - Moderate cross-channel coupling from overlapping dye absorptions
    - Cyan dye has significant red+green absorption
    - Magenta dye has green+some blue absorption
    - Yellow dye mainly blue absorption

    These should be replaced with properly calibrated coefficients from
    a ColorChecker shot when available.
    """
    coeffs = np.zeros((10, 3), dtype=np.float64)

    # Linear terms — primary channel mapping with cross-talk compensation
    # R output (mainly from cyan dye density in R channel)
    coeffs[0, 0] = 1.15    # R density → R
    coeffs[1, 0] = -0.10   # G density → R (compensate cyan dye green leak)
    coeffs[2, 0] = -0.05   # B density → R

    # G output (mainly from magenta dye density in G channel)
    coeffs[0, 1] = -0.08   # R density → G
    coeffs[1, 1] = 1.10    # G density → G
    coeffs[2, 1] = -0.06   # B density → G (compensate magenta blue leak)

    # B output (mainly from yellow dye density in B channel)
    coeffs[0, 2] = -0.03   # R density → B
    coeffs[1, 2] = -0.08   # G density → B
    coeffs[2, 2] = 0.95    # B density → B

    # Quadratic terms — gentle nonlinear correction for dye non-additivity
    coeffs[3, 0] = -0.15   # R² → R (highlight rolloff in red)
    coeffs[4, 1] = -0.12   # G² → G
    coeffs[5, 2] = -0.10   # B² → B

    # Cross terms — inter-channel coupling
    coeffs[6, 1] = 0.04    # RG → G
    coeffs[7, 0] = 0.03    # RB → R

    return coeffs


def default_kodak_portra_coeffs() -> np.ndarray:
    """Approximate coefficients for Kodak Portra 400 on Epson V600.

    Portra has finer grain, different dye set, and less aggressive orange
    mask compared to Gold. Known for accurate skin tones and lower contrast.
    """
    coeffs = np.zeros((10, 3), dtype=np.float64)

    # Linear terms — Portra has less cross-talk than Gold
    coeffs[0, 0] = 1.08
    coeffs[1, 0] = -0.06
    coeffs[2, 0] = -0.03

    coeffs[0, 1] = -0.05
    coeffs[1, 1] = 1.06
    coeffs[2, 1] = -0.04

    coeffs[0, 2] = -0.02
    coeffs[1, 2] = -0.05
    coeffs[2, 2] = 0.92

    # Gentler quadratic correction (Portra has wider latitude)
    coeffs[3, 0] = -0.10
    coeffs[4, 1] = -0.08
    coeffs[5, 2] = -0.07

    coeffs[6, 1] = 0.03
    coeffs[7, 0] = 0.02

    return coeffs
