"""Color space conversions, negadoctor inversion, and sigmoid tone mapping.

Implements darktable's negadoctor and sigmoid algorithms with numba-accelerated
kernels for parallel per-pixel processing.
"""

import numpy as np
from numba import njit, prange


# --- Constants ---

MIDDLE_GREY = 0.1845

# darktable's pipeline works in linear Rec2020 with D50 white point.
# Scanner TIFFs are sRGB (D65). We need to convert before/after processing.

# sRGB linear (D65) -> linear Rec2020 (D50)
# Combines: sRGB->XYZ(D65) -> Bradford adapt D65->D50 -> XYZ(D50)->Rec2020
M_SRGB_TO_REC2020_D50 = np.array([
    [0.62750372, 0.32927550, 0.04330266],
    [0.06910838, 0.91951916, 0.01135963],
    [0.01639405, 0.08801124, 0.89538034],
])

# Inverse: linear Rec2020 (D50) -> sRGB linear (D65)
M_REC2020_D50_TO_SRGB = np.array([
    [ 1.66022695, -0.58754781, -0.07283824],
    [-0.12455353,  1.13292614, -0.00834966],
    [-0.01815511, -0.10060300,  1.11899818],
])

# CIE CAT16 forward matrix (XYZ to LMS)
M_CAT16 = np.array([
    [ 0.401288,  0.650173, -0.051461],
    [-0.250268,  1.204414,  0.045854],
    [-0.002079,  0.048952,  0.953127],
])
M_CAT16_INV = np.linalg.inv(M_CAT16)

# D50 reference white point (darktable's pipeline white)
D50_xy = (0.3457, 0.3585)


# --- Numba kernels ---

@njit(parallel=True, cache=True)
def _negadoctor_kernel(
    img_flat, out_flat, n_pixels,
    Dmin_r, Dmin_g, Dmin_b,
    wb_high_r, wb_high_g, wb_high_b,
    wb_low_r, wb_low_g, wb_low_b,
    D_max, offset, exposure, black_effective, gamma, soft_clip,
):
    """Fused negadoctor kernel -- processes all pixels in parallel via numba."""
    THRESHOLD = 2.3283064365386963e-10
    LOG2_TO_LOG10 = 0.30103

    Dmin = (Dmin_r, Dmin_g, Dmin_b)
    wb_high = (wb_high_r, wb_high_g, wb_high_b)
    wb_low = (wb_low_r, wb_low_g, wb_low_b)

    for i in prange(n_pixels):
        for c in range(3):
            val = img_flat[i, c]

            # Transmission to density ratio
            density = Dmin[c] / max(val, THRESHOLD)

            # Log density (negated = inversion)
            log_density = np.log2(density) * -LOG2_TO_LOG10

            # White balance correction in log space
            wb_high_normed = wb_high[c] / D_max
            offset_precomp = wb_high[c] * offset * wb_low[c]
            corrected = wb_high_normed * log_density + offset_precomp

            # Print exposure + black point
            print_linear = -(exposure * 10.0 ** corrected + black_effective)
            if print_linear < 0.0:
                print_linear = 0.0

            # Gamma (paper grade)
            print_gamma = print_linear ** gamma

            # Soft-clip highlight rolloff
            if print_gamma > soft_clip:
                sc = soft_clip
                excess = print_gamma - sc
                print_gamma = sc + (1.0 - np.exp(-excess / (1.0 - sc))) * (1.0 - sc)

            out_flat[i, c] = print_gamma


@njit(parallel=True, cache=True)
def _sigmoid_kernel(img_flat, out_flat, n_pixels,
                    magnitude, paper_exp, film_fog, film_power, paper_power):
    """Fused sigmoid kernel -- per-pixel generalized log-logistic."""
    for i in prange(n_pixels):
        for c in range(3):
            val = img_flat[i, c]
            if val < 0.0:
                val = 0.0
            film_response = (film_fog + val) ** film_power
            paper_response = magnitude * (
                film_response / (paper_exp + film_response)
            ) ** paper_power
            if np.isnan(paper_response):
                paper_response = magnitude
            out_flat[i, c] = paper_response


@njit(parallel=True, cache=True)
def _srgb_to_linear_kernel(img_flat, out_flat, n_pixels):
    """sRGB gamma to linear, per-pixel parallel."""
    for i in prange(n_pixels):
        for c in range(3):
            v = img_flat[i, c]
            if v <= 0.04045:
                out_flat[i, c] = v / 12.92
            else:
                out_flat[i, c] = ((v + 0.055) / 1.055) ** 2.4


@njit(parallel=True, cache=True)
def _linear_to_srgb_kernel(img_flat, out_flat, n_pixels):
    """Linear to sRGB gamma, per-pixel parallel."""
    for i in prange(n_pixels):
        for c in range(3):
            v = img_flat[i, c]
            if v < 0.0:
                v = 0.0
            if v <= 0.0031308:
                out_flat[i, c] = v * 12.92
            else:
                out_flat[i, c] = 1.055 * v ** (1.0 / 2.4) - 0.055


@njit(parallel=True, cache=True)
def _color_matrix_kernel(img_flat, out_flat, n_pixels,
                         m00, m01, m02, m10, m11, m12, m20, m21, m22):
    """3x3 color matrix multiply, per-pixel parallel."""
    for i in prange(n_pixels):
        r = img_flat[i, 0]
        g = img_flat[i, 1]
        b = img_flat[i, 2]
        out_flat[i, 0] = m00 * r + m01 * g + m02 * b
        out_flat[i, 1] = m10 * r + m11 * g + m12 * b
        out_flat[i, 2] = m20 * r + m21 * g + m22 * b


# --- Color space functions ---

def _apply_color_matrix_fast(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply 3x3 color matrix using numba kernel."""
    h, w, _ = img.shape
    flat = img.reshape(-1, 3).astype(np.float64)
    out = np.empty_like(flat)
    m = matrix
    _color_matrix_kernel(flat, out, flat.shape[0],
                         m[0, 0], m[0, 1], m[0, 2],
                         m[1, 0], m[1, 1], m[1, 2],
                         m[2, 0], m[2, 1], m[2, 2])
    return out.reshape(h, w, 3)


def apply_color_matrix(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 color matrix to a linear RGB image."""
    return _apply_color_matrix_fast(img, matrix)


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB gamma-encoded values to linear light."""
    h, w, c = img.shape
    flat = img.reshape(-1, 3).astype(np.float64)
    out = np.empty_like(flat)
    _srgb_to_linear_kernel(flat, out, flat.shape[0])
    return out.reshape(h, w, c)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear light values to sRGB gamma-encoded."""
    h, w, c = img.shape
    flat = img.reshape(-1, 3).astype(np.float64)
    out = np.empty_like(flat)
    _linear_to_srgb_kernel(flat, out, flat.shape[0])
    return out.reshape(h, w, c)


# --- CAT16 chromatic adaptation ---

def _xy_to_XYZ(x: float, y: float) -> np.ndarray:
    """Convert CIE xy chromaticity to XYZ with Y=1."""
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def compute_cat16_matrix(scene_x: float, scene_y: float) -> np.ndarray:
    """Compute a CAT16 chromatic adaptation matrix.

    darktable's channelmixerrgb works in Rec2020/XYZ space, but we're
    applying this to sRGB-linear data after negadoctor inversion. Since
    negadoctor already flips the color relationships, we need the inverse
    adaptation: D50 -> scene illuminant (reducing blue for warm scenes).

    The full correct pipeline would convert sRGB->XYZ, adapt in XYZ, then
    XYZ->sRGB. We approximate by computing the combined matrix.
    """
    scene_XYZ = _xy_to_XYZ(scene_x, scene_y)
    d50_XYZ = _xy_to_XYZ(*D50_xy)

    scene_lms = M_CAT16 @ scene_XYZ
    d50_lms = M_CAT16 @ d50_XYZ

    # Rec2020 to XYZ (D50) -- darktable's working space
    M_Rec2020_to_XYZ_D65 = np.array([
        [0.6369580, 0.1446169, 0.1688810],
        [0.2627002, 0.6779981, 0.0593017],
        [0.0000000, 0.0280727, 1.0609851],
    ])
    # Bradford D65->D50
    M_Bradford = np.array([
        [ 0.8951,  0.2664, -0.1614],
        [-0.7502,  1.7135,  0.0367],
        [ 0.0389, -0.0685,  1.0296],
    ])
    D65_XYZ = np.array([0.95047, 1.0, 1.08883])
    D50_XYZ_ref = np.array([0.96429568, 1.0, 0.82510460])
    D65_LMS = M_Bradford @ D65_XYZ
    D50_LMS = M_Bradford @ D50_XYZ_ref
    brad_gains = D50_LMS / D65_LMS
    M_D65_to_D50 = np.linalg.inv(M_Bradford) @ np.diag(brad_gains) @ M_Bradford
    M_Rec2020_to_XYZ_D50 = M_D65_to_D50 @ M_Rec2020_to_XYZ_D65
    M_XYZ_D50_to_Rec2020 = np.linalg.inv(M_Rec2020_to_XYZ_D50)

    # CAT16 adaptation: D50 -> scene illuminant in XYZ space
    gains = scene_lms / d50_lms
    M_adapt_XYZ = M_CAT16_INV @ np.diag(gains) @ M_CAT16

    # Combined: Rec2020 -> XYZ(D50) -> CAT16 adapt -> XYZ(D50) -> Rec2020
    return M_XYZ_D50_to_Rec2020 @ M_adapt_XYZ @ M_Rec2020_to_XYZ_D50


# --- Sigmoid tone mapping ---

def _generalized_loglogistic_sigmoid(
    value: np.ndarray,
    magnitude: float,
    paper_exp: float,
    film_fog: float,
    film_power: float,
    paper_power: float,
) -> np.ndarray:
    """Generalized log-logistic sigmoid from darktable."""
    clamped = np.maximum(value, 0.0)
    film_response = np.power(film_fog + clamped, film_power)
    paper_response = magnitude * np.power(
        film_response / (paper_exp + film_response), paper_power
    )
    return np.where(np.isnan(paper_response), magnitude, paper_response)


def sigmoid_commit_params(params: dict) -> dict:
    """Replicate darktable's sigmoid commit_params to derive internal constants."""
    ref_film_power = params["middle_grey_contrast"]
    ref_paper_power = 1.0
    ref_magnitude = 1.0
    ref_film_fog = 0.0
    ref_paper_exposure = (
        (ref_film_fog + MIDDLE_GREY) ** ref_film_power
        * ((ref_magnitude / MIDDLE_GREY) - 1.0)
    )

    delta = 1e-6
    ref_slope = (
        _generalized_loglogistic_sigmoid(
            np.array([MIDDLE_GREY + delta]), ref_magnitude,
            ref_paper_exposure, ref_film_fog, ref_film_power, ref_paper_power
        )[0]
        - _generalized_loglogistic_sigmoid(
            np.array([MIDDLE_GREY - delta]), ref_magnitude,
            ref_paper_exposure, ref_film_fog, ref_film_power, ref_paper_power
        )[0]
    ) / (2.0 * delta)

    paper_power = 5.0 ** (-params["contrast_skewness"])
    white_target = params["display_white_target"]
    black_target = params["display_black_target"]

    temp_film_power = 1.0
    temp_white_grey_relation = (
        (white_target / MIDDLE_GREY) ** (1.0 / paper_power) - 1.0
    )
    temp_paper_exposure = MIDDLE_GREY ** temp_film_power * temp_white_grey_relation
    temp_slope = (
        _generalized_loglogistic_sigmoid(
            np.array([MIDDLE_GREY + delta]), white_target,
            temp_paper_exposure, ref_film_fog, temp_film_power, paper_power
        )[0]
        - _generalized_loglogistic_sigmoid(
            np.array([MIDDLE_GREY - delta]), white_target,
            temp_paper_exposure, ref_film_fog, temp_film_power, paper_power
        )[0]
    ) / (2.0 * delta)

    film_power = ref_slope / temp_slope

    white_grey_relation = (
        (white_target / MIDDLE_GREY) ** (1.0 / paper_power) - 1.0
    )
    white_black_relation = (
        (black_target / white_target) ** (-1.0 / paper_power) - 1.0
    )

    film_fog = (
        MIDDLE_GREY * white_grey_relation ** (1.0 / film_power)
        / (white_black_relation ** (1.0 / film_power)
           - white_grey_relation ** (1.0 / film_power))
    )
    paper_exposure = (
        (film_fog + MIDDLE_GREY) ** film_power * white_grey_relation
    )

    return {
        "white_target": white_target,
        "black_target": black_target,
        "paper_exposure": paper_exposure,
        "film_fog": film_fog,
        "film_power": film_power,
        "paper_power": paper_power,
    }


def apply_sigmoid(img: np.ndarray, params: dict) -> np.ndarray:
    """Apply darktable's sigmoid tone mapping (per-channel mode).

    Maps scene-referred linear values to display-referred [0, white_target].
    """
    d = sigmoid_commit_params(params)

    print(f"  Sigmoid internals: film_power={d['film_power']:.4f} "
          f"paper_power={d['paper_power']:.4f} film_fog={d['film_fog']:.6f} "
          f"paper_exp={d['paper_exposure']:.6f}")

    h, w, _ = img.shape
    flat = img.reshape(-1, 3).astype(np.float64)
    out = np.empty_like(flat)
    _sigmoid_kernel(flat, out, flat.shape[0],
                    d["white_target"], d["paper_exposure"],
                    d["film_fog"], d["film_power"], d["paper_power"])
    return out.reshape(h, w, 3)


# --- Negadoctor ---

def negadoctor(rgb: np.ndarray, params: dict) -> np.ndarray:
    """Apply the negadoctor algorithm to convert a negative to positive.

    Reimplementation of darktable's negadoctor.c _process_pixel().
    All math in float64 for precision on 16-bit data.

    Pipeline:
      1. Normalize to [0,1] and compute transmission-to-density ratio
      2. Log density (log10)
      3. White balance correction in log space
      4. Print exposure + black point
      5. Gamma (paper grade)
      6. Soft-clip highlight rolloff
    """
    Dmin = params["Dmin"]
    wb_high = params["wb_high"]
    wb_low = params["wb_low"]
    D_max = params["D_max"]
    offset = params["offset"]
    black = params["black"]
    gamma = params["gamma"]
    soft_clip = params["soft_clip"]
    exposure = params["exposure"]

    black_effective = -exposure * (1.0 + black)

    h, w, _ = rgb.shape
    img = rgb.astype(np.float64) / 65535.0

    # sRGB to linear (numba parallel)
    flat = img.reshape(-1, 3)
    lin_flat = np.empty_like(flat)
    _srgb_to_linear_kernel(flat, lin_flat, flat.shape[0])

    # Linear sRGB to Rec2020 D50 (numba parallel)
    rec_flat = np.empty_like(lin_flat)
    m = M_SRGB_TO_REC2020_D50
    _color_matrix_kernel(lin_flat, rec_flat, flat.shape[0],
                         m[0, 0], m[0, 1], m[0, 2],
                         m[1, 0], m[1, 1], m[1, 2],
                         m[2, 0], m[2, 1], m[2, 2])

    # Negadoctor (numba parallel)
    out_flat = np.empty_like(rec_flat)
    _negadoctor_kernel(
        rec_flat, out_flat, flat.shape[0],
        Dmin[0], Dmin[1], Dmin[2],
        wb_high[0], wb_high[1], wb_high[2],
        wb_low[0], wb_low[1], wb_low[2],
        D_max, offset, exposure, black_effective, gamma, soft_clip,
    )

    return out_flat.reshape(h, w, 3)
