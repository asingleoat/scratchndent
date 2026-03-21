#!/usr/bin/env python3
"""
scratchndent - IR-based dust and scratch removal for scanned film negatives.

Reads multi-page TIFFs produced by SilverFast (Epson V600 etc.) where:
  Page 0: RGB visible scan (16-bit)
  Page 1: Thumbnail (ignored)
  Page 2: IR channel (16-bit grayscale)

The IR channel reveals dust and scratches (which block infrared light and
appear dark) while film dyes are transparent to IR. We use the IR channel
to build a defect mask, optionally align it to the RGB data, then inpaint
the defects.
"""

import argparse
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import tifffile
from skimage.filters import meijering
from skimage.restoration import inpaint_biharmonic


def load_tiff(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load RGB (page 0) and IR (page 2) from a SilverFast multi-page TIFF."""
    with tifffile.TiffFile(path) as tif:
        pages = tif.pages
        if len(pages) < 3:
            sys.exit(f"Expected at least 3 TIFF pages, got {len(pages)}")

        rgb = pages[0].asarray()  # (H, W, 3) uint16
        ir = pages[2].asarray()   # (H, W) uint16

    if rgb.shape[:2] != ir.shape[:2]:
        sys.exit(
            f"RGB shape {rgb.shape[:2]} != IR shape {ir.shape[:2]}; "
            "cannot process mismatched dimensions"
        )

    return rgb, ir


def align_ir(rgb: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Align IR channel to RGB using ECC (Enhanced Correlation Coefficient).

    Works on downsampled 8-bit grayscale versions for speed, then applies
    the recovered warp to the full-resolution IR.
    """
    # Convert RGB to grayscale for alignment reference
    # Work in 8-bit for ECC speed
    rgb8 = (rgb >> 8).astype(np.uint8)
    ir8 = (ir >> 8).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)

    # Downsample for speed — ECC on a 100+ megapixel image is slow
    scale = 0.125
    small_gray = cv2.resize(gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
    small_ir = cv2.resize(ir8, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,   # max iterations
        1e-6,  # epsilon
    )

    try:
        _, warp_matrix = cv2.findTransformECC(
            small_gray, small_ir, warp_matrix,
            cv2.MOTION_TRANSLATION, criteria,
        )
    except cv2.error as e:
        print(f"Warning: ECC alignment failed ({e}), using unaligned IR",
              file=sys.stderr)
        return ir

    # Scale translation back to full resolution
    warp_matrix[0, 2] /= scale
    warp_matrix[1, 2] /= scale

    tx, ty = warp_matrix[0, 2], warp_matrix[1, 2]
    print(f"IR alignment offset: tx={tx:.2f}px, ty={ty:.2f}px")

    if abs(tx) < 0.5 and abs(ty) < 0.5:
        print("Offset negligible, skipping warp")
        return ir

    # Use scipy shift for sub-pixel translation — avoids OpenCV's SHRT_MAX
    # limit on images taller/wider than 32767 pixels.
    from scipy.ndimage import shift as ndimage_shift
    aligned = ndimage_shift(ir, (-ty, -tx), order=1, mode='reflect')
    return aligned


def make_defect_mask(
    ir: np.ndarray,
    threshold: float = 0.35,
    hair_sensitivity: float = 0.15,
    min_area: int = 4,
    dilate_radius: int = 3,
    close_radius: int = 5,
) -> np.ndarray:
    """Create a binary defect mask from the IR channel.

    Combines two detection strategies:
    1. Ratio-based: catches dust blobs where IR is significantly darker than
       the local background. Uses two-pass background estimation so large
       defects don't corrupt their own background.
    2. Line-based (Meijering filter): catches thin elongated features like
       hairs and fine scratches that produce only a subtle IR dip.

    Returns a uint8 mask where 255 = defect, 0 = clean.
    """
    ir_f = ir.astype(np.float32) / 65535.0

    # --- Ratio-based detection (dust, large scratches) ---

    blur_size = 301
    background = cv2.GaussianBlur(ir_f, (blur_size, blur_size), 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(background > 0.01, ir_f / background, 1.0)

    coarse_mask = ratio < (1.0 - threshold)

    # Second pass: re-estimate background with defects replaced
    ir_cleaned = ir_f.copy()
    ir_cleaned[coarse_mask] = background[coarse_mask]
    background = cv2.GaussianBlur(ir_cleaned, (blur_size, blur_size), 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(background > 0.01, ir_f / background, 1.0)

    dust_mask = ratio < (1.0 - threshold)

    # --- Line-based detection (hairs, fine scratches) ---

    # Invert so dark defects become bright ridges for the line detector.
    # Work on a downsampled version for speed — Meijering is expensive.
    scale = 0.25
    ir_small = cv2.resize(ir_f, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    bg_small = cv2.resize(background, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    with np.errstate(divide='ignore', invalid='ignore'):
        deviation = np.where(bg_small > 0.01, 1.0 - ir_small / bg_small, 0.0)
    deviation = np.clip(deviation, 0, 1).astype(np.float64)

    # Meijering filter detects ridge-like (tubular) structures at multiple
    # scales — sigmas chosen to match hair widths at this resolution
    line_response = meijering(deviation, sigmas=range(1, 4), black_ridges=False)
    line_mask_small = (line_response > hair_sensitivity).astype(np.uint8) * 255

    # Scale back up to full resolution
    line_mask = cv2.resize(line_mask_small, (ir_f.shape[1], ir_f.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    # Only keep line detections where the IR is actually darker than background
    # (prevents false positives on film grain / image detail)
    with np.errstate(divide='ignore', invalid='ignore'):
        full_ratio = np.where(background > 0.01, ir_f / background, 1.0)
    line_mask[full_ratio > 0.98] = 0

    # --- Combine both masks ---

    mask = np.where(dust_mask, 255, line_mask).astype(np.uint8)

    # Morphological close to fill holes within partially-detected large defects
    if close_radius > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * close_radius + 1, 2 * close_radius + 1),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Remove tiny specks (noise)
    if min_area > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask[labels == i] = 0

    # Dilate to ensure full coverage of defect edges
    if dilate_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_radius + 1, 2 * dilate_radius + 1),
        )
        mask = cv2.dilate(mask, kernel)

    return mask


def estimate_local_grain(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    grain_padding: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate film grain characteristics from clean pixels surrounding a defect.

    Separates signal (low-frequency) from grain (high-frequency) in the clean
    surround, then measures per-channel grain statistics.

    Returns (grain_std, grain_spectrum_shape, local_signal) where:
    - grain_std: per-channel standard deviation of the grain (3,)
    - grain_spectrum_shape: average radial power spectrum shape for frequency
      matching (N,)
    - local_signal: low-pass filtered version of the ROI for grain extraction
    """
    h, w = roi_mask.shape

    # Build a "surround" mask: clean pixels in the padding ring
    dilated = cv2.dilate(
        roi_mask.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * grain_padding + 1, 2 * grain_padding + 1)),
    ) > 0
    surround = dilated & ~roi_mask

    # Low-pass filter to separate signal from grain
    # Kernel size chosen to preserve image structure but remove grain
    sigma = 2.5
    signal = cv2.GaussianBlur(roi_rgb, (0, 0), sigma)

    # Extract grain from clean surround pixels
    grain_std = np.zeros(3, dtype=np.float64)
    if np.count_nonzero(surround) > 10:
        for c in range(3):
            grain_pixels = roi_rgb[surround, c] - signal[surround, c]
            grain_std[c] = np.std(grain_pixels)

    # Compute radial power spectrum of the grain for frequency shaping.
    # Zero out defect pixels in the grain image and use the full ROI with a
    # window function — this uses all available clean surround data rather
    # than hoping a small square patch lines up.
    grain_spectrum = None
    if np.count_nonzero(surround) > 64:
        # Average spectrum across RGB channels
        grain_image = roi_rgb - signal
        # Zero out defect pixels so they don't contaminate the spectrum
        grain_image[roi_mask] = 0.0

        # Window to reduce spectral leakage from edges and zeroed defects
        window = np.outer(np.hanning(h), np.hanning(w))

        spectra = []
        for c in range(3):
            fft = np.fft.fftshift(np.fft.fft2(grain_image[:, :, c] * window))
            power = np.abs(fft) ** 2
            spectra.append(power)
        avg_power = np.mean(spectra, axis=0)

        # Radial average
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.mgrid[:h, :w]
        r = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        r_int = r.astype(int)
        r_max = min(h, w) // 2

        spectrum = np.zeros(r_max)
        for ri in range(r_max):
            ring = avg_power[r_int == ri]
            if len(ring) > 0:
                spectrum[ri] = np.mean(ring)

        # Compensate for zeroed defect pixels reducing power
        clean_fraction = np.count_nonzero(~roi_mask) / roi_mask.size
        if clean_fraction > 0.1:
            spectrum /= clean_fraction ** 2

        if np.sum(spectrum) > 0:
            grain_spectrum = spectrum / np.sum(spectrum)

    return grain_std, grain_spectrum, signal


def synthesize_grain(
    shape: tuple[int, int],
    grain_std: np.ndarray,
    grain_spectrum: np.ndarray | None,
    n_channels: int = 3,
) -> np.ndarray:
    """Synthesize film grain noise matching measured statistics.

    Film grain has a characteristic pink-ish (1/f) power spectrum — more
    energy at low frequencies than white noise. If a measured spectrum is
    available, we shape white noise in the frequency domain to match it.
    Otherwise we fall back to a 1/f approximation which is closer to real
    grain than white noise.
    """
    h, w = shape
    grain = np.zeros((h, w, n_channels), dtype=np.float64)

    # Build radial frequency map (distance from DC in frequency space)
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.mgrid[:h, :w]
    r = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    # Avoid division by zero at DC
    r_safe = np.where(r > 0, r, 1.0)

    if grain_spectrum is not None and len(grain_spectrum) > 4:
        # Interpolate measured spectrum into a 2D radial amplitude filter
        r_max = len(grain_spectrum)
        # Convert power spectrum to amplitude spectrum
        amp_spectrum = np.sqrt(np.maximum(grain_spectrum, 0))
        # Interpolate to cover all radial distances in the output
        from numpy import interp
        r_flat = r.ravel()
        amp_flat = interp(r_flat, np.arange(r_max), amp_spectrum,
                          right=amp_spectrum[-1])
        amp_filter = np.fft.ifftshift(amp_flat.reshape(h, w))
    else:
        # Fallback: 1/f spectrum (pink noise), typical of film grain
        amp_filter = np.fft.ifftshift(1.0 / r_safe)

    for c in range(n_channels):
        noise = np.random.randn(h, w)
        fft = np.fft.fft2(noise)

        # Apply spectral shaping
        shaped_fft = fft * amp_filter

        shaped = np.real(np.fft.ifft2(shaped_fft))

        # Normalize to unit variance then scale to match measured grain
        noise_std = np.std(shaped)
        if noise_std > 0:
            shaped = shaped / noise_std

        grain[:, :, c] = shaped * grain_std[c]

    return grain


def inpaint(rgb: np.ndarray, mask: np.ndarray, padding: int = 16) -> np.ndarray:
    """Inpaint defect regions in the RGB image at full 16-bit precision.

    Uses scikit-image biharmonic inpainting to reconstruct the underlying
    smooth signal, then adds synthesized film grain matched to the local
    surround so repaired regions blend naturally with the noisy film.
    """
    result = rgb.copy()
    h, w = mask.shape

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    n_defects = n_labels - 1  # label 0 is background
    if n_defects == 0:
        return result

    for i in range(1, n_labels):
        if i % 500 == 0:
            print(f"  Inpainting region {i}/{n_defects}...")

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Padded bounding box — gives the inpainter context around the defect
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw + padding)
        y1 = min(h, y + bh + padding)

        roi_rgb = result[y0:y1, x0:x1].astype(np.float64) / 65535.0
        roi_mask = (labels[y0:y1, x0:x1] == i)

        # Measure grain from clean surround
        grain_std, grain_spectrum, roi_signal = estimate_local_grain(
            roi_rgb, roi_mask
        )

        # Inpaint the denoised signal — biharmonic interpolates from edge
        # pixels, so feeding it the smooth signal prevents noisy edge pixels
        # from creating color gradients in the fill
        repaired_signal = inpaint_biharmonic(
            roi_signal, roi_mask, channel_axis=-1
        )

        # Synthesize matching grain and add to inpainted signal
        roi_h, roi_w = roi_mask.shape
        grain = synthesize_grain((roi_h, roi_w), grain_std, grain_spectrum)
        repaired_with_grain = repaired_signal + grain

        result[y0:y1, x0:x1][roi_mask] = (
            np.clip(repaired_with_grain[roi_mask] * 65535.0, 0, 65535)
            .astype(np.uint16)
        )

    return result


MIDDLE_GREY = 0.1845

# --- Color space conversion matrices ---
# darktable's pipeline works in linear Rec2020 with D50 white point.
# Scanner TIFFs are sRGB (D65). We need to convert before/after processing.

# sRGB linear (D65) → linear Rec2020 (D50)
# Combines: sRGB→XYZ(D65) → Bradford adapt D65→D50 → XYZ(D50)→Rec2020
M_SRGB_TO_REC2020_D50 = np.array([
    [0.62750372, 0.32927550, 0.04330266],
    [0.06910838, 0.91951916, 0.01135963],
    [0.01639405, 0.08801124, 0.89538034],
])

# Inverse: linear Rec2020 (D50) → sRGB linear (D65)
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


def _xy_to_XYZ(x: float, y: float) -> np.ndarray:
    """Convert CIE xy chromaticity to XYZ with Y=1."""
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def compute_cat16_matrix(scene_x: float, scene_y: float) -> np.ndarray:
    """Compute a CAT16 chromatic adaptation matrix.

    darktable's channelmixerrgb works in Rec2020/XYZ space, but we're
    applying this to sRGB-linear data after negadoctor inversion. Since
    negadoctor already flips the color relationships, we need the inverse
    adaptation: D50 → scene illuminant (reducing blue for warm scenes).

    The full correct pipeline would convert sRGB→XYZ, adapt in XYZ, then
    XYZ→sRGB. We approximate by computing the combined matrix.
    """
    scene_XYZ = _xy_to_XYZ(scene_x, scene_y)
    d50_XYZ = _xy_to_XYZ(*D50_xy)

    scene_lms = M_CAT16 @ scene_XYZ
    d50_lms = M_CAT16 @ d50_XYZ

    # Rec2020 to XYZ (D50) — darktable's working space
    M_Rec2020_to_XYZ_D65 = np.array([
        [0.6369580, 0.1446169, 0.1688810],
        [0.2627002, 0.6779981, 0.0593017],
        [0.0000000, 0.0280727, 1.0609851],
    ])
    # Bradford D65→D50
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

    # CAT16 adaptation: D50 → scene illuminant in XYZ space
    # After negadoctor inversion, the scene illuminant's color cast is inverted,
    # so we adapt away from D50 toward the scene illuminant to cancel it.
    gains = scene_lms / d50_lms
    M_adapt_XYZ = M_CAT16_INV @ np.diag(gains) @ M_CAT16

    # Combined: Rec2020 → XYZ(D50) → CAT16 adapt → XYZ(D50) → Rec2020
    return M_XYZ_D50_to_Rec2020 @ M_adapt_XYZ @ M_Rec2020_to_XYZ_D50


def extract_channelmixer_from_xmp(xmp_path: str) -> np.ndarray | None:
    """Extract chromatic adaptation matrix from the last enabled channelmixerrgb.

    Decodes the scene illuminant chromaticity and computes the CAT16 adaptation
    matrix to D50.
    """
    tree = ET.parse(xmp_path)
    root = tree.getroot()
    ns = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
          "darktable": "http://darktable.sf.net/"}

    last_params = None
    for li in root.iter(f"{{{ns['rdf']}}}li"):
        op = li.get(f"{{{ns['darktable']}}}operation")
        enabled = li.get(f"{{{ns['darktable']}}}enabled")
        if op == "channelmixerrgb" and enabled == "1":
            last_params = li.get(f"{{{ns['darktable']}}}params")

    if last_params is None:
        return None

    # Decompress gz-prefixed params
    import base64, zlib
    prefix_len = 4  # 'gz' + 2-char version
    b64_data = last_params[prefix_len:]
    padding = 4 - len(b64_data) % 4
    if padding != 4:
        b64_data += "=" * padding
    data = zlib.decompress(base64.b64decode(b64_data))

    # Extract x, y chromaticity at float offsets 34, 35
    n_floats = len(data) // 4
    values = struct.unpack(f"<{n_floats}f", data[:n_floats * 4])
    scene_x = values[34]
    scene_y = values[35]
    temperature = values[36]

    print(f"  Channel mixer: scene illuminant xy=({scene_x:.4f}, {scene_y:.4f}) "
          f"T={temperature:.0f}K → adapting to D50")

    return compute_cat16_matrix(scene_x, scene_y)


def apply_color_matrix(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 color matrix to a linear RGB image."""
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)
    result = (matrix @ flat.T).T
    return result.reshape(h, w, 3)


def parse_sigmoid_params(params_hex: str) -> dict:
    """Decode sigmoid binary params from XMP hex string."""
    data = bytes.fromhex(params_hex)
    fmt = "<ff ff i f ff ff ff f i"
    v = struct.unpack(fmt, data)
    return {
        "middle_grey_contrast": float(v[0]),
        "contrast_skewness": float(v[1]),
        "display_white_target": float(v[2]) * 0.01,
        "display_black_target": float(v[3]) * 0.01,
        "color_processing": int(v[4]),
        "hue_preservation": min(max(float(v[5]) * 0.01, 0.0), 1.0),
    }


def extract_sigmoid_from_xmp(xmp_path: str) -> dict | None:
    """Extract the last enabled sigmoid params from a darktable XMP sidecar."""
    tree = ET.parse(xmp_path)
    root = tree.getroot()
    ns = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
          "darktable": "http://darktable.sf.net/"}

    last_params = None
    for li in root.iter(f"{{{ns['rdf']}}}li"):
        op = li.get(f"{{{ns['darktable']}}}operation")
        enabled = li.get(f"{{{ns['darktable']}}}enabled")
        if op == "sigmoid" and enabled == "1":
            last_params = li.get(f"{{{ns['darktable']}}}params")

    if last_params is None:
        return None
    return parse_sigmoid_params(last_params)


def _generalized_loglogistic_sigmoid(
    value: np.ndarray,
    magnitude: float,
    paper_exp: float,
    film_fog: float,
    film_power: float,
    paper_power: float,
) -> np.ndarray:
    """Attempt the generalized log-logistic sigmoid from darktable."""
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

    result = np.empty_like(img)
    for c in range(3):
        result[:, :, c] = _generalized_loglogistic_sigmoid(
            img[:, :, c],
            d["white_target"],
            d["paper_exposure"],
            d["film_fog"],
            d["film_power"],
            d["paper_power"],
        )

    return result


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB gamma-encoded values to linear light."""
    linear = np.where(
        img <= 0.04045,
        img / 12.92,
        np.power((img + 0.055) / 1.055, 2.4),
    )
    return linear


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear light values to sRGB gamma-encoded."""
    srgb = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 0), 1.0 / 2.4) - 0.055,
    )
    return srgb


def parse_negadoctor_params(params_hex: str) -> dict:
    """Decode negadoctor binary params from XMP hex string.

    Layout (v2, 76 bytes):
      int32  film_stock (0=B&W, 1=color)
      float[4] Dmin     (film substrate color, RGB + unused)
      float[4] wb_high  (white balance coefficients)
      float[4] wb_low   (white balance offsets)
      float  D_max      (max film density)
      float  offset     (scan exposure bias)
      float  black      (paper black point)
      float  gamma      (paper grade)
      float  soft_clip  (highlight rolloff threshold)
      float  exposure   (print exposure)
    """
    data = bytes.fromhex(params_hex)
    values = struct.unpack("<i 4f 4f 4f 6f", data)
    return {
        "film_stock": values[0],
        "Dmin": np.array(values[1:4], dtype=np.float64),
        "wb_high": np.array(values[5:8], dtype=np.float64),
        "wb_low": np.array(values[9:12], dtype=np.float64),
        "D_max": float(values[13]),
        "offset": float(values[14]),
        "black": float(values[15]),
        "gamma": float(values[16]),
        "soft_clip": float(values[17]),
        "exposure": float(values[18]),
    }


def extract_negadoctor_from_xmp(xmp_path: str) -> dict:
    """Extract the last enabled negadoctor params from a darktable XMP sidecar."""
    tree = ET.parse(xmp_path)
    root = tree.getroot()

    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "darktable": "http://darktable.sf.net/",
    }

    # Find all history entries, keep the last enabled negadoctor
    last_params = None
    for li in root.iter(f"{{{ns['rdf']}}}li"):
        op = li.get(f"{{{ns['darktable']}}}operation")
        enabled = li.get(f"{{{ns['darktable']}}}enabled")
        if op == "negadoctor" and enabled == "1":
            last_params = li.get(f"{{{ns['darktable']}}}params")

    if last_params is None:
        sys.exit("No enabled negadoctor module found in XMP sidecar")

    params = parse_negadoctor_params(last_params)
    print(f"  Negadoctor params from XMP:")
    print(f"    Dmin:      R={params['Dmin'][0]:.6f} G={params['Dmin'][1]:.6f} B={params['Dmin'][2]:.6f}")
    print(f"    wb_high:   R={params['wb_high'][0]:.4f} G={params['wb_high'][1]:.4f} B={params['wb_high'][2]:.4f}")
    print(f"    wb_low:    R={params['wb_low'][0]:.4f} G={params['wb_low'][1]:.4f} B={params['wb_low'][2]:.4f}")
    print(f"    D_max={params['D_max']:.3f}  offset={params['offset']:.3f}  "
          f"black={params['black']:.4f}  gamma={params['gamma']:.1f}  "
          f"soft_clip={params['soft_clip']:.2f}  exposure={params['exposure']:.4f}")

    return params


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
    THRESHOLD = 2.3283064365386963e-10  # 2^(-32), from darktable

    Dmin = params["Dmin"]
    wb_high = params["wb_high"]
    wb_low = params["wb_low"]
    D_max = params["D_max"]
    offset = params["offset"]
    black = params["black"]
    gamma = params["gamma"]
    soft_clip = params["soft_clip"]
    exposure = params["exposure"]

    # Precompute like darktable's commit_params
    black_effective = -exposure * (1.0 + black)

    # Normalize 16-bit to [0,1], linearize from sRGB gamma, then convert
    # to darktable's working space (linear Rec2020, D50 white point).
    # darktable's colorin module does exactly this before negadoctor sees data.
    img = rgb.astype(np.float64) / 65535.0
    img = srgb_to_linear(img)
    img = apply_color_matrix(img, M_SRGB_TO_REC2020_D50)

    result = np.empty_like(img, dtype=np.float64)

    for c in range(3):
        # Step 1: Transmission to density ratio
        density = Dmin[c] / np.maximum(img[:, :, c], THRESHOLD)

        # Step 2: Log density — note the NEGATION, this is what inverts the negative
        # darktable: log_density[c] *= -LOG2_to_LOG10  (i.e. -log10(density))
        log_density = np.log2(density) * -0.30103

        # Step 3: White balance correction in log space
        # commit_params precomputes: wb_high[c] = p->wb_high[c] / p->D_max
        #                            offset[c]  = p->wb_high[c] * p->offset * p->wb_low[c]
        wb_high_normed = wb_high[c] / D_max
        offset_precomp = wb_high[c] * offset * wb_low[c]
        corrected = wb_high_normed * log_density + offset_precomp

        # Step 4: Print exposure (10^corrected) + black point
        # darktable: print_linear = -(exposure * 10^corrected + black)
        print_linear = -(exposure * np.power(10.0, corrected) + black_effective)
        print_linear = np.maximum(print_linear, 0.0)

        # Step 5: Gamma (paper grade)
        print_gamma = np.power(print_linear, gamma)

        # Step 6: Soft-clip highlight rolloff
        # For values above soft_clip, apply exponential compression
        above = print_gamma > soft_clip
        if np.any(above):
            sc = soft_clip
            excess = print_gamma[above] - sc
            print_gamma[above] = sc + (1.0 - np.exp(-excess / (1.0 - sc))) * (1.0 - sc)

        result[:, :, c] = print_gamma

    # Return scene-referred linear float — sigmoid and sRGB encoding applied later
    return result


def process(
    input_path: str,
    output_path: str,
    *,
    align: bool = True,
    threshold: float = 0.35,
    hair_sensitivity: float = 0.15,
    min_area: int = 4,
    dilate_radius: int = 3,
    close_radius: int = 5,
    inpaint_padding: int = 16,
    save_mask: bool = False,
    invert_xmp: str | None = None,
) -> None:
    """Full pipeline: load, align, mask, inpaint, optionally invert negative."""
    print(f"Loading {input_path}...")
    rgb, ir = load_tiff(input_path)
    print(f"  RGB: {rgb.shape}, IR: {ir.shape}")

    if align:
        print("Aligning IR to RGB...")
        ir = align_ir(rgb, ir)

    print("Building defect mask...")
    mask = make_defect_mask(
        ir,
        threshold=threshold,
        hair_sensitivity=hair_sensitivity,
        min_area=min_area,
        dilate_radius=dilate_radius,
        close_radius=close_radius,
    )
    defect_pixels = np.count_nonzero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    n_labels = cv2.connectedComponentsWithStats(mask, connectivity=8)[0] - 1
    print(f"  Defects: {defect_pixels} pixels in {n_labels} regions "
          f"({100 * defect_pixels / total_pixels:.2f}%)")

    if save_mask:
        mask_path = str(Path(output_path).with_suffix('.mask.tif'))
        tifffile.imwrite(mask_path, mask)
        print(f"  Mask saved to {mask_path}")

    if defect_pixels == 0:
        print("No defects found.")
        result = rgb
    else:
        print(f"Inpainting {n_labels} regions (16-bit biharmonic)...")
        result = inpaint(rgb, mask, padding=inpaint_padding)

    if invert_xmp is not None:
        print(f"Converting negative to positive (negadoctor)...")
        nd_params = extract_negadoctor_from_xmp(invert_xmp)
        # negadoctor returns scene-referred linear float
        result_f = negadoctor(result, nd_params)

        cat_matrix = extract_channelmixer_from_xmp(invert_xmp)
        if cat_matrix is not None:
            print("Applying chromatic adaptation...")
            result_f = apply_color_matrix(result_f, cat_matrix)

        sig_params = extract_sigmoid_from_xmp(invert_xmp)
        if sig_params is not None:
            print("Applying sigmoid tone mapping...")
            result_f = apply_sigmoid(result_f, sig_params)

        # Convert Rec2020(D50) back to sRGB(D65) then apply sRGB gamma
        # (darktable's colorout module)
        result_f = apply_color_matrix(result_f, M_REC2020_D50_TO_SRGB)
        result_f = linear_to_srgb(np.clip(result_f, 0, None))
        result = np.clip(result_f * 65535.0, 0, 65535).astype(np.uint16)

    print(f"Saving result to {output_path}...")
    tifffile.imwrite(output_path, result)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="IR-based dust/scratch removal for scanned film negatives",
    )
    parser.add_argument("input", help="Input multi-page TIFF (RGB + IR)")
    parser.add_argument(
        "-o", "--output",
        help="Output TIFF path (default: <input>.cleaned.tif)",
    )
    parser.add_argument(
        "--no-align", action="store_true",
        help="Skip IR-to-RGB alignment",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Dust detection sensitivity (0-1, higher = more aggressive, default: 0.35)",
    )
    parser.add_argument(
        "--hair-sensitivity", type=float, default=0.15,
        help="Hair/scratch line detection sensitivity (0-1, lower = more sensitive, default: 0.15)",
    )
    parser.add_argument(
        "--min-area", type=int, default=4,
        help="Minimum defect size in pixels (default: 4)",
    )
    parser.add_argument(
        "--dilate", type=int, default=3,
        help="Mask dilation radius in pixels (default: 3)",
    )
    parser.add_argument(
        "--close-radius", type=int, default=5,
        help="Morphological close radius to fill gaps in large defects (default: 5)",
    )
    parser.add_argument(
        "--inpaint-padding", type=int, default=16,
        help="Context padding around each defect region in pixels (default: 16)",
    )
    parser.add_argument(
        "--save-mask", action="store_true",
        help="Save the defect mask as a separate TIFF",
    )
    parser.add_argument(
        "--invert-xmp", type=str, default=None, metavar="XMP",
        help="Convert negative to positive using darktable-cli with the "
             "given XMP sidecar (contains negadoctor + processing settings)",
    )

    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        sys.exit(f"Input file not found: {input_path}")

    output_path = args.output
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_suffix('.cleaned.tif'))

    process(
        input_path,
        output_path,
        align=not args.no_align,
        threshold=args.threshold,
        hair_sensitivity=args.hair_sensitivity,
        min_area=args.min_area,
        dilate_radius=args.dilate,
        close_radius=args.close_radius,
        inpaint_padding=args.inpaint_padding,
        save_mask=args.save_mask,
        invert_xmp=args.invert_xmp,
    )


if __name__ == "__main__":
    main()
