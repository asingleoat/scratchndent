"""IR-based dust and scratch removal for scanned film negatives.

Reads multi-page TIFFs produced by SilverFast (Epson V600 etc.) where:
  Page 0: RGB visible scan (16-bit)
  Page 1: Thumbnail (ignored)
  Page 2: IR channel (16-bit grayscale)

The IR channel reveals dust and scratches (which block infrared light and
appear dark) while film dyes are transparent to IR. We use the IR channel
to build a defect mask, optionally align it to the RGB data, then inpaint
the defects.
"""

import sys

import cv2
import numpy as np
import tifffile
from skimage.filters import meijering
from skimage.restoration import inpaint_biharmonic


def load_tiff(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load RGB (page 0) and IR (page 2) from a SilverFast multi-page TIFF.

    RGB and IR may have different resolutions (e.g. 6400 DPI RGB with
    3200 DPI IR). Downstream functions handle the mismatch.
    """
    with tifffile.TiffFile(path) as tif:
        pages = tif.pages
        if len(pages) < 3:
            sys.exit(f"Expected at least 3 TIFF pages, got {len(pages)}")

        rgb = pages[0].asarray()  # (H, W, 3) uint16
        ir = pages[2].asarray()   # (H, W) uint16

    rgb_h, rgb_w = rgb.shape[:2]
    ir_h, ir_w = ir.shape[:2]
    if rgb_h != ir_h or rgb_w != ir_w:
        print(f"  Note: RGB {rgb_w}x{rgb_h} and IR {ir_w}x{ir_h} differ in resolution")

    return rgb, ir


def align_ir(rgb: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Align IR channel to RGB using ECC (Enhanced Correlation Coefficient).

    Handles different resolutions between RGB and IR (e.g. 6400 DPI RGB
    with 3200 DPI IR). Alignment is computed by downsampling RGB to match
    IR resolution, then the offset is applied to IR in its native resolution.

    Returns the aligned IR at its original (possibly lower) resolution.
    """
    rgb_h, rgb_w = rgb.shape[:2]
    ir_h, ir_w = ir.shape[:2]

    # Resolution ratio between RGB and IR
    res_ratio_y = rgb_h / ir_h
    res_ratio_x = rgb_w / ir_w

    # Convert to 8-bit
    if rgb.dtype == np.uint8:
        rgb8 = rgb
    else:
        rgb8 = (rgb / (rgb.max() / 255.0 + 1e-10)).astype(np.uint8)
    if ir.dtype == np.uint8:
        ir8 = ir
    else:
        ir8 = (ir / (ir.max() / 255.0 + 1e-10)).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)

    # Downsample RGB to match IR resolution for alignment
    if abs(res_ratio_x - 1.0) > 0.01 or abs(res_ratio_y - 1.0) > 0.01:
        print(f"  Downsampling RGB {rgb_w}x{rgb_h} to match IR {ir_w}x{ir_h} for alignment")
        gray = cv2.resize(gray, (ir_w, ir_h), interpolation=cv2.INTER_AREA)

    # Further downsample both for ECC speed
    ecc_scale = 0.125
    small_gray = cv2.resize(gray, None, fx=ecc_scale, fy=ecc_scale,
                            interpolation=cv2.INTER_AREA)
    small_ir = cv2.resize(ir8, None, fx=ecc_scale, fy=ecc_scale,
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

    # Scale translation back to IR resolution (not RGB resolution)
    warp_matrix[0, 2] /= ecc_scale
    warp_matrix[1, 2] /= ecc_scale

    tx, ty = warp_matrix[0, 2], warp_matrix[1, 2]
    print(f"IR alignment offset (IR pixels): tx={tx:.2f}px, ty={ty:.2f}px")

    if abs(tx) < 0.5 and abs(ty) < 0.5:
        print("Offset negligible, skipping warp")
        return ir

    # Apply shift at IR resolution
    from scipy.ndimage import shift as ndimage_shift
    aligned = ndimage_shift(ir, (-ty, -tx), order=1, mode='reflect')
    return aligned


def make_defect_mask(
    ir: np.ndarray,
    rgb: np.ndarray | None = None,
    threshold: float = 0.25,
    hair_sensitivity: float = 0.10,
    min_area: int = 3,
    dilate_radius: int = 4,
    close_radius: int = 6,
    blur_size: int = 301,
    max_coverage: float = 0.03,
) -> np.ndarray:
    """Create a binary defect mask from the IR channel.

    Defects (dust, hairs, scratches) block IR light and appear as dark
    spots in the IR channel relative to their local neighborhood. We use
    an adaptive threshold based on the local standard deviation of the IR
    signal, making detection robust to varying exposure levels across the
    frame — bright and dark areas get proportionally scaled thresholds.

    Two detection strategies:
    1. Adaptive ratio: flags pixels where IR is significantly below the
       local background, scaled by local variability.
    2. Meijering filter: catches thin linear features (hairs, scratches).

    Returns a uint8 mask where 255 = defect, 0 = clean.
    """
    import time as _time

    ir_max = float(ir.max()) if ir.max() > 0 else 1.0
    ir_f = ir.astype(np.float32) / ir_max
    h_ir, w_ir = ir_f.shape[:2]
    print(f"    make_defect_mask: {w_ir}x{h_ir}, {ir_f.size/1e6:.1f}M pixels")

    # --- Adaptive ratio-based detection ---

    t = _time.monotonic()
    background = cv2.GaussianBlur(ir_f, (blur_size, blur_size), 0)
    ir_sq = cv2.GaussianBlur(ir_f ** 2, (blur_size, blur_size), 0)
    local_var = np.maximum(ir_sq - background ** 2, 0.0)
    local_std = np.sqrt(local_var)
    t_blur1 = _time.monotonic() - t

    t = _time.monotonic()
    with np.errstate(divide='ignore', invalid='ignore'):
        deficit = background - ir_f
        n_sigma = np.where(local_std > 1e-4, deficit / local_std, 0.0)
        ratio = np.where(background > 0.01, ir_f / background, 1.0)

    sigma_threshold = 2.5 / threshold
    coarse_mask = (n_sigma > sigma_threshold) & (ratio < (1.0 - threshold * 0.7))
    t_pass1 = _time.monotonic() - t

    # Two-pass refinement
    t = _time.monotonic()
    ir_cleaned = ir_f.copy()
    ir_cleaned[coarse_mask] = background[coarse_mask]
    background2 = cv2.GaussianBlur(ir_cleaned, (blur_size, blur_size), 0)
    ir_sq2 = cv2.GaussianBlur(ir_cleaned ** 2, (blur_size, blur_size), 0)
    local_std2 = np.sqrt(np.maximum(ir_sq2 - background2 ** 2, 0.0))

    with np.errstate(divide='ignore', invalid='ignore'):
        deficit2 = background2 - ir_f
        n_sigma2 = np.where(local_std2 > 1e-4, deficit2 / local_std2, 0.0)
        ratio2 = np.where(background2 > 0.01, ir_f / background2, 1.0)

    dust_mask = (n_sigma2 > sigma_threshold) & (ratio2 < (1.0 - threshold * 0.7))
    t_pass2 = _time.monotonic() - t

    n_dust = np.count_nonzero(dust_mask)
    print(f"    Dust: {n_dust} px ({100*n_dust/dust_mask.size:.2f}%) "
          f"[blur1={t_blur1:.2f}s pass1={t_pass1:.2f}s pass2={t_pass2:.2f}s]")

    # --- Line-based detection (hairs, fine scratches) ---

    t = _time.monotonic()
    scale = 0.25
    nsig_small = cv2.resize(
        np.clip(n_sigma2, 0, None).astype(np.float32),
        None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA,
    ).astype(np.float64)
    nsig_max = np.percentile(nsig_small, 99.9)
    if nsig_max > 0:
        nsig_norm = np.clip(nsig_small / nsig_max, 0, 1)
    else:
        nsig_norm = nsig_small
    t_prep = _time.monotonic() - t

    t = _time.monotonic()
    line_response = meijering(nsig_norm, sigmas=range(1, 9), black_ridges=False)
    t_meijering = _time.monotonic() - t

    t = _time.monotonic()
    line_mask_small = (line_response > hair_sensitivity).astype(np.uint8) * 255
    line_mask = cv2.resize(line_mask_small, (ir_f.shape[1], ir_f.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    line_mask[n_sigma2 < sigma_threshold * 0.2] = 0
    t_gate = _time.monotonic() - t

    n_lines = np.count_nonzero(line_mask)
    print(f"    Lines: {n_lines} px ({100*n_lines/line_mask.size:.2f}%) "
          f"[prep={t_prep:.2f}s meijering={t_meijering:.2f}s gate={t_gate:.2f}s]")

    # --- Combine + morphology ---
    t = _time.monotonic()
    mask = np.where(dust_mask, 255, line_mask).astype(np.uint8)

    if close_radius > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * close_radius + 1, 2 * close_radius + 1),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    if min_area > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask[labels == i] = 0

    if dilate_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_radius + 1, 2 * dilate_radius + 1),
        )
        mask = cv2.dilate(mask, kernel)
    t_morph = _time.monotonic() - t

    coverage = np.count_nonzero(mask) / mask.size
    print(f"    Morphology: {t_morph:.2f}s | final coverage: {100*coverage:.2f}%")

    if coverage > max_coverage:
        print(f"WARNING: defect mask covers {100*coverage:.1f}% of image — "
              f"returning empty mask.", file=sys.stderr)
        return np.zeros_like(mask)

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
    import time as _time
    t_start = _time.monotonic()

    result = rgb.copy()
    rgb_max = float(np.iinfo(rgb.dtype).max) if np.issubdtype(rgb.dtype, np.integer) else 1.0
    h, w = mask.shape

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    n_defects = n_labels - 1  # label 0 is background
    if n_defects == 0:
        return result

    print(f"    Inpainting {n_defects} regions...")
    for i in range(1, n_labels):
        if i % 200 == 0:
            elapsed = _time.monotonic() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_defects - i) / rate if rate > 0 else 0
            print(f"    Inpainting region {i}/{n_defects} "
                  f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Padded bounding box — gives the inpainter context around the defect
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw + padding)
        y1 = min(h, y + bh + padding)

        roi_rgb = result[y0:y1, x0:x1].astype(np.float64) / rgb_max
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
            np.clip(repaired_with_grain[roi_mask] * rgb_max, 0, rgb_max)
            .astype(rgb.dtype)
        )

    t_total = _time.monotonic() - t_start
    print(f"    Inpaint done: {n_defects} regions in {t_total:.2f}s "
          f"({n_defects/t_total:.0f} regions/s)" if t_total > 0 else "")
    return result
