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
import sys
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

    aligned = cv2.warpAffine(
        ir, warp_matrix, (ir.shape[1], ir.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
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

    # Compute radial power spectrum of the grain for frequency shaping
    # Use a square patch from the surround if possible
    grain_spectrum = None
    sz = min(h, w, 64)
    if sz >= 16 and np.count_nonzero(surround) > sz * sz * 0.3:
        # Find a clean square patch in the surround for spectrum estimation
        # Use the center channel (green) as representative
        grain_channel = roi_rgb[:, :, 1] - signal[:, :, 1]
        # Window to avoid edge artifacts
        patch = grain_channel[:sz, :sz]
        window = np.outer(np.hanning(sz), np.hanning(sz))
        fft = np.fft.fft2(patch * window)
        power = np.abs(fft) ** 2
        # Radial average
        cy, cx = sz // 2, sz // 2
        y_grid, x_grid = np.mgrid[:sz, :sz]
        r = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2).astype(int)
        r_max = sz // 2
        spectrum = np.zeros(r_max)
        for ri in range(r_max):
            ring = power[r == ri]
            if len(ring) > 0:
                spectrum[ri] = np.mean(ring)
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

    If a grain spectrum is available, shapes the noise to match the measured
    frequency distribution. Otherwise falls back to Gaussian noise with
    matched standard deviation.
    """
    h, w = shape
    grain = np.zeros((h, w, n_channels), dtype=np.float64)

    for c in range(n_channels):
        noise = np.random.randn(h, w)

        if grain_spectrum is not None and len(grain_spectrum) > 1:
            # Shape noise in frequency domain to match grain spectrum
            fft = np.fft.fft2(noise)
            cy, cx = h // 2, w // 2
            y_grid, x_grid = np.mgrid[:h, :w]
            r = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

            # Build radial filter from measured spectrum
            r_max = len(grain_spectrum)
            target_power = np.zeros((h, w), dtype=np.float64)
            for ri in range(r_max):
                ring = (r >= ri) & (r < ri + 1)
                target_power[ring] = grain_spectrum[ri]
            # Smooth transition beyond measured range
            target_power[r >= r_max] = grain_spectrum[-1]

            # Apply as amplitude filter
            with np.errstate(divide='ignore', invalid='ignore'):
                current_power = np.abs(fft) ** 2
                current_power_shifted = np.fft.fftshift(current_power)
                target_shifted = np.fft.fftshift(target_power)
                # Avoid division by zero
                scale = np.where(
                    current_power_shifted > 0,
                    np.sqrt(target_shifted / current_power_shifted),
                    0,
                )
            fft_shifted = np.fft.fftshift(fft)
            shaped = np.fft.ifftshift(fft_shifted * scale)
            noise = np.real(np.fft.ifft2(shaped))

            # Normalize to unit variance then scale
            noise_std = np.std(noise)
            if noise_std > 0:
                noise = noise / noise_std

        grain[:, :, c] = noise * grain_std[c]

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
) -> None:
    """Full pipeline: load, align, mask, inpaint, save."""
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
        print("No defects found, saving original.")
        tifffile.imwrite(output_path, rgb)
    else:
        print(f"Inpainting {n_labels} regions (16-bit biharmonic)...")
        result = inpaint(rgb, mask, padding=inpaint_padding)
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
    )


if __name__ == "__main__":
    main()
