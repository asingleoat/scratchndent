"""Automatic frame detection for scanned film strips.

Uses strong domain priors (known format, frame count, aspect ratio) combined
with edge-based fitting to locate individual frames on a scanned film strip.

Each frame is parameterized as (cx, cy, w, h, angle) and optimized to align
its boundaries with image edges, subject to regularization that keeps frames
similar in size, evenly spaced, and similarly oriented.
"""

import math
import time as _time

import cv2
import numpy as np
from scipy.signal import find_peaks


# Film format definitions.
# - frame_mm: (width, height) of the exposed frame in mm
# - pitch_mm: distance between frame centers along the strip in mm
#   (frame height + inter-frame gap)
# - strip_width_mm: total width of the film strip including borders/sprockets
# All formats: the strip's narrow dimension is strip_width_mm, frames are
# spaced along the long dimension with pitch_mm spacing.
FORMATS = {
    "35mm": {
        "frame_mm": (36, 24),
        "pitch_mm": 38,         # 36mm frame + ~2mm gap
        "strip_width_mm": 35,   # includes sprocket holes
        "description": "35mm (135 film)",
    },
    "645": {
        "frame_mm": (56, 41.5),
        "pitch_mm": 60,
        "strip_width_mm": 61.5,
        "description": "645 medium format",
    },
    "6x6": {
        "frame_mm": (56, 56),
        "pitch_mm": 60,
        "strip_width_mm": 61.5,
        "description": "6x6 medium format",
    },
    "6x7": {
        "frame_mm": (56, 69),
        "pitch_mm": 73,
        "strip_width_mm": 61.5,
        "description": "6x7 medium format",
    },
    "6x9": {
        "frame_mm": (56, 84),
        "pitch_mm": 88,
        "strip_width_mm": 61.5,
        "description": "6x9 medium format",
    },
}



def _compute_strip_profiles(gray: np.ndarray, is_vertical: bool
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute three 1D strip profiles from narrow bands avoiding sprocket holes.

    Takes three narrow strips at 30%, 50%, and 70% across the frame width.
    All three are well inside the exposed frame area and far from sprocket
    holes (which live at 0-16% and 84-100% of the strip width for 35mm).

    The outer two profiles (A at 30%, B at 70%) are used for angle
    estimation. The center profile (C at 50%) provides an additional
    measurement. Using three profiles allows median-filtering to reject
    outlier peak positions from noise or image content.

    Also returns a cross profile for centering.

    Returns (profile_a, profile_b, profile_c, cross_profile)
    """
    h, w = gray.shape[:2]
    img_f = gray.astype(np.float64)

    if is_vertical:
        band_width = max(1, w // 15)
        centers = [int(w * 0.30), int(w * 0.50), int(w * 0.70)]
        profiles = []
        for c in centers:
            sl = img_f[:, c - band_width // 2:c + band_width // 2]
            profiles.append(sl.mean(axis=1))
        cross_profile = img_f.mean(axis=0)
    else:
        band_width = max(1, h // 15)
        centers = [int(h * 0.30), int(h * 0.50), int(h * 0.70)]
        profiles = []
        for c in centers:
            sl = img_f[c - band_width // 2:c + band_width // 2, :]
            profiles.append(sl.mean(axis=0))
        cross_profile = img_f.mean(axis=1)

    # Smooth to remove grain noise
    for arr in profiles + [cross_profile]:
        k = max(3, len(arr) // 100) | 1
        arr[:] = cv2.GaussianBlur(arr.reshape(1, -1), (k, 1), 0).ravel()

    return profiles[0], profiles[1], profiles[2], cross_profile


def _detect_film_extent(image: np.ndarray
                        ) -> tuple[float, float, float] | None:
    """Detect the actual film strip extent and rotation via rotated bbox.

    The image dimensions overestimate the film strip when the strip is
    angled on the scanner bed: a strip rotated by theta produces an
    axis-aligned bounding box wider than the strip itself by L*sin(theta).
    For example, a 35mm strip 190mm long rotated by 1.7 deg has an
    axis-aligned bbox 40.6mm wide -- a 16% overestimate that breaks the
    pitch-ratio frame count formula.

    Methodology (borrowed from epdaughter's detect_film_area):
      1. Otsu threshold separates dark film from bright scanner background
      2. Morphological closing merges frame interiors and inter-frame gaps
         into a single connected blob
      3. Largest connected component = the film strip
      4. cv2.minAreaRect gives a rotated bounding rectangle whose narrow
         side is the actual film width and whose long-edge direction is
         the coarse strip rotation

    The returned angle is a *coarse* estimate of the overall strip
    rotation. Per-frame Theil-Sen still runs in step 7 to refine each
    frame's exact angle (frames may be slightly misaligned from the
    strip).

    Sign convention: positive angle = strip leans upper-left to
    lower-right in image coordinates -- matches the per-frame angle
    convention used by step 7 / step 8.

    Returns (strip_narrow_px, strip_long_px, strip_angle_rad) of the
    rotated bounding rect, or None if no clear film/background
    separation is found (tightly cropped scan), in which case callers
    fall back to image dimensions and zero rotation.
    """
    # 8-bit grayscale (no inversion -- film is darker than scanner bg)
    if image.ndim == 3:
        if image.dtype == np.uint16:
            gray = (image.mean(axis=2) / 256).astype(np.uint8)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.dtype == np.uint16:
        gray = (image / 256).astype(np.uint8)
    else:
        gray = image

    # Otsu split: scanner background (bright) vs film (darker)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    film_mask = (binary == 0).astype(np.uint8) * 255

    # Close gaps within the film region (frame interiors are dark, gaps
    # are slightly less dark) so the strip becomes one connected blob.
    # Kernel must exceed the inter-frame gap (~2mm) but stay smaller than
    # the spacing between separate strips. ~5% of the smaller image
    # dimension works for typical scans.
    h, w = gray.shape
    k = max(3, min(h, w) // 20)
    k |= 1
    kernel = np.ones((k, k), np.uint8)
    film_mask = cv2.morphologyEx(film_mask, cv2.MORPH_CLOSE, kernel)

    # Largest connected film blob
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        film_mask, connectivity=8)
    if n_labels < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    if areas[largest_idx - 1] < film_mask.size * 0.10:
        return None

    # Rotated bounding rect of the largest blob
    component = (labels == largest_idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (_, _), (rw, rh), _ = rect
    if rw < 1 or rh < 1:
        return None

    # Compute strip rotation from the long edge of the rotated rect.
    # cv2.minAreaRect's own angle field has a confusing convention that
    # depends on which side is considered "width"; using the box edges
    # directly is unambiguous.
    box = cv2.boxPoints(rect)
    edge_lengths = [
        float(np.linalg.norm(box[(i + 1) % 4] - box[i])) for i in range(4)
    ]
    longest_idx = int(np.argmax(edge_lengths))
    pa = box[longest_idx]
    pb = box[(longest_idx + 1) % 4]
    dx = float(pb[0] - pa[0])
    dy = float(pb[1] - pa[1])

    is_vertical_image = h > w
    if is_vertical_image:
        # Long axis should point downward; flip if needed
        if dy < 0:
            dx, dy = -dx, -dy
        # Theil-Sen convention (step 7): positive angle = frame edge
        # tilts down-right = frame rotated CW visually = strip leans
        # upper-right -> lower-left, i.e. long-axis vector has dx < 0.
        # atan2(dx, dy) gives the deviation from straight-down with the
        # opposite sign convention (positive for dx > 0 = UL->LR), so
        # negate to match Theil-Sen.
        strip_angle = -math.atan2(dx, dy)
    else:
        # Long axis should point rightward; flip if needed
        if dx < 0:
            dx, dy = -dx, -dy
        # Same sign reasoning for horizontal strip: a CW-rotated
        # horizontal strip has long-axis vector with dy < 0, so
        # atan2(dy, dx) is negative for the case Theil-Sen calls
        # positive. Negate to match.
        strip_angle = -math.atan2(dy, dx)

    return float(min(rw, rh)), float(max(rw, rh)), float(strip_angle)


def _analyze_strip(img_h: int, img_w: int, fmt: dict,
                   film_extent: tuple[float, float] | None = None) -> dict:
    """Compute frame count, dimensions, and orientation from image + format.

    If `film_extent` is provided (from _detect_film_extent), the actual
    film strip dimensions are used instead of the image dimensions. This
    matters when the strip is angled on the scanner bed: the image bbox
    overestimates the strip width, which breaks the frame count formula.

    Returns a dict with:
      n_frames: int
      frame_w, frame_h: frame dimensions in image pixels
      is_vertical: whether the strip runs vertically
      pitch_px: center-to-center frame spacing in pixels
    """
    frame_w_mm, frame_h_mm = fmt["frame_mm"]
    pitch_mm = fmt["pitch_mm"]
    strip_width_mm = fmt["strip_width_mm"]

    # Determine strip orientation: long dimension is the strip axis
    is_vertical = img_h > img_w

    if film_extent is not None:
        # Use measured film extent (rotation-corrected)
        strip_narrow_px, strip_long_px = film_extent[0], film_extent[1]
    elif is_vertical:
        strip_narrow_px = img_w
        strip_long_px = img_h
    else:
        strip_narrow_px = img_h
        strip_long_px = img_w

    # Pixels per mm based on the strip's narrow dimension
    px_per_mm = strip_narrow_px / strip_width_mm

    # Frame dimensions in image pixel coordinates.
    # The frame's narrow physical dimension maps to the scan's narrow dimension,
    # and the wide dimension maps to the scan's long dimension.
    narrow_mm = min(frame_w_mm, frame_h_mm)
    wide_mm = max(frame_w_mm, frame_h_mm)
    if is_vertical:
        frame_w_px = narrow_mm * px_per_mm   # narrow across strip (image x)
        frame_h_px = wide_mm * px_per_mm     # wide along strip (image y)
    else:
        frame_w_px = wide_mm * px_per_mm     # wide along strip (image x)
        frame_h_px = narrow_mm * px_per_mm   # narrow across strip (image y)
    pitch_px = pitch_mm * px_per_mm

    # Frame count: the strip's long/narrow dimension ratio, scaled by the
    # film's strip-width-to-frame-pitch ratio. For 35mm this is 35/36
    # (strip is 35mm wide, frames are 36mm along the strip), so a strip
    # that's 4.3x longer than wide contains int(4.3 * 35/36) = 4 frames.
    pitch_ratio = strip_width_mm / max(frame_w_mm, frame_h_mm)
    n_frames = max(1, int(pitch_ratio * strip_long_px / strip_narrow_px))

    return {
        "n_frames": n_frames,
        "frame_w": frame_w_px,
        "frame_h": frame_h_px,
        "pitch_px": pitch_px,
        "is_vertical": is_vertical,
    }


def _initial_placement(img_h: int, img_w: int, n_frames: int,
                       strip_info: dict, strip_angle: float) -> np.ndarray:
    """Generate initial frame positions assuming even spacing.

    Returns array of shape (n_frames, 5): [cx, cy, w, h, angle]
    """
    frame_w = strip_info["frame_w"]
    frame_h = strip_info["frame_h"]
    pitch = strip_info["pitch_px"]
    is_vertical = strip_info["is_vertical"]

    frames = np.zeros((n_frames, 5))

    # frame_w and frame_h are already in image coordinates
    if is_vertical:
        cx = img_w / 2
        total_span = pitch * n_frames
        y_offset = (img_h - total_span) / 2 + pitch / 2
        for i in range(n_frames):
            frames[i] = [cx, y_offset + i * pitch,
                         frame_w, frame_h, strip_angle]
    else:
        cy = img_h / 2
        total_span = pitch * n_frames
        x_offset = (img_w - total_span) / 2 + pitch / 2
        for i in range(n_frames):
            frames[i] = [x_offset + i * pitch, cy,
                         frame_w, frame_h, strip_angle]

    return frames


def _pack_params(frames: np.ndarray) -> np.ndarray:
    """Flatten frame array to 1D parameter vector."""
    return frames.ravel()


def _unpack_params(params: np.ndarray, n_frames: int) -> np.ndarray:
    """Reshape 1D parameter vector back to (n_frames, 5)."""
    return params.reshape(n_frames, 5)


def _cost_function(params: np.ndarray, n_frames: int,
                   grad_a: np.ndarray, grad_b: np.ndarray,
                   cross_profile: np.ndarray,
                   is_vertical: bool, target_aspect: float,
                   cross_sep: float,
                   reg_weight: float = 1.0) -> float:
    """Cost function using absolute gradient of dual 1D profiles.

    Frame boundaries sit at the steepest transitions in the strip profile
    (maximum absolute gradient), not at peaks or valleys. This correctly
    locates the edge of inter-frame gaps regardless of whether gaps are
    brighter or darker than frame content.

    grad_a, grad_b: precomputed absolute gradient of the two strip profiles.
    cross_sep: pixel distance between the two profile sample bands.
    """
    frames = _unpack_params(params, n_frames)
    grad_len = len(grad_a)
    cross_len = len(cross_profile)

    boundary_cost = 0.0
    for i in range(n_frames):
        cx, cy, w, h, angle = frames[i]
        if is_vertical:
            strip_dim = h
            strip_pos = cy
            cross_pos = cx
        else:
            strip_dim = w
            strip_pos = cx
            cross_pos = cy

        shift_a = -cross_sep / 2 * math.sin(angle)
        shift_b = cross_sep / 2 * math.sin(angle)

        for sign in (-1, 1):
            pos_a = int(round(strip_pos + sign * strip_dim / 2 + shift_a))
            pos_b = int(round(strip_pos + sign * strip_dim / 2 + shift_b))
            if 0 <= pos_a < grad_len:
                boundary_cost -= grad_a[pos_a]
            if 0 <= pos_b < grad_len:
                boundary_cost -= grad_b[pos_b]

        # Cross-strip centering
        cpos = int(round(cross_pos))
        if 0 <= cpos < cross_len:
            boundary_cost += cross_profile[cpos] * 0.1

    # --- Regularization ---
    mean_w = frames[:, 2].mean()
    mean_h = frames[:, 3].mean()
    mean_angle = frames[:, 4].mean()

    reg_cost = 0.0
    for i in range(n_frames):
        reg_cost += ((frames[i, 2] - mean_w) / max(mean_w, 1)) ** 2
        reg_cost += ((frames[i, 3] - mean_h) / max(mean_h, 1)) ** 2
        reg_cost += (frames[i, 4] - mean_angle) ** 2 * 100
        actual_aspect = frames[i, 2] / max(frames[i, 3], 1)
        reg_cost += (actual_aspect - target_aspect) ** 2 * 50

    # Non-overlap along strip axis
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            if is_vertical:
                dist = abs(frames[i, 1] - frames[j, 1])
                min_dist = (frames[i, 3] + frames[j, 3]) / 2
            else:
                dist = abs(frames[i, 0] - frames[j, 0])
                min_dist = (frames[i, 2] + frames[j, 2]) / 2
            if dist < min_dist:
                reg_cost += (min_dist - dist) ** 2 * 10

    return boundary_cost + reg_weight * reg_cost


def detect_frames(
    image: np.ndarray,
    format_name: str,
    n_frames: int | None = None,
) -> list[dict]:
    """Detect frame positions on a scanned film strip.

    Parameters
    ----------
    image : HxWx3 uint8 or uint16
        The scan image (can be downscaled for speed).
    format_name : str
        One of the keys in FORMATS.
    n_frames : int, optional
        Override the frame count (otherwise computed from dimensions).

    Returns
    -------
    frames : list of dict
        Each dict has keys: cx, cy, w, h, angle (in image pixel coordinates).
        Also includes 'aspect' with the format's aspect ratio as w:h string.
    """
    if format_name not in FORMATS:
        raise ValueError(f"Unknown format '{format_name}'. "
                         f"Available: {list(FORMATS.keys())}")

    fmt = FORMATS[format_name]
    orig_h, orig_w = image.shape[:2]

    # Detect actual film extent + coarse strip rotation. Corrects for the
    # strip being angled on the scanner bed: an angled strip inflates the
    # axis-aligned bounding box width by L*sin(theta), which breaks the
    # frame-count formula and degrades 1D projection profiles. Falls
    # back to image dimensions and zero rotation if film/background
    # separation can't be found.
    film_extent = _detect_film_extent(image)

    # If we got a meaningful strip rotation, rotate the input image so
    # the strip is axis-aligned. This is a coarse alignment -- per-frame
    # Theil-Sen still runs in step 7 to refine each frame's exact angle
    # (frames may be slightly misaligned from the strip). All internal
    # detection runs on the rotated image; output coords are transformed
    # back to native scan coords at the end.
    rotation_threshold_rad = math.radians(0.1)
    if film_extent is not None and abs(film_extent[2]) > rotation_threshold_rad:
        strip_angle_rad = film_extent[2]
        # Rotate the input image to undo the strip's lean so it becomes
        # axis-aligned. strip_angle is in Theil-Sen convention: positive
        # = frame/strip rotated CW visually. To undo a CW lean we rotate
        # the image content CCW visually. cv2's positive angle parameter
        # is visual CCW (verified empirically), so we pass +strip_angle.
        center = (orig_w / 2.0, orig_h / 2.0)
        rot_M = cv2.getRotationMatrix2D(
            center, math.degrees(strip_angle_rad), 1.0)
        cos_v = abs(rot_M[0, 0])
        sin_v = abs(rot_M[0, 1])
        new_w = int(orig_h * sin_v + orig_w * cos_v)
        new_h = int(orig_h * cos_v + orig_w * sin_v)
        rot_M[0, 2] += new_w / 2.0 - center[0]
        rot_M[1, 2] += new_h / 2.0 - center[1]
        # rot_M (returned by getRotationMatrix2D + canvas-expansion shift)
        # is the FORWARD transform mapping src (original) coords to dst
        # (rotated/expanded) coords. warpAffine internally inverts it to
        # do the actual pixel sampling. We need the inverse for our
        # output coordinate transform (rotated -> original).
        image = cv2.warpAffine(
            image, rot_M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rotation_M_rot_to_orig = cv2.invertAffineTransform(rot_M)
    else:
        strip_angle_rad = 0.0
        rotation_M_rot_to_orig = None

    img_h, img_w = image.shape[:2]
    strip_info = _analyze_strip(img_h, img_w, fmt, film_extent=film_extent)
    if n_frames is not None:
        strip_info["n_frames"] = n_frames
    actual_n = strip_info["n_frames"]

    extent_note = ""
    if film_extent is not None:
        extent_note = (
            f", film extent {film_extent[1]:.0f}x{film_extent[0]:.0f}px "
            f"(image {max(orig_h,orig_w)}x{min(orig_h,orig_w)}px), "
            f"strip rotation {math.degrees(film_extent[2]):+.2f} deg"
        )
    print(f"  Auto-detect: {format_name}, "
          f"{'vertical' if strip_info['is_vertical'] else 'horizontal'} strip, "
          f"{actual_n} frames, "
          f"frame {float(strip_info['frame_w']):.0f}x{float(strip_info['frame_h']):.0f}px"
          f"{extent_note}")

    t_total = _time.monotonic()
    _t = _time.monotonic()

    # Convert to grayscale 8-bit, invert, and contrast-stretch.
    # On a raw negative the inter-frame gaps are barely visible under the
    # orange mask. Inverting + CLAHE gives high-contrast frame boundaries.
    if image.dtype == np.uint16:
        gray = (image.mean(axis=2) / 256).astype(np.uint8) if image.ndim == 3 else (image / 256).astype(np.uint8)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Invert — makes frame content bright, gaps dark
    gray = 255 - gray
    # Keep a non-CLAHE copy for cross-strip edge detection (CLAHE creates
    # tile-boundary artifacts that bias edge positions)
    gray_raw_inv = gray.copy()
    # CLAHE for strip-axis detection (needs contrast in low-contrast gaps)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Work at reduced resolution for speed
    # Work at full preview resolution — our 1D projection + DTW approach
    # is fast enough that downsampling isn't needed, and higher resolution
    # gives better sub-pixel edge and angle detection.
    work_size = max(img_h, img_w)
    work_scale = min(work_size / max(img_h, img_w), 1.0)
    if work_scale < 1.0:
        gray_small = cv2.resize(gray, (int(img_w * work_scale), int(img_h * work_scale)),
                                interpolation=cv2.INTER_AREA)
        gray_raw_small = cv2.resize(gray_raw_inv, (int(img_w * work_scale), int(img_h * work_scale)),
                                    interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray
        gray_raw_small = gray_raw_inv
        work_scale = 1.0

    sh, sw = gray_small.shape[:2]
    print(f"  Preprocessing ({_time.monotonic()-_t:.2f}s, work res {sw}x{sh})")

    # Step 1: Initial placement at work resolution.
    # Angle is initialized to 0 — per-frame angles are computed precisely
    # in step 7 (Theil-Sen), making a global estimate unnecessary.
    work_info = {
        "frame_w": strip_info["frame_w"] * work_scale,
        "frame_h": strip_info["frame_h"] * work_scale,
        "pitch_px": strip_info["pitch_px"] * work_scale,
        "is_vertical": strip_info["is_vertical"],
    }
    frames = _initial_placement(sh, sw, actual_n, work_info, 0.0)

    _t = _time.monotonic()
    # Step 3: Compute dual 1D projection profiles (avoiding sprocket holes)
    is_vert = strip_info["is_vertical"]
    prof_a, prof_c, prof_b, cross_prof = _compute_strip_profiles(gray_small, is_vert)
    # A is at 30%, C at 50%, B at 70% across the strip
    cross_dim = sw if is_vert else sh
    # Separation between A and B bands for angle computation
    cross_sep = cross_dim * 0.40  # 70% - 30% = 40% of strip width

    # Compute absolute gradient for each profile
    grads = []
    for prof in (prof_a, prof_b, prof_c):
        g = np.abs(np.gradient(prof))
        g[0] = 0; g[-1] = 0
        gk = max(3, len(g) // 200) | 1
        g = cv2.GaussianBlur(g.reshape(1, -1), (gk, 1), 0).ravel()
        grads.append(g)
    grad_a, grad_b, grad_c = grads

    # Average all three for boundary finding (more robust than two)
    grad_avg = (grad_a + grad_b + grad_c) / 3
    print(f"  Profiles + gradients ({_time.monotonic()-_t:.2f}s)")

    _t = _time.monotonic()
    # Step 4: DTW-based frame boundary alignment.
    # Downsample the gradient profile for DTW speed (O(n*m*band) is expensive
    # at full resolution). DTW finds coarse edge positions which are then
    # refined by snapping to peaks in the full-resolution gradient.
    dtw_max_len = 1000
    dtw_scale = min(dtw_max_len / len(grad_avg), 1.0)
    if dtw_scale < 1.0:
        grad_avg_dtw = cv2.resize(grad_avg.reshape(1, -1),
                                  (int(len(grad_avg) * dtw_scale), 1),
                                  interpolation=cv2.INTER_AREA).ravel()
    else:
        grad_avg_dtw = grad_avg
        dtw_scale = 1.0

    #
    # Build a synthetic template of what the gradient profile should look
    # like: [edge, frame_silence, edge, gap_silence, edge, frame_silence, ...]
    # Then use DTW to align this template to the observed gradient,
    # allowing warping in the silence (frame/gap) regions to handle
    # variable film advance.

    if is_vert:
        strip_len = sh
        frame_strip_dim = work_info["frame_h"]
    else:
        strip_len = sw
        frame_strip_dim = work_info["frame_w"]

    # Build DTW template using physical mm ratios, not pixel estimates.
    # The template only needs correct proportions — DTW warps to match
    # the actual signal. Using mm ratios avoids error from the scan being
    # wider than the actual film strip (which inflates px_per_mm by ~5%).
    frame_mm_along = max(fmt["frame_mm"])  # frame dim along strip axis
    gap_mm = fmt["pitch_mm"] - frame_mm_along
    total_mm = actual_n * fmt["pitch_mm"]
    # Template length in DTW samples, proportional to physical length
    dtw_obs_len = len(grad_avg_dtw) if dtw_scale < 1.0 else len(grad_avg)
    mm_to_dtw = dtw_obs_len / (strip_len * 1.0)  # approximate, DTW handles mismatch
    effective_frame_dim = max(1, int(frame_mm_along * mm_to_dtw * strip_len / (strip_len)))
    # Simpler: just use the ratio of frame to pitch to set template proportions
    # at a fixed template resolution
    template_len_target = min(dtw_obs_len, 1000)
    samples_per_mm = template_len_target / total_mm
    effective_frame_dim = max(1, int(frame_mm_along * samples_per_mm))
    effective_gap = max(1, int(gap_mm * samples_per_mm))
    template_parts = []
    for i in range(actual_n):
        template_parts.append(1.0)                                    # frame start edge
        template_parts.extend([0.0] * effective_frame_dim)            # frame interior
        template_parts.append(1.0)                                    # frame end edge
        if i < actual_n - 1:
            template_parts.extend([0.0] * effective_gap)              # inter-frame gap
    template = np.array(template_parts, dtype=np.float64)

    # Normalize both signals to [0, 1]
    obs = grad_avg_dtw.copy()
    if obs.max() > 0:
        obs = obs / obs.max()
    if template.max() > 0:
        template = template / template.max()


    # DTW: find optimal alignment of template to observation.
    # We use a simple DTW that maps each template index to an observation index.
    # Cost: squared difference between template[i] and obs[j].
    # Constraint: monotonic, and the mapping can stretch/compress by up to 2x.

    n_t = len(template)
    n_o = len(obs)

    # Band constraint: template position i should map near i * (n_o / n_t)
    scale_ratio = n_o / n_t
    # Band must be wide enough to accommodate strip margins (frames don't
    # start at the image edge) and variable film advance
    band = int(max(n_o * 0.1, effective_frame_dim * 0.5))

    # Subsequence DTW: template can start anywhere in the observation.
    # cost[0, j] = 0 for all j (free start position).
    INF = 1e18
    cost = np.full((n_t + 1, n_o + 1), INF, dtype=np.float64)
    cost[0, :] = 0.0  # free start: template can begin at any observation position
    parent = np.zeros((n_t + 1, n_o + 1, 2), dtype=np.int32)

    for i in range(1, n_t + 1):
        expected_j = int(i * scale_ratio)
        j_lo = max(1, expected_j - band)
        j_hi = min(n_o, expected_j + band)
        for j in range(j_lo, j_hi + 1):
            d = (template[i - 1] - obs[j - 1]) ** 2
            candidates = []
            if cost[i - 1, j - 1] < INF:
                candidates.append((cost[i - 1, j - 1] + d, i - 1, j - 1))
            if cost[i, j - 1] < INF:
                candidates.append((cost[i, j - 1] + d * 0.5, i, j - 1))
            if cost[i - 1, j] < INF:
                candidates.append((cost[i - 1, j] + d * 0.5, i - 1, j))
            if candidates:
                best = min(candidates, key=lambda x: x[0])
                if best[0] < cost[i, j]:
                    cost[i, j] = best[0]
                    parent[i, j] = [best[1], best[2]]

    # Find best end position (template fully consumed, any observation position)
    end_j = int(np.argmin(cost[n_t, :]))
    end_i = n_t

    # Trace back
    alignment = {}  # template_idx -> obs_idx
    i, j = end_i, end_j
    while i > 0 and j > 0:
        alignment[i - 1] = j - 1
        pi, pj = parent[i, j]
        i, j = int(pi), int(pj)

    # Extract edge positions: template indices where template == 1.0
    edge_template_indices = [k for k in range(len(template_parts))
                            if template_parts[k] == 1.0]
    edge_obs_positions = []
    for ti in edge_template_indices:
        if ti in alignment:
            pos_dtw = alignment[ti]
        else:
            nearest = min(alignment.keys(), key=lambda k: abs(k - ti))
            pos_dtw = alignment[nearest]
        # Scale back to full resolution
        edge_obs_positions.append(int(round(pos_dtw / dtw_scale)))


    # Snap each edge to the nearest strong gradient peak (full resolution).
    # Frame-start edges (even indices) search forward-biased.
    # Frame-end edges (odd indices) search backward-biased.
    # This prevents end-of-frame edges from jumping into the next
    # frame's start region in tight inter-frame gaps.
    snap_radius = int(frame_strip_dim * 0.15)
    n_edges = len(edge_obs_positions)
    snapped = []
    for idx in range(n_edges):
        pos = edge_obs_positions[idx]
        is_frame_end = (idx % 2 == 1)
        is_internal = (idx > 0 and idx < n_edges - 1)
        min_pos = (snapped[-1] + 3) if snapped else 0

        lo = max(int(min_pos), pos - snap_radius)
        hi = min(len(grad_avg), pos + snap_radius + 1)

        if hi > lo:
            window = grad_avg[lo:hi]
            if is_internal and is_frame_end:
                # Frame-end in a gap: find the first prominent peak
                # (closest to the frame interior), not the global max
                # which may be the next frame's start edge.
                from scipy.signal import find_peaks as _fp
                pks, _ = _fp(window, prominence=window.max() * 0.3)
                if len(pks) > 0:
                    best = lo + pks[0]  # first peak = closest to frame
                else:
                    best = lo + int(np.argmax(window))
            elif is_internal and not is_frame_end:
                # Frame-start in a gap: find the last prominent peak
                # (closest to the frame interior)
                from scipy.signal import find_peaks as _fp
                pks, _ = _fp(window, prominence=window.max() * 0.3)
                if len(pks) > 0:
                    best = lo + pks[-1]  # last peak = closest to frame
                else:
                    best = lo + int(np.argmax(window))
            else:
                best = lo + int(np.argmax(window))
            snapped.append(best)
        else:
            snapped.append(max(int(min_pos), pos))
    edge_obs_positions = snapped

    # Fix the last frame's edges using inductive bias from earlier frames.
    # The first N-1 frames give us reliable median pitch (start-to-start
    # spacing) and median frame dimension. DTW can drift on the final
    # frame -- the strip's actual length often differs slightly from the
    # template's, and the residual error accumulates -- placing the last
    # frame's start far from the true position. The default snap radius
    # (15% of frame dim) is then too small to recover. Re-derive both
    # edges of the last frame from the earlier-frame median statistics
    # and snap each to the nearest gradient peak using a wider window.
    if actual_n >= 3 and len(edge_obs_positions) == 2 * actual_n:
        # Median pitch and dim from the first N-1 frames
        starts = [edge_obs_positions[2 * i] for i in range(actual_n)]
        dims = [edge_obs_positions[2*i+1] - edge_obs_positions[2*i]
                for i in range(actual_n - 1)]
        pitches = [starts[i + 1] - starts[i] for i in range(actual_n - 2)]
        median_pitch = int(np.median(pitches))
        expected_dim = int(np.median(dims))

        # Predict last frame's start from earlier-frame pitch
        expected_start = starts[-2] + median_pitch
        # Wider snap window than the default -- DTW drift may put the
        # initial position well outside the standard radius
        wide_radius = max(snap_radius, expected_dim // 4)

        def _snap_to_peak_near(target: int, radius: int, sigma: float) -> int:
            lo = max(0, target - radius)
            hi = min(len(grad_avg), target + radius + 1)
            if hi <= lo:
                return target
            window = grad_avg[lo:hi].copy()
            # Gaussian weighting around the target position
            for j in range(len(window)):
                dist = abs((lo + j) - target)
                window[j] *= math.exp(-0.5 * (dist / sigma) ** 2)
            return lo + int(np.argmax(window))

        # Start snap: median pitch from earlier frames is highly
        # consistent (typically std < 1% of pitch), so the predicted
        # expected_start is already very close to the truth. Use a wide
        # search radius (to recover from arbitrary DTW drift) but a
        # tight sigma so the snap stays near expected_start and only
        # refines to a nearby gradient peak.
        new_start = _snap_to_peak_near(
            expected_start, wide_radius,
            sigma=max(expected_dim * 0.01, 1.0))
        # Enforce monotonicity vs previous frame's end
        new_start = max(new_start, edge_obs_positions[-3] + 3)

        # End snap: the last frame's trailing edge is close to the
        # strip's end, where the film-to-background transition produces
        # a strong gradient just past the actual frame edge. A wide
        # search would get pulled into that strip-end gradient. But the
        # DTW-drift problem only affects the start; with a reliable
        # new_start in hand, the predicted end (new_start + expected_dim)
        # is accurate to within a few pixels because the earlier-frame
        # dims are extremely consistent (typically std < 3 px). Search
        # a very tight window around the predicted end to refine to the
        # actual gradient peak -- the radius needs to cover the
        # earlier-frame dim variance and a bit more, but stay well
        # inside the inter-frame distance to the strip-end gradient.
        end_target = new_start + expected_dim
        # Use 1.5x the observed earlier-frame dim std, with a small
        # floor. The radius is intentionally much smaller than the
        # snap_radius used for general edges.
        dim_std = float(np.std(dims)) if len(dims) > 1 else 3.0
        end_radius = max(int(dim_std * 1.5) + 2, 4)
        end_lo = max(new_start + expected_dim // 2,
                     end_target - end_radius)
        end_hi = min(len(grad_avg), end_target + end_radius + 1)
        if end_hi > end_lo:
            new_end = end_lo + int(np.argmax(grad_avg[end_lo:end_hi]))
        else:
            new_end = end_target

        edge_obs_positions[-2] = new_start
        edge_obs_positions[-1] = new_end

    # Place frames from edge pairs, enforcing aspect ratio
    target_aspect = work_info["frame_w"] / max(work_info["frame_h"], 1)
    cross_center = (sw / 2) if is_vert else (sh / 2)

    if len(edge_obs_positions) == 2 * actual_n:
        for i in range(actual_n):
            e_start = edge_obs_positions[2 * i]
            e_end = edge_obs_positions[2 * i + 1]
            strip_center = (e_start + e_end) / 2
            strip_dim = e_end - e_start
            # Cross dimension is always the narrow side of the frame
            cross_dim_frame = strip_dim * min(fmt["frame_mm"]) / max(fmt["frame_mm"])
            if is_vert:
                frames[i, 0] = cross_center  # cx centered on strip
                frames[i, 1] = strip_center  # cy from DTW
                frames[i, 2] = cross_dim_frame  # w from aspect ratio
                frames[i, 3] = strip_dim     # h from DTW edges
            else:
                frames[i, 0] = strip_center
                frames[i, 1] = cross_center
                frames[i, 2] = strip_dim
                frames[i, 3] = cross_dim_frame
        print(f"  Frames placed via DTW alignment ({_time.monotonic()-_t:.2f}s)")
    else:
        print(f"  DTW produced {len(edge_obs_positions)} edges, expected {2*actual_n}, "
              f"using initial placement ({_time.monotonic()-_t:.2f}s)")

    _t = _time.monotonic()
    # Per-frame angle estimation using many narrow projection strips.
    # Sample 10 strips evenly across the safe zone (25%-75% of strip width)
    # to get robust statistics. Each strip produces a gradient profile;
    # for each frame edge, find the peak position in each strip.
    # Fit a line (peak position vs strip x-position) to get the angle.
    # This is robust to individual strips hitting image content.
    n_angle_strips = 20
    cross_dim_px = sw if is_vert else sh
    strip_positions = np.linspace(0.20, 0.80, n_angle_strips)
    angle_band_width = max(1, cross_dim_px // 20)

    img_f = gray_small.astype(np.float64)
    angle_grads = []
    angle_x_positions = []
    for frac in strip_positions:
        center = int(cross_dim_px * frac)
        if is_vert:
            band = img_f[:, max(0, center - angle_band_width // 2):
                            center + angle_band_width // 2]
            prof = band.mean(axis=1)
        else:
            band = img_f[max(0, center - angle_band_width // 2):
                           center + angle_band_width // 2, :]
            prof = band.mean(axis=0)
        k = max(3, len(prof) // 100) | 1
        prof = cv2.GaussianBlur(prof.reshape(1, -1), (k, 1), 0).ravel()
        g = np.abs(np.gradient(prof))
        g[0] = 0; g[-1] = 0
        gk = max(3, len(g) // 200) | 1
        g = cv2.GaussianBlur(g.reshape(1, -1), (gk, 1), 0).ravel()
        angle_grads.append(g)
        angle_x_positions.append(center)

    angle_x = np.array(angle_x_positions, dtype=np.float64)
    search_r = max(5, int(frame_strip_dim * 0.025))

    for i in range(actual_n):
        edge_peak_positions = []  # list of (x_pos, peak_y) per strip per edge
        for edge_idx in [2 * i, 2 * i + 1]:
            if edge_idx >= len(edge_obs_positions):
                continue
            pos_int = edge_obs_positions[edge_idx]
            lo = max(0, pos_int - search_r)
            hi = min(len(angle_grads[0]), pos_int + search_r + 1)
            if hi <= lo:
                continue
            for s_idx in range(n_angle_strips):
                g = angle_grads[s_idx]
                idx = lo + int(np.argmax(g[lo:hi]))
                # Sub-pixel refinement via quadratic interpolation
                if lo < idx < hi - 1:
                    a, b, c = g[idx - 1], g[idx], g[idx + 1]
                    denom = a - 2 * b + c
                    if abs(denom) > 1e-10:
                        peak = idx + 0.5 * (a - c) / denom
                    else:
                        peak = float(idx)
                else:
                    peak = float(idx)
                edge_peak_positions.append((angle_x[s_idx], peak))

        if len(edge_peak_positions) >= 4:
            xs = np.array([p[0] for p in edge_peak_positions])
            ys = np.array([p[1] for p in edge_peak_positions])
            # Robust line fit: use median-based slope (Theil-Sen)
            slopes = []
            for j in range(len(xs)):
                for k in range(j + 1, len(xs)):
                    if abs(xs[k] - xs[j]) > 1:
                        slopes.append((ys[k] - ys[j]) / (xs[k] - xs[j]))
            if slopes:
                angle = math.atan(float(np.median(slopes)))
                max_angle = math.radians(5)
                angle = max(-max_angle, min(max_angle, angle))
                frames[i, 4] = angle

    print(f"  Angle estimation ({_time.monotonic()-_t:.2f}s)")

    _t = _time.monotonic()
    # Per-frame cross-strip edge detection.
    # Uses the non-CLAHE inverted image to avoid tile-boundary artifacts
    # that bias edge positions. CLAHE is great for finding inter-frame gaps
    # along the strip but creates asymmetric edges across the strip.
    img_cross = gray_raw_small.astype(np.float64)
    n_cross_samples = 15
    cross_dim_px = sw if is_vert else sh
    cross_search_r = max(3, int(cross_dim_px * 0.02))

    for i in range(actual_n):
        cx, cy = frames[i, 0], frames[i, 1]
        angle = frames[i, 4]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        strip_dim = frames[i, 3] if is_vert else frames[i, 2]
        # Derive expected cross dimension from detected strip dimension.
        # The cross dimension is always the narrow side of the frame.
        narrow_mm = min(fmt["frame_mm"])
        wide_mm = max(fmt["frame_mm"])
        cross_dim_est = strip_dim * narrow_mm / wide_mm

        # Sample positions along the frame's strip axis (middle 60%)
        margin_frac = 0.2
        offsets = np.linspace(-strip_dim / 2 * (1 - margin_frac),
                               strip_dim / 2 * (1 - margin_frac),
                               n_cross_samples)

        left_edges = []
        right_edges = []

        for offset in offsets:
            # Point along the frame's strip axis (rotated)
            if is_vert:
                # Strip axis is roughly y; offset moves along y
                sample_y = cy + offset * cos_a
                sample_x_base = cx + offset * sin_a
            else:
                sample_x_base = cx + offset * cos_a
                sample_y = cy + offset * sin_a

            # Sample a 1D line perpendicular to the strip axis through this point
            # For a vertical strip, "perpendicular" is roughly x-direction,
            # rotated by the frame angle
            n_pts = int(cross_dim_px)
            # Sample coordinates: pixel offsets from the sample center point
            # t[i] is the offset in pixels, with t=0 at the center
            t = np.linspace(-(n_pts - 1) / 2, (n_pts - 1) / 2, n_pts)

            if is_vert:
                # Cross direction is roughly x, rotated
                xs = sample_x_base + t * cos_a
                ys = sample_y - t * sin_a
            else:
                xs = sample_x_base - t * sin_a
                ys = sample_y + t * cos_a

            # Clip to image bounds
            valid = (xs >= 0) & (xs < sw - 1) & (ys >= 0) & (ys < sh - 1)
            if valid.sum() < n_pts * 0.5:
                continue

            # Sample using bilinear interpolation
            row = cv2.remap(
                img_cross.astype(np.float32),
                xs.astype(np.float32).reshape(1, -1),
                ys.astype(np.float32).reshape(1, -1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).ravel()

            # Smooth and compute SIGNED gradient.
            # Left frame edge: dark→bright (outside→inside) = positive gradient
            # Right frame edge: bright→dark (inside→outside) = negative gradient
            # Using signed gradient rejects sprocket holes which create
            # gradients of both signs near the frame edge.
            k = max(3, len(row) // 50) | 1
            row = cv2.GaussianBlur(row.reshape(1, -1), (k, 1), 0).ravel()
            g_signed = np.gradient(row)
            g_signed[:2] = 0; g_signed[-2:] = 0

            half_w = cross_dim_est / 2

            def t_to_idx(t_val):
                return int(round(t_val + (n_pts - 1) / 2))

            def _subpixel_peak(g, lo, hi):
                """Find sub-pixel peak via quadratic interpolation."""
                idx = lo + int(np.argmax(g[lo:hi]))
                if 0 < idx - lo < hi - lo - 1:
                    a = g[idx - 1]
                    b = g[idx]
                    c = g[idx + 1]
                    denom = a - 2 * b + c
                    if abs(denom) > 1e-10:
                        offset = 0.5 * (a - c) / denom
                        return idx + offset
                return float(idx)

            # Paired product indexed by CENTER position:
            #   score[c] = g_pos[c - hw] * g_neg[c + hw]
            # High only where both edges exist at the expected separation,
            # symmetric around c. The peak of this array is the frame center.
            g_pos = np.maximum(g_signed, 0)
            g_neg = np.maximum(-g_signed, 0)

            hw_idx = int(round(half_w * (n_pts - 1) / (t[-1] - t[0])))

            if hw_idx > 0 and hw_idx < len(g_pos) // 2:
                lo_c = hw_idx
                hi_c = len(g_pos) - hw_idx
                paired = g_pos[lo_c - hw_idx:hi_c - hw_idx] * g_neg[lo_c + hw_idx:hi_c + hw_idx]

                # Search near expected center
                center_target_idx = t_to_idx(0) - lo_c
                search_lo = max(0, center_target_idx - cross_search_r)
                search_hi = min(len(paired), center_target_idx + cross_search_r + 1)

                if search_hi > search_lo and paired[search_lo:search_hi].max() > 0:
                    # Find coarse center from paired product
                    coarse_k = search_lo + int(np.argmax(paired[search_lo:search_hi]))
                    # Left edge image index and right edge image index
                    left_img_idx = coarse_k  # = coarse_k + lo_c - hw_idx = coarse_k
                    right_img_idx = coarse_k + 2 * hw_idx

                    # Sub-pixel refine each edge independently using signed gradients
                    left_f = _subpixel_peak(g_pos,
                                            max(0, left_img_idx - 2),
                                            min(len(g_pos), left_img_idx + 3))
                    right_f = _subpixel_peak(g_neg,
                                             max(0, right_img_idx - 2),
                                             min(len(g_neg), right_img_idx + 3))

                    # Convert from image index to t-offset (pixel offset from
                    # the sample line center).
                    #
                    # ARGMAX BIAS CORRECTION (+1.0):
                    # np.argmax returns the first index when values tie. For
                    # a symmetric gradient peak spanning two pixels (common at
                    # step edges where the central-difference gradient is equal
                    # at positions i and i+1), argmax always returns i — a
                    # consistent 0.5px bias toward lower indices.
                    #
                    # This bias compounds through two stages:
                    # 1. The paired product g_pos[i]*g_neg[i+w] uses argmax to
                    #    find the coarse center → ~0.5px leftward bias
                    # 2. Sub-pixel refinement uses argmax on narrow windows
                    #    around the coarse positions → another ~0.5px leftward
                    #
                    # Total: ~1.0px at work resolution. Verified empirically by
                    # running detection on a horizontally flipped copy of the
                    # scan — without correction, both original and flipped show
                    # the same leftward shift (proving it's algorithmic, not
                    # content-dependent). The +1.0 correction zeroes out the
                    # original/flipped asymmetry.
                    left_t = left_f + 1.0 - (n_pts - 1) / 2
                    right_t = right_f + 1.0 - (n_pts - 1) / 2
                    left_edges.append(left_t)
                    right_edges.append(right_t)

        if left_edges and right_edges:
            # Median to reject sprocket-hole outliers
            left_offset = float(np.median(left_edges))
            right_offset = float(np.median(right_edges))
            cross_w = right_offset - left_offset
            cross_center_offset = (left_offset + right_offset) / 2

            if cross_w > 0:
                # Shift frame center in the cross direction
                if is_vert:
                    frames[i, 0] = cx + cross_center_offset * cos_a
                    frames[i, 2] = cross_w
                else:
                    frames[i, 1] = cy + cross_center_offset * cos_a
                    frames[i, 3] = cross_w

    # Scale from work resolution back to full preview coords
    frames[:, :4] /= work_scale

    # If we rotated the input image so the strip would be axis-aligned,
    # transform frame centers from rotated coords back to original-image
    # coords and add the global strip rotation to each frame's per-frame
    # angle. The per-frame angles found by Theil-Sen (step 7) are
    # residuals relative to the rotated strip; the original-coords angle
    # is the sum of the residual and the global rotation.
    if rotation_M_rot_to_orig is not None:
        M = rotation_M_rot_to_orig  # 2x3, rotated coords -> original coords
        for i in range(actual_n):
            rcx, rcy = float(frames[i, 0]), float(frames[i, 1])
            ocx = M[0, 0] * rcx + M[0, 1] * rcy + M[0, 2]
            ocy = M[1, 0] * rcx + M[1, 1] * rcy + M[1, 2]
            frames[i, 0] = ocx
            frames[i, 1] = ocy
            frames[i, 4] = float(frames[i, 4]) + strip_angle_rad

    # Aspect ratio string matches the selection orientation (w:h in image coords)
    narrow_mm = min(fmt["frame_mm"])
    wide_mm = max(fmt["frame_mm"])
    if strip_info["is_vertical"]:
        aspect_str = f"{narrow_mm}:{wide_mm}"   # portrait: w < h
    else:
        aspect_str = f"{wide_mm}:{narrow_mm}"   # landscape: w > h

    # Convert to list of dicts
    result_frames = []
    for i in range(actual_n):
        cx, cy, w, h, angle = frames[i]
        result_frames.append({
            "cx": float(cx),
            "cy": float(cy),
            "w": float(abs(w)),
            "h": float(abs(h)),
            "angle": float(angle),
        })

    # Return profiles for debug visualization, scaled to full preview coords.
    # These are the work-resolution profiles — scale positions to match preview.
    print(f"  Cross-strip edges ({_time.monotonic()-_t:.2f}s)")
    print(f"  Detected {len(result_frames)} frames (aspect {aspect_str}) "
          f"in {_time.monotonic()-t_total:.2f}s total")
    return {
        "frames": result_frames,
        "aspect": aspect_str,
    }
