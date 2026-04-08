# Automatic Frame Detection Algorithm

## Overview

This document describes the automatic frame detection algorithm used to locate individual photographic frames on scanned film strips. The algorithm takes a single image of a scanned film strip and a film format specification, and returns the position, size, and rotation of each frame.

The key insight is that the problem is heavily constrained by physical properties of film: frame count is determined by strip dimensions, frame aspect ratio is fixed by the film format, and frame boundaries produce characteristic gradient signatures in 1D projections of the image. These constraints reduce what could be a difficult 2D object detection problem to a sequence of 1D signal processing steps.

## Coordinate Systems

Three coordinate systems are used:

1. **Full-resolution image coordinates**: The original scan pixels. The input image may be a downscaled preview of a much larger scan. Frame positions are returned in this coordinate system.

2. **Work-resolution coordinates**: Equal to the input image resolution (no additional downsampling except for the DTW step). All profile computation, angle estimation, and cross-strip edge detection operate at this resolution.

3. **DTW-resolution coordinates**: The 1D gradient profile is downsampled to at most 2000 samples for the DTW step, since DTW has O(n*m*band) complexity. Edge positions found by DTW are scaled back to work resolution and refined by snapping to full-resolution gradient peaks.

## Input

- **image**: An HxWx3 array (uint8 or uint16). Typically a downscaled preview of the full scan (e.g. 1900x8192 for a 9820x42332 original).
- **format_name**: One of "35mm", "645", "6x6", "6x7", "6x9". Defines the physical frame dimensions, inter-frame pitch, and strip width.
- **n_frames** (optional): Override the computed frame count.

## Output

For each frame: `{cx, cy, w, h, angle}` in input image pixel coordinates, plus an aspect ratio string.

## Algorithm Steps

### Step 0: Geometry Analysis

Given the input image dimensions and the film format's physical specifications:

- **Strip orientation**: The image's longer dimension is the strip axis. If height > width, the strip is vertical (frames stacked top-to-bottom).

- **Pixels per mm**: `px_per_mm = strip_narrow_px / strip_width_mm`, where `strip_narrow_px` is the image's shorter dimension and `strip_width_mm` is the film strip's physical width (e.g. 35mm for 135 film).

- **Frame dimensions in pixels**: For a vertical strip, the frame's narrow physical dimension (e.g. 24mm for 35mm film) maps to the image's x-axis (width), and the wide dimension (e.g. 36mm) maps to the y-axis (height). So:
  - `frame_w_px = narrow_mm * px_per_mm` (across strip)
  - `frame_h_px = wide_mm * px_per_mm` (along strip)

  For a horizontal strip, swap w and h.

- **Frame count**: `n = floor(pitch_ratio * strip_long_px / strip_narrow_px)`, where `pitch_ratio = strip_width_mm / max(frame_w_mm, frame_h_mm)`. For 35mm this is `35/36 = 0.972`, so a strip that's 4.3x longer than wide contains `floor(4.3 * 0.972) = 4` frames.

### Step 1: Image Preprocessing

1. Convert to 8-bit grayscale (mean of RGB channels for uint16 input, or `cvtColor` for uint8).

2. Invert: `gray = 255 - gray`. On a color negative, frame content is dark and inter-frame gaps are bright (film base). Inverting makes frame content bright.

3. Create two copies:
   - **CLAHE-enhanced**: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) with clipLimit=3.0 and 8x8 tile grid. This is used for strip-axis edge detection where inter-frame gaps may have low contrast.
   - **Raw inverted**: No CLAHE. Used for cross-strip edge detection because CLAHE introduces tile-boundary artifacts that create asymmetric edge profiles, biasing detected positions.

### Step 2: Strip Orientation Estimation

Apply Canny edge detection (thresholds 30, 100) to the CLAHE image, then Hough line detection (`HoughLinesP` with threshold=100, minLineLength=image_width/4).

Classify detected lines as horizontal (|angle| < 45 deg) or vertical. The group with more total line length is the strip axis direction. The length-weighted mean angle of that group gives the strip rotation angle.

This angle is typically near 0 deg (horizontal strip) or 90 deg (vertical strip), with small deviations from scanner/film misalignment.

### Step 3: 1D Projection Profiles

Three narrow bands are sampled perpendicular to the strip axis, at positions 30%, 50%, and 70% across the strip width. Each band is ~1/15 of the strip width.

For a vertical strip, each band is a narrow vertical slice of columns. The mean pixel value across each band's columns produces a 1D profile along the strip axis (y). For a horizontal strip, it's rows along x.

These band positions (30%, 50%, 70%) are chosen to be well inside the exposed frame area and far from sprocket holes, which occupy roughly 0-16% and 84-100% of the strip width on 35mm film.

Each profile is smoothed with a Gaussian kernel (size = profile_length / 100, forced odd).

The absolute gradient of each smoothed profile is computed. Boundary artifacts at the first and last sample are zeroed before smoothing to prevent the image border from appearing as a false gradient peak.

The three gradient profiles are averaged to produce `grad_avg`, which is used for frame boundary detection.

### Step 4: DTW Frame Boundary Detection

#### Template Construction

A synthetic 1D template is constructed representing the expected gradient pattern of N frames:

```
[1, 0, 0, ..., 0, 1, 0, 0, ..., 0, 1, 0, 0, ..., 0, 1, ...]
 ^   frame (90%)  ^   gap (~10%+gap)  ^   frame     ^
```

- Each `1` represents an expected frame boundary (high gradient).
- Zeros represent frame interior (low gradient) and inter-frame gaps.
- Frame interior length = 90% of the nominal frame dimension (the actual exposed area is slightly smaller than the physical gate).
- Gap length = remaining 10% of frame dimension + the physical inter-frame gap.

The template and observation (`grad_avg`, downsampled to max 2000 samples) are both normalized to [0, 1].

#### Subsequence DTW

A banded subsequence DTW aligns the template to the observation. "Subsequence" means the template can start at any position in the observation (the first row of the cost matrix is initialized to 0), accommodating strip margins before the first frame.

The cost matrix is `(n_template + 1) x (n_observation + 1)`. For each template position `i`, only observation positions within a band of `max(strip_length * 0.1, frame_dim * 0.5)` around the expected position `i * (n_obs / n_template)` are evaluated.

Three transitions are allowed at each cell:
- Diagonal (i-1, j-1) → (i, j): cost = d (match)
- Horizontal (i, j-1) → (i, j): cost = d * 0.5 (skip observation sample)
- Vertical (i-1, j) → (i, j): cost = d * 0.5 (stretch template)

where d = (template[i] - obs[j])^2.

The best end position is the `j` with minimum cost at `i = n_template` (template fully consumed). Backtracing through parent pointers recovers the alignment: a mapping from each template index to an observation index.

#### Edge Extraction

Template indices where the value is 1.0 (the boundary markers) are extracted. Their aligned observation positions give the coarse frame boundary locations in DTW-resolution space.

These positions are scaled back to full work resolution: `pos_full = round(pos_dtw / dtw_scale)`.

### Step 5: Edge Snapping

The DTW-derived edge positions are approximate. Each is refined by snapping to the nearest strong gradient peak in the full-resolution `grad_avg`.

The snap search radius is 15% of the frame strip dimension. Within this window:

- **First and last edges** (strip boundaries): snap to the global maximum gradient peak.

- **Internal frame-end edges** (odd indices, except the last): Find all prominent peaks (prominence > 30% of window max) using `scipy.signal.find_peaks`. Select the **first** peak — the one closest to the frame interior. This prevents the snap from jumping across a narrow inter-frame gap to the next frame's start edge, which is often a stronger peak.

- **Internal frame-start edges** (even indices, except the first): Select the **last** prominent peak — closest to the frame interior, for the same reason.

A monotonicity constraint ensures each snapped edge is at least 3 pixels past the previous one.

### Step 6: Frame Placement (Strip Axis)

Consecutive edge pairs define frames: edges [0,1] = frame 1, [2,3] = frame 2, etc. For each frame:

- Strip-axis center = midpoint of its two edges.
- Strip-axis dimension = distance between its two edges.
- Cross-strip dimension = strip-axis dimension * target_aspect_ratio (derived from frame_w / frame_h in the initial geometry analysis).
- Cross-strip center = image center along the cross-strip axis (temporary; refined in step 8).

### Step 7: Per-Frame Angle Estimation

Twenty narrow projection strips are sampled at evenly spaced positions from 20% to 80% across the strip width, each ~1/20 of the strip width. Each produces a gradient profile along the strip axis (using the CLAHE image for contrast).

For each frame, at each of its two detected edge positions (from step 5), the gradient peak position is found in each of the 20 strips using sub-pixel quadratic interpolation:

Given the integer argmax position `idx` and its neighbors `a = g[idx-1], b = g[idx], c = g[idx+1]`:
```
offset = 0.5 * (a - c) / (a - 2*b + c)
peak = idx + offset
```

This produces up to 40 (x_position, peak_y_position) pairs per frame. A Theil-Sen robust line fit computes the slope: for every pair of points (i, j) with different x-positions, compute `slope = (y_j - y_i) / (x_j - x_i)`. The median of all pairwise slopes gives the frame angle via `angle = atan(median_slope)`.

Theil-Sen is robust to outliers — if some strips hit image content that shifts the gradient peak, the median rejects them. With 40 points this produces up to 780 pairwise slopes, making the estimate very stable.

The angle is clamped to +/- 5 degrees.

### Step 8: Cross-Strip Edge Detection

This step refines the frame's position and width across the strip (the dimension perpendicular to the strip axis, where sprocket holes create complications).

**Why a separate step**: The strip-axis detection (steps 3-6) uses the average of three projection profiles which naturally wash out localized features like sprocket holes. But cross-strip detection requires probing individual scan lines that pass directly over sprocket holes. Different techniques are needed.

**Image used**: The raw inverted image (no CLAHE) is used because CLAHE's tile boundaries create asymmetric edge profiles that systematically bias the detected center position.

For each frame, 15 sample lines are taken perpendicular to the strip axis from within the frame interior (middle 60%, avoiding the frame boundaries themselves). Each line is rotated by the frame's detected angle using `cv2.remap` with bilinear interpolation, so the sampling accounts for frame rotation.

**Sampling geometry**: For a vertical strip with a frame at (cx, cy) with angle `a`, sample line offsets are distributed along the strip axis. For each offset `d`:
- Sample center point: `(cx + d*sin(a), cy + d*cos(a))`
- Sample line direction: perpendicular to strip axis, rotated by `a`
- Sample coordinates: `xs = center_x + t*cos(a)`, `ys = center_y - t*sin(a)`, where `t` ranges from `-(n-1)/2` to `+(n-1)/2` for `n` sample points spanning the strip width.

**Signed gradient**: The 1D gradient along each sample line is computed (not the absolute gradient). This is critical because:
- The left frame edge is a dark→bright transition (positive gradient)
- The right frame edge is a bright→dark transition (negative gradient)
- Sprocket holes create both positive and negative gradients

By searching for positive peaks at the expected left-edge position and negative peaks at the expected right-edge position, sprocket hole gradients are rejected because they have the wrong sign at the expected separation.

**Paired product**: Rather than finding left and right edges independently, a paired product array is computed indexed by the center position `c`:

```
paired[c] = max(gradient, 0)[c - hw] * max(-gradient, 0)[c + hw]
```

where `hw` is half the expected frame width (derived from aspect ratio and the detected strip-axis dimension). This product is high only where a positive gradient (left edge) and negative gradient (right edge) exist at exactly the right separation. The peak of this array gives the frame center.

The expected width `hw` is computed as: `strip_dim * target_aspect / 2`, where `strip_dim` is the frame's strip-axis dimension from step 6 and `target_aspect = frame_w / frame_h` from the format geometry.

**Sub-pixel refinement**: The coarse center from the paired product peak is refined by finding the exact peak positions of the left and right edges independently using quadratic interpolation on narrow (5-pixel) windows of the signed gradient.

**Argmax bias correction**: A systematic +1.0 pixel correction is added to both edge positions. This corrects for a compounding bias from `np.argmax`, which returns the first index when values tie. For symmetric gradient peaks (common at step edges), this consistently shifts the detected position by -0.5 pixels per argmax call. The bias was verified empirically by running detection on a horizontally flipped copy of a scan: without correction, both original and flipped show the same leftward shift; with +1.0 correction, the original/flipped asymmetry is eliminated.

**Robust aggregation**: The 15 per-frame center measurements (one per sample line) are aggregated using the median, which rejects outliers from sample lines that crossed sprocket holes.

### Step 9: Output Assembly

Frame parameters (cx, cy, w, h, angle) are converted from work-resolution to input-image coordinates by dividing the positional values by `work_scale` (which is 1.0 when operating at full preview resolution).

The width is taken from the cross-strip detection (step 8). The height is from the strip-axis DTW detection (step 6). The center and angle are from steps 8 and 7 respectively.

An aspect ratio string is generated from the film format's physical dimensions, oriented to match the frame orientation (e.g. "24:36" for portrait 35mm).

## Performance Characteristics

- **Preprocessing** (grayscale, invert, CLAHE): ~0.2s at 1900x8192
- **Strip orientation** (Canny + Hough): ~0.8s
- **1D profiles + gradients**: ~0.01s
- **DTW** (at 2000px downsampled): ~1-2s
- **Angle estimation** (20 strips, Theil-Sen): ~0.02s
- **Cross-strip edges** (15 samples per frame, paired product): ~0.2s
- **Total**: ~2-3s for a 1900x8192 preview image

## Accuracy

Tested against manually placed frames on a 35mm 4-frame strip (1900x8192 preview of a 9820x42332 6400 DPI scan):

- **Y-center (strip axis)**: mean absolute error 2.7px (0.03% of image height)
- **X-center (cross strip)**: mean absolute error 3.1px (excluding one content-dependent outlier)
- **Width**: mean absolute error 3.1px
- **Height**: mean absolute error 4.6px
- **Rotation**: verified symmetric (no algorithmic bias) via horizontal flip test
- **Algorithmic position bias**: eliminated to <1px via the argmax correction

Remaining errors are content-dependent (image features near frame edges shifting gradient peaks) and are within the precision of manual selection.

## Key Design Decisions

1. **CLAHE for strip-axis, raw for cross-strip**: CLAHE enhances contrast in low-contrast inter-frame gaps but introduces tile-boundary artifacts that bias cross-strip edge positions by ~5px.

2. **Signed gradient for cross-strip**: Using the sign of the gradient (positive for left edge, negative for right edge) rejects sprocket hole contamination that absolute gradient cannot distinguish from frame edges.

3. **Paired product for center finding**: Finding the paired product peak `g_pos[c-hw] * g_neg[c+hw]` is equivalent to finding the position where both edges exist at the expected separation, in a single array scan. This is more robust than finding edges independently and hoping they're correctly paired.

4. **DTW at reduced resolution, snapping at full resolution**: DTW is O(n*m*band) and dominates computation time. Running it at 2000px max then snapping to full-resolution gradient peaks gives the same accuracy as full-resolution DTW at ~10x the speed.

5. **Theil-Sen for angle**: With 20 strips x 2 edges = 40 measurements, the median of 780 pairwise slopes is extremely robust to outliers from image content hitting individual strips.

6. **Template at 90% frame dimension**: The actual exposed frame is slightly smaller than the physical gate size, and users typically crop slightly inside the frame border. Using 90% prevents the DTW from overshooting to the physical edge.
