#!/usr/bin/env python3
"""Quick CLI harness to run detect_frames on a single scan without the GUI.

Mirrors the server's detection path: load full TIFF, downscale to preview
size, run detect_frames at preview resolution, print results in the same
format as the server console output.

Usage:
    python test_detect.py [path_to_scan.tiff] [--format 35mm] [--n N]

Defaults to scan_0038_rgbir_6400dpi.tiff and 35mm format.
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from scratchndent.auto_frame import detect_frames

DEFAULT_SCAN = Path.home() / "code/epdaughter/scans/scan_0038_rgbir_6400dpi.tiff"
PREVIEW_SIZE = 8192  # matches PARAM_DEFAULTS["preview_size"] in extract.py

# Hand-picked ground truth from scan_0038 (preview pixel coords, scale 0.1697)
GROUND_TRUTH = [
    {"x": 445.0, "y":   71.5, "w": 1045.5, "h": 1563.0, "angle": 0.0297},
    {"x": 407.5, "y": 1701.0, "w": 1035.6, "h": 1556.0, "angle": 0.0290},
    {"x": 359.9, "y": 3342.8, "w": 1036.4, "h": 1554.7, "angle": 0.0290},
    {"x": 320.9, "y": 4954.5, "w": 1036.4, "h": 1554.7, "angle": 0.0290},
    {"x": 284.3, "y": 6572.4, "w": 1036.4, "h": 1554.7, "angle": 0.0290},
]


def load_preview(path: Path, preview_size: int) -> tuple[np.ndarray, float]:
    """Load TIFF page 0 and downscale to preview size. Returns (uint16 RGB, scale)."""
    with tifffile.TiffFile(path) as tif:
        img = tif.pages[0].asarray()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    scale = min(preview_size / max(h, w), 1.0)
    if scale < 1.0:
        small = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img
    if small.dtype != np.uint16:
        small = (small.astype(np.uint16) * 257)
    return small, scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", nargs="?", default=str(DEFAULT_SCAN))
    ap.add_argument("--format", default="35mm")
    ap.add_argument("--n", type=int, default=None,
                    help="Override frame count")
    ap.add_argument("--preview-size", type=int, default=PREVIEW_SIZE)
    args = ap.parse_args()

    path = Path(args.scan)
    if not path.exists():
        sys.exit(f"Not found: {path}")

    print(f"Loading {path.name} (preview size {args.preview_size})...")
    preview, scale = load_preview(path, args.preview_size)
    h, w = preview.shape[:2]
    print(f"  Preview: {w}x{h}, scale={scale:.4f}")

    result = detect_frames(preview, args.format, n_frames=args.n)
    frames = result["frames"]

    print()
    print(f"  === Detected {len(frames)} frames ===")
    for i, f in enumerate(frames):
        print(f"  Frame {i+1}: cx={f['cx']:.1f} cy={f['cy']:.1f} "
              f"w={f['w']:.1f} h={f['h']:.1f} "
              f"angle={f['angle']:.4f} ({math.degrees(f['angle']):+.2f} deg)")

    # Compare against ground truth (scan_0038 only)
    if path.name == DEFAULT_SCAN.name and len(frames) == len(GROUND_TRUTH):
        print()
        print("  === vs ground truth ===")
        print("  (gt: x,y are top-left; auto: cx,cy are center)")
        for i, (gt, f) in enumerate(zip(GROUND_TRUTH, frames)):
            gt_cx = gt["x"] + gt["w"] / 2
            gt_cy = gt["y"] + gt["h"] / 2
            dcx = f["cx"] - gt_cx
            dcy = f["cy"] - gt_cy
            dw = f["w"] - gt["w"]
            dh = f["h"] - gt["h"]
            dang = math.degrees(f["angle"] - gt["angle"])
            print(f"  Frame {i+1}: dcx={dcx:+6.1f} dcy={dcy:+6.1f}  "
                  f"dw={dw:+6.1f} dh={dh:+6.1f}  dangle={dang:+5.2f} deg")


if __name__ == "__main__":
    main()
