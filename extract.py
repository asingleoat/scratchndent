#!/usr/bin/env python3
"""
Frame extraction tool — browser-based UI for selecting and exporting
rectangular regions from scanned film strips.

Usage:
    python extract.py <input.tif> [--port 8888] [--output-dir frames/]

Opens a browser with a zoomable preview of the scan. Draw selection
rectangles, adjust aspect ratio/rotation, then export cropped frames
at full resolution.
"""

import argparse
import io
import json
import math
import sys
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import tifffile
from PIL import Image

from scratchndent import (
    align_ir,
    make_defect_mask,
    inpaint,
    extract_negadoctor_from_xmp,
    negadoctor,
    extract_channelmixer_from_xmp,
    apply_color_matrix,
    extract_sigmoid_from_xmp,
    apply_sigmoid,
    linear_to_srgb,
    M_REC2020_D50_TO_SRGB,
)


# ---------------------------------------------------------------------------
# Globals set at startup
# ---------------------------------------------------------------------------
INPUT_PATH: str = ""
OUTPUT_DIR: Path = Path(".")
FULL_IMG: np.ndarray | None = None       # full-res processed image (lazy, None until export)
FULL_IMG_READY: bool = False              # True once full-res processing is done
PREVIEW_JPEG: bytes = b""                # downscaled JPEG for the browser
PREVIEW_SCALE: float = 1.0               # preview pixels / full pixels
FULL_WIDTH: int = 0                       # full-res dimensions (from raw load)
FULL_HEIGHT: int = 0
IMAGE_LIST: list[str] = []               # all image paths in folder
IMAGE_IDX: int = 0                        # current index into IMAGE_LIST
INVERT_XMP: str | None = None
IR_CLEAN: bool = True                     # auto-detect by default
PREVIEW_SIZE: int = 2400
LOADING: bool = False                     # True while switching images
HAS_IR: bool = False                      # whether current image has IR channel
PROGRESS: str = ""                        # current export/processing status for GUI


def set_progress(msg: str) -> None:
    """Update progress status for both stdout and GUI polling."""
    global PROGRESS
    PROGRESS = msg
    print(f"  [{msg}]")


def apply_inversion(img: np.ndarray) -> np.ndarray:
    """Apply negadoctor + CAT16 + sigmoid pipeline. Input: uint16, output: uint16."""
    nd_params = extract_negadoctor_from_xmp(INVERT_XMP)
    result_f = negadoctor(img, nd_params)

    cat_matrix = extract_channelmixer_from_xmp(INVERT_XMP)
    if cat_matrix is not None:
        result_f = apply_color_matrix(result_f, cat_matrix)

    sig_params = extract_sigmoid_from_xmp(INVERT_XMP)
    if sig_params is not None:
        result_f = apply_sigmoid(result_f, sig_params)

    result_f = apply_color_matrix(result_f, M_REC2020_D50_TO_SRGB)
    result_f = linear_to_srgb(np.clip(result_f, 0, None))
    return np.clip(result_f * 65535.0, 0, 65535).astype(np.uint16)


def apply_ir_clean(rgb: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Run IR-based dust/scratch removal. Returns cleaned rgb."""
    ir_aligned = align_ir(rgb, ir)
    mask = make_defect_mask(ir_aligned)
    if np.count_nonzero(mask) > 0:
        return inpaint(rgb, mask)
    return rgb


def ensure_ir_cleaned() -> np.ndarray:
    """Load and IR-clean the current image at full resolution (no inversion).

    Inversion is deferred to per-crop at export time for speed.
    """
    global FULL_IMG, FULL_IMG_READY
    if FULL_IMG_READY:
        return FULL_IMG

    set_progress("Loading full-resolution image...")
    rgb, ir = load_image(INPUT_PATH)
    h, w = rgb.shape[:2]
    set_progress(f"Loaded {w}x{h} image")

    if IR_CLEAN and ir is not None:
        set_progress("Aligning IR channel...")
        ir_aligned = align_ir(rgb, ir)
        set_progress("Detecting defects...")
        mask = make_defect_mask(ir_aligned)
        n_defects = np.count_nonzero(mask)
        if n_defects > 0:
            set_progress(f"Inpainting {n_defects} defect pixels...")
            rgb = inpaint(rgb, mask)
        else:
            set_progress("No defects found")

    FULL_IMG = rgb
    FULL_IMG_READY = True
    set_progress("IR cleaning complete")
    return FULL_IMG


def switch_to_image(idx: int) -> None:
    """Load image, generate preview at reduced resolution for speed."""
    global INPUT_PATH, FULL_IMG, FULL_IMG_READY, PREVIEW_JPEG, PREVIEW_SCALE
    global IMAGE_IDX, LOADING, FULL_WIDTH, FULL_HEIGHT, HAS_IR
    LOADING = True
    FULL_IMG = None
    FULL_IMG_READY = False
    IMAGE_IDX = idx
    INPUT_PATH = IMAGE_LIST[idx]

    print(f"Loading {INPUT_PATH} ({idx + 1}/{len(IMAGE_LIST)})...")
    rgb, ir = load_image(INPUT_PATH)
    h, w = rgb.shape[:2]
    FULL_WIDTH, FULL_HEIGHT = w, h
    HAS_IR = ir is not None
    print(f"  Image: {w}x{h}, {rgb.dtype}")

    # Downscale for fast preview processing — use longer dimension for quality
    proc_dim = max(PREVIEW_SIZE, 6000)
    proc_scale = min(proc_dim / max(h, w), 1.0)
    preview_scale = min(PREVIEW_SIZE / max(h, w), 1.0)
    if proc_scale < 1.0:
        pw, ph = int(w * proc_scale), int(h * proc_scale)
        small_rgb = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
        if ir is not None:
            small_ir = cv2.resize(ir, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            small_ir = None
    else:
        small_rgb = rgb
        small_ir = ir

    # Skip IR cleaning for preview — it's cosmetic and the full-res export
    # handles it properly. This keeps small_rgb in uint16 for accurate inversion.

    # Inversion at preview resolution (uint16 preserved for tonal accuracy)
    if INVERT_XMP:
        print("  Inverting (preview)...")
        small_rgb = apply_inversion(small_rgb)

    # Generate JPEG — downscale processed result to preview size
    if small_rgb.dtype == np.uint16:
        preview8 = (small_rgb >> 8).astype(np.uint8)
    else:
        preview8 = small_rgb
    # Further downscale if we processed at higher res than preview
    if proc_scale > preview_scale:
        final_w = int(w * preview_scale)
        final_h = int(h * preview_scale)
        preview8 = cv2.resize(preview8, (final_w, final_h), interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(preview8)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    PREVIEW_JPEG = buf.getvalue()
    PREVIEW_SCALE = preview_scale

    print(f"  Preview ready ({preview8.shape[1]}x{preview8.shape[0]}, scale={preview_scale:.4f})")
    LOADING = False


def load_image(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load image, returning (rgb, ir) where ir is None if not available."""
    ir = None
    if path.lower().endswith((".tif", ".tiff")):
        with tifffile.TiffFile(path) as tif:
            img = tif.pages[0].asarray()
            if len(tif.pages) >= 3:
                ir_page = tif.pages[2].asarray()
                if ir_page.ndim == 2 and ir_page.shape[:2] == img.shape[:2]:
                    ir = ir_page
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        sys.exit(f"Could not load image: {path}")
    return img, ir



def crop_rotated_rect(
    img: np.ndarray,
    cx: float, cy: float,
    w: float, h: float,
    angle_deg: float,
) -> np.ndarray:
    """Crop a rotated rectangle from the full-resolution image.

    Parameters are in full-image pixel coordinates.
    angle_deg: rotation in degrees (clockwise).
    """
    img_h, img_w = img.shape[:2]

    # Compute bounding box of the rotated rect to extract a sub-region first,
    # avoiding warpAffine on the full (potentially >32767px) image.
    diag = math.hypot(w, h) / 2
    margin = int(math.ceil(diag)) + 4
    x0 = max(int(cx) - margin, 0)
    y0 = max(int(cy) - margin, 0)
    x1 = min(int(cx) + margin, img_w)
    y1 = min(int(cy) + margin, img_h)
    sub = img[y0:y1, x0:x1]
    local_cx = cx - x0
    local_cy = cy - y0

    pad = 2
    out_w = int(math.ceil(w)) + pad * 2
    out_h = int(math.ceil(h)) + pad * 2

    M = cv2.getRotationMatrix2D((local_cx, local_cy), -angle_deg, 1.0)
    M[0, 2] += out_w / 2 - local_cx
    M[1, 2] += out_h / 2 - local_cy

    rotated = cv2.warpAffine(sub, M, (out_w, out_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)

    return rotated[pad:pad + int(h), pad:pad + int(w)]


# ---------------------------------------------------------------------------
# HTML / JS UI (embedded)
# ---------------------------------------------------------------------------
def get_html() -> str:
    html_path = Path(__file__).parent / "extract_ui.html"
    return html_path.read_text()



# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence request logs

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._respond(200, "text/html", get_html().encode())
        elif parsed.path == "/preview":
            self._respond(200, "image/jpeg", PREVIEW_JPEG)
        elif parsed.path == "/info":
            info = {
                "full_width": FULL_WIDTH,
                "full_height": FULL_HEIGHT,
                "preview_scale": PREVIEW_SCALE,
                "filename": Path(INPUT_PATH).name,
                "image_idx": IMAGE_IDX,
                "image_count": len(IMAGE_LIST),
                "loading": LOADING,
            }
            self._respond(200, "application/json", json.dumps(info).encode())
        elif parsed.path == "/progress":
            self._respond(200, "application/json",
                          json.dumps({"status": PROGRESS}).encode())
        elif parsed.path == "/images":
            data = {
                "images": [Path(p).name for p in IMAGE_LIST],
                "current": IMAGE_IDX,
            }
            self._respond(200, "application/json", json.dumps(data).encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def do_POST(self):
        if self.path == "/export":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            result = handle_export(body)
            self._respond(200, "application/json", json.dumps(result).encode())
        elif self.path == "/switch":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            idx = body.get("idx", 0)
            if 0 <= idx < len(IMAGE_LIST):
                switch_to_image(idx)
                info = {
                    "full_width": FULL_WIDTH, "full_height": FULL_HEIGHT,
                    "preview_scale": PREVIEW_SCALE,
                    "filename": Path(INPUT_PATH).name,
                    "image_idx": IMAGE_IDX,
                    "image_count": len(IMAGE_LIST),
                }
                self._respond(200, "application/json", json.dumps(info).encode())
            else:
                self._respond(400, "application/json",
                              json.dumps({"error": "Invalid index"}).encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def _respond(self, code, content_type, data):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def handle_export(body: dict) -> dict:
    # Default basename from input filename (e.g. "untitled" from "untitled.tif")
    default_base = Path(INPUT_PATH).stem
    basename = body.get("basename", default_base)
    rects = body.get("rects", [])
    n_rects = len(rects)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exported = []

    set_progress(f"Preparing full-res export ({n_rects} frame{'s' if n_rects != 1 else ''})...")
    cleaned = ensure_ir_cleaned()

    for i, r in enumerate(rects):
        set_progress(f"Cropping frame {i + 1}/{n_rects}...")
        cx, cy, w, h = r["cx"], r["cy"], r["w"], r["h"]
        angle = r.get("angle", 0)

        cropped = crop_rotated_rect(cleaned, cx, cy, w, h, angle)

        if INVERT_XMP:
            set_progress(f"Inverting frame {i + 1}/{n_rects}...")
            cropped = apply_inversion(cropped)

        out_name = f"{basename}_{i + 1:02d}.tif"
        out_path = OUTPUT_DIR / out_name
        # Don't overwrite existing files
        n = 1
        while out_path.exists():
            out_name = f"{basename}_{i + 1:02d}_{n}.tif"
            out_path = OUTPUT_DIR / out_name
            n += 1
        set_progress(f"Writing {out_name} ({cropped.shape[1]}x{cropped.shape[0]})...")
        tifffile.imwrite(str(out_path), cropped)
        exported.append(out_name)

    msg = f"Exported {len(exported)} frame{'s' if len(exported) != 1 else ''} to {OUTPUT_DIR}/"
    set_progress(msg)
    return {"message": msg, "files": exported}


TIFF_EXTS = {".tif", ".tiff"}


def find_images(path: Path) -> list[str]:
    """Find all TIFFs in a directory, sorted by name."""
    if path.is_file():
        parent = path.parent
    else:
        parent = path
    files = sorted(
        p for p in parent.iterdir()
        if p.is_file() and p.suffix.lower() in TIFF_EXTS
    )
    return [str(p) for p in files]


def main():
    global OUTPUT_DIR, IMAGE_LIST, INVERT_XMP, IR_CLEAN, PREVIEW_SIZE

    parser = argparse.ArgumentParser(
        description="Browser-based frame extraction from scanned film strips",
    )
    parser.add_argument("input", help="Input TIFF file or directory of TIFFs")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--output-dir", type=str, default="frames",
        help="Directory for exported frames (default: frames/)",
    )
    parser.add_argument(
        "--preview-size", type=int, default=2400,
        help="Max dimension of browser preview in pixels (default: 2400)",
    )
    parser.add_argument(
        "--invert-xmp", type=str, default=None,
        help="darktable XMP sidecar for negative-to-positive conversion",
    )
    parser.add_argument(
        "--no-ir-clean", action="store_true",
        help="Disable automatic IR dust/scratch removal",
    )
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    INVERT_XMP = args.invert_xmp
    IR_CLEAN = not args.no_ir_clean
    PREVIEW_SIZE = args.preview_size

    input_path = Path(args.input)
    IMAGE_LIST = find_images(input_path)
    if not IMAGE_LIST:
        sys.exit(f"No TIFF files found in {input_path}")

    # Start at the specified file, or first file in directory
    start_idx = 0
    if input_path.is_file():
        resolved = str(input_path.resolve())
        for i, p in enumerate(IMAGE_LIST):
            if str(Path(p).resolve()) == resolved:
                start_idx = i
                break

    print(f"Found {len(IMAGE_LIST)} image(s)")
    switch_to_image(start_idx)

    server = ThreadedHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Serving at {url}")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
