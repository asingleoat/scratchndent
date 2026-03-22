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
)
from scratchndent.film_inversion import compute_dmin, invert_negative
from scratchndent.film_render import render_to_display


CONFIG_FILE = Path("scratchndent_config.json")


def load_config() -> dict:
    """Load persisted settings from config file."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(updates: dict) -> None:
    """Merge updates into the config file."""
    cfg = load_config()
    cfg.update(updates)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Globals set at startup
# ---------------------------------------------------------------------------
INPUT_PATH: str = ""
OUTPUT_DIR: Path = Path(".")
FULL_IMG: np.ndarray | None = None       # full-res raw RGB (lazy, None until export)
FULL_IR: np.ndarray | None = None        # full-res aligned IR (None if no IR)
FULL_IMG_READY: bool = False              # True once full-res load + alignment is done
DMIN: np.ndarray | None = None            # per-channel Dmin from full strip
PREVIEW_JPEG: bytes = b""                # downscaled JPEG for the browser
PREVIEW_SCALE: float = 1.0               # preview pixels / full pixels
FULL_WIDTH: int = 0                       # full-res dimensions (from raw load)
FULL_HEIGHT: int = 0
IMAGE_LIST: list[str] = []               # all image paths in folder
IMAGE_IDX: int = 0                        # current index into IMAGE_LIST
FILM_STOCK: str | None = None              # film stock for inversion (None = no inversion)
FILM_PROFILE: str | None = None            # path to calibration profile (overrides stock)
FILM_CONTRAST: float = 1.6                 # rendering contrast
IR_CLEAN: bool = True                     # auto-detect by default
PREVIEW_SIZE: int = 2400
LOADING: bool = False                     # True while switching images
HAS_IR: bool = False                      # whether current image has IR channel
PROGRESS: str = ""                        # current export/processing status for GUI
REBATE_RECT: dict | None = None           # {x, y, w, h} in full-res coords


def set_progress(msg: str) -> None:
    """Update progress status for both stdout and GUI polling."""
    global PROGRESS
    PROGRESS = msg
    print(f"  [{msg}]")


def _rebate_in_bounds(shape: tuple, rect: dict) -> bool:
    """Check if rebate rect fits within an image."""
    h, w = shape[:2]
    r = rect
    return (r["x"] >= 0 and r["y"] >= 0 and
            r["x"] + r["w"] <= w and r["y"] + r["h"] <= h)


def make_rebate_mask(h: int, w: int) -> np.ndarray | None:
    """Build a boolean mask from the rebate rectangle, if set."""
    if REBATE_RECT is None:
        return None
    r = REBATE_RECT
    mask = np.zeros((h, w), dtype=bool)
    y0 = max(0, r["y"])
    y1 = min(h, r["y"] + r["h"])
    x0 = max(0, r["x"])
    x1 = min(w, r["x"] + r["w"])
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = True
    return mask


def apply_inversion(img: np.ndarray) -> np.ndarray:
    """Film inversion pipeline. Input: uint16, output: uint16."""
    scene_linear = invert_negative(
        img,
        dmin=DMIN,
        stock=FILM_STOCK or "kodak_gold",
        profile_path=FILM_PROFILE,
    )
    return render_to_display(
        scene_linear,
        contrast=FILM_CONTRAST,
    )


def ensure_loaded() -> None:
    """Load the full-res image and align IR (if present).

    Alignment uses the full strip for best feature matching. Defect
    detection and inpainting are deferred to per-crop at export time
    so we only process the regions the user actually selected.

    Dmin is computed from the full strip (using rebate if available)
    so per-crop inversion has a correct film base reference.
    """
    global FULL_IMG, FULL_IR, FULL_IMG_READY, DMIN
    if FULL_IMG_READY:
        return

    set_progress("Loading full-resolution image...")
    rgb, ir = load_image(INPUT_PATH)
    h, w = rgb.shape[:2]
    set_progress(f"Loaded {w}x{h} image")

    # Store raw IR for potential later use; alignment is deferred to export
    # so we skip the cost entirely if the user unchecks IR clean.
    FULL_IR = ir  # None if no IR page

    # Compute Dmin if we don't already have one from a previous image.
    # Dmin persists across the roll — set once from a good rebate area.
    if FILM_STOCK and DMIN is None:
        if REBATE_RECT and _rebate_in_bounds(rgb.shape, REBATE_RECT):
            r = REBATE_RECT
            set_progress("Computing Dmin from rebate selection...")
            rebate_rgb = rgb[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]]
            DMIN = compute_dmin(rebate_rgb)
        else:
            set_progress("Computing Dmin from full image (no rebate set)...")
            DMIN = compute_dmin(rgb)

    FULL_IMG = rgb
    FULL_IMG_READY = True


def ir_clean_region(
    rgb_region: np.ndarray,
    ir_region: np.ndarray,
) -> np.ndarray:
    """Detect and inpaint defects in a cropped region."""
    mask = make_defect_mask(ir_region, rgb=rgb_region)
    n_defects = np.count_nonzero(mask)
    if n_defects > 0:
        print(f"    {n_defects} defect pixels in region")
        return inpaint(rgb_region, mask)
    return rgb_region


def switch_to_image(idx: int) -> None:
    """Load image, generate preview at reduced resolution for speed."""
    global INPUT_PATH, FULL_IMG, FULL_IMG_READY, PREVIEW_JPEG, PREVIEW_SCALE
    global IMAGE_IDX, LOADING, FULL_WIDTH, FULL_HEIGHT, HAS_IR
    LOADING = True
    FULL_IMG = None
    FULL_IR = None
    FULL_IMG_READY = False
    REBATE_RECT = None
    # DMIN persists across image switches — same roll = same film base
    IMAGE_IDX = idx
    INPUT_PATH = IMAGE_LIST[idx]

    print(f"Loading {INPUT_PATH} ({idx + 1}/{len(IMAGE_LIST)})...")
    rgb, ir = load_image(INPUT_PATH)
    h, w = rgb.shape[:2]
    FULL_WIDTH, FULL_HEIGHT = w, h
    HAS_IR = ir is not None
    print(f"  Image: {w}x{h}, {rgb.dtype}")

    # Downscale for preview
    preview_scale = min(PREVIEW_SIZE / max(h, w), 1.0)
    if preview_scale < 1.0:
        pw, ph = int(w * preview_scale), int(h * preview_scale)
        small_rgb = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        small_rgb = rgb

    # Preview just needs to be usable for identifying frames and selecting
    # rebate — not a proper inversion. Simple invert + histogram equalization.
    print("  Generating quick preview...")
    if small_rgb.dtype == np.uint16:
        preview8 = (small_rgb >> 8).astype(np.uint8)
    else:
        preview8 = small_rgb
    # Invert (negative → rough positive)
    preview8 = 255 - preview8
    # Per-channel CLAHE for local contrast (makes frames visible regardless of exposure)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for c in range(3):
        preview8[:, :, c] = clahe.apply(preview8[:, :, c])
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
                "has_dmin": DMIN is not None,
            }
            self._respond(200, "application/json", json.dumps(info).encode())
        elif parsed.path == "/progress":
            self._respond(200, "application/json",
                          json.dumps({"status": PROGRESS}).encode())
        elif parsed.path == "/settings":
            cfg = load_config()
            self._respond(200, "application/json", json.dumps(cfg).encode())
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
        elif self.path == "/rebate":
            global REBATE_RECT
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            REBATE_RECT = {
                "x": int(body["x"]),
                "y": int(body["y"]),
                "w": int(body["w"]),
                "h": int(body["h"]),
            }
            print(f"  Rebate set: {REBATE_RECT}")
            # Compute Dmin from the rebate region and persist
            if FULL_IMG is not None and FILM_STOCK:
                global DMIN
                r = REBATE_RECT
                rebate_rgb = FULL_IMG[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]]
                DMIN = compute_dmin(rebate_rgb)
                print(f"  Dmin updated: R={DMIN[0]:.4f} G={DMIN[1]:.4f} B={DMIN[2]:.4f}")
                save_config({"dmin": DMIN.tolist()})
            self._respond(200, "application/json",
                          json.dumps({"ok": True}).encode())
        elif self.path == "/settings":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            save_config(body)
            self._respond(200, "application/json",
                          json.dumps({"ok": True}).encode())
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
    ir_clean = body.get("ir_clean", True)
    n_rects = len(rects)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exported = []

    set_progress(f"Preparing full-res export ({n_rects} frame{'s' if n_rects != 1 else ''})...")
    ensure_loaded()
    if DMIN is not None:
        print(f"  Dmin: R={DMIN[0]:.4f} G={DMIN[1]:.4f} B={DMIN[2]:.4f}")
    if REBATE_RECT:
        print(f"  Rebate: {REBATE_RECT}")

    # Align IR once if IR cleaning is requested and we have an IR channel
    aligned_ir = None
    if ir_clean and FULL_IR is not None:
        set_progress("Aligning IR channel (full strip)...")
        aligned_ir = align_ir(FULL_IMG, FULL_IR)

    for i, r in enumerate(rects):
        set_progress(f"Cropping frame {i + 1}/{n_rects}...")
        cx, cy, w, h = r["cx"], r["cy"], r["w"], r["h"]
        angle = r.get("angle", 0)

        cropped = crop_rotated_rect(FULL_IMG, cx, cy, w, h, angle)

        # Per-crop IR cleaning (detect + inpaint only the frame region)
        if aligned_ir is not None:
            set_progress(f"IR cleaning frame {i + 1}/{n_rects}...")
            ir_cropped = crop_rotated_rect(aligned_ir, cx, cy, w, h, angle)
            cropped = ir_clean_region(cropped, ir_cropped)

        if FILM_STOCK:
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
        # Embed processing metadata in TIFF Software tag (single-valued,
        # avoids duplicate ImageDescription that breaks Apple Preview)
        meta = {
            "source": Path(INPUT_PATH).name,
            "stock": FILM_STOCK,
            "contrast": FILM_CONTRAST,
            "ir_clean": ir_clean and aligned_ir is not None,
            "dmin": DMIN.tolist() if DMIN is not None else None,
            "rebate_rect": REBATE_RECT,
            "crop": {"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle},
        }
        meta_json = json.dumps(meta)
        tifffile.imwrite(str(out_path), cropped, extratags=[
            # Tag 65000: private tag for scratchndent metadata
            (65000, 's', 0, meta_json, True),
        ])
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
    global OUTPUT_DIR, IMAGE_LIST, FILM_STOCK, FILM_PROFILE, FILM_CONTRAST
    global IR_CLEAN, PREVIEW_SIZE, DMIN

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
        "--stock", type=str, default=None,
        help="Film stock for inversion (kodak_gold, kodak_portra)",
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Path to calibration profile JSON (overrides --stock)",
    )
    parser.add_argument(
        "--contrast", type=float, default=1.6,
        help="Rendering contrast (default: 1.6, range ~0.8-2.0)",
    )
    parser.add_argument(
        "--no-ir-clean", action="store_true",
        help="Disable automatic IR dust/scratch removal",
    )
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    FILM_STOCK = args.stock
    FILM_PROFILE = args.profile
    FILM_CONTRAST = args.contrast

    # Load persisted settings from config file
    cfg = load_config()
    if cfg.get("dmin") and FILM_STOCK:
        DMIN = np.array(cfg["dmin"], dtype=np.float64)
        print(f"Loaded Dmin from config: R={DMIN[0]:.4f} G={DMIN[1]:.4f} B={DMIN[2]:.4f}")
    IR_CLEAN = cfg.get("ir_clean", not args.no_ir_clean)
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
