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
import time
import webbrowser
from concurrent.futures import ProcessPoolExecutor, as_completed
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


CONFIG_FILE = Path("scratchndent_config.toml")

# Reference DPI for pixel-based parameters. All pixel params are defined
# at this resolution and scaled automatically for the actual scan DPI.
REFERENCE_DPI = 800

# Default algorithm parameters — overridden by config file values.
# Pixel-based params are defined at REFERENCE_DPI.
PARAM_DEFAULTS = {
    # IR dust/scratch detection (pixel params at 800 DPI reference)
    "ir_threshold": 0.25,
    "ir_hair_sensitivity": 0.10,
    "ir_min_area": 3,              # pixels at 800 DPI
    "ir_dilate_radius": 4,         # pixels at 800 DPI
    "ir_close_radius": 6,          # pixels at 800 DPI
    "ir_blur_size": 301,           # pixels at 800 DPI
    "ir_max_coverage": 0.03,

    # Inpainting
    "inpaint_padding": 16,         # pixels at 800 DPI

    # Film rendering
    "render_contrast": 1.2,
    "render_curve_k": 5.0,
    "render_percentile_lo": 0.5,
    "render_percentile_hi": 99.5,

    # Film inversion
    "exposure_compensation": 0.0,

    # UI / preview
    "preview_size": 2400,
    "clahe_clip": 2.0,
}

# Parameters that scale with DPI (pixel-based)
DPI_SCALED_PARAMS = {
    "ir_min_area",       # scales as area (DPI ratio squared)
    "ir_dilate_radius",  # scales linearly
    "ir_close_radius",
    "ir_blur_size",
    "inpaint_padding",
}
# Area-based params scale quadratically
DPI_AREA_PARAMS = {"ir_min_area"}

# Comments for each parameter, written to TOML for self-documentation
PARAM_COMMENTS = {
    "stock": "Active film stock name — must match a [stocks.<name>] section below",
    "preview_size": "Max preview dimension in pixels (default 2400)",
    "ir_threshold": "Defect detection sensitivity (lower = more aggressive, default 0.25)",
    "ir_hair_sensitivity": "Meijering line filter threshold for hairs/scratches (lower = more sensitive, default 0.10)",
    "ir_min_area": "Minimum defect size in pixels at 800 DPI — auto-scaled for scan resolution (default 3)",
    "ir_dilate_radius": "Mask dilation in pixels at 800 DPI — auto-scaled for scan resolution (default 4)",
    "ir_close_radius": "Morphological close in pixels at 800 DPI — auto-scaled for scan resolution (default 6)",
    "ir_blur_size": "Background blur kernel in pixels at 800 DPI — auto-scaled for scan resolution (default 301)",
    "ir_max_coverage": "Sanity cap: max fraction of image flagged as defects before giving up (default 0.03 = 3%)",
    "inpaint_padding": "Context padding in pixels at 800 DPI — auto-scaled for scan resolution (default 16)",
    "render_contrast": "S-curve contrast strength: 1.0 = linear, 1.5 = moderate, 2.0 = punchy (default 1.2)",
    "render_curve_k": "S-curve steepness multiplier (default 5.0)",
    "render_percentile_lo": "Low percentile for display range normalization (default 0.5)",
    "render_percentile_hi": "High percentile for display range normalization (default 99.5)",
    "exposure_compensation": "Density-domain exposure shift: positive = brighter output (default 0.0)",
    "clahe_clip": "CLAHE clip limit for preview contrast enhancement (default 2.0)",
    "dmin": "Film base density [R, G, B] — set via rebate selection in the UI",
    "ir_clean": "Enable IR dust/scratch removal (true/false)",
    "invert": "Enable film negative inversion (true/false)",
    "aspect": "Last used aspect ratio for frame selection",
}

# Map flat keys to TOML sections
PARAM_SECTIONS = {
    "ir_threshold": "dust_removal",
    "ir_hair_sensitivity": "dust_removal",
    "ir_min_area": "dust_removal",
    "ir_dilate_radius": "dust_removal",
    "ir_close_radius": "dust_removal",
    "ir_blur_size": "dust_removal",
    "ir_max_coverage": "dust_removal",
    "inpaint_padding": "dust_removal",
    "render_contrast": "render",
    "render_curve_k": "render",
    "render_percentile_lo": "render",
    "render_percentile_hi": "render",
    "exposure_compensation": "inversion",
}

# Built-in film stock definitions. Each maps to polynomial coefficients (10x3).
# Basis: [R, G, B, R², G², B², RG, RB, GB, 1] → [R_out, G_out, B_out]
from scratchndent.film_calibration import default_kodak_gold_coeffs, default_kodak_portra_coeffs

BUILTIN_STOCKS = {
    "kodak_gold": {
        "description": "Kodak Gold 200 on Epson V600",
        "coeffs": default_kodak_gold_coeffs().tolist(),
    },
    "kodak_portra": {
        "description": "Kodak Portra 400 on Epson V600",
        "coeffs": default_kodak_portra_coeffs().tolist(),
    },
}


def get_available_stocks() -> dict[str, dict]:
    """Get all available film stocks: built-in + config-defined."""
    stocks = dict(BUILTIN_STOCKS)
    cfg = load_config()
    # Config stocks override built-ins
    if "_stocks" in cfg:
        stocks.update(cfg["_stocks"])
    return stocks


def get_stock_coeffs(name: str) -> np.ndarray:
    """Get polynomial coefficients for a film stock by name."""
    stocks = get_available_stocks()
    if name in stocks:
        return np.array(stocks[name]["coeffs"], dtype=np.float64)
    raise ValueError(f"Unknown film stock '{name}'. "
                     f"Available: {list(stocks.keys())}")


def load_config() -> dict:
    """Load persisted settings from TOML config file.

    Returns a flat dict with parameter sections flattened.
    Stock definitions are stored under the special key "_stocks".
    """
    if CONFIG_FILE.exists():
        try:
            import tomllib
            with open(CONFIG_FILE, "rb") as f:
                raw = tomllib.load(f)
            flat = {}
            stocks = {}
            for k, v in raw.items():
                if k == "stocks" and isinstance(v, dict):
                    stocks = v
                elif isinstance(v, dict):
                    flat.update(v)
                else:
                    flat[k] = v
            if stocks:
                flat["_stocks"] = stocks
            return flat
        except (Exception,):
            pass
    return {}


def _format_toml_value(v) -> str:
    """Format a Python value as TOML."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        items = ", ".join(_format_toml_value(x) for x in v)
        return f"[{items}]"
    return repr(v)


def save_config(updates: dict) -> None:
    """Merge updates into the TOML config file with comments."""
    cfg = load_config()
    cfg.update(updates)

    # Separate stock definitions from flat params
    stocks = cfg.pop("_stocks", {})

    lines = ["# scratchndent configuration", "# Edit values below or uncomment to override defaults", ""]

    # Top-level keys
    top_keys = ["stock", "preview_size", "dmin", "ir_clean", "invert", "aspect"]
    for k in top_keys:
        comment = PARAM_COMMENTS.get(k)
        if comment:
            lines.append(f"# {comment}")
        if k in cfg:
            lines.append(f"{k} = {_format_toml_value(cfg[k])}")
        elif k in PARAM_DEFAULTS:
            lines.append(f"# {k} = {_format_toml_value(PARAM_DEFAULTS[k])}")
        lines.append("")

    # Parameter sections
    for section_name in ["dust_removal", "inversion", "render"]:
        section_keys = [k for k, s in PARAM_SECTIONS.items() if s == section_name]
        if not section_keys:
            continue
        lines.append(f"[{section_name}]")
        for k in section_keys:
            comment = PARAM_COMMENTS.get(k)
            if comment:
                lines.append(f"# {comment}")
            if k in cfg:
                lines.append(f"{k} = {_format_toml_value(cfg[k])}")
            else:
                lines.append(f"# {k} = {_format_toml_value(PARAM_DEFAULTS[k])}")
        lines.append("")

    # Film stock definitions
    # Write all available stocks (built-in + custom)
    all_stocks = dict(BUILTIN_STOCKS)
    all_stocks.update(stocks)
    lines.append("# Film stock profiles. Each defines polynomial coefficients for the")
    lines.append("# density-to-scene transform. Basis: [R, G, B, R2, G2, B2, RG, RB, GB, 1] -> [R, G, B]")
    lines.append("# Rows are the 10 basis terms, columns are R/G/B output channels.")
    lines.append("# Edit coefficients to tune color response for your scanner+stock combination.")
    lines.append("")
    for name, stock_def in all_stocks.items():
        lines.append(f"[stocks.{name}]")
        if "description" in stock_def:
            lines.append(f'description = "{stock_def["description"]}"')
        coeffs = stock_def["coeffs"]
        basis_labels = ["R", "G", "B", "R2", "G2", "B2", "RG", "RB", "GB", "bias"]
        lines.append("coeffs = [")
        for i, row in enumerate(coeffs):
            row_str = ", ".join(f"{v:8.4f}" for v in row)
            lines.append(f"    [{row_str}],  # {basis_labels[i]}")
        lines.append("]")
        lines.append("")

    CONFIG_FILE.write_text("\n".join(lines))


def read_tiff_dpi(path: str) -> int | None:
    """Read DPI from TIFF XResolution tag. Returns None if not available."""
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            res_tag = page.tags.get(282)  # XResolution
            unit_tag = page.tags.get(296)  # ResolutionUnit
            if res_tag is None:
                return None
            # XResolution is a rational (numerator, denominator)
            val = res_tag.value
            if isinstance(val, tuple):
                dpi = val[0] / val[1] if val[1] != 0 else val[0]
            else:
                dpi = float(val)
            # ResolutionUnit: 2=inches, 3=centimeters
            if unit_tag and unit_tag.value == 3:
                dpi *= 2.54  # convert from dots/cm to DPI
            return int(round(dpi))
    except Exception:
        return None


CURRENT_DPI: int | None = None  # set when an image is loaded


def get_dpi_scale() -> float:
    """Get the scale factor from reference DPI to current scan DPI."""
    if CURRENT_DPI and CURRENT_DPI > 0:
        return CURRENT_DPI / REFERENCE_DPI
    return 1.0


def get_param(name: str) -> float | int:
    """Get a parameter value, scaled for DPI if applicable."""
    cfg = load_config()
    raw = cfg[name] if name in cfg else PARAM_DEFAULTS[name]

    if name in DPI_SCALED_PARAMS:
        scale = get_dpi_scale()
        if name in DPI_AREA_PARAMS:
            raw = raw * scale * scale  # area scales quadratically
        else:
            raw = raw * scale
        # Blur size must be odd
        if name == "ir_blur_size":
            raw = int(raw) | 1  # ensure odd
            return raw

    return type(PARAM_DEFAULTS[name])(raw)


def get_active_stock() -> str | None:
    """Get the active film stock name from config, or None if not set."""
    cfg = load_config()
    return cfg.get("stock")


def get_preview_size() -> int:
    """Get the preview size from config."""
    cfg = load_config()
    return int(cfg.get("preview_size", PARAM_DEFAULTS["preview_size"]))


# ---------------------------------------------------------------------------
# Globals set at startup
# ---------------------------------------------------------------------------
INPUT_PATH: str = ""
INPUT_DIR: Path = Path(".")               # directory to scan for images
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
IR_CLEAN: bool = True                     # auto-detect by default
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
    stock = get_active_stock()
    coeffs = get_stock_coeffs(stock) if stock else None
    scene_linear = invert_negative(
        img,
        dmin=DMIN,
        stock=stock or "kodak_gold",
        coeffs=coeffs,
        exposure_compensation=get_param("exposure_compensation"),
    )
    return render_to_display(
        scene_linear,
        contrast=get_param("render_contrast"),
        curve_k=get_param("render_curve_k"),
        percentile_lo=get_param("render_percentile_lo"),
        percentile_hi=get_param("render_percentile_hi"),
    )


def ensure_loaded() -> None:
    """Load the full-res image and align IR (if present).

    Alignment uses the full strip for best feature matching. Defect
    detection and inpainting are deferred to per-crop at export time
    so we only process the regions the user actually selected.

    Dmin is computed from the full strip (using rebate if available)
    so per-crop inversion has a correct film base reference.
    """
    global FULL_IMG, FULL_IR, FULL_IMG_READY, DMIN, CURRENT_DPI
    if FULL_IMG_READY:
        return

    # Ensure DPI is set for parameter scaling
    if CURRENT_DPI is None:
        CURRENT_DPI = read_tiff_dpi(INPUT_PATH)

    set_progress("Loading full-resolution image...")
    rgb, ir = load_image(INPUT_PATH)
    h, w = rgb.shape[:2]
    set_progress(f"Loaded {w}x{h} image")

    # Store raw IR for potential later use; alignment is deferred to export
    # so we skip the cost entirely if the user unchecks IR clean.
    FULL_IR = ir  # None if no IR page

    # Compute Dmin if we don't already have one from a previous image.
    # Dmin persists across the roll — set once from a good rebate area.
    if get_active_stock() and DMIN is None:
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
    """Detect defects at IR resolution, inpaint at RGB resolution.

    Handles different resolutions between RGB and IR — the mask is
    computed at IR resolution then upscaled to RGB resolution for
    inpainting at full detail.
    """
    rgb_h, rgb_w = rgb_region.shape[:2]
    ir_h, ir_w = ir_region.shape[:2]

    # Detect defects at IR resolution using config parameters
    mask_ir = make_defect_mask(
        ir_region,
        threshold=get_param("ir_threshold"),
        hair_sensitivity=get_param("ir_hair_sensitivity"),
        min_area=get_param("ir_min_area"),
        dilate_radius=get_param("ir_dilate_radius"),
        close_radius=get_param("ir_close_radius"),
        blur_size=get_param("ir_blur_size"),
        max_coverage=get_param("ir_max_coverage"),
    )
    n_defects = np.count_nonzero(mask_ir)
    if n_defects == 0:
        return rgb_region

    # Upscale mask to RGB resolution if needed
    if rgb_h != ir_h or rgb_w != ir_w:
        mask = cv2.resize(mask_ir, (rgb_w, rgb_h),
                          interpolation=cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel)
    else:
        mask = mask_ir

    n_final = np.count_nonzero(mask)
    print(f"    {n_defects} defect pixels (IR) → {n_final} pixels (RGB)")
    return inpaint(rgb_region, mask, padding=get_param("inpaint_padding"))


def switch_to_image(idx: int) -> None:
    """Load image, generate preview at reduced resolution for speed."""
    global INPUT_PATH, FULL_IMG, FULL_IMG_READY, PREVIEW_JPEG, PREVIEW_SCALE
    global IMAGE_IDX, LOADING, FULL_WIDTH, FULL_HEIGHT, HAS_IR, CURRENT_DPI
    LOADING = True
    FULL_IMG = None
    FULL_IR = None
    FULL_IMG_READY = False
    REBATE_RECT = None
    # DMIN persists across image switches — same roll = same film base
    IMAGE_IDX = idx
    INPUT_PATH = IMAGE_LIST[idx]

    # Read DPI from file metadata for parameter scaling
    CURRENT_DPI = read_tiff_dpi(INPUT_PATH)
    print(f"Loading {INPUT_PATH} ({idx + 1}/{len(IMAGE_LIST)})...")
    rgb, ir = load_image(INPUT_PATH)
    h, w = rgb.shape[:2]
    FULL_WIDTH, FULL_HEIGHT = w, h
    HAS_IR = ir is not None
    dpi_str = f", {CURRENT_DPI} DPI (scale {get_dpi_scale():.1f}x)" if CURRENT_DPI else ""
    print(f"  Image: {w}x{h}, {rgb.dtype}{dpi_str}")

    # Downscale for preview
    preview_scale = min(get_preview_size() / max(h, w), 1.0)
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
    # Mask out sprocket holes before inversion — they're near-white in the
    # raw scan (full scanner light, no film) and would become black after
    # inversion, skewing the contrast stretch.
    gray_raw = cv2.cvtColor(preview8, cv2.COLOR_RGB2GRAY)
    content_mask = gray_raw < 240  # exclude sprocket holes / scanner light

    # Invert (negative → rough positive)
    preview8 = 255 - preview8

    # Per-channel contrast stretch using only film content pixels
    if np.count_nonzero(content_mask) > 100:
        for c in range(3):
            ch = preview8[:, :, c]
            lo = np.percentile(ch[content_mask], 1)
            hi = np.percentile(ch[content_mask], 99)
            if hi > lo:
                preview8[:, :, c] = np.clip(
                    (ch.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255
                ).astype(np.uint8)
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
                if ir_page.ndim == 2:
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
                "dpi": CURRENT_DPI,
                "dpi_scale": round(get_dpi_scale(), 2),
            }
            self._respond(200, "application/json", json.dumps(info).encode())
        elif parsed.path == "/progress":
            self._respond(200, "application/json",
                          json.dumps({"status": PROGRESS}).encode())
        elif parsed.path == "/settings":
            cfg = load_config()
            cfg.pop("_stocks", None)  # don't send raw stock defs in flat settings
            self._respond(200, "application/json", json.dumps(cfg).encode())
        elif parsed.path == "/stocks":
            stocks = get_available_stocks()
            data = {
                "active": get_active_stock(),
                "stocks": {k: v.get("description", k) for k, v in stocks.items()},
            }
            self._respond(200, "application/json", json.dumps(data).encode())
        elif parsed.path == "/images":
            rescan_images()
            data = {
                "images": [Path(p).name for p in IMAGE_LIST],
                "current": IMAGE_IDX,
            }
            self._respond(200, "application/json", json.dumps(data).encode())
        elif parsed.path == "/gallery":
            gallery_path = Path(__file__).parent / "gallery.html"
            self._respond(200, "text/html", gallery_path.read_bytes())
        elif parsed.path == "/gallery/list":
            export_files = sorted(
                p.name for p in OUTPUT_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in TIFF_EXTS
            ) if OUTPUT_DIR.exists() else []
            self._respond(200, "application/json",
                          json.dumps({"files": export_files}).encode())
        elif parsed.path.startswith("/gallery/thumb/"):
            name = parsed.path[len("/gallery/thumb/"):]
            self._serve_export_jpeg(name, max_dim=200)
        elif parsed.path.startswith("/gallery/full/"):
            name = parsed.path[len("/gallery/full/"):]
            self._serve_export_jpeg(name, max_dim=2400)
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
            # Compute Dmin from the rebate region. Use cached full image
            # if available, otherwise read just the region from disk.
            global DMIN
            if get_active_stock():
                r = REBATE_RECT
                if FULL_IMG is not None:
                    rebate_rgb = FULL_IMG[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]]
                else:
                    print(f"  Reading rebate region from {INPUT_PATH}...")
                    with tifffile.TiffFile(INPUT_PATH) as tif:
                        page = tif.pages[0]
                        full = page.asarray()
                        rebate_rgb = full[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]].copy()
                        del full
                DMIN = compute_dmin(rebate_rgb)
                print(f"  Dmin updated: R={DMIN[0]:.4f} G={DMIN[1]:.4f} B={DMIN[2]:.4f}")
                save_config({"dmin": DMIN.tolist()})
            self._respond(200, "application/json",
                          json.dumps({"ok": True, "dmin": DMIN.tolist() if DMIN is not None else None}).encode())
        elif self.path == "/settings":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            save_config(body)
            self._respond(200, "application/json",
                          json.dumps({"ok": True}).encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def _serve_export_jpeg(self, name: str, max_dim: int = 2400):
        """Serve an exported TIFF as a JPEG preview."""
        from urllib.parse import unquote
        name = unquote(name)
        path = OUTPUT_DIR / name
        if not path.exists() or not path.is_file():
            self._respond(404, "text/plain", b"Not found")
            return
        try:
            with tifffile.TiffFile(str(path)) as tif:
                img = tif.pages[0].asarray()
            if img.dtype == np.uint16:
                img = (img >> 8).astype(np.uint8)
            h, w = img.shape[:2]
            scale = min(max_dim / max(h, w), 1.0)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            self._respond(200, "image/jpeg", buf.getvalue())
        except Exception as e:
            self._respond(500, "text/plain", str(e).encode())

    def _respond(self, code, content_type, data):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _process_frame(
    frame_idx: int,
    rect: dict,
    rgb_img: np.ndarray,
    aligned_ir: np.ndarray | None,
    ir_scale_x: float,
    ir_scale_y: float,
    film_stock: str | None,
    dmin: np.ndarray | None,
    out_path: str,
    meta: dict,
) -> dict:
    """Process and export a single frame. Designed to run in a worker process."""
    timings = {}
    t0 = time.monotonic()

    cx, cy, w, h = rect["cx"], rect["cy"], rect["w"], rect["h"]
    angle = rect.get("angle", 0)

    t = time.monotonic()
    cropped = crop_rotated_rect(rgb_img, cx, cy, w, h, angle)
    timings["crop"] = time.monotonic() - t

    # Per-crop IR cleaning
    if aligned_ir is not None:
        t = time.monotonic()
        ir_cropped = crop_rotated_rect(
            aligned_ir,
            cx * ir_scale_x, cy * ir_scale_y,
            w * ir_scale_x, h * ir_scale_y,
            angle,
        )
        cropped = ir_clean_region(cropped, ir_cropped)
        timings["ir_clean"] = time.monotonic() - t

    if film_stock:
        t = time.monotonic()
        scene_linear = invert_negative(
            cropped,
            dmin=dmin,
            stock=film_stock,
            exposure_compensation=get_param("exposure_compensation"),
        )
        timings["invert"] = time.monotonic() - t

        t = time.monotonic()
        cropped = render_to_display(
            scene_linear,
            contrast=get_param("render_contrast"),
            curve_k=get_param("render_curve_k"),
            percentile_lo=get_param("render_percentile_lo"),
            percentile_hi=get_param("render_percentile_hi"),
        )
        timings["render"] = time.monotonic() - t

    # Apply output rotation
    rotation = rect.get("rotation", 0)
    if rotation == 90:
        cropped = np.rot90(cropped, k=-1)
    elif rotation == 180:
        cropped = np.rot90(cropped, k=2)
    elif rotation == 270:
        cropped = np.rot90(cropped, k=1)

    t = time.monotonic()
    meta_json = json.dumps(meta)
    tifffile.imwrite(out_path, cropped, extratags=[
        (65000, 's', 0, meta_json, True),
    ])
    timings["write"] = time.monotonic() - t

    timings["total"] = time.monotonic() - t0
    return {"out_path": out_path, "timings": timings,
            "shape": (cropped.shape[1], cropped.shape[0])}


def handle_export(body: dict) -> dict:
    default_base = Path(INPUT_PATH).stem
    basename = body.get("basename", default_base)
    rects = body.get("rects", [])
    ir_clean = body.get("ir_clean", True)
    invert = body.get("invert", True)
    n_rects = len(rects)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    set_progress(f"Preparing full-res export ({n_rects} frame{'s' if n_rects != 1 else ''})...")

    t = time.monotonic()
    ensure_loaded()
    t_load = time.monotonic() - t
    if DMIN is not None:
        print(f"  Dmin: R={DMIN[0]:.4f} G={DMIN[1]:.4f} B={DMIN[2]:.4f}")

    # Align IR once (shared across all frames)
    aligned_ir = None
    ir_scale_x = ir_scale_y = 1.0
    t_align = 0.0
    if ir_clean and FULL_IR is not None:
        set_progress("Aligning IR channel (full strip)...")
        t = time.monotonic()
        aligned_ir = align_ir(FULL_IMG, FULL_IR)
        t_align = time.monotonic() - t
        rgb_h, rgb_w = FULL_IMG.shape[:2]
        ir_h, ir_w = aligned_ir.shape[:2]
        ir_scale_x = ir_w / rgb_w
        ir_scale_y = ir_h / rgb_h

    # Build output paths and metadata for each frame
    frame_jobs = []
    for i, r in enumerate(rects):
        cx, cy, w, h = r["cx"], r["cy"], r["w"], r["h"]
        angle = r.get("angle", 0)

        out_name = f"{basename}_{i + 1:02d}.tif"
        out_path = OUTPUT_DIR / out_name
        n = 1
        while out_path.exists():
            out_name = f"{basename}_{i + 1:02d}_{n}.tif"
            out_path = OUTPUT_DIR / out_name
            n += 1

        meta = {
            "source": Path(INPUT_PATH).name,
            "stock": get_active_stock() if invert else None,
            "contrast": get_param("render_contrast") if invert else None,
            "ir_clean": ir_clean and aligned_ir is not None,
            "invert": invert,
            "dmin": DMIN.tolist() if DMIN is not None and invert else None,
            "rebate_rect": REBATE_RECT,
            "crop": {"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle},
        }
        frame_jobs.append((i, r, str(out_path), out_name, meta))

    # Process frames in parallel
    set_progress(f"Processing {n_rects} frame{'s' if n_rects != 1 else ''} in parallel...")
    exported = [None] * n_rects
    all_timings = [None] * n_rects

    film_stock = get_active_stock() if invert else None

    if n_rects == 1:
        # Single frame — no pool overhead
        i, r, out_path, out_name, meta = frame_jobs[0]
        result = _process_frame(
            i, r, FULL_IMG, aligned_ir, ir_scale_x, ir_scale_y,
            film_stock, DMIN, out_path, meta,
        )
        exported[0] = out_name
        all_timings[0] = result["timings"]
        set_progress(f"Wrote {out_name} ({result['shape'][0]}x{result['shape'][1]})")
    else:
        # Multiple frames — use thread pool (numpy/numba release GIL)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(n_rects, 4)) as pool:
            futures = {}
            for i, r, out_path, out_name, meta in frame_jobs:
                fut = pool.submit(
                    _process_frame,
                    i, r, FULL_IMG, aligned_ir, ir_scale_x, ir_scale_y,
                    film_stock, DMIN, out_path, meta,
                )
                futures[fut] = (i, out_name)

            for fut in as_completed(futures):
                i, out_name = futures[fut]
                result = fut.result()
                exported[i] = out_name
                all_timings[i] = result["timings"]
                set_progress(f"Wrote {out_name} ({result['shape'][0]}x{result['shape'][1]})")

    t_total = time.monotonic() - t_start

    # Print timing summary
    print(f"\n  === Export timing summary ===")
    print(f"  Load:      {t_load:.2f}s")
    if t_align > 0:
        print(f"  IR align:  {t_align:.2f}s")
    for i, timings in enumerate(all_timings):
        if timings:
            parts = " | ".join(f"{k}: {v:.2f}s" for k, v in timings.items())
            print(f"  Frame {i+1}:   {parts}")
    print(f"  Total:     {t_total:.2f}s\n")

    msg = f"Exported {len(exported)} frame{'s' if len(exported) != 1 else ''} to {OUTPUT_DIR}/ ({t_total:.1f}s)"
    set_progress(msg)
    return {"message": msg, "files": exported}


TIFF_EXTS = {".tif", ".tiff"}


def rescan_images() -> None:
    """Re-scan the input directory for new/removed files."""
    global IMAGE_LIST, IMAGE_IDX
    old_current = IMAGE_LIST[IMAGE_IDX] if IMAGE_IDX < len(IMAGE_LIST) else None
    IMAGE_LIST = find_images(INPUT_DIR)
    # Try to keep the current image selected
    if old_current and old_current in IMAGE_LIST:
        IMAGE_IDX = IMAGE_LIST.index(old_current)
    elif IMAGE_IDX >= len(IMAGE_LIST):
        IMAGE_IDX = max(0, len(IMAGE_LIST) - 1)


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
    global OUTPUT_DIR, INPUT_DIR, IMAGE_LIST
    global IR_CLEAN, DMIN

    parser = argparse.ArgumentParser(
        description="Browser-based frame extraction from scanned film strips",
        epilog="Examples:\n"
               "  python extract.py scan.tiff           # single file\n"
               "  python extract.py /path/to/scans/     # whole directory\n"
               "\n"
               "Film stock, IR cleaning, contrast, and other settings are\n"
               "configured in the GUI or scratchndent_config.toml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Input TIFF file or directory of TIFFs to process",
    )
    parser.add_argument("--port", type=int, default=8888,
                        help="Server port (default: 8888)")
    parser.add_argument(
        "--output-dir", type=str, default="frames",
        help="Directory for exported frames (default: frames/)",
    )
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit(1)

    OUTPUT_DIR = Path(args.output_dir)

    # Load persisted settings from config file
    cfg = load_config()
    if cfg.get("dmin") and get_active_stock():
        DMIN = np.array(cfg["dmin"], dtype=np.float64)
        print(f"Loaded Dmin from config: R={DMIN[0]:.4f} G={DMIN[1]:.4f} B={DMIN[2]:.4f}")
    IR_CLEAN = cfg.get("ir_clean", True)

    input_path = Path(args.input)
    INPUT_DIR = input_path.parent if input_path.is_file() else input_path
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
