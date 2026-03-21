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
    return r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Frame Extractor</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a1a; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }
#toolbar {
    position: fixed; top: 0; left: 0; right: 0; height: 48px; z-index: 100;
    background: #2a2a2a; display: flex; align-items: center; padding: 0 16px; gap: 12px;
    border-bottom: 1px solid #444;
}
#toolbar label { font-size: 13px; color: #aaa; }
#toolbar select, #toolbar button, #toolbar input {
    font-size: 13px; padding: 4px 8px; border-radius: 4px; border: 1px solid #555;
    background: #333; color: #eee;
}
#toolbar button { cursor: pointer; }
#toolbar button:hover { background: #444; }
#toolbar button.primary { background: #2d6; color: #111; border-color: #2d6; font-weight: 600; }
#toolbar button.primary:hover { background: #3e7; }
#toolbar button.danger { background: #d44; color: #fff; border-color: #d44; }
#toolbar button.danger:hover { background: #e55; }
#sidebar {
    position: fixed; top: 48px; right: 0; width: 240px; bottom: 0;
    background: #222; border-left: 1px solid #444; overflow-y: auto; padding: 8px;
    z-index: 90;
}
.sel-item {
    padding: 8px; margin-bottom: 4px; border-radius: 4px; background: #2a2a2a;
    border: 1px solid #444; font-size: 12px; cursor: pointer;
}
.sel-item.active { border-color: #2d6; }
.sel-item .del { float: right; cursor: pointer; color: #d44; }
#canvas-wrap {
    position: fixed; top: 48px; left: 0; right: 240px; bottom: 0; overflow: hidden;
}
canvas { position: absolute; top: 0; left: 0; cursor: crosshair; }
#status {
    position: fixed; bottom: 0; left: 0; right: 240px; height: 24px;
    background: #2a2a2a; border-top: 1px solid #444; padding: 0 12px;
    font-size: 12px; line-height: 24px; color: #888; z-index: 100;
}
</style>
</head>
<body>
<div id="toolbar">
    <button id="btn-prev" title="Previous image">&larr;</button>
    <span id="img-name" style="font-size:13px;min-width:120px;text-align:center"></span>
    <button id="btn-next" title="Next image">&rarr;</button>
    <span style="width:1px;height:24px;background:#555;margin:0 4px"></span>
    <label>Aspect:</label>
    <select id="aspect">
        <option value="free">Free</option>
        <option value="3:2" selected>3:2</option>
        <option value="2:3">2:3</option>
        <option value="4:3">4:3</option>
        <option value="3:4">3:4</option>
        <option value="5:4">5:4</option>
        <option value="1:1">1:1</option>
    </select>
    <button id="btn-add">+ New selection</button>
    <button id="btn-clear" class="danger">Clear all</button>
    <span style="flex:1"></span>
    <label>Base name:</label>
    <input id="basename" value="" style="width:100px">
    <button id="btn-export" class="primary">Export all</button>
</div>
<div id="loading-overlay" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;
    background:rgba(0,0,0,0.7);z-index:200;align-items:center;justify-content:center;
    font-size:20px;color:#eee;">Loading...</div>
<div id="canvas-wrap">
    <canvas id="canvas"></canvas>
</div>
<div id="sidebar">
    <div style="padding:4px 0 8px;font-size:13px;color:#888;">Selections</div>
    <div id="sel-list"></div>
</div>
<div id="status">Ready</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('canvas-wrap');
const selList = document.getElementById('sel-list');
const statusEl = document.getElementById('status');

let img = new window.Image();
let imgW = 0, imgH = 0;
let scale = 1, offsetX = 0, offsetY = 0;
let previewScale = 1; // preview px / full px

let selections = [];
let activeIdx = -1;
let mode = 'idle'; // idle, drawing, moving, resizing, rotating
let drawStart = null;
let dragInfo = null;

const HANDLE_SIZE = 8;
const ROT_HANDLE_DIST = 30;

// --- Image loading ---
img.onload = () => {
    imgW = img.width; imgH = img.height;
    fitImage();
    render();
};
img.src = '/preview';
let imageIdx = 0, imageCount = 1;
const imgNameEl = document.getElementById('img-name');
const basenameEl = document.getElementById('basename');
const overlay = document.getElementById('loading-overlay');

function updateImageInfo(info) {
    previewScale = info.preview_scale;
    imageIdx = info.image_idx;
    imageCount = info.image_count;
    const name = info.filename || '';
    imgNameEl.textContent = `${imageIdx + 1}/${imageCount}: ${name}`;
    basenameEl.value = name.replace(/\.[^.]+$/, '');
    statusEl.textContent = `${name} | ${info.full_width}x${info.full_height} | Preview scale: ${previewScale.toFixed(3)}`;
    document.getElementById('btn-prev').disabled = imageIdx <= 0;
    document.getElementById('btn-next').disabled = imageIdx >= imageCount - 1;
}

fetch('/info').then(r => r.json()).then(updateImageInfo);

async function switchImage(idx) {
    overlay.style.display = 'flex';
    selections = []; activeIdx = -1; updateSelList();
    const resp = await fetch('/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ idx }),
    });
    const info = await resp.json();
    updateImageInfo(info);
    // Reload preview (cache-bust)
    img.src = '/preview?' + Date.now();
    // img.onload will call fitImage + render, then hide overlay
}

const origOnload = img.onload;
img.onload = () => {
    imgW = img.width; imgH = img.height;
    fitImage();
    render();
    overlay.style.display = 'none';
};

function fitImage() {
    const ww = wrap.clientWidth, wh = wrap.clientHeight;
    scale = Math.min(ww / imgW, wh / imgH) * 0.95;
    offsetX = (ww - imgW * scale) / 2;
    offsetY = (wh - imgH * scale) / 2;
}

// --- Coordinate transforms ---
function imgToScreen(x, y) {
    return [x * scale + offsetX, y * scale + offsetY];
}
function screenToImg(sx, sy) {
    return [(sx - offsetX) / scale, (sy - offsetY) / scale];
}

// --- Aspect ratio ---
function getAspect() {
    const v = document.getElementById('aspect').value;
    if (v === 'free') return null;
    const [a, b] = v.split(':').map(Number);
    return a / b;
}

// --- Selection model ---
let lastAngle = 0, lastW = 0, lastH = 0;
function addSelection(x, y, w, h, angle = lastAngle) {
    selections.push({ x, y, w, h, angle });
    activeIdx = selections.length - 1;
    updateSelList();
}

function updateSelList() {
    selList.innerHTML = '';
    selections.forEach((s, i) => {
        const div = document.createElement('div');
        div.className = 'sel-item' + (i === activeIdx ? ' active' : '');
        const fw = (s.w / previewScale)|0, fh = (s.h / previewScale)|0;
        div.innerHTML = `<span class="del" data-i="${i}">&times;</span>#${i + 1}: ${fw}&times;${fh}`;
        div.addEventListener('click', (e) => {
            if (e.target.classList.contains('del')) {
                selections.splice(i, 1);
                if (activeIdx >= selections.length) activeIdx = selections.length - 1;
                updateSelList(); render();
            } else {
                activeIdx = i; updateSelList(); render();
            }
        });
        selList.appendChild(div);
    });
}

// --- Hit testing ---
function getHandles(s) {
    const cos = Math.cos(s.angle), sin = Math.sin(s.angle);
    const hw = s.w / 2, hh = s.h / 2;
    const cx = s.x + hw, cy = s.y + hh;
    function rot(lx, ly) {
        return [cx + lx * cos - ly * sin, cy + lx * sin + ly * cos];
    }
    return {
        tl: rot(-hw, -hh), tr: rot(hw, -hh),
        bl: rot(-hw, hh), br: rot(hw, hh),
        tm: rot(0, -hh), bm: rot(0, hh),
        ml: rot(-hw, 0), mr: rot(hw, 0),
        rot_t: rot(0, -hh - ROT_HANDLE_DIST / scale),
        rot_b: rot(0, hh + ROT_HANDLE_DIST / scale),
        rot_l: rot(-hw - ROT_HANDLE_DIST / scale, 0),
        rot_r: rot(hw + ROT_HANDLE_DIST / scale, 0),
        center: [cx, cy],
    };
}

function hitTest(mx, my) {
    const threshold = HANDLE_SIZE / scale;
    for (let i = selections.length - 1; i >= 0; i--) {
        const s = selections[i];
        const h = getHandles(s);
        // Rotation handles (all 4 sides)
        for (const rk of ['rot_t', 'rot_b', 'rot_l', 'rot_r']) {
            const [rx, ry] = h[rk];
            if (Math.hypot(mx - rx, my - ry) < threshold * 1.5) {
                return { type: 'rotate', idx: i };
            }
        }
        // Corner/edge handles
        for (const name of ['tl', 'tr', 'bl', 'br', 'tm', 'bm', 'ml', 'mr']) {
            const [px, py] = h[name];
            if (Math.hypot(mx - px, my - py) < threshold) {
                return { type: 'resize', idx: i, handle: name };
            }
        }
        // Inside test (unrotated space)
        const cos = Math.cos(-s.angle), sin = Math.sin(-s.angle);
        const cx = s.x + s.w / 2, cy = s.y + s.h / 2;
        const dx = mx - cx, dy = my - cy;
        const lx = dx * cos - dy * sin, ly = dx * sin + dy * cos;
        if (Math.abs(lx) < s.w / 2 && Math.abs(ly) < s.h / 2) {
            return { type: 'move', idx: i };
        }
    }
    return null;
}

// --- Rendering ---
function render() {
    const ww = wrap.clientWidth, wh = wrap.clientHeight;
    canvas.width = ww; canvas.height = wh;
    ctx.clearRect(0, 0, ww, wh);

    // Draw image
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);
    ctx.restore();

    // Draw selections
    selections.forEach((s, i) => {
        const isActive = i === activeIdx;
        const cx = s.x + s.w / 2, cy = s.y + s.h / 2;
        const [scx, scy] = imgToScreen(cx, cy);

        ctx.save();
        ctx.translate(scx, scy);
        ctx.rotate(s.angle);

        // Rectangle
        const sw = s.w * scale, sh = s.h * scale;
        ctx.strokeStyle = isActive ? '#2d6' : '#fff';
        ctx.lineWidth = isActive ? 2 : 1;
        ctx.strokeRect(-sw / 2, -sh / 2, sw, sh);

        // Semi-transparent fill
        ctx.fillStyle = isActive ? 'rgba(34,221,102,0.08)' : 'rgba(255,255,255,0.05)';
        ctx.fillRect(-sw / 2, -sh / 2, sw, sh);

        if (isActive) {
            // Handles
            ctx.fillStyle = '#2d6';
            const hs = HANDLE_SIZE;
            for (const [hx, hy] of [
                [-sw/2, -sh/2], [sw/2, -sh/2], [-sw/2, sh/2], [sw/2, sh/2],
                [0, -sh/2], [0, sh/2], [-sw/2, 0], [sw/2, 0],
            ]) {
                ctx.fillRect(hx - hs/2, hy - hs/2, hs, hs);
            }
            // Rotation handles (all 4 sides)
            const rotPositions = [
                [0, -sh/2, 0, -sh/2 - ROT_HANDLE_DIST],
                [0, sh/2, 0, sh/2 + ROT_HANDLE_DIST],
                [-sw/2, 0, -sw/2 - ROT_HANDLE_DIST, 0],
                [sw/2, 0, sw/2 + ROT_HANDLE_DIST, 0],
            ];
            for (const [x1, y1, x2, y2] of rotPositions) {
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.strokeStyle = '#2d6';
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(x2, y2, hs * 0.7, 0, Math.PI * 2);
                ctx.fillStyle = '#2d6';
                ctx.fill();
            }

            // Label
            ctx.fillStyle = '#2d6';
            ctx.font = '12px system-ui';
            ctx.textAlign = 'center';
            const fw = (s.w / previewScale)|0, fh = (s.h / previewScale)|0;
            ctx.fillText(`#${i+1}: ${fw}×${fh}`, 0, -sh/2 - ROT_HANDLE_DIST - 14);
        }

        ctx.restore();
    });
}

// --- Mouse handling ---
canvas.addEventListener('mousedown', (e) => {
    const [mx, my] = screenToImg(e.offsetX, e.offsetY);
    const hit = hitTest(mx, my);

    if (hit) {
        activeIdx = hit.idx;
        const s = selections[hit.idx];
        if (hit.type === 'move') {
            mode = 'moving';
            dragInfo = { startMx: mx, startMy: my, origX: s.x, origY: s.y };
        } else if (hit.type === 'resize') {
            mode = 'resizing';
            dragInfo = { handle: hit.handle, startMx: mx, startMy: my,
                         origX: s.x, origY: s.y, origW: s.w, origH: s.h };
        } else if (hit.type === 'rotate') {
            mode = 'rotating';
            const cx = s.x + s.w / 2, cy = s.y + s.h / 2;
            dragInfo = { cx, cy, startAngle: Math.atan2(my - cy, mx - cx), origAngle: s.angle };
        }
        updateSelList(); render();
    } else {
        // Start drawing new selection
        mode = 'drawing';
        drawStart = { x: mx, y: my };
    }
});

canvas.addEventListener('mousemove', (e) => {
    const [mx, my] = screenToImg(e.offsetX, e.offsetY);

    if (mode === 'drawing' && drawStart) {
        let w = mx - drawStart.x, h = my - drawStart.y;
        const aspect = getAspect();
        if (aspect) {
            const sign = Math.sign(h) || 1;
            h = sign * Math.abs(w) / aspect;
        }
        // Update or create temp selection
        if (activeIdx >= 0 && selections[activeIdx]._temp) {
            const s = selections[activeIdx];
            s.x = w >= 0 ? drawStart.x : drawStart.x + w;
            s.y = h >= 0 ? drawStart.y : drawStart.y + h;
            s.w = Math.abs(w); s.h = Math.abs(h);
        } else {
            const sx = w >= 0 ? drawStart.x : drawStart.x + w;
            const sy = h >= 0 ? drawStart.y : drawStart.y + h;
            addSelection(sx, sy, Math.abs(w), Math.abs(h));
            selections[activeIdx]._temp = true;
        }
        render();
    } else if (mode === 'moving' && dragInfo) {
        const s = selections[activeIdx];
        s.x = dragInfo.origX + (mx - dragInfo.startMx);
        s.y = dragInfo.origY + (my - dragInfo.startMy);
        render(); updateSelList();
    } else if (mode === 'resizing' && dragInfo) {
        const s = selections[activeIdx];
        const cos = Math.cos(-s.angle), sin = Math.sin(-s.angle);
        const cx = dragInfo.origX + dragInfo.origW / 2;
        const cy = dragInfo.origY + dragInfo.origH / 2;
        const dx = mx - dragInfo.startMx, dy = my - dragInfo.startMy;
        // Rotate delta into local space
        const ldx = dx * cos - dy * sin;
        const ldy = dx * sin + dy * cos;
        const h = dragInfo.handle;
        let nw = dragInfo.origW, nh = dragInfo.origH;
        let nox = 0, noy = 0;
        if (h.includes('r')) { nw += ldx; }
        if (h.includes('l')) { nw -= ldx; nox = ldx; }
        if (h.includes('b')) { nh += ldy; }
        if (h.includes('t')) { nh -= ldy; noy = ldy; }
        // Enforce aspect ratio
        const aspect = getAspect();
        if (aspect) {
            if (h === 'ml' || h === 'mr') { nh = nw / aspect; }
            else if (h === 'tm' || h === 'bm') { nw = nh * aspect; }
            else { nh = nw / aspect; }
        }
        nw = Math.max(nw, 20); nh = Math.max(nh, 20);
        // Adjust position for the offset (rotate back)
        const rcos = Math.cos(s.angle), rsin = Math.sin(s.angle);
        s.x = dragInfo.origX + nox * rcos - noy * rsin;
        s.y = dragInfo.origY + nox * rsin + noy * rcos;
        s.w = nw; s.h = nh;
        render(); updateSelList();
    } else if (mode === 'rotating' && dragInfo) {
        const s = selections[activeIdx];
        const angle = Math.atan2(my - dragInfo.cy, mx - dragInfo.cx);
        s.angle = dragInfo.origAngle + (angle - dragInfo.startAngle);
        render();
    } else {
        // Update cursor
        const hit = hitTest(mx, my);
        if (hit) {
            if (hit.type === 'rotate') canvas.style.cursor = 'grab';
            else if (hit.type === 'move') canvas.style.cursor = 'move';
            else canvas.style.cursor = 'nwse-resize';
        } else {
            canvas.style.cursor = 'crosshair';
        }
    }
});

canvas.addEventListener('mouseup', () => {
    if (mode === 'drawing' && activeIdx >= 0 && selections[activeIdx]._temp) {
        const s = selections[activeIdx];
        delete s._temp;
        if (s.w < 10 || s.h < 10) {
            selections.splice(activeIdx, 1);
            activeIdx = selections.length - 1;
        }
        updateSelList();
    }
    if (activeIdx >= 0) {
        const s = selections[activeIdx];
        lastAngle = s.angle;
        lastW = s.w;
        lastH = s.h;
    }
    mode = 'idle'; dragInfo = null; drawStart = null;
    render();
});

// --- Zoom/pan with scroll ---
canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const mx = e.offsetX, my = e.offsetY;
    offsetX = mx - (mx - offsetX) * factor;
    offsetY = my - (my - offsetY) * factor;
    scale *= factor;
    render();
}, { passive: false });

// --- Middle mouse pan ---
let panning = false, panStart = null;
canvas.addEventListener('mousedown', (e) => {
    if (e.button === 1) { panning = true; panStart = { x: e.offsetX - offsetX, y: e.offsetY - offsetY }; e.preventDefault(); }
});
canvas.addEventListener('mousemove', (e) => {
    if (panning && panStart) { offsetX = e.offsetX - panStart.x; offsetY = e.offsetY - panStart.y; render(); }
});
canvas.addEventListener('mouseup', (e) => {
    if (e.button === 1) { panning = false; panStart = null; }
});

// --- Buttons ---
document.getElementById('btn-add').addEventListener('click', () => {
    let w, h;
    if (lastW > 0 && lastH > 0) {
        w = lastW; h = lastH;
    } else {
        const aspect = getAspect() || 1.5;
        h = imgH * 0.3;
        w = h * aspect;
    }
    // Place in center of current viewport
    const vcx = (canvas.width / 2 - offsetX) / scale;
    const vcy = (canvas.height / 2 - offsetY) / scale;
    addSelection(vcx - w / 2, vcy - h / 2, w, h);
    render();
});

document.getElementById('btn-prev').addEventListener('click', () => {
    if (imageIdx > 0) switchImage(imageIdx - 1);
});
document.getElementById('btn-next').addEventListener('click', () => {
    if (imageIdx < imageCount - 1) switchImage(imageIdx + 1);
});

document.getElementById('btn-clear').addEventListener('click', () => {
    selections = []; activeIdx = -1; updateSelList(); render();
});

document.getElementById('btn-export').addEventListener('click', async () => {
    if (selections.length === 0) { statusEl.textContent = 'No selections to export'; return; }
    const exportBtn = document.getElementById('btn-export');
    exportBtn.disabled = true;
    exportBtn.textContent = 'Exporting...';
    const basename = document.getElementById('basename').value || 'frame';
    statusEl.textContent = 'Starting export...';

    // Poll progress while export runs
    const pollId = setInterval(async () => {
        try {
            const pr = await fetch('/progress');
            const pj = await pr.json();
            if (pj.status) statusEl.textContent = pj.status;
        } catch {}
    }, 500);

    const rects = selections.map(s => ({
        cx: (s.x + s.w / 2) / previewScale,
        cy: (s.y + s.h / 2) / previewScale,
        w: s.w / previewScale,
        h: s.h / previewScale,
        angle: s.angle * 180 / Math.PI,
    }));
    try {
        const resp = await fetch('/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ basename, rects }),
        });
        const result = await resp.json();
        statusEl.textContent = result.message;
    } catch (e) {
        statusEl.textContent = 'Export failed: ' + e.message;
    }
    clearInterval(pollId);
    exportBtn.disabled = false;
    exportBtn.textContent = 'Export all';
});

// --- Resize handling ---
window.addEventListener('resize', () => { render(); });

// --- Keyboard ---
document.addEventListener('keydown', (e) => {
    if (e.key === 'Delete' || e.key === 'Backspace') {
        if (activeIdx >= 0 && document.activeElement.tagName !== 'INPUT') {
            selections.splice(activeIdx, 1);
            activeIdx = Math.min(activeIdx, selections.length - 1);
            updateSelList(); render();
        }
    }
});
</script>
</body>
</html>"""


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
