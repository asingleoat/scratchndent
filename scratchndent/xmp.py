"""darktable XMP sidecar parsing for negadoctor, sigmoid, and channel mixer."""

import base64
import struct
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from scratchndent.color import compute_cat16_matrix


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
          f"T={temperature:.0f}K -> adapting to D50")

    return compute_cat16_matrix(scene_x, scene_y)
