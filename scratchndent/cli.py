"""Command-line interface for the scratchndent pipeline."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

from scratchndent.ir_clean import load_tiff, align_ir, make_defect_mask, inpaint
from scratchndent.color import (
    negadoctor, apply_color_matrix, apply_sigmoid,
    linear_to_srgb, M_REC2020_D50_TO_SRGB,
)
from scratchndent.xmp import (
    extract_negadoctor_from_xmp, extract_channelmixer_from_xmp,
    extract_sigmoid_from_xmp,
)


def process(
    input_path: str,
    output_path: str,
    *,
    align: bool = True,
    threshold: float = 0.25,
    hair_sensitivity: float = 0.10,
    min_area: int = 3,
    dilate_radius: int = 4,
    close_radius: int = 6,
    inpaint_padding: int = 16,
    save_mask: bool = False,
    invert_xmp: str | None = None,
) -> None:
    """Full pipeline: load, align, mask, inpaint, optionally invert negative."""
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
        print("No defects found.")
        result = rgb
    else:
        print(f"Inpainting {n_labels} regions (16-bit biharmonic)...")
        result = inpaint(rgb, mask, padding=inpaint_padding)

    if invert_xmp is not None:
        print(f"Converting negative to positive (negadoctor)...")
        nd_params = extract_negadoctor_from_xmp(invert_xmp)
        # negadoctor returns scene-referred linear float
        result_f = negadoctor(result, nd_params)

        cat_matrix = extract_channelmixer_from_xmp(invert_xmp)
        if cat_matrix is not None:
            print("Applying chromatic adaptation...")
            result_f = apply_color_matrix(result_f, cat_matrix)

        sig_params = extract_sigmoid_from_xmp(invert_xmp)
        if sig_params is not None:
            print("Applying sigmoid tone mapping...")
            result_f = apply_sigmoid(result_f, sig_params)

        # Convert Rec2020(D50) back to sRGB(D65) then apply sRGB gamma
        # (darktable's colorout module)
        result_f = apply_color_matrix(result_f, M_REC2020_D50_TO_SRGB)
        result_f = linear_to_srgb(np.clip(result_f, 0, None))
        result = np.clip(result_f * 65535.0, 0, 65535).astype(np.uint16)

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
    parser.add_argument(
        "--invert-xmp", type=str, default=None, metavar="XMP",
        help="Convert negative to positive using darktable-cli with the "
             "given XMP sidecar (contains negadoctor + processing settings)",
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
        invert_xmp=args.invert_xmp,
    )


if __name__ == "__main__":
    main()
