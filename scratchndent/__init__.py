"""scratchndent - IR-based dust/scratch removal and negative inversion for scanned film."""

from scratchndent.ir_clean import (
    load_tiff,
    align_ir,
    make_defect_mask,
    estimate_local_grain,
    synthesize_grain,
    inpaint,
)
from scratchndent.color import (
    MIDDLE_GREY,
    M_SRGB_TO_REC2020_D50,
    M_REC2020_D50_TO_SRGB,
    M_CAT16,
    M_CAT16_INV,
    D50_xy,
    compute_cat16_matrix,
    apply_color_matrix,
    srgb_to_linear,
    linear_to_srgb,
    negadoctor,
    apply_sigmoid,
    sigmoid_commit_params,
)
from scratchndent.xmp import (
    parse_negadoctor_params,
    extract_negadoctor_from_xmp,
    parse_sigmoid_params,
    extract_sigmoid_from_xmp,
    extract_channelmixer_from_xmp,
)
from scratchndent.cli import process, main
