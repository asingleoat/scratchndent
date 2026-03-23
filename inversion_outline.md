I have the shape of the algorithm; I’m grabbing the Kodak/ACES primary references so the sketch lines up with how film people actually define the measurement space, not just “looks good” inversion.

Thought for 50s

Hand Claude this model:

Do not “invert RGB.” Treat the scan as a 3-channel measurement of film transmittance, convert to density, remove base+fog / orange mask as D-min, then apply a calibrated stock+scanner transform into a print-density-like space, and only after that do a rendering transform to display RGB. That is the closest thing to a mechanistic pipeline in Kodak/ACES land. Kodak explicitly says the orange mask lives in D-min and compensates unwanted dye absorptions, Kodak publishes spectral dye-density curves as useful for optimizing scanners, and ACES formalizes a print-density target with APD/ADX.

A good mental split is:

measurement model: raw sensor/scanner values → transmittance → density
film model: subtract D-min, then map scanner-density space to stock/print-density space
rendering model: print-density-like values → pleasing display RGB

Kodak’s Cineon material and later APD/ADX docs are the clean formal references here, though they are published mainly for motion-picture negatives rather than consumer still C-41 docs. Kodak also notes that extended-latitude negatives can clip in limited-range scans, which is why you want raw linear capture and lots of headroom.

Here is the high-level numpy-style sketch.

import numpy as np

EPS = 1e-8

def percentile_img(img, q):
    # q can be scalar or per-channel tuple/list
    arr = img.reshape(-1, img.shape[-1])
    return np.percentile(arr, q, axis=0)

def safe_log10(x):
    return np.log10(np.clip(x, EPS, None))

def normalize_transmittance(raw_rgb, dark_rgb, light_rgb):
    """
    raw_rgb:   HxWx3 raw linear scan of the negative
    dark_rgb:  1x1x3 or HxWx3 dark frame / black level
    light_rgb: 1x1x3 or HxWx3 blank light field through film holder
               but with no film image content; ideally same optical path
    Returns per-channel transmittance in [0, 1+] before clipping.
    """
    num = raw_rgb - dark_rgb
    den = light_rgb - dark_rgb
    T = num / np.clip(den, EPS, None)
    return np.clip(T, EPS, 1.5)

def transmittance_to_density(T):
    """
    Optical density D = -log10(T)
    """
    return -safe_log10(T)

def estimate_dmin_from_rebate(density_rgb, rebate_mask=None):
    """
    D-min = film base + fog + mask.
    Best is an explicit mask over unexposed rebate.
    Fallback is a low-percentile estimate, but explicit rebate is better.
    """
    if rebate_mask is not None:
        px = density_rgb[rebate_mask]
        return np.median(px, axis=0)
    return percentile_img(density_rgb, 1.0)

def subtract_dmin(density_rgb, dmin_rgb):
    """
    Net image density after removing film base/orange mask baseline.
    """
    net = density_rgb - dmin_rgb[None, None, :]
    return np.clip(net, 0.0, None)

def exposure_anchor(net_density_rgb, neutral_mask=None):
    """
    Optional global exposure normalization.
    For a target-based calibration, use a known neutral patch.
    Otherwise use a heuristic scene anchor.
    """
    if neutral_mask is not None:
        anchor = np.median(net_density_rgb[neutral_mask], axis=0)
    else:
        anchor = percentile_img(net_density_rgb, 50.0)
    return anchor

def apply_density_normalization(net_density_rgb, anchor_rgb, target_anchor_rgb):
    """
    Optional affine normalization before the stock/scanner transform.
    """
    delta = target_anchor_rgb - anchor_rgb
    return net_density_rgb + delta[None, None, :]

def poly_features(x):
    """
    x: (..., 3)
    Second-order polynomial basis often works much better than a pure 3x3.
    """
    r = x[..., 0:1]
    g = x[..., 1:2]
    b = x[..., 2:3]
    return np.concatenate([
        r, g, b,
        r*r, g*g, b*b,
        r*g, r*b, g*b,
        np.ones_like(r),
    ], axis=-1)

def scanner_density_to_apd_like(net_density_rgb, coeffs):
    """
    Learned stock+scanner transform.
    coeffs: shape [10, 3] for quadratic basis -> 3 output channels
    Output is "APD-like" or print-density-like space, not display RGB yet.
    """
    feats = poly_features(net_density_rgb)
    return feats @ coeffs

def apd_like_to_linear_rgb(apd_rgb, render_lut=None, matrix=None):
    """
    Rendering stage: map print-density-like values to display-linear RGB.
    This is not the inversion itself; it is the output rendering.
    """
    x = apd_rgb
    if render_lut is not None:
        x = render_lut(x)   # placeholder for 3D LUT / CLF / OCIO style transform
    if matrix is not None:
        x = x @ matrix.T
    return np.clip(x, 0.0, None)

def tone_map(rgb_linear, gamma=2.2):
    """
    Placeholder display rendering.
    """
    return np.power(np.clip(rgb_linear, 0.0, 1.0), 1.0 / gamma)

def invert_negative(
    raw_rgb,
    dark_rgb,
    light_rgb,
    stock_scanner_coeffs,
    rebate_mask=None,
    neutral_mask=None,
    target_anchor_rgb=None,
    render_lut=None,
    render_matrix=None,
):
    # 1) measurement
    T = normalize_transmittance(raw_rgb, dark_rgb, light_rgb)
    D = transmittance_to_density(T)

    # 2) remove D-min / orange mask baseline
    dmin = estimate_dmin_from_rebate(D, rebate_mask=rebate_mask)
    netD = subtract_dmin(D, dmin)

    # 3) optional exposure normalization
    if target_anchor_rgb is not None:
        anchor = exposure_anchor(netD, neutral_mask=neutral_mask)
        netD = apply_density_normalization(netD, anchor, target_anchor_rgb)

    # 4) calibrated stock+scanner transform into print-density-like space
    apd_like = scanner_density_to_apd_like(netD, stock_scanner_coeffs)

    # 5) rendering to display RGB
    rgb_linear = apd_like_to_linear_rgb(
        apd_like,
        render_lut=render_lut,
        matrix=render_matrix,
    )
    rgb_display = tone_map(rgb_linear)

    return {
        "transmittance": T,
        "density": D,
        "dmin": dmin,
        "net_density": netD,
        "apd_like": apd_like,
        "rgb_linear": rgb_linear,
        "rgb_display": rgb_display,
    }

The calibration target for scanner_density_to_apd_like(...) is the important part. A pure 3x3 is usually too weak because the scanner channels do not line up with the film dye absorptions; Kodak’s own docs are basically telling you that the dye spectra matter, and APD exists because “film density” needs a defined spectral meaning. So in practice you fit a quadratic polynomial, 3D LUT, or small monotone neural net per stock+scanner+illuminant combination.

The minimum viable “principled” calibration loop is:

def fit_stock_scanner_transform(
    measured_net_density_rgb,   # Nx3 from your scanner/camera scan
    target_apd_like_rgb,        # Nx3 target in print-density-like space
):
    X = poly_features(measured_net_density_rgb)   # Nx10
    Y = target_apd_like_rgb                       # Nx3

    # ordinary least squares; swap for ridge if needed
    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coeffs

Where target_apd_like_rgb comes from one of these, in descending order of rigor:

Best: a calibration negative on the exact stock, exposed from known spectra / color patches, with a known target print-density or colorimetric reference.
Good: photograph a ColorChecker on that stock under a controlled illuminant, then fit scanner-density → scene/display target jointly. This is less pure, because it folds rendering into inversion, but it works.
Most mechanistic: use measured scanner spectral sensitivities + illuminant SPD + Kodak spectral dye-density curves to synthesize the forward model and solve the inverse numerically. Kodak publishes the dye-density curves because they matter for scanning/printing optimization.

If Claude is going to build this, I’d tell it to separate the code into these modules:

film_measurement.py
    raw_to_transmittance()
    transmittance_to_density()
    estimate_dmin()

film_calibration.py
    extract_patch_means()
    fit_density_transform()
    save/load calibration

film_inversion.py
    apply_density_transform()
    optional exposure anchor
    neutral balance in density domain

film_render.py
    print-density-like -> display-linear
    tone mapping / gamut mapping

A few implementation notes matter:

Use raw linear capture. No scanner auto inversion, no auto white balance, no hidden tone curves. Kodak’s scanning docs explicitly warn that insufficient encoding range loses highlight density information.
Measure the light field through the same optical path. Flat-fielding is part of the measurement model.
Use rebate when available. That is the cleanest estimate of D-min / orange mask baseline.
Do neutral balancing in density space, not after RGB inversion.
Keep inversion and rendering separate. “Make it positive” and “make it look nice” are different problems.

The physically strongest forward model is roughly:

𝑚
𝑐
=
∫
𝑆
𝑐
(
𝜆
)
 
𝐸
(
𝜆
)
 
10
−
𝐷
(
𝜆
)
 
𝑑
𝜆
m
c
	​

=∫S
c
	​

(λ)E(λ)10
−D(λ)
dλ

where 
𝑚
𝑐
m
c
	​

 is scanner channel response, 
𝑆
𝑐
(
𝜆
)
S
c
	​

(λ) is scanner spectral sensitivity, 
𝐸
(
𝜆
)
E(λ) is illuminant SPD, and 
𝐷
(
𝜆
)
D(λ) is film spectral density from the cyan/magenta/yellow dyes plus D-min. That is the real reason a generic channelwise invert is wrong. Kodak’s dye-density curves and the APD definition both point straight at this spectral view.

My recommendation for a first version is not the full spectral solver. Build this in two stages:

v1:
raw linear scan
-> flat field / dark subtraction
-> transmittance
-> density
-> rebate D-min subtraction
-> quadratic stock+scanner calibration
-> render LUT

v2:
replace quadratic calibration with spectral forward model + fitted inverse

That gets you something grounded enough to be defensible, without disappearing into a thesis project.

I can also give you a concrete patch-based calibration procedure next, including how to shoot the target and how to fit the coefficients.

---

## Implementation status (as of 2026-03-21)

### Done

- Transmittance -> density conversion (film_measurement.py)
- Dmin estimation from user-selected rebate area, persisted across images/sessions (film_measurement.py + UI)
- Dmin subtraction (film_measurement.py)
- Polynomial stock+scanner transform with per-stock default coefficients (film_calibration.py)
- Profile save/load as JSON (film_calibration.py)
- Rendering pipeline separate from inversion (film_render.py)
- Module structure matching the outline's recommendation (film_measurement, film_calibration, film_inversion, film_render)

### Not yet implemented

1. **Dark frame / light field calibration** — the outline calls for explicit dark_rgb (scanner black level) and light_rgb (blank light field, no film) for proper flat-fielding. We default to 0 and 65535. A UI to capture or specify these would improve accuracy, especially for scanners with uneven illumination.

2. **Actual calibration fitting from a target** — fit_density_transform() exists in film_calibration.py but has never been used with real data. The outline describes photographing a ColorChecker on each stock and fitting the polynomial from measured density -> known target values. Our current coefficients are hand-tuned estimates derived from density statistics of real scans, not from ground-truth calibration targets.

3. **extract_patch_means()** — the outline mentions a function to extract color patch averages from a calibration target scan. Not built. Needed for #2.

4. **Render LUT / 3D LUT / OCIO support** — the outline's apd_like_to_linear_rgb includes a render_lut parameter for a CLF/OCIO-style lookup table. We use a simple logistic S-curve for contrast instead. A proper render LUT would allow matching specific film print looks.

5. **Render matrix** — the outline includes a matrix parameter in the render stage for a final color space conversion matrix (e.g. to a specific display profile). Not implemented.

6. **Exposure anchor with neutral mask** — we implemented neutral balance but removed it because the grey-world fallback (scene median) was destructive to color without a real neutral reference. The outline envisions a UI where the user selects a known grey patch. Could be re-added properly with a "select neutral" tool similar to the rebate selection tool.

7. **v2: spectral forward model** — replacing the polynomial calibration with a physics-based model using Kodak's published spectral dye-density curves + scanner spectral sensitivities + illuminant SPD to synthesize the forward model mc = integral(Sc(lambda) * E(lambda) * 10^-D(lambda) dlambda) and solve the inverse numerically. This is the most rigorous approach but requires spectral data for each scanner/stock/illuminant combination.

### Priorities

The biggest practical improvement would be #2 (proper calibration fitting via ColorChecker) since it would replace the hand-tuned polynomial coefficients with data-driven ones. #1 (flat-fielding) would help with scanner uniformity. #6 (neutral selection) would allow per-scene white balance correction in density space. The rest are refinements.
