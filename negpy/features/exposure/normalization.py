import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R, ImageBuffer
from negpy.features.process.models import ProcessMode
from negpy.kernel.image.validation import ensure_image

if TYPE_CHECKING:
    from negpy.features.process.models import ProcessConfig

# Above this size the block-median is threaded over row strips (np.median frees the GIL).
_BLOCK_MEDIAN_PARALLEL_MIN_PIXELS = 2_000_000

# Contrast-mask spacer, per-cent of the analysis grid's short side. Of the grid, not of
# the render, so preview and export mask alike. Under the minimum the mask stops being
# unsharp and collapses into a plain (1-g) reduction, which is Grade.
MASK_SPACER_DEFAULT = 4.0
MASK_SPACER_MIN = 2.0
MASK_SPACER_MAX = 6.0


@njit(cache=True, fastmath=True)
def _normalize_log_image_jit(img_log: np.ndarray, floors: np.ndarray, ceils: np.ndarray) -> np.ndarray:
    """
    Log -> ~0.0-1.0 (Linear stretch, unclamped: out-of-bounds densities are
    rolled off by the downstream characteristic curve).
    Supports both f < c (Negative) and f > c (Positive) mapping.
    """
    h, w, c = img_log.shape
    res = np.empty_like(img_log)
    epsilon = 1e-6

    for y in range(h):
        for x in range(w):
            for ch in range(3):
                f = floors[ch]
                c_val = ceils[ch]
                delta = c_val - f

                denom = delta
                if abs(delta) < epsilon:
                    if delta >= 0:
                        denom = epsilon
                    else:
                        denom = -epsilon

                res[y, x, ch] = (img_log[y, x, ch] - f) / denom
    return res


class LogNegativeBounds:
    """
    D-min / D-max container.
    """

    def __init__(self, floors: Tuple[float, float, float], ceils: Tuple[float, float, float]):
        self.floors = floors
        self.ceils = ceils


def resolve_analysis_region(
    image_shape: tuple[int, ...],
    active_roi: Optional[tuple[int, int, int, int]],
    analysis_buffer: float,
    analysis_rect: Optional[tuple[float, float, float, float]],
) -> tuple[Optional[tuple[int, int, int, int]], float]:
    """Resolve the (roi, buffer) the meters should read.

    A freehand `analysis_rect` (normalized in the transformed image) wins over the crop
    ROI + centered buffer: it maps directly to a pixel ROI and disables the symmetric
    inset. Otherwise the crop `active_roi` and the `analysis_buffer` slider apply as before.
    """
    if analysis_rect is not None:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = analysis_rect
        roi = (
            int(min(y1, y2) * h),
            int(max(y1, y2) * h),
            int(min(x1, x2) * w),
            int(max(x1, x2) * w),
        )
        # Degenerate rect (zero area) is ignored so a stray click can't blank analysis.
        if roi[1] - roi[0] >= 2 and roi[3] - roi[2] >= 2:
            return roi, 0.0
    return active_roi, analysis_buffer


def get_analysis_crop(img: ImageBuffer, buffer_ratio: float) -> ImageBuffer:
    """
    Returns a center crop of the image for analysis purposes.
    The buffer_ratio (0.0 to 0.25) defines how much of the border to exclude.
    """
    if buffer_ratio <= 0:
        return img

    h, w = img.shape[:2]
    safe_buffer = min(max(buffer_ratio, 0.0), 0.3)

    cut_h = int(h * safe_buffer)
    cut_w = int(w * safe_buffer)

    return img[cut_h : h - cut_h, cut_w : w - cut_w]


def _block_median_grid(img_log: ImageBuffer) -> ImageBuffer:
    """
    Block-median prefilter to a fixed target grid: isolated extremes (speculars,
    dust pinholes) vanish inside their block's median, and statistics become nearly
    resolution-invariant since the grid size is constant.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    h, w = img_log.shape[:2]
    grid = int(EXPOSURE_CONSTANTS["analysis_grid"])
    b = int(np.ceil(max(h, w) / grid))
    if b <= 1 or h < b or w < b:
        return img_log

    hb, wb = (h // b) * b, (w // b) * b
    arr = img_log[:hb, :wb]
    grid_rows, c = hb // b, arr.shape[2]

    def _median(rows: np.ndarray) -> np.ndarray:
        v = rows.reshape(rows.shape[0] // b, b, wb // b, b, c)
        if b == 2:
            # Median of 4 = (sum - min - max) / 2; strided views, no partition.
            p00, p01, p10, p11 = v[:, 0, :, 0], v[:, 0, :, 1], v[:, 1, :, 0], v[:, 1, :, 1]
            s = p00.astype(np.float64) + p01 + p10 + p11
            mn = np.minimum(np.minimum(p00, p01), np.minimum(p10, p11))
            mx = np.maximum(np.maximum(p00, p01), np.maximum(p10, p11))
            return ((s - mn - mx) * 0.5).astype(rows.dtype, copy=False)
        return np.median(v, axis=(1, 3))

    workers = min(os.cpu_count() or 1, grid_rows)
    if workers < 2 or hb * wb < _BLOCK_MEDIAN_PARALLEL_MIN_PIXELS:
        return _median(arr)

    # Block-aligned strips -> per-cell median identical to the single pass.
    rows_per = -(-grid_rows // workers)
    strips = [arr[i * b : min(grid_rows, i + rows_per) * b] for i in range(0, grid_rows, rows_per)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_median, strips))
    return np.concatenate(parts, axis=0)


def prefilter_log_grid(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> ImageBuffer:
    """
    Shared meter prefilter (log10 -> crop -> block-median grid), computed once and
    fed to the *_from_log meters. Re-prefiltering it (roi=None, buffer=0) is a no-op
    (_block_median_grid early-returns when b<=1), so results stay bit-exact.
    """
    img_log = to_log_density(image)
    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)
    return _block_median_grid(img_log)


def measure_clip_fractions(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> tuple[float, float, float]:
    """
    Per-channel fraction of pixels at/above the sensor-white clip level (linear
    input). In a negative scan the film base and scene shadows sit near sensor
    white, so a clipped scan silently collapses distinct densities to D=0 —
    this feeds the scan-exposure warning, not any render math.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    img = image
    if roi:
        y1, y2, x1, x2 = roi
        img = img[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img = get_analysis_crop(img, analysis_buffer)
    # Stride-subsampled: a warning metric doesn't need every pixel.
    img = img[::4, ::4]
    level = float(EXPOSURE_CONSTANTS["scan_clip_level"])
    clipped = np.mean(img >= level, axis=(0, 1))
    return (float(clipped[0]), float(clipped[1]), float(clipped[2]))


def effective_crosstalk_matrix(process: "ProcessConfig", process_mode: Optional[str]) -> Optional[np.ndarray]:
    """
    Mode-aware resolve: the configured unmix only when the profile was derived for the
    film being processed, otherwise None.

    Mirrors effective_paper_profile. A crosstalk matrix describes a dye set's unwanted
    absorptions, and C-41 and E-6 do not share one — every bundled profile is a color
    negative stock, so without this gate a slide silently gets a negative's correction.
    Legacy configs carry no `crosstalk_process` and default to C-41, which is what every
    profile that predates the field actually is.

    Also composes in fade restoration, gated the same way on `fade_process`: a C-41 and
    an E-6 dye set do not share side absorptions either. Fade restores the *original*
    densities of the film the crosstalk unmix already recovered as *faded*, so it applies
    after: `fade @ unmix`. Neither gate implies the other, so either factor, both or
    neither can be None.

    When both are active for the same dye set, `fade_delta` is dropped (survival ratios
    still apply): a crosstalk profile and `fade_delta` both describe the same physical
    quantity -- a dye set's inherent side absorption -- so composing both would apply it
    twice. See `fade_delta_conflict_reason`.
    """
    from negpy.features.process.models import ProcessMode

    profile_mode = str(getattr(process, "crosstalk_process", ProcessMode.C41) or ProcessMode.C41)
    unmix = None
    if process_mode is None or profile_mode == str(process_mode):
        unmix = resolve_crosstalk_matrix(process.crosstalk_strength, process.crosstalk_matrix)

    fade_mode = str(getattr(process, "fade_process", ProcessMode.E6) or ProcessMode.E6)
    fade = None
    if process_mode is None or fade_mode == str(process_mode):
        delta = None if (unmix is not None and fade_delta_conflict_reason(process, process_mode)) else process.fade_delta
        fade = resolve_fade_matrix(process.fade_strength, process.fade_ratio_r, process.fade_ratio_g, process.fade_ratio_b, delta)

    if fade is None:
        return unmix
    return fade if unmix is None else fade @ unmix


def fade_delta_conflict_reason(process: "ProcessConfig", process_mode: Optional[str]) -> str:
    """Why `fade_delta` is being dropped from the composition in favor of the crosstalk
    unmix already active for this dye set, or "" when there is no conflict.

    The reason is a domain mismatch, not double-counting a correction. `resolve_fade_matrix`
    builds its inverse to act on *measured* densities -- the domain the raw scan is in, and
    the domain the fade profile's own delta was measured in (see fade/README.md's `bands`).
    But when a crosstalk unmix runs first, the data reaching the fade factor is no longer
    measured densities: the unmix has already moved it toward *dye concentrations*, and
    fading scales concentrations directly with no cross-channel mixing term at all. The
    correct fade operator in that domain is the plain diagonal `diag(1, 1/ag, 1/ab)` -- which
    is exactly what dropping delta produces (`resolve_fade_matrix(..., delta=None)` reduces
    to S = identity). Keeping delta would apply a measured-density operator to
    concentration-space data, a real, checked-numerically error rather than the identity's
    harmless no-op that the ratio_g == ratio_b == 1 case gives regardless. Survival ratios
    have no crosstalk equivalent and are never affected.
    """
    from negpy.features.process.models import ProcessMode

    profile_mode = str(getattr(process, "crosstalk_process", ProcessMode.C41) or ProcessMode.C41)
    crosstalk_active = (process_mode is None or profile_mode == str(process_mode)) and (
        resolve_crosstalk_matrix(process.crosstalk_strength, process.crosstalk_matrix) is not None
    )
    if not crosstalk_active:
        return ""
    fade_mode = str(getattr(process, "fade_process", ProcessMode.E6) or ProcessMode.E6)
    fade_mode_matches = process_mode is None or fade_mode == str(process_mode)
    delta_nonzero = process.fade_delta is not None and any(float(v) != 0.0 for v in process.fade_delta)
    if fade_mode_matches and delta_nonzero and float(process.fade_strength) > 0.0:
        return "a crosstalk profile is already active for this dye set — its side-absorption profile is ignored to avoid double-correcting (survival ratios still apply)"
    return ""


def resolve_crosstalk_matrix(strength: float, matrix: Optional[tuple]) -> Optional[np.ndarray]:
    """
    Effective spectral-crosstalk (dye-unmix) matrix — identity↔calibration blend
    by strength, row-normalized so neutral gray is preserved (rows redistribute
    channel differences only) — or None when off. Applied to raw NEGATIVE log
    densities before any metering/stretch; since the op is linear and
    img_log = -D, applying it to log values is exact.
    """
    if float(strength) <= 0.0:
        return None
    from negpy.features.process.models import DEFAULT_CROSSTALK_MATRIX

    m = matrix if matrix is not None else DEFAULT_CROSSTALK_MATRIX
    cal = np.array(m, dtype=np.float64).reshape(3, 3)
    applied = np.eye(3) * (1.0 - float(strength)) + cal * float(strength)
    row_sums = np.sum(applied, axis=1, keepdims=True)
    return applied / np.maximum(row_sums, 1e-6)


#: Above this condition number the restoration matrix is refused rather than inverted:
#: the fade parameters at that point are undetermined enough that inverting it
#: would amplify noise rather than recover signal.
FADE_CONDITION_LIMIT = 50.0


def resolve_fade_matrix(strength: float, ratio_r: float, ratio_g: float, ratio_b: float, delta: Optional[tuple]) -> Optional[np.ndarray]:
    """Restoration operator inv(F) for a faded dye set, or None when off.

    Deliberately not row-normalized, unlike resolve_crosstalk_matrix: fade changes each
    layer's neutral density and that difference is the entire signal. Neutral image content
    only ever constrains the *ratios* green/red and blue/red, never red's own absolute
    survival -- so `ratio_g`/`ratio_b` are estimable from a frame and `ratio_r` is not; it is
    a third, independent parameter, not derivable from the other two. `delta` (the six
    side-absorption ratios) is treated as zero when no profile supplies one, so the ratio
    sliders alone still give a real, if purely diagonal, correction. Strength scales all
    three survival ratios toward 1.0 (a less-faded film) before F is built, not the output --
    delta is a measurement property of the dye set and the scanner's bands and does not vary
    with fade extent, so it is never strength-scaled (see `_fade_forward_matrix`). Refuses
    (returns None) rather than inverting when F is singular or ill-conditioned for the
    requested parameters.
    """
    if float(strength) <= 0.0:
        return None
    f = _fade_forward_matrix(strength, ratio_r, ratio_g, ratio_b, delta)
    if f is None:
        return None
    det = np.linalg.det(f)
    if abs(det) < 1e-6 or np.linalg.cond(f) > FADE_CONDITION_LIMIT:
        return None
    return np.linalg.inv(f)


def _fade_forward_matrix(strength: float, ratio_r: float, ratio_g: float, ratio_b: float, delta: Optional[tuple]) -> Optional[np.ndarray]:
    """The forward fade operator, shared by resolve_fade_matrix and fade_reject_reason so
    the two cannot disagree on what F is.

    F = S @ diag(ar, ar*ag, ar*ab) @ inv(S), a similarity transform on measured densities:
    S is the dye set's side-absorption matrix (`delta`, fixed by the dye set and the
    scanner's bands -- never strength-scaled). `ag`/`ab` are green/red and blue/red
    survival *ratios* (the tooltip's "relative to red"), so the true, absolute green and
    blue survivals are `ar * ag` and `ar * ab`, not `ag` and `ab` on their own -- ratio
    times reference, not the ratio alone. A similarity transform is exactly the identity
    whenever ar == ag == ab == 1, for any S -- strength 0 is then a true no-op by
    construction, not by delta happening to scale to zero too, and composing a real fade
    with an already-active crosstalk unmix of the same dye set cannot double-apply delta,
    since an unfaded fade factor carries no net crosstalk correction to begin with.
    `ratio_r` is red's own absolute survival fraction, not a ratio to another channel
    (there is nothing to take a ratio against): pinning it at 1.0 asserts red never faded,
    which understates the correction on the dye layer E-6 fades fastest (cyan, read on the
    red channel) precisely when the feature matters most -- see resolve_fade_matrix.
    Returns None when S itself is not invertible (a degenerate hand-entered profile)
    rather than raising.
    """
    s = float(strength)
    ar = 1.0 + s * (float(ratio_r) - 1.0)
    ag = 1.0 + s * (float(ratio_g) - 1.0)
    ab = 1.0 + s * (float(ratio_b) - 1.0)
    s_matrix = _fade_side_absorption_matrix(delta)
    if s_matrix is None:
        return None
    return s_matrix @ np.diag([ar, ar * ag, ar * ab]) @ np.linalg.inv(s_matrix)


def _fade_side_absorption_matrix(delta: Optional[tuple]) -> Optional[np.ndarray]:
    """S, the dye set's side-absorption matrix (1 on the diagonal, `delta` off it) --
    the domain resolve_fade_matrix's similarity transform acts in: `F = S @ D @ inv(S)`
    on measured densities. Shared with `fade_side_absorption_unmix` so both agree on
    what S is. None when S itself is not invertible (a degenerate hand-entered profile)."""
    d = delta if delta is not None else (0.0,) * 6
    d_gr, d_br, d_rg, d_bg, d_rb, d_gb = (float(x) for x in d)
    s_matrix = np.array(
        [
            [1.0, d_gr, d_br],
            [d_rg, 1.0, d_bg],
            [d_rb, d_gb, 1.0],
        ],
        dtype=np.float64,
    )
    if abs(np.linalg.det(s_matrix)) < 1e-6:
        return None
    return s_matrix


def fade_side_absorption_unmix(delta: Optional[tuple]) -> Optional[np.ndarray]:
    """inv(S): recovers per-layer dye concentration from measured density, given a fade
    profile's own side-absorption `delta`. Not row-normalized, so it shifts overall
    channel brightness along with removing the mixing -- fine for matrix algebra
    (`test_dropping_delta_is_the_correct_operator...`), wrong for feeding a fixed-band
    luma detector like `measure_neutral_axis_from_log`, which needs `fade_measurement_unmix`
    instead. None when delta is absent or S is degenerate."""
    s_matrix = _fade_side_absorption_matrix(delta)
    if s_matrix is None:
        return None
    return np.linalg.inv(s_matrix)


def fade_measurement_unmix(delta: Optional[tuple]) -> Optional[tuple[np.ndarray, tuple[float, float, float]]]:
    """(neutral-preserving unmix, correction factors) for reading a survival-ratio
    estimate off `measure_neutral_axis_from_log`'s refs in concentration space.

    Row-normalizing inv(S) -- the same technique `resolve_crosstalk_matrix` uses --
    preserves neutral gray, so overall channel brightness doesn't move and the neutral-axis
    detector's fixed luma bands keep finding pixels; the raw `fade_side_absorption_unmix`
    shifts brightness enough that the detector returns nothing. Row-normalizing introduces
    a per-channel bias exactly equal to inv(S)'s own row sums (the second element here),
    which the caller must divide back out of a channel-to-red spread ratio -- ignoring it
    is itself a real, double-digit-percent error, not a rounding correction. None when
    delta is absent, S is degenerate, or a row sum is too close to zero to divide by."""
    s_matrix = _fade_side_absorption_matrix(delta)
    if s_matrix is None:
        return None
    s_inv = np.linalg.inv(s_matrix)
    row_sums = s_inv.sum(axis=1)
    if np.any(np.abs(row_sums) < 1e-6):
        return None
    normalized = s_inv / row_sums[:, None]
    return normalized, (float(row_sums[0]), float(row_sums[1]), float(row_sums[2]))


def fade_reject_reason(strength: float, ratio_r: float, ratio_g: float, ratio_b: float, delta: Optional[tuple]) -> str:
    """Why resolve_fade_matrix declined to build a restoration operator for these
    parameters, or "" when it didn't. Strength <= 0 is not reported: that is the
    ordinary off state, not a rejection."""
    if float(strength) <= 0.0:
        return ""
    f = _fade_forward_matrix(strength, ratio_r, ratio_g, ratio_b, delta)
    if f is None:
        return "the dye-set side-absorption profile is degenerate (its own matrix is singular) -- check the delta values"
    det = np.linalg.det(f)
    if abs(det) < 1e-6:
        return f"the restoration matrix is singular for this strength and profile (det≈{det:.2g})"
    cond = np.linalg.cond(f)
    if cond > FADE_CONDITION_LIMIT:
        return f"the restoration matrix is too ill-conditioned to invert safely (condition {cond:.0f}, limit {FADE_CONDITION_LIMIT:.0f})"
    return ""


def unmix_log_image(img_log: ImageBuffer, matrix: Optional[np.ndarray]) -> ImageBuffer:
    """Apply the unmix matrix to a (H, W, 3) log-density image; identity when None."""
    if matrix is None:
        return img_log
    return np.einsum("hwc,kc->hwk", img_log.astype(np.float32, copy=False), matrix.astype(np.float32))


def sorted_channel_grid(img_log: ImageBuffer) -> np.ndarray:
    """
    Per-channel sorted values of a prefiltered grid, shape (n, 3). The clip percentiles
    index into it, so a clip drag re-reads one sort instead of re-partitioning the grid.
    """
    return np.sort(img_log.reshape(-1, 3), axis=0)


def percentile_from_sorted(sorted_grid: np.ndarray, q: float) -> np.ndarray:
    """
    The three channels' `np.percentile(grid[:, :, ch], q)`, read off `sorted_channel_grid`.

    Bit-exact needs numpy's arithmetic, not its definition: virtual index `(n-1)*(q/100)`
    in float64, the far half interpolated back from the upper sample, and a weight that
    keeps `q`'s own type -- numpy interpolates in float32 for a Python `q` and float64 for
    an `np.float64` one, and both kinds of caller exist here.
    """
    n = sorted_grid.shape[0]
    virtual = (n - 1) * np.true_divide(q, 100)
    lo = min(max(int(np.floor(virtual)), 0), n - 1)
    hi = min(lo + 1, n - 1)
    t = virtual - lo
    if type(q) in (int, float):
        t = float(t)
    a, b = sorted_grid[lo], sorted_grid[hi]
    diff = b - a
    return b - diff * (1 - t) if t >= 0.5 else a + diff * t


def to_log_density(image: ImageBuffer) -> ImageBuffer:
    """
    Linear scan -> log10 density, the domain every meter reads.

    fmin/fmax rather than clip: they drop a NaN in favour of the bound, so one clamp covers
    the NaN and infinity fixup as well.
    """
    epsilon = 1e-6
    return np.log10(np.fmin(np.fmax(image, epsilon), 1.0))


def measure_shadow_refs_from_log(
    img_log: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    sorted_grid: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Per-channel shadow reference density: a high percentile of the prefiltered
    log image — the tones just inside print black (thin negative side for C-41).
    Channel differences here are the residual shadow cast that auto
    shadow-neutral cancels.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi or analysis_buffer > 0:
        sorted_grid = None
    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    p = float(EXPOSURE_CONSTANTS["shadow_neutral_percentile"])
    if sorted_grid is not None:
        refs = [float(v) for v in percentile_from_sorted(sorted_grid, p)]
    else:
        img_log = _block_median_grid(img_log)
        refs = [float(np.percentile(img_log[:, :, ch], p)) for ch in range(3)]
    return (refs[0], refs[1], refs[2])


def measure_shadow_log_refs(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Linear-image wrapper around measure_shadow_refs_from_log.
    """
    img_log = to_log_density(image)
    return measure_shadow_refs_from_log(img_log, roi, analysis_buffer)


def _rms_chroma(triplets: np.ndarray) -> np.ndarray:
    """Distance from the neutral axis in normalized-density space (pairwise RMS):
    rotation-symmetric around grey, so the near-neutral ranking is hue-uniform
    (max-min scores an opposed R/B split double a same-side deviation)."""
    r, g, b = triplets[..., 0], triplets[..., 1], triplets[..., 2]
    return np.sqrt(((r - g) ** 2 + (g - b) ** 2 + (r - b) ** 2) / 3.0)


def measure_neutral_axis_from_log(
    img_log: ImageBuffer,
    bounds: "LogNegativeBounds",
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float]], float]]:
    """
    Per-channel neutral axis: median raw-log density at a highlight, a midtone and a shadow
    luma band, over each band's lowest-chroma pixels. Two passes: pass 1 selects through the
    residual cast under a loose cap (a strong but correctable cast must not collapse the axis;
    saturated content still fails it), then the affine R/B->G correction implied by its
    mid+shadow refs re-ranks chroma so pass 2 selects true neutrals under the strict cap.
    Returns (midtone, shadow, highlight, confidence) — highlight is None when that band has no
    trustworthy neutral set (callers then fit a 2-point line); confidence in [0,1] combines the
    grey sets' corrected tightness, the midtone sample size and mid<->shadow deviation agreement
    (drives Auto Cast Removal). None overall when midtone or shadow is missing (shadow tie).
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)
    norm = normalize_log_image(img_log, bounds)
    luma = LUMA_R * norm[:, :, 0] + LUMA_G * norm[:, :, 1] + LUMA_B * norm[:, :, 2]

    flat_log = img_log.reshape(-1, 3)
    norm_f = norm.reshape(-1, 3)
    luma_f = luma.reshape(-1)
    chroma_f = _rms_chroma(norm_f)

    c = EXPOSURE_CONSTANTS
    q = float(c["neutral_axis_chroma_quantile"])
    cap = float(c["neutral_axis_chroma_cap"])
    pass1_cap = float(c["neutral_axis_first_pass_cap"])
    min_px = int(c["neutral_axis_min_pixels"])
    epsilon = 1e-6

    def _band_refs(
        lo: float, hi: float, chroma_vals: np.ndarray, cap_val: float
    ) -> Optional[Tuple[Tuple[float, float, float], float, int]]:
        band = (luma_f >= lo) & (luma_f <= hi)
        if int(band.sum()) < min_px:
            return None
        band_chroma = chroma_vals[band]
        thr = float(np.quantile(band_chroma, q))
        keep = band_chroma <= thr
        idx = np.nonzero(band)[0][keep]
        near_neutral_chroma = float(np.median(band_chroma[keep])) if idx.size else cap_val
        if idx.size < min_px or near_neutral_chroma > cap_val:
            return None
        # One gather for all three channels: the per-channel fancy index is the cost here.
        sel = flat_log[idx]
        refs = (float(np.median(sel[:, 0])), float(np.median(sel[:, 1])), float(np.median(sel[:, 2])))
        return (refs, near_neutral_chroma, int(idx.size))

    def _norm_ref(refs: Tuple[float, float, float]) -> Tuple[float, float, float]:
        out = []
        for ch in range(3):
            denom = bounds.ceils[ch] - bounds.floors[ch]
            if abs(denom) < epsilon:
                denom = epsilon if denom >= 0 else -epsilon
            out.append((refs[ch] - bounds.floors[ch]) / denom)
        return (out[0], out[1], out[2])

    hb = c["neutral_axis_highlight_band"]
    mb = c["neutral_axis_mid_band"]
    sb = c["neutral_axis_shadow_band"]
    mid1 = _band_refs(float(mb[0]), float(mb[1]), chroma_f, pass1_cap)
    sh1 = _band_refs(float(sb[0]), float(sb[1]), chroma_f, pass1_cap)
    if mid1 is None or sh1 is None:
        return None

    nm, ns = _norm_ref(mid1[0]), _norm_ref(sh1[0])
    corrected = norm_f.copy()
    for ch in (0, 2):
        du = nm[ch] - ns[ch]
        if abs(du) < epsilon:
            a, b = 1.0, nm[1] - nm[ch]
        else:
            a = (nm[1] - ns[1]) / du
            b = nm[1] - a * nm[ch]
        corrected[:, ch] = a * norm_f[:, ch] + b
    chroma2_f = _rms_chroma(corrected)

    mid = _band_refs(float(mb[0]), float(mb[1]), chroma2_f, cap)
    shadow = _band_refs(float(sb[0]), float(sb[1]), chroma2_f, cap)
    if mid is None or shadow is None:
        return None
    highlight = _band_refs(float(hb[0]), float(hb[1]), chroma2_f, cap)

    # Confidence: the corrected tightness of the grey sets, times the midtone sample size,
    # times the mid/shadow deviation agreement, where a dead zone passes plausible crossover.
    n0 = float(c["neutral_axis_confidence_n0"])
    dead = float(c["neutral_axis_agreement_deadzone"])
    scale = float(c["neutral_axis_agreement_scale"])
    tight = float(np.clip(1.0 - max(mid[1], shadow[1]) / cap, 0.0, 1.0))
    size_term = mid[2] / (mid[2] + n0)
    dm, ds = _norm_ref(mid[0]), _norm_ref(shadow[0])
    spread = max(abs((dm[ch] - dm[1]) - (ds[ch] - ds[1])) for ch in (0, 2))
    agree = 1.0 - min(max(spread - dead, 0.0) / scale, 1.0)
    confidence = float(np.clip(tight * size_term * agree, 0.0, 1.0))
    return (mid[0], shadow[0], highlight[0] if highlight is not None else None, confidence)


def measure_neutral_axis(
    image: ImageBuffer,
    bounds: "LogNegativeBounds",
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float]], float]]:
    """Linear-image wrapper around measure_neutral_axis_from_log."""
    img_log = to_log_density(image)
    return measure_neutral_axis_from_log(img_log, bounds, roi, analysis_buffer)


def luminance_density_range(bounds: LogNegativeBounds) -> float:
    """
    Single global density range as a Rec.709 luminance weighting of the
    per-channel ranges. Replaces the green-only range so frames with a strong
    single-channel cast don't swing the slope as hard, while green still
    dominates so calibrated grade behaviour barely shifts. abs() keeps it
    sign-safe for E6's reversed (f > c) bounds.
    """
    rr = abs(bounds.ceils[0] - bounds.floors[0])
    rg = abs(bounds.ceils[1] - bounds.floors[1])
    rb = abs(bounds.ceils[2] - bounds.floors[2])
    return float(LUMA_R * rr + LUMA_G * rg + LUMA_B * rb)


def measure_anchor_from_log(
    img_log: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Per-frame exposure anchor: where this negative's midtone sits in [0, 1],
    replacing the fixed assumed_anchor. Block-median prefiltered (speculars/dust
    rejected).

    Partial metering: the anchor moves only anchor_meter_strength of the way from
    assumed_anchor toward the metered median, so a deliberately low-key (dark) or
    high-key (bright) scene keeps most of its intended key instead of being
    forced to mid-gray, while gross mis-exposure is still pulled toward correct.
    A linear pull (no key-dependent amplification) keeps it predictable. Finally
    clamped to assumed_anchor +/- anchor_meter_band as a hard safety guard.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    epsilon = 1e-6
    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)

    norm = np.empty_like(img_log)
    for ch in range(3):
        f = bounds.floors[ch]
        denom = bounds.ceils[ch] - f
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        norm[:, :, ch] = (img_log[:, :, ch] - f) / denom

    lum = LUMA_R * norm[:, :, 0] + LUMA_G * norm[:, :, 1] + LUMA_B * norm[:, :, 2]
    p = float(EXPOSURE_CONSTANTS["anchor_meter_percentile"])
    measured = float(np.percentile(lum, p))

    assumed = float(EXPOSURE_CONSTANTS["assumed_anchor"])
    strength = float(EXPOSURE_CONSTANTS["anchor_meter_strength"])
    band = float(EXPOSURE_CONSTANTS["anchor_meter_band"])
    anchor = assumed + strength * (measured - assumed)
    return float(min(max(anchor, assumed - band), assumed + band))


def measure_anchor(
    image: ImageBuffer,
    bounds: LogNegativeBounds,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Linear-image wrapper around measure_anchor_from_log.
    """
    img_log = to_log_density(image)
    return measure_anchor_from_log(img_log, bounds, roi, analysis_buffer)


def measure_textural_range_from_log(
    img_log: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Per-frame textural density range: the P10-P90 luminance spread of the
    prefiltered log image, in log10-density units. This is the *useful* scene
    range that grade selection fits to paper — block-median prefiltering and the
    inner percentiles reject speculars / film-base / dust, so it is far more
    outlier-robust than the floor-to-ceil extreme range.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]
    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)

    lum = LUMA_R * img_log[:, :, 0] + LUMA_G * img_log[:, :, 1] + LUMA_B * img_log[:, :, 2]
    clip = float(EXPOSURE_CONSTANTS["textural_range_clip"])
    lo, hi = np.percentile(lum, [clip, 100.0 - clip])
    return float(abs(hi - lo))


def measure_textural_range(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
) -> float:
    """
    Linear-image wrapper around measure_textural_range_from_log.
    """
    img_log = to_log_density(image)
    return measure_textural_range_from_log(img_log, roi, analysis_buffer)


def normalized_roi(roi: Optional[Tuple[int, int, int, int]], shape: Tuple[int, int]) -> Optional[Tuple[float, float, float, float]]:
    """A pixel ROI as fractions of `shape`, replayable on any downsampled copy. None
    and a full-frame ROI both stay None."""
    if roi is None:
        return None
    h, w = shape
    y1, y2, x1, x2 = roi
    if (y1, x1, y2, x2) == (0, 0, h, w):
        return None
    return (y1 / float(h), y2 / float(h), x1 / float(w), x2 / float(w))


def contrast_mask_plane(
    image: ImageBuffer,
    bounds: LogNegativeBounds,
    unmix: Optional[np.ndarray],
    rotation: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    distortion_k1: float = 0.0,
    converge_v: float = 0.0,
    converge_h: float = 0.0,
    roi_norm: Optional[Tuple[float, float, float, float]] = None,
    spacer: float = MASK_SPACER_DEFAULT,
) -> Tuple[np.ndarray, float]:
    """
    The blurred low-gamma plane an unsharp mask is built from, zero-mean, on the analysis
    grid, plus the val it was centred on. The gamma's sign picks the mask's polarity and
    so the direction; the centre is the val a flat area rotates about, which the Analysis
    chart needs to draw the mask's band.

    Takes the linear frame *before* geometry and replays it on the downsampled copy, so
    both engines call this on the same array. `roi_norm` is the printed frame as
    (y1, y2, x1, x2) fractions: a rebate or surround blurred in prints as a vignette.
    `spacer` is per-cent of the grid: the scale above which tones are masked.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS
    from negpy.features.geometry.logic import apply_fine_rotation, apply_keystone, apply_radial_distortion

    h, w = image.shape[:2]
    grid = int(EXPOSURE_CONSTANTS["analysis_grid"])
    if max(h, w) > grid:
        scale = grid / float(max(h, w))
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)

    if rotation:
        image = np.rot90(image, k=rotation)
    if flip_horizontal:
        image = np.fliplr(image)
    if flip_vertical:
        image = np.flipud(image)
    image = np.ascontiguousarray(image)
    if fine_rotation != 0.0:
        image = apply_fine_rotation(image, fine_rotation)
    if distortion_k1 != 0.0:
        image = apply_radial_distortion(image, distortion_k1)
    image = apply_keystone(image, converge_v, converge_h)

    if roi_norm is not None:
        gh, gw = image.shape[:2]
        y1 = max(0, min(gh - 1, int(round(roi_norm[0] * gh))))
        y2 = max(y1 + 1, min(gh, int(round(roi_norm[1] * gh))))
        x1 = max(0, min(gw - 1, int(round(roi_norm[2] * gw))))
        x2 = max(x1 + 1, min(gw, int(round(roi_norm[3] * gw))))
        image = np.ascontiguousarray(image[y1:y2, x1:x2])

    # A frame that never metered normalizes to huge values; no stretch, no mask.
    if luminance_density_range(bounds) < 1e-6:
        return np.zeros(image.shape[:2], dtype=np.float32), 0.5

    val = normalize_log_image(unmix_log_image(prefilter_log_grid(image, None, 0.0), unmix), bounds)
    lum = LUMA_R * val[:, :, 0] + LUMA_G * val[:, :, 1] + LUMA_B * val[:, :, 2]
    sigma = min(max(spacer, MASK_SPACER_MIN), MASK_SPACER_MAX) * 0.01 * min(lum.shape[:2])
    blurred = cv2.GaussianBlur(np.ascontiguousarray(lum, dtype=np.float32), (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    centre = float(blurred.mean())
    return blurred - centre, centre


def normalize_log_image(img_log: ImageBuffer, bounds: LogNegativeBounds) -> ImageBuffer:
    """
    Stretches log-data to fit [0, 1].
    """
    floors = np.ascontiguousarray(np.array(bounds.floors, dtype=np.float32))
    ceils = np.ascontiguousarray(np.array(bounds.ceils, dtype=np.float32))

    return ensure_image(_normalize_log_image_jit(np.ascontiguousarray(img_log.astype(np.float32)), floors, ceils))


def _sample_log_bounds(
    img_log: np.ndarray,
    percentile_clip: float,
    base: float,
    process_mode: str,
    e6_normalize: bool,
    sorted_grid: Optional[np.ndarray] = None,
) -> tuple[list, list]:
    """
    Per-channel (floors, ceils) at one clip level. `base` is the robust baseline
    clip added on top of the slider value; negative slider values expand outward
    by a log-density margin instead.
    """
    if percentile_clip >= 0:
        clip = max(0.00001, min(50.0, percentile_clip + base))
        margin = 0.0
    else:
        # Margin mode expands from the same robust basis, so the slider stays continuous through
        # its neutral position.
        clip = base
        margin = -percentile_clip
    p_low, p_high = np.float64(clip), np.float64(100.0 - clip)
    fixed_range = 3.0

    if process_mode == ProcessMode.E6:
        p_low, p_high = p_high, p_low
        fixed_range = -3.0

    def _pct(p) -> list:
        if sorted_grid is not None:
            return [float(v) for v in percentile_from_sorted(sorted_grid, p)]
        return [float(np.percentile(img_log[:, :, ch], p)) for ch in range(3)]

    floors = _pct(p_low)

    if process_mode != ProcessMode.E6 or e6_normalize:
        ceils = _pct(p_high)
    else:
        ceils = [floors[ch] + fixed_range for ch in range(3)]

    if margin > 0.0:
        # Expand outward; per-channel sign handles both f < c and f > c (E6).
        for ch in range(3):
            if ceils[ch] >= floors[ch]:
                floors[ch] -= margin
                ceils[ch] += margin
            else:
                floors[ch] += margin
                ceils[ch] -= margin

    return floors, ceils


def _same_pixel_color_floor_refs(
    img_log: ImageBuffer,
    luma_floors: list,
    luma_ceils: list,
    base_refs: Tuple[float, float, float],
    color_clip: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Dense-end (print-white) color refs from one shared pixel set: the luma-extreme
    band's lowest-chroma subset, chroma measured base-anchored (offsets from the
    thin-end refs, per-channel span as provisional gamma, refined once from the
    band medians). Independent per-channel percentiles read a different scene
    object per channel, so colored highlight content masquerades as film cast;
    a shared chroma-gated set cannot. The thin end needs no such treatment —
    density on real film is bounded below by base, anchoring per-channel ceils.
    None when the band's neutral set is too small or too chromatic (caller falls
    back to the percentile pass).
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    c = EXPOSURE_CONSTANTS
    q = float(c["neutral_axis_chroma_quantile"])
    cap = float(c["neutral_axis_chroma_cap"])
    min_px = int(c["neutral_axis_min_pixels"])
    width = float(c["color_bounds_band_width"])
    epsilon = 1e-6

    flat = img_log.reshape(-1, 3).astype(np.float64)
    base = np.asarray(base_refs, dtype=np.float64)
    norm = np.empty_like(flat)
    for ch in range(3):
        denom = luma_ceils[ch] - luma_floors[ch]
        if abs(denom) < epsilon:
            denom = epsilon if denom >= 0 else -epsilon
        norm[:, ch] = (flat[:, ch] - luma_floors[ch]) / denom
    luma = LUMA_R * norm[:, 0] + LUMA_G * norm[:, 1] + LUMA_B * norm[:, 2]

    clip = max(0.00001, min(50.0 - width, float(color_clip)))
    lo, hi = np.percentile(luma, [clip, clip + width])
    band = (luma >= lo) & (luma <= hi)
    if int(band.sum()) < min_px:
        return None
    d = flat[band] - base[None, :]

    def _select(gamma: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        g = gamma.copy()
        g[np.abs(g) < epsilon] = epsilon
        chroma = _rms_chroma(d / g[None, :])
        thr = float(np.quantile(chroma, q))
        sel = chroma <= thr
        if int(sel.sum()) < min_px:
            return None
        return sel, float(np.median(chroma[sel]))

    spans = np.array([luma_floors[ch] - base_refs[ch] for ch in range(3)], dtype=np.float64)
    first = _select(spans)
    # Pass-1 loose cap: a homogeneous colored cluster would otherwise be self-normalized to
    # zero chroma by pass 2 and read as neutral.
    if first is None or first[1] > float(c["neutral_axis_first_pass_cap"]):
        return None
    provisional = np.median(d[first[0]], axis=0)
    if np.any(np.abs(provisional) < epsilon):
        return None
    second = _select(provisional)
    if second is None or second[1] > cap:
        return None
    refs = base + np.median(d[second[0]], axis=0)
    return (float(refs[0]), float(refs[1]), float(refs[2]))


def analyze_log_exposure_bounds(
    image: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    process_mode: str = ProcessMode.C41,
    e6_normalize: bool = True,
    percentile_clip: float = 0.0,
    color_clip: float = 0.0,
    unmix: Optional[np.ndarray] = None,
) -> LogNegativeBounds:
    """
    Performs full analysis pass on a linear image to find density floors/ceils.

    Two independent axes are sampled and recombined:
      - percentile_clip (luma): drives the overall black/white-point luminance and
        span (ceil-floor) — i.e. dynamic range / highlight headroom. Sampled at the
        gentle base_luma_clip baseline; slider semantics are:
          > 0  clips the histogram tails (added on top of the baseline clip).
          = 0  robust extremes (block-median prefilter + baseline clip).
          < 0  outward headroom: bounds pushed BEYOND the robust extremes by the margin.
      - color_clip (color): the absolute per-tail clip percentile for the per-channel
        color deviation (white balance / orange-mask cast). A tighter (larger) clip
        gives a more robust channel balance; a gentler (smaller) clip samples nearer
        the extremes. Default neutral is base_color_clip.
    The luminance centre+span comes from the luma sampling, the per-channel color
    offsets from the color sampling, so the cast clip is tunable without compressing
    highlights. Identical channels (mono) give zero deviation at any clip.
    """
    img_log = to_log_density(image)
    img_log = unmix_log_image(img_log, unmix)
    return analyze_log_exposure_bounds_from_log(img_log, roi, analysis_buffer, process_mode, e6_normalize, percentile_clip, color_clip)


def analyze_log_exposure_bounds_from_log(
    img_log: ImageBuffer,
    roi: Optional[tuple[int, int, int, int]] = None,
    analysis_buffer: float = 0.0,
    process_mode: str = ProcessMode.C41,
    e6_normalize: bool = True,
    percentile_clip: float = 0.0,
    color_clip: float = 0.0,
    sorted_grid: Optional[np.ndarray] = None,
) -> LogNegativeBounds:
    """Log-image core of analyze_log_exposure_bounds (skips the log10).

    ``sorted_grid`` is `sorted_channel_grid(img_log)` from a caller that already holds it;
    it stands in for both percentile passes, so it is only valid when no ROI or analysis
    buffer is left to apply here.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    if roi or analysis_buffer > 0:
        sorted_grid = None
    if roi:
        y1, y2, x1, x2 = roi
        img_log = img_log[y1:y2, x1:x2]

    if analysis_buffer > 0:
        img_log = get_analysis_crop(img_log, analysis_buffer)

    img_log = _block_median_grid(img_log)

    base_luma = float(EXPOSURE_CONSTANTS["base_luma_clip"])

    floors, ceils = _sample_log_bounds(img_log, percentile_clip, base_luma, process_mode, e6_normalize, sorted_grid)

    # Color pass: per-channel deviations recombined onto the luma mean centre and span. The
    # ceils (thin end, base-anchored) come from per-channel percentiles at color_clip. The
    # floors (dense end, scene content) prefer the same-pixel chroma-gated band refs and fall
    # back to the percentile pass when the band holds no trustworthy neutrals, and always for
    # E-6 and margin-mode clips.
    c_floors, c_ceils = _sample_log_bounds(img_log, color_clip, 0.0, process_mode, e6_normalize, sorted_grid)
    if process_mode != ProcessMode.E6 and color_clip >= 0:
        sp = _same_pixel_color_floor_refs(img_log, floors, ceils, (c_ceils[0], c_ceils[1], c_ceils[2]), color_clip)
        if sp is not None:
            c_floors = [sp[0], sp[1], sp[2]]
    mean_lf, mean_lc = sum(floors) / 3.0, sum(ceils) / 3.0
    mean_cf, mean_cc = sorted(c_floors)[1], sorted(c_ceils)[1]
    floors = [mean_lf + (c_floors[ch] - mean_cf) for ch in range(3)]
    ceils = [mean_lc + (c_ceils[ch] - mean_cc) for ch in range(3)]

    return LogNegativeBounds(
        (floors[0], floors[1], floors[2]),
        (ceils[0], ceils[1], ceils[2]),
    )


def mix_luma_color_bounds(luma_src: LogNegativeBounds, color_src: LogNegativeBounds) -> LogNegativeBounds:
    """
    Luma-weighted centre+range from one bounds, per-channel color cast from
    another. Keeps the color source's per-channel shape but shifts it so the
    result's luma-weighted centre and range (the brightness/anchor and the H&D
    slope drivers — see luminance_density_range) match the luma source. So
    color-average moves only the per-channel cast, never contrast/brightness.
    Identity when luma_src is color_src (mirrors analyze_log_exposure_bounds'
    recombination), which also keeps a persisted self-mix from stacking edits.
    """
    if luma_src == color_src:
        return luma_src
    w = (LUMA_R, LUMA_G, LUMA_B)
    centre = lambda b: sum(w[c] * (b.floors[c] + b.ceils[c]) / 2.0 for c in range(3))  # noqa: E731
    rng = lambda b: sum(w[c] * (b.ceils[c] - b.floors[c]) for c in range(3))  # noqa: E731
    dC = centre(luma_src) - centre(color_src)
    dR = rng(luma_src) - rng(color_src)
    df, dc = dC - dR / 2.0, dC + dR / 2.0
    cf, cc = color_src.floors, color_src.ceils
    return LogNegativeBounds(
        (cf[0] + df, cf[1] + df, cf[2] + df),
        (cc[0] + dc, cc[1] + dc, cc[2] + dc),
    )


def resolve_bounds(process, analyze_fn) -> LogNegativeBounds:
    """Final bounds for rendering. See resolve_bounds_detailed for the per-frame base."""
    return resolve_bounds_detailed(process, analyze_fn)[0]


def resolve_bounds_detailed(process, analyze_fn) -> tuple[LogNegativeBounds, LogNegativeBounds]:
    """
    Returns (final, base): the final mixed bounds to render with, and the per-frame
    base (local/analyzed) to persist. Persist the base, not the mix — re-feeding a
    mix as the next base stacks edits (mean-vs-median drift; color-only roll).
    Picks luma + color from the roll baseline (locked) or the per-frame base, then
    mixes. analyze_fn() supplies the base and is called only when actually needed.
    """
    roll_luma = process.use_luma_average and process.is_locked_initialized
    roll_color = process.use_color_average and process.is_locked_initialized
    locked = LogNegativeBounds(process.locked_floors, process.locked_ceils)
    if roll_luma and roll_color:
        return locked, locked
    base = LogNegativeBounds(process.local_floors, process.local_ceils) if process.is_local_initialized else analyze_fn()
    final = mix_luma_color_bounds(locked if roll_luma else base, locked if roll_color else base)
    return final, base


def luma_source_bounds(process, base: LogNegativeBounds) -> LogNegativeBounds:
    """
    Bounds the luma/exposure reading (metered anchor) must come from: the roll
    baseline when luma-average is on, else the per-frame base. The anchor is a
    luma-weighted percentile and so reacts non-linearly to the per-channel cast;
    measuring it here, not on the final mix, keeps brightness independent of the
    color-average toggle (mix_luma_color_bounds already pins centre+range).
    """
    if process.use_luma_average and process.is_locked_initialized:
        return LogNegativeBounds(process.locked_floors, process.locked_ceils)
    return base
