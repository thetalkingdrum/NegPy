"""NegativeConverter's expired-film rescue, color table and tone curve, for an A/B against Fade
Restoration. A faithful port of negative2positive/src/pipeline/expiredRescue.js (MIT, Copyright (c)
2025 lex; NOTICE.md): `analyzeExpiredFilm`, `offsetsFromBands`, `shapeCubic`,
`defaultExpiredRescueParams`, `buildExpiredRescueStages` and `applyExpiredTone`, on display values,
and the uneven-fog part of the spatial stage: app/expiredRescueOpenCv.js's sample and maps, with
negpy_core in place of OpenCV.js, then `fitFloor`, `fitFogSurface`, `surfaceRange`,
`fitExpiredSpatial` and `applyExpiredSpatial`. It keeps the source's arithmetic and its artifacts;
no semantic anchors and no local contrast."""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np

HIST_BINS = 1024
LOW_PERCENTILE = 0.005
HIGH_PERCENTILE = 0.995
BAND_EDGES = (0.0, 0.06, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0001)
BAND_MOVE_LIMIT = 0.35
MIN_BAND_SHARE = 0.004
NEUTRAL_CHROMA = 0.12
CLUSTER_RADIUS = 0.06
MIN_SAMPLES = 400
TARGET_MIDTONE = 0.42
OFFSET_BINS = 64
RESCUE_LUMA = (0.2126, 0.7152, 0.0722)
MAX_SAMPLES = 200000
BORDER_BUFFER = 0.02


def _js_round(v: float) -> float:
    """`Math.round`: halves go up."""
    return math.floor(v + 0.5)


def _round4(v: float) -> float:
    return _js_round(v * 10000.0) / 10000.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class Band(NamedTuple):
    lum: float
    mean: tuple[float, float, float]
    share: float


class Analysis(NamedTuple):
    """The plain object `analyzeExpiredFilm` returns, the fields the stages read."""

    lum_low: float
    lum_high: float
    lum_median: float
    leveled_median: float
    bands: tuple[Optional[Band], ...]
    lean: tuple[float, float, float]
    neutral_share: float
    samples: int


def _luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _neutral_weight(s: np.ndarray) -> np.ndarray:
    chroma = (s.max(axis=1) - s.min(axis=1)) / NEUTRAL_CHROMA
    return 1.0 / (1.0 + chroma * chroma)


def _percentile_from_histogram(hist: np.ndarray, total: int, fraction: float) -> float:
    target = fraction * total
    acc = np.cumsum(hist)
    hit = np.nonzero(acc >= target)[0]
    return 1.0 if len(hit) == 0 else (float(hit[0]) + 0.5) / len(hist)


def _plane16(display: np.ndarray) -> np.ndarray:
    """Their 16-bit plane of a display-encoded positive in 0..1."""
    return np.round(np.clip(np.asarray(display, dtype=np.float64)[..., :3], 0.0, 1.0) * 65535.0)


def _bounds(h: int, w: int) -> tuple[int, int, int, int]:
    """`resolveBounds` with no region: (left, top, width, height) inside a BORDER_BUFFER inset."""
    inset = math.floor(min(w, h) * BORDER_BUFFER)
    return inset, inset, max(1, w - inset * 2), max(1, h - inset * 2)


def analyze(display: np.ndarray, spatial: Optional["SpatialStage"] = None) -> Optional[Analysis]:
    """`analyzeExpiredFilm` on a display-encoded (H, W, 3) positive in 0..1, read as their 16-bit plane;
    with *spatial*, each sample is flattened first, as `options.spatial` does."""
    h, w = display.shape[:2]
    left, top, bw, bh = _bounds(h, w)
    stride = max(1, math.ceil(math.sqrt((bw * bh) / MAX_SAMPLES)))
    plane = _plane16(display)
    picked = plane[top : top + bh : stride, left : left + bw : stride] * (1.0 / 65535.0)
    if spatial is not None:
        ys = np.arange(top, top + bh, stride)
        xs = np.arange(left, left + bw, stride)
        u = np.broadcast_to(((xs + 0.5) / w)[None, :], picked.shape[:2])
        v = np.broadcast_to(((ys + 0.5) / h)[:, None], picked.shape[:2])
        picked = apply_spatial(spatial, (picked * 255.0).astype(np.float32), u, v).astype(np.float64) / 255.0
    picked = picked.reshape(-1, 3)
    count = len(picked)
    if count < MIN_SAMPLES:
        return None
    lum_in = _luminance(picked[:, 0], picked[:, 1], picked[:, 2])
    hist = np.bincount(np.minimum(HIST_BINS - 1, (lum_in * HIST_BINS).astype(np.int64)), minlength=HIST_BINS)
    # The samples are held in a Float32Array; every later step reads them back from it.
    samples = picked.astype(np.float32).astype(np.float64)

    lum_low = _percentile_from_histogram(hist, count, LOW_PERCENTILE)
    lum_high = _percentile_from_histogram(hist, count, HIGH_PERCENTILE)
    if lum_high - lum_low < 0.02:
        lum_low, lum_high = 0.0, 1.0
    lum_median = _percentile_from_histogram(hist, count, 0.5)
    leveled_median = _clamp((lum_median - lum_low) / (lum_high - lum_low), 0.0, 1.0)

    n_bands = len(BAND_EDGES) - 1
    span = max(0.02, lum_high - lum_low)
    lum = _luminance(samples[:, 0], samples[:, 1], samples[:, 2])
    t = np.clip((lum - lum_low) / span, 0.0, 1.0)
    band = np.searchsorted(np.asarray(BAND_EDGES[1:-1]), t, side="right")
    neutral = _neutral_weight(samples)
    band_sum = np.zeros((n_bands, 3))
    np.add.at(band_sum, band, neutral[:, None] * samples)
    band_lum = np.bincount(band, neutral * lum, n_bands)
    band_weight = np.bincount(band, neutral, n_bands)
    band_total = float(neutral.sum())
    neutral_population = band_total
    band_mean: list[Optional[np.ndarray]] = [band_sum[k] / band_weight[k] if band_weight[k] > 0 else None for k in range(n_bands)]
    for _ in range(3):
        means = np.array([m if m is not None else (np.nan, np.nan, np.nan) for m in band_mean])
        live = ~np.isnan(means[band, 0])
        distance = np.abs(samples - np.where(live[:, None], means[band], 0.0)).max(axis=1) / CLUSTER_RADIUS
        wt = np.where(live, neutral / (1.0 + distance * distance), 0.0)
        shift_sum = np.zeros((n_bands, 3))
        np.add.at(shift_sum, band, wt[:, None] * samples)
        shift_lum = np.bincount(band, wt * lum, n_bands)
        shift_weight = np.bincount(band, wt, n_bands)
        for k in range(n_bands):
            if band_mean[k] is None or shift_weight[k] < 8:
                continue
            band_mean[k] = shift_sum[k] / shift_weight[k]
            band_lum[k] = shift_lum[k]
            band_weight[k] = shift_weight[k]
    bands: list[Optional[Band]] = []
    for k in range(n_bands):
        share = band_weight[k] / band_total if band_total > 0 else 0.0
        mean = band_mean[k]
        if mean is None or share < MIN_BAND_SHARE or band_weight[k] < 8:
            bands.append(None)
            continue
        bands.append(Band(_round4(band_lum[k] / band_weight[k]), (_round4(mean[0]), _round4(mean[1]), _round4(mean[2])), _round4(share)))
    lean = [0.0, 0.0, 0.0]
    lean_weight = 0.0
    for b in bands:
        if b is None:
            continue
        for c in range(3):
            lean[c] += b.share * (b.mean[c] - b.lum)
        lean_weight += b.share
    lean = [v / lean_weight if lean_weight > 0 else 0.0 for v in lean]
    return Analysis(
        _round4(lum_low),
        _round4(lum_high),
        _round4(lum_median),
        _round4(leveled_median),
        tuple(bands),
        (_round4(lean[0]), _round4(lean[1]), _round4(lean[2])),
        _round4(neutral_population / count if count else 0.0),
        count,
    )


def _shape_cubic(xs: list[float], ys: list[float]):
    """`shapeCubic`: Fritsch-Carlson slopes, flat beyond the ends."""
    n = len(xs)
    if n == 1:
        return lambda x: ys[0]
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    delta = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    d = [0.0] * n
    d[0] = delta[0]
    d[n - 1] = delta[n - 2]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    def curve(x: float) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[n - 1]:
            return ys[n - 1]
        i = 0
        while i < n - 2 and x > xs[i + 1]:
            i += 1
        t = (x - xs[i]) / h[i]
        t2 = t * t
        t3 = t2 * t
        return (
            (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * d[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * d[i + 1]
        )

    return curve


def offsets_from_bands(analysis: Analysis, neutralize: float = 1.0, crossover: float = 1.0) -> Optional[np.ndarray]:
    """`offsetsFromBands`: OFFSET_BINS x 3 offsets in 0..255 units over luminance, or None when nothing leans."""
    points = [b for b in analysis.bands if b is not None]
    if not points:
        return None
    xs = [b.lum for b in points]
    lo, hi = xs[0], xs[-1]
    table = np.zeros((OFFSET_BINS, 3), dtype=np.float32)
    moved = False
    for c in range(3):
        curve = _shape_cubic(xs, [b.mean[c] - b.lum for b in points])
        for i in range(OFFSET_BINS):
            band_lean = curve(_clamp(i / (OFFSET_BINS - 1), lo, hi))
            delta = _clamp(-(analysis.lean[c] * neutralize + (band_lean - analysis.lean[c]) * crossover), -BAND_MOVE_LIMIT, BAND_MOVE_LIMIT)
            if abs(delta) > 0.002:
                moved = True
            table[i, c] = delta * 255
    return table if moved else None


class ToneParams(NamedTuple):
    levels: float
    brightness: float
    contrast: float


IDENTITY_TONE = ToneParams(0.0, 0.0, 0.0)


def default_tone(analysis: Analysis) -> ToneParams:
    """`defaultExpiredRescueParams`: levels at 100, brightness toward a TARGET_MIDTONE midtone, contrast from the range."""
    median = _clamp(analysis.leveled_median, 0.02, 0.98)
    gamma_needed = math.log(TARGET_MIDTONE) / math.log(median)
    brightness = _js_round(_clamp(-100 * math.log2(gamma_needed), -25, 60))
    contrast = _js_round(_clamp(20 + (1 - (analysis.lum_high - analysis.lum_low)) * 30, 10, 45))
    return ToneParams(100.0, brightness, contrast)


def _soft_contrast(x: float, strength: float) -> float:
    s = 4
    shaped = 0.5 + math.tanh((x - 0.5) * s) / (2 * math.tanh(s / 2))
    return x + strength * (shaped - x)


def tone_curve(analysis: Analysis, params: ToneParams) -> np.ndarray:
    """The 256-entry shared tone curve of `buildExpiredRescueStages`, in 0..255."""
    levels = params.levels / 100
    gamma = math.pow(2, -params.brightness / 100)
    contrast = params.contrast / 100
    lo = analysis.lum_low * levels
    hi = 1 - (1 - analysis.lum_high) * levels
    span = max(0.02, hi - lo)
    tone = np.zeros(256, dtype=np.float32)
    for v in range(256):
        x = _clamp((v / 255 - lo) / span, 0.0, 1.0)
        x = math.pow(x, gamma)
        if contrast > 0:
            x = _soft_contrast(x, contrast)
        tone[v] = _clamp(x * 255, 0.0, 255.0)
    return tone


def apply_tone(values255: np.ndarray, offsets: Optional[np.ndarray], tone: np.ndarray) -> np.ndarray:
    """`applyExpiredTone` on (N, 3) values in 0..255: the luminance-indexed offsets, then the tone curve."""
    px = np.asarray(values255, dtype=np.float64).reshape(-1, 3)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    if offsets is not None:
        t = offsets.astype(np.float64)
        pos = np.clip(_luminance(r, g, b) / 255, 0.0, 1.0) * (OFFSET_BINS - 1)
        i0 = pos.astype(np.int64)
        i1 = np.where(i0 + 1 < OFFSET_BINS, i0 + 1, i0)
        f = pos - i0
        r = np.clip(r + t[i0, 0] * (1 - f) + t[i1, 0] * f, 0, 255)
        g = np.clip(g + t[i0, 1] * (1 - f) + t[i1, 1] * f, 0, 255)
        b = np.clip(b + t[i0, 2] * (1 - f) + t[i1, 2] * f, 0, 255)
    curve = tone.astype(np.float64)

    def lerp(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 255.0)
        i = np.minimum(x.astype(np.int64), 254)
        return curve[i] + (curve[i + 1] - curve[i]) * (x - i)

    return np.stack([lerp(r), lerp(g), lerp(b)], axis=1).reshape(np.shape(values255))


def rescue(display: np.ndarray, colour: bool, tone: bool, fog: bool = False) -> np.ndarray:
    """Their stages on a display-encoded (H, W, 3) positive in 0..1: the uneven-fog stage at 100 % when
    *fog*, then the color table with Neutralise and Crossover at 100 % when *colour* and the default
    tone curve when *tone*, both measured on the flattened frame. Unchanged with nothing to do."""
    spatial = fog_stage(display) if fog else None
    px = np.asarray(display, dtype=np.float64)[..., :3] * 255.0
    if spatial is not None:
        h, w = display.shape[:2]
        u = np.broadcast_to(((np.arange(w) + 0.5) / w)[None, :], (h, w))
        v = np.broadcast_to(((np.arange(h) + 0.5) / h)[:, None], (h, w))
        px = apply_spatial(spatial, px.astype(np.float32), u, v).astype(np.float64)
    analysis = analyze(display, spatial) if colour or tone else None
    if analysis is None:
        return display if spatial is None else (px / 255.0).astype(np.float32)
    offsets = offsets_from_bands(analysis) if colour else None
    curve = tone_curve(analysis, default_tone(analysis) if tone else IDENTITY_TONE)
    out = apply_tone(px, offsets, curve) / 255.0
    return out.astype(np.float32)


# Uneven fog: the spatial stage's fog part.

#: The area-averaged sample's width and the grid's, as `measureExpiredSpatialMaps` defaults them.
FOG_WORKING_WIDTH = 160
FOG_GRID_WIDTH = 32
#: Erosion kernel width and blur sigma, as fractions of the sample's width.
FOG_KERNEL = 0.22
FOG_BLUR = 0.04
FOG_SURFACE_LIMIT = 0.15
FOG_SURFACE_PLAUSIBLE = 0.25


class FogMaps(NamedTuple):
    """`measureExpiredSpatialMapsFromSample`'s `low` grids (float32), grid size and placement."""

    low: tuple[np.ndarray, np.ndarray, np.ndarray]
    grid_width: int
    grid_height: int
    fraction: tuple[float, float, float, float]


class SpatialStage(NamedTuple):
    """`buildExpiredSpatialStage` for the fog alone: float32 coefficients and offsets, the fraction."""

    coefficients: np.ndarray
    offset: np.ndarray
    fraction: tuple[float, float, float, float]


def _odd(v: float) -> int:
    n = max(3, int(_js_round(v)))
    return n if n % 2 else n + 1


def fog_sample(display: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """(uint8 RGB, fraction): `createBoundsSample` and `downsampleBoundsRows` of the 16-bit plane at
    FOG_WORKING_WIDTH, and the region's placement in the frame."""
    h, w = display.shape[:2]
    left, top, bw, bh = _bounds(h, w)
    width = FOG_WORKING_WIDTH
    height = max(8, int(_js_round(bh * (width / bw))))
    plane = _plane16(display)
    sy0 = top + np.floor(np.arange(height) / height * bh).astype(np.int64)
    sy1 = np.maximum(sy0 + 1, top + np.floor((np.arange(height) + 1) / height * bh).astype(np.int64))
    sx0 = left + np.floor(np.arange(width) / width * bw).astype(np.int64)
    sx1 = np.maximum(sx0 + 1, left + np.floor((np.arange(width) + 1) / width * bw).astype(np.int64))
    # Exact integer block sums through a summed-area table.
    sat = np.zeros((h + 1, w + 1, 3), dtype=np.int64)
    sat[1:, 1:] = plane.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    y0, y1, x0, x1 = sy0[:, None], sy1[:, None], sx0[None, :], sx1[None, :]
    total = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
    n = ((y1 - y0) * (x1 - x0))[..., None]
    # A Uint8ClampedArray store rounds half to even.
    rgb = np.clip(np.round((total / n) * (1.0 / 257.0)), 0, 255).astype(np.uint8)
    fraction = (left / w, top / h, bw / w, bh / h)
    return rgb, fraction


def fog_maps(display: np.ndarray) -> FogMaps:
    """The OpenCV half: each channel of the sample eroded by an ellipse FOG_KERNEL of its width, blurred
    with sigma FOG_BLUR of it (replicated border) and area-resized to the FOG_GRID_WIDTH grid."""
    import negpy_core

    rgb, fraction = fog_sample(display)
    height, width = rgb.shape[:2]
    grid_h = max(3, int(_js_round(FOG_GRID_WIDTH * height / width)))
    f32 = (rgb.astype(np.float64) * (1.0 / 255.0)).astype(np.float32)
    k = _odd(width * FOG_KERNEL)
    kernel = negpy_core.ellipse_kernel((k, k))
    low = []
    for c in range(3):
        eroded = negpy_core.morphology_ex(np.ascontiguousarray(f32[..., c]), negpy_core.MORPH_ERODE, kernel)
        blurred = negpy_core.gaussian_blur(np.ascontiguousarray(eroded), (0, 0), width * FOG_BLUR, 0.0, 1)
        low.append(np.asarray(negpy_core.resize_area(np.ascontiguousarray(blurred), (FOG_GRID_WIDTH, grid_h)), dtype=np.float32))
    return FogMaps((low[0], low[1], low[2]), FOG_GRID_WIDTH, grid_h, fraction)


def _quadratic(c, u, v):
    return c[0] + c[1] * u + c[2] * v + c[3] * u * u + c[4] * v * v + c[5] * u * v


def _solve_least_squares(rows: np.ndarray, values: np.ndarray, weights: np.ndarray, dof: int) -> Optional[list[float]]:
    """`solveLeastSquares`: the normal equations summed in the source's order, a 1e-7 ridge and
    Gauss-Jordan elimination with partial pivoting."""
    live = weights > 0
    r, val, wt = rows[live], values[live], weights[live]
    m = [[0.0] * (dof + 1) for _ in range(dof)]
    for p in range(dof):
        rp = r[:, p] * wt
        for q in range(dof):
            m[p][q] = float(np.cumsum(rp * r[:, q])[-1]) if len(rp) else 0.0
        m[p][dof] = float(np.cumsum(rp * val)[-1]) if len(rp) else 0.0
    for p in range(dof):
        m[p][p] += 1e-7
    for col in range(dof):
        pivot = col
        for rr in range(col + 1, dof):
            if abs(m[rr][col]) > abs(m[pivot][col]):
                pivot = rr
        if abs(m[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            m[pivot], m[col] = m[col], m[pivot]
        for rr in range(dof):
            if rr == col:
                continue
            factor = m[rr][col] / m[col][col]
            if not factor:
                continue
            for q in range(col, dof + 1):
                m[rr][q] -= factor * m[col][q]
    out = [0.0] * 6
    for p in range(dof):
        out[p] = m[p][dof] / m[p][p]
    return out if all(math.isfinite(v) for v in out) else None


def _fit_floor(values: np.ndarray, us: np.ndarray, vs: np.ndarray, rows: np.ndarray, dof: int) -> Optional[list[float]]:
    """`fitFloor`: an asymmetric robust fit to a grid of local minima, cells above it lose weight fast."""
    median = float(np.sort(values)[len(values) >> 1])
    weights = np.where(values <= median, 1.0, 0.25)
    coefficients = _solve_least_squares(rows, values, weights, dof)
    if coefficients is None:
        return None
    for _ in range(5):
        residual = values - _quadratic(coefficients, us, vs)
        scaled = np.where(residual > 0, residual / 0.012, residual / 0.04)
        weights = 1.0 / (1.0 + scaled * scaled)
        nxt = _solve_least_squares(rows, values, weights, dof)
        if nxt is None:
            break
        coefficients = nxt
    return coefficients


def _surface_range(c, fraction) -> tuple[float, float]:
    left, top, width, height = fraction
    t = np.arange(9) / 8
    u = (left + t * width)[None, :]
    v = (top + t * height)[:, None]
    value = _quadratic(c, u, v)
    return float(value.min()), float(value.max())


def _fit_fog_surface(values, us, vs, fraction) -> Optional[list[float]]:
    rows = np.stack([np.ones_like(us), us, vs, us * us, vs * vs, us * vs], axis=1)
    for dof in (6, 3):
        fit = _fit_floor(values, us, vs, rows, dof)
        if fit is None:
            continue
        lo, hi = _surface_range(fit, fraction)
        if hi - lo <= FOG_SURFACE_PLAUSIBLE:
            return fit
    return None


def fit_spatial(maps: FogMaps) -> dict:
    """`fitExpiredSpatial`'s fog part: per channel, the rounded coefficients, offset and amplitude."""
    gw, gh = maps.grid_width, maps.grid_height
    left, top, width, height = maps.fraction
    i = np.tile(np.arange(gw), gh)
    j = np.repeat(np.arange(gh), gw)
    # Held in Float32Arrays.
    us = (left + ((i + 0.5) / gw) * width).astype(np.float32).astype(np.float64)
    vs = (top + ((j + 0.5) / gh) * height).astype(np.float32).astype(np.float64)
    coefficients, offset, amplitude = [], [], []
    for c in range(3):
        grid = np.asarray(maps.low[c], dtype=np.float32).reshape(-1).astype(np.float64)
        fit = _fit_fog_surface(grid, us, vs, maps.fraction) if gw * gh >= 12 else None
        if fit is None:
            fit = [float(np.cumsum(grid)[-1]) / len(grid), 0.0, 0.0, 0.0, 0.0, 0.0]
        lo, hi = _surface_range(fit, maps.fraction)
        span = hi - lo
        if not span >= 0.005:
            fit = [lo, 0.0, 0.0, 0.0, 0.0, 0.0]
        coefficients.append([_round4(v) for v in fit])
        offset.append(_round4(lo))
        amplitude.append(_round4(max(0.0, span if span >= 0.005 else 0.0)))
    fraction = tuple(_round4(v) for v in maps.fraction)
    return {"coefficients": coefficients, "offset": offset, "amplitude": amplitude, "fraction": fraction}


def spatial_stage(spatial: dict) -> Optional[SpatialStage]:
    """`sanitizeExpiredSpatial` then `buildExpiredSpatialStage` at Uneven fog 100 % and no local
    contrast; None when no channel has an amplitude."""
    coefficients = np.clip(np.asarray(spatial["coefficients"], dtype=np.float64), -20, 20)
    offset = np.clip(np.asarray(spatial["offset"], dtype=np.float64), -1, 2)
    amplitude = np.clip(np.asarray(spatial["amplitude"], dtype=np.float64), 0, 1)
    if not (amplitude > 0).any():
        return None
    return SpatialStage(coefficients.astype(np.float32), offset.astype(np.float32), spatial["fraction"])


def fog_stage(display: np.ndarray) -> Optional[SpatialStage]:
    """The fog stage their app builds for a display-encoded positive, or None with no fog."""
    return spatial_stage(fit_spatial(fog_maps(display)))


def apply_spatial(stage: SpatialStage, px: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """`applyExpiredSpatial`'s fog part on float32 values in 0..255 at frame coordinates *u*, *v*."""
    left, top, width, height = stage.fraction
    u = np.clip(u, left, left + width)
    v = np.clip(v, top, top + height)
    out = np.array(px, dtype=np.float32, copy=True)
    for ch in range(3):
        c = stage.coefficients[ch].astype(np.float64)
        fog = np.clip(_quadratic(c, u, v) - float(stage.offset[ch]), 0.0, FOG_SURFACE_LIMIT) * 255.0
        value = (out[..., ch].astype(np.float64) - fog) / (1.0 - fog / 255.0)
        out[..., ch] = np.where(fog > 0, np.maximum(value, 0.0), out[..., ch]).astype(np.float32)
    return out


def nc_stage_flags(process) -> tuple[bool, bool, bool]:
    """(colour, tone, fog): the NegativeConverter stages a slide render runs (`FadeABModel`,
    `FadeABTone`, `FadeABFog`)."""
    from negpy.features.process.models import FadeABFog, FadeABModel, FadeABTone
    from negpy.features.process.path import RenderPath, render_path

    if render_path(process) is RenderPath.PRINT:
        return False, False, False
    return process.fade_ab_model == FadeABModel.NC, process.fade_ab_tone == FadeABTone.NC, process.fade_ab_fog == FadeABFog.NC


def apply_nc_stage(img_linear: np.ndarray, process) -> np.ndarray:
    """Their stages on a scene-linear working-space positive, read and written through the working TRC,
    as their app reads a display-encoded positive. Unchanged when no stage acts."""
    from negpy.kernel.image.logic import working_oetf_decode, working_oetf_encode

    colour, tone, fog = nc_stage_flags(process)
    if not (colour or tone or fog):
        return img_linear
    display = np.clip(working_oetf_encode(np.ascontiguousarray(img_linear[..., :3], dtype=np.float32)), 0.0, 1.0)
    out = rescue(display, colour, tone, fog)
    return np.ascontiguousarray(working_oetf_decode(np.ascontiguousarray(out, dtype=np.float32)), dtype=np.float32)
