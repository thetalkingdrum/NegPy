"""NegativeConverter's expired-film rescue color table, for an A/B against Fade Restoration. A
faithful port of negative2positive/src/pipeline/expiredRescue.js (MIT, Copyright (c) 2025 lex;
NOTICE.md): `analyzeExpiredFilm`, `offsetsFromBands`, `shapeCubic`, `buildExpiredRescueStages` and
`applyExpiredTone` with the identity tone, on display values. It keeps the source's arithmetic and
its artifacts; no semantic anchors, no tone curve, no spatial stage."""

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


def analyze(display: np.ndarray) -> Optional[Analysis]:
    """`analyzeExpiredFilm` on a display-encoded (H, W, 3) positive in 0..1, read as their 16-bit plane."""
    h, w = display.shape[:2]
    left, top, bw, bh = _bounds(h, w)
    stride = max(1, math.ceil(math.sqrt((bw * bh) / MAX_SAMPLES)))
    plane = _plane16(display)
    picked = plane[top : top + bh : stride, left : left + bw : stride] * (1.0 / 65535.0)
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


def rescue(display: np.ndarray) -> np.ndarray:
    """Their color table with Neutralise and Crossover at 100 % on a display-encoded (H, W, 3) positive
    in 0..1. Unchanged with nothing to read."""
    analysis = analyze(display)
    if analysis is None:
        return display
    px = np.asarray(display, dtype=np.float64)[..., :3] * 255.0
    out = apply_tone(px, offsets_from_bands(analysis), tone_curve(analysis, IDENTITY_TONE)) / 255.0
    return out.astype(np.float32)


def nc_stage_active(process) -> bool:
    """Whether a slide render runs their color table (`FadeABModel.NC`)."""
    from negpy.features.process.models import FadeABModel
    from negpy.features.process.path import RenderPath, render_path

    return render_path(process) is not RenderPath.PRINT and process.fade_ab_model == FadeABModel.NC


def apply_nc_stage(img_linear: np.ndarray, process) -> np.ndarray:
    """Their color table on a scene-linear working-space positive, read and written through the working
    TRC, as their app reads a display-encoded positive. Unchanged when it does not act."""
    from negpy.kernel.image.logic import working_oetf_decode, working_oetf_encode

    if not nc_stage_active(process):
        return img_linear
    display = np.clip(working_oetf_encode(np.ascontiguousarray(img_linear[..., :3], dtype=np.float32)), 0.0, 1.0)
    out = rescue(display)
    return np.ascontiguousarray(working_oetf_decode(np.ascontiguousarray(out, dtype=np.float32)), dtype=np.float32)
