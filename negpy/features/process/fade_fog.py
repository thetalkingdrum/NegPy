"""Uneven fog, NegPy's design: a source bake that removes the part of an aged slide's fog that changes
across the frame, measured and removed in density on the linear source after Flat Field.

Fog adds density, so the thinnest film shows base plus fog: the local floor is the local minimum
density. One shared surface is fitted to the floor of the channel mean with NegativeConverter's
asymmetric robust fit (quadratic, then plane, then even; MIT, NOTICE.md), so content in one channel
cannot tilt it. Color is at most a small plane per channel."""

import math
from typing import NamedTuple, Optional

import numpy as np

from negpy.domain.types import ImageBuffer

#: The small copy's width and the floor grid's, as NegativeConverter measures its maps.
FOG_WORKING_WIDTH = 160
FOG_GRID_WIDTH = 32
#: Erosion window and anti-alias blur sigma, as fractions of the small copy's width.
FOG_KERNEL = 0.22
FOG_BLUR = 0.04
#: Border left out of the measurement, as a fraction of the short side.
FOG_BORDER = 0.02
#: The robust fit's scales (D): a cell this far above the surface is content and drops out fast, one
#: this far below keeps its say. Their 0.012 and 0.04 display units at the floor, where a slide's
#: thinnest film shows about 0.9 display units per D.
FOG_ABOVE = 0.013
FOG_BELOW = 0.044
#: A quadratic or plane that bends more than this (D) over the region is content, not fog: their 0.25.
FOG_PLAUSIBLE = 0.28
#: A grid cell whose densest channel floor sits more than FOG_PLAUSIBLE above this percentile of the
#: grid holds no clear film in its window (a colored area is thin in one channel only), so it says
#: nothing about the fog and is left out.
FOG_CLEAR_PCT = 5.0
#: On clear film one pixel is the thinnest in all three channels, so the floor of the channel mean
#: equals the mean of the channel floors. A cell whose two differ by more than this (D) took its
#: channel floors from different areas, such as the edge of a colored area, and is left out.
FOG_COHERENT = 0.02
#: Below this amplitude (D) the frame has no uneven fog and the bake does nothing: under the grain of
#: a scan and under 1 dE on a mid gray.
FOG_FLOOR = 0.02
#: The most density the bake removes anywhere (D), half a stop: a larger step across a slide is more
#: often the picture, and this keeps a wrong fit a mild error, as their 0.15 limit does.
FOG_CAP = 0.15
#: Each channel's color plane is shrunk by this factor and held to FOG_COLOR_CAP (D) over the region.
FOG_COLOR_SHRINK = 0.5
FOG_COLOR_CAP = 0.03
#: A pixel at or above this linear value in every channel is light past the film, not a density.
FOG_CLIP = 0.99
#: Weighted least-squares passes of the robust fit after the first.
FOG_ITERATIONS = 5


class FogSurface(NamedTuple):
    """The uneven part to remove, in D: shared quadratic coefficients, a zero-sum color plane per
    channel (3 coefficients each), the shared surface's minimum over the region and the region."""

    shared: tuple[float, ...]
    color: tuple[tuple[float, float, float], ...]
    offset: float
    fraction: tuple[float, float, float, float]

    def amplitude(self) -> float:
        lo, hi = _surface_range(np.asarray(self.shared), self.fraction)
        return min(hi - lo, FOG_CAP)


def fog_active(process) -> bool:
    """Whether the NegPy fog bake runs: Uneven fog NegPy on a slide."""
    from negpy.features.process.models import FadeABFog
    from negpy.features.process.path import RenderPath, render_path

    return process.fade_ab_fog == FadeABFog.NEGPY and render_path(process) is not RenderPath.PRINT


def fog_token(process) -> str:
    """The source-hash token of the bake, "" when it does not run."""
    return "|fog-negpy" if fog_active(process) else ""


def _quadratic(c: np.ndarray, u, v):
    return c[0] + c[1] * u + c[2] * v + c[3] * u * u + c[4] * v * v + c[5] * u * v


def _features(u: np.ndarray, v: np.ndarray, dof: int) -> np.ndarray:
    return np.stack([np.ones_like(u), u, v, u * u, v * v, u * v], axis=1)[:, :dof]


def _robust_floor(values: np.ndarray, u: np.ndarray, v: np.ndarray, dof: int) -> Optional[np.ndarray]:
    """NegativeConverter's `fitFloor` in D: cells above the median start at a quarter weight, then
    Cauchy weights with FOG_ABOVE above the surface and FOG_BELOW below it."""
    a = _features(u, v, dof)
    weights = np.where(values <= np.sort(values)[len(values) >> 1], 1.0, 0.25)
    coef = None
    for _ in range(FOG_ITERATIONS + 1):
        sw = np.sqrt(weights)
        try:
            sol, *_ = np.linalg.lstsq(a * sw[:, None], values * sw, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(sol)):
            break
        coef = np.zeros(6)
        coef[:dof] = sol
        residual = values - _quadratic(coef, u, v)
        scaled = np.where(residual > 0, residual / FOG_ABOVE, residual / FOG_BELOW)
        weights = 1.0 / (1.0 + scaled * scaled)
    return coef


def _surface_range(c: np.ndarray, fraction: tuple[float, float, float, float]) -> tuple[float, float]:
    left, top, width, height = fraction
    t = np.arange(9) / 8
    value = _quadratic(c, (left + t * width)[None, :], (top + t * height)[:, None])
    return float(value.min()), float(value.max())


def _floor_grid(density: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The local floor of one density plane of the small copy: erosion by an ellipse FOG_KERNEL of its
    width, an anti-alias blur and an area resize onto the FOG_GRID_WIDTH grid. A cell with no film
    reading is left out of the minimum."""
    import negpy_core

    h, w = density.shape
    k = max(3, int(round(w * FOG_KERNEL)) | 1)
    plane = np.where(valid, density, np.inf).astype(np.float32)
    eroded = negpy_core.morphology_ex(np.ascontiguousarray(plane), negpy_core.MORPH_ERODE, negpy_core.ellipse_kernel((k, k)))
    finite = np.isfinite(eroded)
    fill = float(np.median(eroded[finite])) if finite.any() else 0.0
    eroded = np.where(finite, eroded, fill).astype(np.float32)
    blurred = negpy_core.gaussian_blur(np.ascontiguousarray(eroded), (0, 0), w * FOG_BLUR, 0.0, 1)
    grid_h = max(3, int(round(FOG_GRID_WIDTH * h / w)))
    return np.asarray(negpy_core.resize_area(np.ascontiguousarray(blurred), (FOG_GRID_WIDTH, grid_h)), dtype=np.float64)


def _small_density(linear: ImageBuffer) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """(density (h, w, 3), valid (h, w), fraction): the region inside FOG_BORDER, area-averaged in
    linear light to FOG_WORKING_WIDTH wide without the pixels past the film, then as density."""
    import negpy_core

    img = np.asarray(linear, dtype=np.float32)[..., :3]
    h, w = img.shape[:2]
    inset = int(min(h, w) * FOG_BORDER)
    region = img[inset : h - inset or None, inset : w - inset or None]
    rh, rw = region.shape[:2]
    sw = FOG_WORKING_WIDTH
    sh = max(8, int(round(rh * sw / rw)))
    film: np.ndarray = np.logical_not(np.all(region >= FOG_CLIP, axis=2))
    weight = np.ascontiguousarray(film.astype(np.float32))
    summed = negpy_core.resize_area(
        np.ascontiguousarray(np.where(film[..., None], np.maximum(region, 1e-6), 0.0).astype(np.float32)), (sw, sh)
    )
    count = np.asarray(negpy_core.resize_area(weight, (sw, sh)), dtype=np.float32)
    valid = count > 0.25
    mean = np.asarray(summed) / np.maximum(count, 1e-6)[..., None]
    density = -np.log10(np.clip(mean, 1e-6, None))
    return density, valid, (inset / w, inset / h, rw / w, rh / h)


class FogFit(NamedTuple):
    """One fog measurement and what it read, for the bake and its diagnostics. *surface* is None when
    the frame has no uneven fog or no surface fits; *reason* names why ("" with a surface). The grid
    holds the cell centres (*u*, *v*, fractions of the frame), each cell's shared floor (D), and
    which cells were coherent and clear (*kept*). *amplitude* is the fitted surface's range over its
    region before FOG_CAP, None when no surface fitted."""

    surface: Optional[FogSurface]
    reason: str
    shape: tuple[int, int]
    u: np.ndarray
    v: np.ndarray
    floor: np.ndarray
    coherent: np.ndarray
    kept: np.ndarray
    amplitude: Optional[float]

    @property
    def kept_share(self) -> float:
        return float(self.kept.mean()) if self.kept.size else 0.0

    @property
    def cap_hit(self) -> bool:
        return self.amplitude is not None and self.amplitude > FOG_CAP

    def metrics(self) -> dict:
        """The measure-only numbers a render reports."""
        return {"amplitude": self.amplitude, "cap_hit": self.cap_hit, "kept_share": self.kept_share, "reason": self.reason}


def _sides_clear(u: np.ndarray, v: np.ndarray, clear: np.ndarray, shared: np.ndarray, fraction) -> bool:
    """Whether clear cells lie on both sides of the fitted surface's gradient across the region."""
    left, top, width, height = fraction
    cu, cv = left + 0.5 * width, top + 0.5 * height
    gu = shared[1] + 2.0 * shared[3] * cu + shared[5] * cv
    gv = shared[2] + 2.0 * shared[4] * cv + shared[5] * cu
    norm = math.hypot(gu, gv)
    if norm < 1e-9:
        return True
    along = ((u[clear] - cu) * gu + (v[clear] - cv) * gv) / norm
    reach = 0.25 * math.hypot(width * gu / norm, height * gv / norm)
    return bool((along < -reach).any() and (along > reach).any())


def fit_fog(
    linear: ImageBuffer,
    plausible: float = FOG_PLAUSIBLE,
    min_cells: int = 12,
    both_sides: bool = False,
    clear_band: Optional[float] = None,
) -> FogFit:
    """The frame's uneven fog with what the fit read. *plausible* (the bend limit), *clear_band* (how far
    above the grid's thin end a clear cell may sit, *plausible* when None), *min_cells* (clear cells
    the fit needs) and *both_sides* (clear cells on both sides of the gradient) set how strictly it
    rejects content; the bake uses the defaults."""
    empty = np.zeros(0)
    density, valid, region = _small_density(linear)
    if valid.sum() < 64:
        return FogFit(None, "too little film", (0, 0), empty, empty, empty, empty.astype(bool), empty.astype(bool), None)
    left, top, width, height = region
    channel_grids = [_floor_grid(density[..., c], valid) for c in range(3)]
    gh, gw = channel_grids[0].shape
    channel_floors = [g.ravel() for g in channel_grids]
    grid_shared = _floor_grid(density.mean(axis=2), valid).ravel()
    u = np.tile(left + (np.arange(gw) + 0.5) / gw * width, gh)
    v = np.repeat(top + (np.arange(gh) + 0.5) / gh * height, gw)
    densest = np.max(channel_floors, axis=0)
    coherent = grid_shared - np.mean(channel_floors, axis=0) <= FOG_COHERENT
    band = plausible if clear_band is None else clear_band
    clear = coherent & (densest <= np.percentile(densest[coherent], FOG_CLEAR_PCT) + band) if coherent.any() else coherent

    def fit(surface, reason, amplitude=None):
        return FogFit(surface, reason, (gh, gw), u, v, grid_shared, coherent, clear, amplitude)

    if clear.sum() < min_cells:
        return fit(None, "too few clear-film cells")
    # The surface is trusted only over the cells that measured it, and held at their edges.
    cell_w, cell_h = width / gw, height / gh
    u0, u1 = float(u[clear].min()) - 0.5 * cell_w, float(u[clear].max()) + 0.5 * cell_w
    v0, v1 = float(v[clear].min()) - 0.5 * cell_h, float(v[clear].max()) + 0.5 * cell_h
    fraction = (u0, v0, u1 - u0, v1 - v0)
    left, top, width, height = fraction
    uc, vc, shared_c = u[clear], v[clear], grid_shared[clear]
    shared = None
    for dof in (6, 3):
        coef = _robust_floor(shared_c, uc, vc, dof)
        if coef is not None:
            lo, hi = _surface_range(coef, fraction)
            if hi - lo <= plausible:
                shared = coef
                break
    if shared is None:
        return fit(None, "no plausible surface")
    lo, hi = _surface_range(shared, fraction)
    if hi - lo < FOG_FLOOR:
        return fit(None, "under the floor", hi - lo)
    if both_sides and not _sides_clear(u, v, clear, shared, fraction):
        return fit(None, "clear film on one side only", hi - lo)
    planes = []
    for c in range(3):
        coef = _robust_floor(channel_floors[c][clear], uc, vc, 3)
        planes.append(np.zeros(3) if coef is None else coef[:3])
    color = np.asarray(planes) - np.mean(planes, axis=0)
    color[:, 0] = 0.0
    color *= FOG_COLOR_SHRINK
    for c in range(3):
        span = abs(color[c, 1]) * width + abs(color[c, 2]) * height
        if span > FOG_COLOR_CAP:
            color[c] *= FOG_COLOR_CAP / span
    # Each plane is zero at the region's centre, so it moves no channel there.
    cu, cv = left + 0.5 * width, top + 0.5 * height
    color[:, 0] = -(color[:, 1] * cu + color[:, 2] * cv)
    surface = FogSurface(tuple(float(x) for x in shared), tuple(tuple(float(x) for x in p) for p in color), lo, fraction)
    return fit(surface, "", hi - lo)


def measure_fog(linear: ImageBuffer) -> Optional[FogSurface]:
    """The frame's uneven fog, or None when it is under FOG_FLOOR or no surface fits."""
    return fit_fog(linear).surface


def fog_map(linear: ImageBuffer, surface: FogSurface) -> np.ndarray:
    """The shared density the bake removes at each pixel of *linear* (H, W), capped as `remove_fog` caps it."""
    h, w = np.asarray(linear).shape[:2]
    left, top, width, height = surface.fraction
    u = np.clip((np.arange(w, dtype=np.float64) + 0.5) / w, left, left + width)[None, :]
    v = np.clip((np.arange(h, dtype=np.float64) + 0.5) / h, top, top + height)[:, None]
    return np.clip(_quadratic(np.asarray(surface.shared), u, v) - surface.offset, 0.0, FOG_CAP)


def remove_fog(linear: ImageBuffer, surface: FogSurface) -> ImageBuffer:
    """*linear* with the surface's uneven part taken out of every pixel's density: the shared surface
    above its minimum, capped at FOG_CAP, plus the channel's color plane, held at the region's edges."""
    img = np.asarray(linear, dtype=np.float32)
    h, w = img.shape[:2]
    left, top, width, height = surface.fraction
    u = np.clip((np.arange(w, dtype=np.float64) + 0.5) / w, left, left + width)[None, :]
    v = np.clip((np.arange(h, dtype=np.float64) + 0.5) / h, top, top + height)[:, None]
    uneven = np.clip(_quadratic(np.asarray(surface.shared), u, v) - surface.offset, 0.0, FOG_CAP)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        if c < 3:
            p = surface.color[c]
            gain = np.power(10.0, uneven + (p[0] + p[1] * u + p[2] * v)).astype(np.float32)
            out[..., c] = img[..., c] * gain
        else:
            out[..., c] = img[..., c]
    return out


def bake_fog(linear: ImageBuffer, process) -> ImageBuffer:
    """The NegPy fog bake: *linear* itself when it does not run or the frame has no uneven fog."""
    return bake_fog_with_fit(linear, process)[0]


def bake_fog_with_fit(linear: ImageBuffer, process) -> tuple[ImageBuffer, Optional[FogFit]]:
    """`bake_fog` and its measurement, None when the bake does not run."""
    if not fog_active(process):
        return linear, None
    fit = fit_fog(linear)
    return (linear if fit.surface is None else remove_fog(linear, fit.surface)), fit
