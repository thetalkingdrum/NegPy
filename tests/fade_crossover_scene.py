"""A slide scene and film model for the Fade Restoration crossover table, built apart from the table's
band model: each dye layer has its own logistic characteristic curve, so a fade bends the gray scale
differently at each density, which no band offset made it."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

#: Log10 scene exposure of each patch's three channels, relative to a mid gray at 0.
SKY = (-0.45, -0.3, -0.1)
FOLIAGE = (-0.85, -0.6, -1.1)
SKIN = (-0.22, -0.38, -0.48)
RED_SUBJECT = (-0.2, -1.2, -1.3)
#: Patch name -> scene log exposure, and the offsets each patch is shown at.
PATCHES = {"sky": SKY, "foliage": FOLIAGE, "skin": SKIN}
PATCH_STOPS = (-0.6, 0.0, 0.45)

RAMP_STEPS = 32
#: Scene log exposure of the gray ramp, darkest step first.
RAMP_RANGE = (-2.4, 0.9)


@dataclass(frozen=True)
class Layer:
    """One dye layer's logistic characteristic curve in log exposure: density falls from d_max to
    d_min around e0, with gamma as the slope at e0 (in D per log E)."""

    d_min: float = 0.12
    d_max: float = 3.3
    gamma: float = 1.7
    e0: float = -0.6

    def density(self, log_e: np.ndarray) -> np.ndarray:
        span = self.d_max - self.d_min
        return self.d_min + span / (1.0 + np.exp(4.0 * self.gamma / span * (log_e - self.e0)))


@dataclass(frozen=True)
class Film:
    red: Layer = Layer()
    green: Layer = Layer()
    blue: Layer = Layer()

    def scan(self, log_e: np.ndarray) -> np.ndarray:
        """Linear transmittance of the three layers for (..., 3) scene log exposure."""
        d = np.stack([layer.density(log_e[..., c]) for c, layer in enumerate((self.red, self.green, self.blue))], axis=-1)
        return np.power(10.0, -d)


REFERENCE = Film()
#: A crossover fade: every layer keeps its own toe and shoulder, so the lean changes with density.
CROSSOVER = Film(
    red=Layer(d_max=3.2, gamma=1.65, e0=-0.6),
    green=Layer(d_max=2.8, gamma=1.4, e0=-0.5),
    blue=Layer(d_max=2.5, gamma=1.85, e0=-0.72),
)


def linear_fade(film: Film, survivals: tuple[float, float, float]) -> "ScaledFilm":
    """*film* with each layer's whole density scaled: the fade the survivals model exactly."""
    return ScaledFilm(film, survivals)


@dataclass(frozen=True)
class ScaledFilm:
    film: Film
    survivals: tuple[float, float, float]

    def scan(self, log_e: np.ndarray) -> np.ndarray:
        d = -np.log10(self.film.scan(log_e))
        return np.power(10.0, -d * np.asarray(self.survivals))


#: A broadband (white LED) scanner: each channel reads 15 to 25 % of its neighbors' dyes.
BROADBAND_MIX = ((0.80, 0.20, 0.00), (0.18, 0.64, 0.18), (0.00, 0.22, 0.78))
#: Log offset of clear film below full scale on a raw Plustek / SilverFast HDR file: 1.3 stops.
PLUSTEK_WHITE = -1.3 * float(np.log10(2.0))


@dataclass(frozen=True)
class ScannerFilm:
    """*film* as a scanner reads it: each channel's density a *mix* of the dye densities, and clear
    film *white* (log) below full scale."""

    film: object
    mix: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    white: float = 0.0

    def scan(self, log_e: np.ndarray) -> np.ndarray:
        d = -np.log10(self.film.scan(log_e))
        return np.power(10.0, -(d @ np.asarray(self.mix).T) + self.white)


def plustek(film) -> ScannerFilm:
    return ScannerFilm(film, BROADBAND_MIX, PLUSTEK_WHITE)


def coolscan(film) -> ScannerFilm:
    return ScannerFilm(film)


@dataclass(frozen=True)
class Scene:
    """Scene log exposure (H, W, 3) and the masks of its regions."""

    log_e: np.ndarray
    ramp: list[np.ndarray]  # one mask per ramp step, darkest first
    patches: dict[str, np.ndarray]
    subject: Optional[np.ndarray] = None
    groups: Optional[tuple[np.ndarray, np.ndarray]] = None


def build_scene(width: int = 384, height: int = 288, subject: bool = False, opposite_groups: bool = False) -> Scene:
    """The gray ramp on the top band, colored patches below it, and, with *subject*, a saturated red
    subject over a large part of the frame. *opposite_groups* adds two near-neutral groups at one
    exposure that lean opposite ways in green, the larger one magenta."""
    log_e = np.zeros((height, width, 3))
    ramp_h = height // 3
    steps = np.linspace(*RAMP_RANGE, RAMP_STEPS)
    edges = np.linspace(0, width, RAMP_STEPS + 1).astype(int)
    ramp = []
    for i, e in enumerate(steps):
        mask = np.zeros((height, width), bool)
        mask[:ramp_h, edges[i] : edges[i + 1]] = True
        log_e[mask] = e
        ramp.append(mask)
    rest = np.zeros((height, width), bool)
    rest[ramp_h:] = True
    # The ground behind the patches is a second, darker gray scale, as a real frame holds grays at
    # many densities.
    cols = np.linspace(RAMP_RANGE[0] + 0.3, RAMP_RANGE[1] - 0.3, width)
    log_e[ramp_h:] = cols[None, :, None]
    patches: dict[str, np.ndarray] = {}
    pw, ph = width // 10, height // 9
    y0 = ramp_h + ph // 2
    for row, (name, colour) in enumerate(PATCHES.items()):
        for col, stop in enumerate(PATCH_STOPS):
            mask = np.zeros((height, width), bool)
            y = y0 + row * (ph + ph // 3)
            x = width // 20 + col * (pw + pw // 3)
            mask[y : y + ph, x : x + pw] = True
            log_e[mask] = np.asarray(colour) + stop
            patches[f"{name} {stop:+.2f}"] = mask
    subj = None
    if subject:
        subj = np.zeros((height, width), bool)
        subj[ramp_h + 4 :, width // 2 :] = True
        log_e[subj] = RED_SUBJECT
    groups = None
    if opposite_groups:
        big = np.zeros((height, width), bool)
        small = np.zeros((height, width), bool)
        big[ramp_h + 4 :, width // 2 : width // 2 + width // 3] = True
        small[ramp_h + 4 :, width // 2 + width // 3 :] = True
        # A green offset in exposure leaves a lean in density near the curve's middle.
        log_e[big] = (-0.5, -0.62, -0.5)
        log_e[small] = (-0.5, -0.38, -0.5)
        groups = (big, small)
    return Scene(log_e, ramp, patches, subj, groups)


def scan_image(film, scene: Scene, noise: float = 0.004, seed: int = 0) -> np.ndarray:
    """The film's linear scan of *scene*, float32, with a little log noise as grain."""
    rng = np.random.default_rng(seed)
    t = film.scan(scene.log_e) * np.power(10.0, rng.normal(0.0, noise, scene.log_e.shape))
    return np.clip(t, 1e-6, 1.0).astype(np.float32)


def render(image: np.ndarray, process_changes: dict) -> np.ndarray:
    """The CPU engine's display-encoded render of a slide *image* with *process_changes* (a scanner
    file passes positive_source=True, as it opens)."""
    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.models import ProcessMode, mode_aware_exposure_reset
    from negpy.services.rendering.engine import DarkroomEngine

    base = WorkspaceConfig()
    process = replace(base.process, process_mode=ProcessMode.E6, **process_changes)
    # A slide's own defaults: no meter and no Cast Removal, so nothing else reads the frame.
    settings = replace(base, process=process, exposure=mode_aware_exposure_reset(ProcessMode.E6, base.exposure))
    return np.asarray(DarkroomEngine().process(image, settings, source_hash="fade-crossover"))


def to_lab(encoded: np.ndarray) -> np.ndarray:
    from negpy.kernel.image.logic import rgb_to_lab_working, working_oetf_decode

    return rgb_to_lab_working(working_oetf_decode(np.ascontiguousarray(encoded[..., :3], dtype=np.float32)))


def region_lab(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean Lab of a region's inner part, away from edges the engine blurs."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    my, mx = max(1, (y1 - y0) // 4), max(1, (x1 - x0) // 4)
    return lab[y0 + my : y1 - my + 1, x0 + mx : x1 - mx + 1].reshape(-1, 3).mean(axis=0)


def chroma(lab: np.ndarray) -> float:
    return float(np.hypot(lab[1], lab[2]))


def hue(lab: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(lab[2], lab[1])))


def estimate_grid(image: np.ndarray, **process) -> np.ndarray:
    """The grid the panel's Estimate reads: the slide base stage's own, with the fade at identity."""
    from negpy.domain.interfaces import PipelineContext
    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.models import ProcessMode
    from negpy.services.rendering.engine import base_processor

    base = WorkspaceConfig()
    settings = replace(base, process=replace(base.process, process_mode=ProcessMode.E6, **process))
    ctx = PipelineContext(original_size=image.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6, wants_fade_grid=True)
    base_processor(settings).process(image, ctx)
    return ctx.metrics["transfer_grid"]


def estimate(image: np.ndarray, **process):
    """Fade Estimate (the Midtone method, the panel's default) on *image*'s analysis grid, with the
    render's white space as the panel passes it."""
    from negpy.features.process.fade import estimate_fade, render_white_space
    from negpy.features.process.models import ProcessConfig, ProcessMode

    space = render_white_space(replace(ProcessConfig(), process_mode=ProcessMode.E6, **process), ProcessMode.E6, None, None)
    return estimate_fade(estimate_grid(image, **process), white_space=space)


def survival_changes(est, crossover: float, with_table: bool = True, **process_fields) -> dict:
    """The process fields of an Estimate, with its white and table tagged as the Estimate button
    stores them, for a scanner file with *process_fields* (Positive, a scanner zero)."""
    from negpy.features.process.fade import crossover_space, fade_density, render_white_space
    from negpy.features.process.models import ProcessConfig, ProcessMode

    changes = dict(
        process_fields,
        fade_ratio_r=1.0 if est.red is None else est.red,
        fade_ratio_g=est.green,
        fade_ratio_b=est.blue,
        fade_crossover=crossover,
    )
    process = replace(ProcessConfig(), process_mode=ProcessMode.E6, **changes)
    space = render_white_space(process, ProcessMode.E6, None, None)
    if space and est.white is not None:
        white = dict(zip(("fade_white_r", "fade_white_g", "fade_white_b"), est.white), fade_white_space=space)
        changes.update(white)
        process = replace(process, **white)
    if with_table and est.crossover is not None:
        changes.update(
            fade_crossover_table=est.crossover,
            fade_crossover_space=crossover_space(fade_density(process, ProcessMode.E6, space), space, process),
        )
    return changes


def ramp_chroma(lab: np.ndarray, scene: Scene, bands: int = 8) -> list[tuple[float, float]]:
    """(max, mean) C* of the ramp steps in *bands* equal groups, darkest first."""
    c = np.array([chroma(region_lab(lab, m)) for m in scene.ramp])
    return [(float(g.max()), float(g.mean())) for g in np.array_split(c, bands)]


@dataclass(frozen=True)
class FogScene:
    """A scanner file of the unfaded film with and without uneven fog, and its gray cells."""

    fogged: np.ndarray
    clean: np.ndarray
    gray: np.ndarray


#: A saturated blue sky, as a narrowband scanner separates it.
SKY_BLUE = (-0.9, -0.5, -0.1)
WARM = (-0.15, -0.55, -1.25)


def fog_scene(layout: str = "fog", width: int = 384, height: int = 256) -> FogScene:
    """Scenes on the unfaded film, scanned at full scale with no channel mix (Coolscan-like).

    "fog": NegativeConverter's fog test scene, a textured gray ramp over the whole range with a large
    light-gray wall in the lower right that must not read as fog, under fog density that falls from
    the left edge and a bump in the top-right corner. "warm" and "sky" hold no fog: a light gray wall
    with a dark doorway, under a large warm area or a blue sky over the top third. The clean scan
    holds the fog's even part, which no stage removes."""
    y, x = np.mgrid[0:height, 0:width]
    if layout == "fog":
        t = ((x * 7 + y * 3) % 40) / 39.0
        log_e = np.repeat((RAMP_RANGE[0] + t * (RAMP_RANGE[1] - RAMP_RANGE[0]))[..., None], 3, axis=-1)
        other = (x > width * 0.6) & (y > height * 0.6)
        log_e[other] = -0.3
    else:
        log_e = np.full((height, width, 3), -0.6)
        door = (y > height * 0.7) & (np.abs(x - width * 0.5) < width * 0.08)
        log_e[door] = -2.6
        if layout == "warm":
            other = (x > width * 0.5) & (y > height * 0.33)
        else:
            other = y < height * 0.35
        log_e[other] = WARM if layout == "warm" else SKY_BLUE
        other = other | door
    density = -np.log10(REFERENCE.scan(log_e))
    if layout == "fog":
        corner = np.exp(-(((x - width) / (0.25 * width)) ** 2 + (y / (0.25 * width)) ** 2))
        fog = 0.03 + 0.12 * (1.0 - x / (width - 1)) + 0.08 * corner
    else:
        fog = np.full((height, width), 0.03)
    grain = np.random.default_rng(1).normal(0.0, 0.004, density.shape)

    def scan(d: np.ndarray) -> np.ndarray:
        return np.clip(np.power(10.0, -(d + grain)), 1e-6, 1.0).astype(np.float32)

    return FogScene(scan(density + fog[..., None]), scan(density + float(fog.min())), ~other)


#: Log exposure offsets (r, g, b) of the warm near-neutrals, per unit of warmth: a painted wall, sand, skin.
WALL = (0.03, 0.0, -0.05)
SAND = (0.04, -0.02, -0.14)


@dataclass(frozen=True)
class WarmScene:
    """Scene log exposure (H, W, 3), the true gray ramp's steps and the named regions."""

    log_e: np.ndarray
    ramp: list[np.ndarray]
    regions: dict[str, np.ndarray]


def warm_scene(warmth: float = 1.0, gray_share: float = 0.1, layout: str = "wall", width: int = 384, height: int = 288) -> WarmScene:
    """A frame whose near-neutrals are mostly warm, with a true gray ramp over *gray_share* of the height.

    "wall": a large wall in light and shade, sand, skin, and some sky and foliage.
    "sun": gray objects lit warm by the sun and cool by the sky in the shade, the ramp only over the
    midtones; the frame's extremes are a warm highlight and a cool shadow.
    "sky": the wall layout under a saturated blue sky over the top 40 %, the case the sky sector is for.
    "bright": the wall layout with the ramp only over the midtones and a warm sunlit area as the
    frame's brightest part, so the warm content sets the frame's own stretch.
    *warmth* scales the colored offsets; 0 makes the wall, sand and sun/shade objects true grays."""
    log_e = np.zeros((height, width, 3))
    ramp_h = max(8, int(height * gray_share))
    steps = np.linspace(-2.4, 0.9, 16) if layout == "wall" else np.linspace(-1.4, 0.2, 16)
    edges = np.linspace(0, width, len(steps) + 1).astype(int)
    ramp = []
    for i, e in enumerate(steps):
        m = np.zeros((height, width), bool)
        m[:ramp_h, edges[i] : edges[i + 1]] = True
        log_e[m] = e
        ramp.append(m)
    y, x = np.mgrid[0:height, 0:width]
    body = y >= ramp_h
    regions: dict[str, np.ndarray] = {}
    if layout == "sun":
        # Gray objects across the frame: the left half in sun (warm, bright), the right half in shade.
        level = -0.9 + 1.6 * ((x * 7 + y * 5) % 37) / 36.0
        lit = body & (x < width * 0.5)
        shade = body & ~lit
        log_e[lit] = (level[lit] + 0.5)[:, None] + warmth * np.asarray(WALL)
        log_e[shade] = (level[shade] - 0.9)[:, None] - warmth * np.asarray(WALL)
        regions["wall"] = lit
        regions["shade"] = shade
        return WarmScene(log_e, ramp, regions)
    # The wall: the left 55 %, lit to shaded from top to bottom.
    wall = body & (x < width * 0.55)
    shade = -1.6 + 2.0 * (1.0 - (y - ramp_h) / max(1, height - ramp_h))
    log_e[wall] = shade[wall][:, None] + warmth * np.asarray(WALL)
    regions["wall"] = wall
    sand = body & (x >= width * 0.55) & (y >= height * 0.7)
    top = 1.2 if layout == "bright" else 0.3
    log_e[sand] = (-0.3 + (top + 0.3) * (x[sand] - width * 0.55) / (width * 0.45))[:, None] + warmth * np.asarray(SAND)
    regions["sand"] = sand
    skin = body & (x >= width * 0.55) & (x < width * 0.75) & (y < height * 0.7)
    log_e[skin] = SKIN
    regions["skin"] = skin
    sky = body & (x >= width * 0.75) & (y < height * 0.45)
    log_e[sky] = SKY
    if layout == "sky":
        sky = body & (y < height * 0.4)
        log_e[sky] = SKY_BLUE
    regions["sky"] = sky
    foliage = body & (x >= width * 0.75) & (y >= height * 0.45) & (y < height * 0.7)
    log_e[foliage] = FOLIAGE
    regions["foliage"] = foliage
    return WarmScene(log_e, ramp, regions)


def scan_warm(film, scene: WarmScene) -> np.ndarray:
    """The film's linear scan of a warm scene, as `scan_image` does."""
    rng = np.random.default_rng(0)
    t = film.scan(scene.log_e) * np.power(10.0, rng.normal(0.0, 0.004, scene.log_e.shape))
    return np.clip(t, 1e-6, 1.0).astype(np.float32)
