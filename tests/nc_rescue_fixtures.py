"""Inputs for the NegativeConverter port check: 16-bit RGB positives that experiments/fade_crossover/
port_check runs through their expiredRescue.js under Node, and tests/test_nc_rescue.py through ours."""

from __future__ import annotations

import numpy as np

W, H = 256, 96
#: The ageing model of their expiredRescue.test.mjs: fog, top of range and gamma per channel.
AGE_FOG = (0.30, 0.36, 0.42)
AGE_TOP = (0.86, 0.76, 0.90)
AGE_GAMMA = (0.85, 1.05, 0.95)


def _their_scene() -> np.ndarray:
    """The scene of their test: a textured neutral ramp, an orange patch, grass and a dark corner (8-bit)."""
    out = np.zeros((H, W, 3), dtype=np.int64)
    for y in range(H):
        for x in range(W):
            if y < 64:
                v = round(255 * (x / (W - 1)) ** 1.1)
                tex = ((x * 7 + y * 13) % 11) - 5
                rgb = (v + tex,) * 3
            elif x < 96:
                rgb = (190, 110, 40)
            elif x < 176:
                rgb = (60, 140, 50)
            else:
                rgb = (30, 30, 30)
            out[y, x] = rgb
    return np.clip(out, 0, 255)


def faded_positive() -> np.ndarray:
    """Their scene aged by their model without the crossover term, as a uint16 plane."""
    x = _their_scene() / 255.0
    y = np.stack([AGE_FOG[c] + (AGE_TOP[c] - AGE_FOG[c]) * np.power(x[..., c], AGE_GAMMA[c]) for c in range(3)], axis=-1)
    return np.round(np.clip(y, 0.0, 1.0) * 65535.0).astype(np.uint16)


def crossover_positive() -> np.ndarray:
    """The crossover film of `fade_crossover_scene` as a display-encoded positive, uint16."""
    from fade_crossover_scene import CROSSOVER, build_scene, scan_image

    t = scan_image(CROSSOVER, build_scene(W, H))
    return np.round(np.clip(np.power(t, 1.0 / 2.2), 0.0, 1.0) * 65535.0).astype(np.uint16)


#: Pixels sampled for the comparison: (y, x).
PROBES = tuple((y, x) for y in (5, 30, 60, 70, 90) for x in (3, 40, 100, 150, 200, 250))

CASES = {"faded": faded_positive, "crossover": crossover_positive}
#: Stage variants: (colour table, tone curve).
VARIANTS = {"colour": (True, False), "tone": (False, True), "both": (True, True)}


FOG_W, FOG_H = 240, 160


def _fog_at(x: np.ndarray) -> np.ndarray:
    """Their test's fog: a uniform floor and a gradient toward the left edge (light piping)."""
    return 0.10 + 0.12 * (1 - x / (FOG_W - 1))


def fog_wall_positive() -> np.ndarray:
    """Their expiredRescueOpenCv.test.mjs scene, a textured neutral ramp with a bright wall in the
    lower right, aged by their fog, as a uint16 plane."""
    y, x = np.mgrid[0:FOG_H, 0:FOG_W]
    v = 255.0 * ((x * 7 + y * 3) % 40) / 39.0
    v = np.where((x > FOG_W * 0.6) & (y > FOG_H * 0.6), 220.0, v)
    scene = np.round(v)
    f = _fog_at(x)
    aged = np.round((f + (0.9 - f) * (scene / 255.0)) * 255.0)
    return np.round(np.repeat(aged[..., None], 3, axis=-1) / 255.0 * 65535.0).astype(np.uint16)


def coolscan_clean_positive() -> np.ndarray:
    """A clean narrowband slide scan with no fog: deep shadows, a large warm area and a gray ramp,
    from the film model of `fade_crossover_scene`, display-encoded, uint16."""
    from fade_crossover_scene import REFERENCE, build_scene, coolscan, scan_image

    scene = build_scene(FOG_W, FOG_H)
    log_e = scene.log_e.copy()
    yy, xx = np.mgrid[0:FOG_H, 0:FOG_W]
    # A warm wall over the right half of the lower part, and a deep shadow along the bottom left.
    log_e[(yy > FOG_H // 3) & (xx > FOG_W // 2)] = (-0.15, -0.55, -1.25)
    log_e[(yy > FOG_H * 3 // 4) & (xx < FOG_W // 3)] = (-2.6, -2.6, -2.6)
    t = scan_image(coolscan(REFERENCE), type(scene)(log_e, scene.ramp, scene.patches))
    return np.round(np.clip(np.power(t, 1.0 / 2.2), 0.0, 1.0) * 65535.0).astype(np.uint16)


FOG_CASES = {"fog_wall": fog_wall_positive, "coolscan_clean": coolscan_clean_positive}
FOG_PROBES = tuple((y, x) for y in (6, 40, 80, 120, 150) for x in (2, 50, 120, 170, 236))
#: Fog variants: (colour table, tone curve) after the fog stage.
FOG_VARIANTS = {"fog": (False, False), "fog_both": (True, True)}
