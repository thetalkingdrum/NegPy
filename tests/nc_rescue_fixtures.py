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
