"""Gray finders for the cold-lean study (phase 4): a NumPy copy of NegPy's `neutral_axis` (negpy_core) with
each suspect switchable, so one change at a time shows how far it moves the midtone gray.

`Finder()` with no options is NegPy's finder; `check_replica` asserts it matches negpy_core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.normalization import LUMA_B, LUMA_G, LUMA_R, _block_median_grid
from negpy.features.process.fade import (
    CLIPPED_LOG,
    COLOR_EXTREME_PCT,
    CROSSOVER_BAND_EDGES,
    CROSSOVER_RANGE_PCT,
    _measured_slide_bounds,
)

C = EXPOSURE_CONSTANTS


@dataclass(frozen=True)
class Finder:
    """One gray finder. The defaults are NegPy's.

    subjects: drop foliage- and sky-sector cells first.
    caps:     reject a band whose near-neutral set's median chroma passes the caps.
    soft:     None for NegPy's hard set (the lowest-chroma 30 %, medians); else the spread (D) of
              NC's soft weight 1/(1 + ((max - min) / soft)^2), with weighted means and NC's mean-shift.
    bands:    3 for NegPy's normalized-luma bands; 8 for NC's bands over the mean log (mid = bands 3
              and 4, shadow = 1 and 2, highlight = 5 and 6).
    raw:      measure chroma as the raw log spread (max - min) instead of the per-channel stretch.
    """

    subjects: object = True  # True, False, "foliage" or "sky": the sectors dropped
    caps: bool = True
    soft: Optional[float] = None
    bands: int = 3
    raw: bool = False

    def label(self) -> str:
        parts = []
        if not self.subjects:
            parts.append("no subject sectors")
        elif self.subjects is not True:
            parts.append(f"{self.subjects} sector only")
        if not self.caps:
            parts.append("no chroma caps")
        if self.soft is not None:
            parts.append(f"soft weight {self.soft:g} D")
        if self.bands != 3:
            parts.append(f"{self.bands} bands")
        if self.raw:
            parts.append("raw-log chroma")
        return ", ".join(parts) or "NegPy"


def _normalize(grid: np.ndarray, bounds) -> np.ndarray:
    f = np.asarray(bounds.floors, dtype=np.float32)
    d = np.asarray(bounds.ceils, dtype=np.float32) - f
    d = np.where(np.abs(d) < 1e-6, np.where(d >= 0, 1e-6, -1e-6), d)
    return ((grid - f).astype(np.float64) / d.astype(np.float64)).astype(np.float32)


def _is_subject(n: np.ndarray, which: object = True) -> np.ndarray:
    dr, db = n[:, 0] - n[:, 1], n[:, 2] - n[:, 1]
    t, s = (dr + db) / 2.0, (dr - db) / 2.0
    chroma = np.hypot(t.astype(np.float64), s.astype(np.float64)).astype(np.float32)
    hue = np.degrees(np.arctan2(-s, t))
    out = np.zeros(len(n), bool)
    for key in ("foliage", "sky") if which is True else (which,):
        lo, hi = C[f"neutral_axis_{key}_hue"]
        out |= (chroma > float(C[f"neutral_axis_{key}_chroma"])) & (hue > lo) & (hue < hi)
    return out


def _rms(n: np.ndarray) -> np.ndarray:
    r, g, b = n[:, 0], n[:, 1], n[:, 2]
    return np.sqrt(((r - g) ** 2 + (g - b) ** 2 + (r - b) ** 2) / 3.0)


def _soft_refs(log: np.ndarray, sel: np.ndarray, spread_d: float) -> Optional[np.ndarray]:
    from negpy.features.process.fade import CROSSOVER_CLUSTER_RADIUS, CROSSOVER_MEAN_SHIFT_STEPS

    cells = log[sel].astype(np.float64)
    if len(cells) < int(C["neutral_axis_min_pixels"]):
        return None
    s = (cells.max(axis=1) - cells.min(axis=1)) / spread_d
    w = 1.0 / (1.0 + s * s)
    centre = (w[:, None] * cells).sum(axis=0) / w.sum()
    for _ in range(CROSSOVER_MEAN_SHIFT_STEPS):
        d = np.abs(cells - centre).max(axis=1) / CROSSOVER_CLUSTER_RADIUS
        ww = w / (1.0 + d * d)
        centre = (ww[:, None] * cells).sum(axis=0) / ww.sum()
    return centre


def find(grid: np.ndarray, finder: Finder = Finder()) -> Optional[tuple]:
    """(mid, shadow, highlight or None) log refs, or None."""
    grid = np.ascontiguousarray(_block_median_grid(grid), dtype=np.float32)
    bounds = _measured_slide_bounds(grid, COLOR_EXTREME_PCT)
    log = grid.reshape(-1, 3)
    norm = _normalize(grid, bounds).reshape(-1, 3)
    if finder.subjects:
        keep = ~_is_subject(norm, finder.subjects)
        log, norm = log[keep], norm[keep]
    q, cap, cap1 = (float(C[k]) for k in ("neutral_axis_chroma_quantile", "neutral_axis_chroma_cap", "neutral_axis_first_pass_cap"))
    if not finder.caps:
        cap = cap1 = np.inf
    n_min = int(C["neutral_axis_min_pixels"])
    if finder.bands == 3:
        lum = LUMA_R * norm[:, 0] + LUMA_G * norm[:, 1] + LUMA_B * norm[:, 2]
        band_sel = {
            k: (lum >= lo) & (lum <= hi)
            for k, (lo, hi) in (
                ("mid", C["neutral_axis_mid_band"]),
                ("shadow", C["neutral_axis_shadow_band"]),
                ("highlight", C["neutral_axis_highlight_band"]),
            )
        }
    else:
        live = (log > CLIPPED_LOG).all(axis=1) & (log < 0.0).any(axis=1)
        mean = log.astype(np.float64).mean(axis=1)
        lo, hi = np.percentile(mean[live], CROSSOVER_RANGE_PCT)
        t = np.clip((mean - lo) / max(hi - lo, 0.05), 0.0, 1.0)
        band = np.searchsorted(np.asarray(CROSSOVER_BAND_EDGES[1:-1]), t, side="right")
        band_sel = {k: live & np.isin(band, idx) for k, idx in (("mid", (3, 4)), ("shadow", (1, 2)), ("highlight", (5, 6)))}

    def hard(sel: np.ndarray, chroma: np.ndarray, cap_: float) -> Optional[np.ndarray]:
        idx = np.nonzero(sel)[0]
        if len(idx) < n_min:
            return None
        thr = np.quantile(chroma[idx], q)
        idx = idx[chroma[idx] <= thr]
        if len(idx) < n_min or float(np.median(chroma[idx])) > cap_:
            return None
        g = float(np.median(log[idx, 1]))
        return np.array([g + float(np.median(log[idx, 0] - log[idx, 1])), g, g + float(np.median(log[idx, 2] - log[idx, 1]))])

    if finder.soft is not None:
        refs = {k: _soft_refs(log, s, finder.soft) for k, s in band_sel.items()}
    elif finder.raw:
        spread = (log.max(axis=1) - log.min(axis=1)).astype(np.float32)
        refs = {k: hard(s, spread, np.inf) for k, s in band_sel.items()}
    else:
        chroma1 = _rms(norm)
        mid1, sh1 = hard(band_sel["mid"], chroma1, cap1), hard(band_sel["shadow"], chroma1, cap1)
        if mid1 is None or sh1 is None:
            return None
        f, c = np.asarray(bounds.floors, np.float64), np.asarray(bounds.ceils, np.float64)
        nm, ns = (mid1 - f) / (c - f), (sh1 - f) / (c - f)
        n2 = norm.copy()
        for ch in (0, 2):
            du = nm[ch] - ns[ch]
            a = 1.0 if abs(du) < 1e-6 else (nm[1] - ns[1]) / du
            b = nm[1] - nm[ch] if abs(du) < 1e-6 else nm[1] - a * nm[ch]
            n2[:, ch] = np.float32(a) * norm[:, ch] + np.float32(b)
        chroma2 = _rms(n2)
        refs = {k: hard(s, chroma2, cap) for k, s in band_sel.items()}
    if refs["mid"] is None or refs["shadow"] is None:
        return None
    return refs["mid"], refs["shadow"], refs["highlight"]


def check_replica(grid: np.ndarray, atol: float = 2e-4) -> None:
    from negpy.features.exposure.normalization import measure_neutral_axis_from_log

    core = measure_neutral_axis_from_log(grid, _measured_slide_bounds(grid, COLOR_EXTREME_PCT), None, 0.0)
    mine = find(grid)
    assert (core is None) == (mine is None), (core, mine)
    if core is not None and mine is not None:
        for a, b in zip(core[:2], mine[:2]):
            assert np.allclose(a, b, atol=atol), (a, b)
