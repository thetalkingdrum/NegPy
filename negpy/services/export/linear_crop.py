"""Apply crop for Linear Output: the frame's crop as a slice of the unresampled buffer.

The crop rect is normalized in the engine's transformed frame: rotation, flips, fine
rotation, distortion and Tilt/Swing. The linear buffer carries only the rotation and flips,
plus the lens warps when Apply lens correction is on. The rest is mapped back, and the slice
is the bounding box of the mapped crop, so the file keeps the tilt.
"""

import math
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from negpy.features.geometry.logic import (
    autocrop_detection_key,
    compute_geometry_crop_rect,
    get_manual_rect_coords,
    keystone_matrix,
    radial_source_points,
)
from negpy.features.geometry.models import GeometryConfig
from negpy.features.lens.models import LensCorrections, LensMetadata

ROI = tuple[int, int, int, int]
# (N, 2) edge-based x, y points in, the same points in another frame out.
PointMap = Callable[[np.ndarray], np.ndarray]

_EDGE_SAMPLES = 64
# Covers the half-pixel difference between the edge-based and index-based point
# conventions of the mapped transforms, and the rounding of the embedded warp lookup.
_PAD_PX = 2

_EXIF_OPS: dict[int, tuple[str, ...]] = {
    2: ("fliplr",),
    3: ("rot90", "rot90"),
    4: ("flipud",),
    5: ("transpose",),
    6: ("rot90", "rot90", "rot90"),
    7: ("transpose", "rot90", "rot90"),
    8: ("rot90",),
}


def crop_unresolved(geometry: GeometryConfig) -> bool:
    """An armed Auto Crop that no preview has resolved: its rect is missing or stale."""
    return geometry.crop_from_auto and (geometry.crop_rect is None or geometry.crop_detect_key != autocrop_detection_key(geometry))


def dihedral_matrix(ops: Sequence[str], h: int, w: int) -> tuple[np.ndarray, tuple[int, int]]:
    """3x3 map of an edge-based point through numpy rot90/flips/transpose, and the final (h, w)."""
    m = np.eye(3)
    for op in ops:
        if op == "rot90":
            step = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, w], [0.0, 0.0, 1.0]])
            h, w = w, h
        elif op == "fliplr":
            step = np.array([[-1.0, 0.0, w], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        elif op == "flipud":
            step = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, h], [0.0, 0.0, 1.0]])
        elif op == "transpose":
            step = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            h, w = w, h
        else:
            raise ValueError(op)
        m = step @ m
    return m, (h, w)


def user_geometry_ops(geometry: GeometryConfig) -> tuple[str, ...]:
    """The array ops of `_apply_user_geometry`, in order."""
    return ("rot90",) * (geometry.rotation % 4) + ("fliplr",) * geometry.flip_horizontal + ("flipud",) * geometry.flip_vertical


def _transform(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ m.T
    return h[:, :2] / h[:, 2:3]


def _lens_source(lens: LensMetadata, shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    """Where corrected sensor points sample the uncorrected sensor, through the distortion warps."""
    h, w = shape
    distortion = LensCorrections(distortion=True)
    for warp in reversed(lens.warps):
        if not warp.has_distortion:
            continue
        rows = np.clip(np.floor(pts[:, 1]).astype(int), 0, h - 1)
        cols = np.clip(np.floor(pts[:, 0]).astype(int), 0, w - 1)
        out = np.empty_like(pts)
        for row in np.unique(rows):
            mx, my = warp.remap(lens, (h, w), int(row), int(row) + 1, 1, distortion)
            sel = rows == row
            out[sel, 0] = mx[0, cols[sel]] + 0.5
            out[sel, 1] = my[0, cols[sel]] + 0.5
        pts = out
    return pts


def embedded_unwarp(
    lens: LensMetadata,
    orientation: int,
    source_shape: tuple[int, int],
    offset: tuple[int, int],
    half_shape: tuple[int, int],
    geometry: GeometryConfig,
) -> PointMap:
    """Map points in the post-geometry frame from the embedded-corrected image to the uncorrected one.

    ``source_shape`` is the EXIF-oriented decode, ``offset`` and ``half_shape`` the (y, x) origin
    and size of the half-frame slice taken from it.
    """
    ops = _EXIF_OPS.get(orientation, ())
    sensor_shape = (source_shape[1], source_shape[0]) if ops.count("rot90") % 2 or "transpose" in ops else source_shape
    exif_m, _ = dihedral_matrix(ops, *sensor_shape)
    user_m, _ = dihedral_matrix(user_geometry_ops(geometry), *half_shape)
    shift = np.array([offset[1], offset[0]], dtype=np.float64)

    def unwarp(pts: np.ndarray) -> np.ndarray:
        p = _transform(np.linalg.inv(user_m), pts) + shift
        p = _transform(np.linalg.inv(exif_m), p)
        p = _lens_source(lens, sensor_shape, p)
        p = _transform(exif_m, p) - shift
        return _transform(user_m, p)

    return unwarp


def _leftover_inverse(geometry: GeometryConfig, w: int, h: int, k1_applied: bool, unwarp: Optional[PointMap]) -> Optional[PointMap]:
    """Map the engine's final frame back to the linear buffer, or None when they are the same frame."""
    steps: list[PointMap] = []
    if geometry.converge_v != 0.0 or geometry.converge_h != 0.0:
        inv_keystone = np.linalg.inv(keystone_matrix(geometry.converge_v, geometry.converge_h, w, h))
        steps.append(lambda p: _transform(inv_keystone, p))
    k1 = geometry.distortion_k1
    if k1 != 0.0 and not k1_applied:
        steps.append(lambda p: radial_source_points(p, k1, w, h))
    if geometry.fine_rotation != 0.0:
        rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), geometry.fine_rotation, 1.0)
        inv_rot = np.vstack([cv2.invertAffineTransform(rot), [0.0, 0.0, 1.0]])
        steps.append(lambda p: _transform(inv_rot, p))
    if unwarp is not None:
        steps.append(unwarp)
    if not steps:
        return None

    def inverse(p: np.ndarray) -> np.ndarray:
        for step in steps:
            p = step(p)
        return p

    return inverse


def _boundary(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, _EDGE_SAMPLES + 1)
    xs, ys = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
    return np.concatenate(
        [
            np.stack([xs, np.full_like(xs, y1)], axis=1),
            np.stack([xs, np.full_like(xs, y2)], axis=1),
            np.stack([np.full_like(ys, x1), ys], axis=1),
            np.stack([np.full_like(ys, x2), ys], axis=1),
        ]
    )


def linear_crop_roi(
    shape: tuple[int, ...],
    geometry: GeometryConfig,
    scale: float,
    k1_applied: bool,
    unwarp: Optional[PointMap] = None,
) -> Optional[ROI]:
    """The (y1, y2, x1, x2) slice of a post-geometry linear buffer, or None for no crop.

    ``scale`` is the export's margin scale (source long edge over the preview render size).
    Resolve an armed Auto Crop first (`crop_unresolved`); this reads only a rect already set.
    """
    h, w = shape[:2]
    if geometry.crop_rect:
        rect = geometry.crop_rect
    elif geometry.crop_to_valid and not geometry.crop_from_auto:
        rect = compute_geometry_crop_rect(geometry.fine_rotation, geometry.converge_v, geometry.converge_h, w, h)
    elif geometry.autocrop_offset > 0:
        rect = (0.0, 0.0, 1.0, 1.0)
    else:
        return None
    roi = get_manual_rect_coords((h, w), rect, offset_px=geometry.autocrop_offset, scale_factor=scale)
    inverse = _leftover_inverse(geometry, w, h, k1_applied, unwarp)
    if inverse is None:
        return roi
    y1, y2, x1, x2 = roi
    src = inverse(_boundary(x1, y1, x2, y2))
    bx1 = max(0, math.floor(float(src[:, 0].min())) - _PAD_PX)
    bx2 = min(w, math.ceil(float(src[:, 0].max())) + _PAD_PX)
    by1 = max(0, math.floor(float(src[:, 1].min())) - _PAD_PX)
    by2 = min(h, math.ceil(float(src[:, 1].max())) + _PAD_PX)
    return by1, by2, bx1, bx2
