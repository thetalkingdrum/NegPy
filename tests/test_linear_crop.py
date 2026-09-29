"""Linear Output's Apply crop: the crop as a slice of the unresampled linear buffer."""

import os
from unittest import mock

import numpy as np
import pytest
import tifffile

from negpy.domain.interfaces import PipelineContext
from negpy.features.geometry.logic import compute_geometry_crop_rect, get_manual_rect_coords, map_coords_to_geometry
from negpy.features.geometry.models import GeometryConfig
from negpy.features.geometry.processor import CropProcessor, GeometryProcessor
from negpy.features.lens.models import LensMetadata
from negpy.kernel.image.logic import apply_exif_orientation
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.export.linear_crop import dihedral_matrix, linear_crop_roi, user_geometry_ops
from negpy.services.export.linear_output import _apply_user_geometry, _CameraWB, _decode_linear, _SourceMeta, export_linear_output

_H, _W = 120, 200
_WB = _CameraWB(as_shot=(1.0, 1.0, 1.0, 1.0), daylight=(1.0, 1.0, 1.0, 1.0))


def _ids(h: int = _H, w: int = _W) -> np.ndarray:
    """A float buffer whose every pixel is unique, so a slice can be traced back to the source."""
    ids = np.arange(h * w, dtype=np.float32).reshape(h, w) / float(h * w)
    return np.repeat(ids[:, :, None], 3, axis=2)


def _write_tiff(tmp_path: str, data: np.ndarray, name: str = "scan.tif") -> str:
    path = os.path.join(str(tmp_path), name)
    tifffile.imwrite(path, (data * 65535).astype(np.uint16), photometric="rgb")
    return path


class TestDihedralMatrix:
    @pytest.mark.parametrize(
        "ops",
        [("rot90",), ("rot90", "rot90"), ("rot90",) * 3, ("fliplr",), ("flipud",), ("transpose",), ("rot90", "fliplr", "flipud")],
    )
    def test_matches_the_array_ops(self, ops) -> None:
        arr = np.arange(5 * 7).reshape(5, 7)
        out = arr
        for op in ops:
            out = {"rot90": np.rot90, "fliplr": np.fliplr, "flipud": np.flipud, "transpose": np.transpose}[op](out)
        m, shape = dihedral_matrix(ops, 5, 7)
        assert shape == out.shape
        for y in range(5):
            for x in range(7):
                px, py, _ = m @ np.array([x + 0.5, y + 0.5, 1.0])
                assert out[int(np.floor(py)), int(np.floor(px))] == arr[y, x]

    @pytest.mark.parametrize("orientation", range(1, 9))
    def test_exif_ops_match_apply_exif_orientation(self, orientation: int) -> None:
        from negpy.services.export.linear_crop import _EXIF_OPS

        arr = np.arange(5 * 7).reshape(5, 7)
        out = apply_exif_orientation(arr, orientation)
        m, shape = dihedral_matrix(_EXIF_OPS.get(orientation, ()), 5, 7)
        assert shape == out.shape
        for y in range(5):
            for x in range(7):
                px, py, _ = m @ np.array([x + 0.5, y + 0.5, 1.0])
                assert out[int(np.floor(py)), int(np.floor(px))] == arr[y, x]

    def test_user_geometry_ops_match(self) -> None:
        g = GeometryConfig(rotation=3, flip_horizontal=True)
        arr = np.arange(5 * 7).reshape(5, 7)
        out = _apply_user_geometry(arr, g)
        m, shape = dihedral_matrix(user_geometry_ops(g), 5, 7)
        assert shape == out.shape
        px, py, _ = m @ np.array([1.5, 2.5, 1.0])
        assert out[int(py), int(px)] == arr[2, 1]


class TestUnwarpedCrop:
    @pytest.mark.parametrize(
        "geometry",
        [
            GeometryConfig(crop_rect=(0.1, 0.2, 0.7, 0.9)),
            GeometryConfig(rotation=1, flip_horizontal=True, crop_rect=(0.05, 0.1, 0.8, 0.95), autocrop_offset=4),
            GeometryConfig(rotation=2, autocrop_offset=6),
        ],
    )
    def test_matches_crop_processor(self, geometry: GeometryConfig) -> None:
        img = _ids()
        scale = 3.0
        ctx = PipelineContext(original_size=(_W, _H), scale_factor=scale)
        engine = CropProcessor(geometry).process(GeometryProcessor(geometry).process(img, ctx), ctx)

        linear = _apply_user_geometry(img, geometry)
        y1, y2, x1, x2 = linear_crop_roi(linear.shape, geometry, scale, k1_applied=False)
        np.testing.assert_array_equal(linear[y1:y2, x1:x2], engine)

    def test_no_crop_is_none(self) -> None:
        assert linear_crop_roi((_H, _W), GeometryConfig(), 1.0, k1_applied=False) is None

    def test_margin_uses_the_export_scale(self, tmp_path: str) -> None:
        big = np.zeros((600, 900, 3), dtype=np.float32)
        path = _write_tiff(tmp_path, big)
        geometry = GeometryConfig(crop_rect=(0.1, 0.1, 0.9, 0.9), autocrop_offset=20)
        rgb, _, _, _ = _decode_linear(path, geometry, apply_crop=True)
        scale = 900 / float(APP_CONFIG.preview_render_size)
        y1, y2, x1, x2 = get_manual_rect_coords((600, 900), geometry.crop_rect, offset_px=20, scale_factor=scale)
        assert rgb.shape[:2] == (y2 - y1, x2 - x1)


class TestWarpedCrop:
    @pytest.mark.parametrize("k1", [-0.08, 0.05])
    def test_radial_source_points_inverts_map_point_radial(self, k1: float) -> None:
        from negpy.features.geometry.logic import map_point_radial, radial_source_points

        pts = np.array([[10.0, 7.0], [150.0, 30.0], [100.0, 60.0], [190.0, 115.0]])
        back = np.array([map_point_radial(float(x), float(y), k1, _W, _H) for x, y in radial_source_points(pts, k1, _W, _H)])
        np.testing.assert_allclose(back, pts, atol=1e-4)

    @pytest.mark.parametrize(
        "geometry",
        [
            GeometryConfig(fine_rotation=2.5, crop_rect=(0.15, 0.2, 0.8, 0.85)),
            GeometryConfig(converge_v=4.0, converge_h=-3.0, crop_rect=(0.1, 0.1, 0.9, 0.9)),
            GeometryConfig(distortion_k1=0.06, crop_rect=(0.05, 0.05, 0.95, 0.95)),
            GeometryConfig(fine_rotation=-1.5, distortion_k1=-0.04, converge_v=2.0, crop_rect=(0.2, 0.1, 0.9, 0.7)),
        ],
    )
    def test_bounding_box_holds_the_crop_and_stays_in_bounds(self, geometry: GeometryConfig) -> None:
        h, w = _H, _W
        roi = linear_crop_roi((h, w), geometry, 1.0, k1_applied=False)
        assert roi is not None
        by1, by2, bx1, bx2 = roi
        assert 0 <= by1 < by2 <= h and 0 <= bx1 < bx2 <= w

        cy1, cy2, cx1, cx2 = get_manual_rect_coords((h, w), geometry.crop_rect)
        inside = []
        for y in np.linspace(0.5, h - 0.5, 41):
            for x in np.linspace(0.5, w - 0.5, 67):
                nx, ny = map_coords_to_geometry(
                    x / w,
                    y / h,
                    (h, w),
                    fine_rotation=geometry.fine_rotation,
                    distortion_k1=geometry.distortion_k1,
                    converge_v=geometry.converge_v,
                    converge_h=geometry.converge_h,
                )
                if cx1 <= nx * w <= cx2 and cy1 <= ny * h <= cy2:
                    inside.append((x, y))
        pts = np.array(inside)
        assert pts[:, 0].min() >= bx1 and pts[:, 0].max() <= bx2
        assert pts[:, 1].min() >= by1 and pts[:, 1].max() <= by2
        # Tight: no more than the pad plus one sample step beyond the traced source.
        assert pts[:, 0].min() - bx1 <= 6 and bx2 - pts[:, 0].max() <= 6
        assert pts[:, 1].min() - by1 <= 6 and by2 - pts[:, 1].max() <= 6

    def test_crop_to_valid_uses_the_valid_rect(self) -> None:
        geometry = GeometryConfig(fine_rotation=3.0, crop_to_valid=True)
        roi = linear_crop_roi((_H, _W), geometry, 1.0, k1_applied=False)
        valid = compute_geometry_crop_rect(3.0, 0.0, 0.0, _W, _H)
        explicit = linear_crop_roi((_H, _W), GeometryConfig(fine_rotation=3.0, crop_rect=valid), 1.0, k1_applied=False)
        # The valid rect's corners sit on the source edges, so its box is the whole frame.
        assert roi == explicit == (0, _H, 0, _W)


class _MirrorWarp:
    has_distortion = True
    has_ca = False

    def remap(self, lens, shape, start, stop, channel, corrections):
        h, w = shape[:2]
        ys, xs = np.meshgrid(np.arange(start, stop, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        return (w - 1 - xs).astype(np.float32), ys


class TestEmbeddedWarpLeftOver:
    """Embedded Distortion on in Optics, Apply lens correction off: the crop maps back through the profile."""

    def _decode(self, tmp_path: str, geometry: GeometryConfig, apply_lens: bool):
        path = os.path.join(str(tmp_path), "frame.arw")
        open(path, "wb").close()
        lens = LensMetadata(source="test", warps=(_MirrorWarp(),))
        with mock.patch(
            "negpy.services.export.linear_output._decode_camera_raw_buffer",
            return_value=(_ids(), _WB, _SourceMeta(), lens),
        ):
            return _decode_linear(path, geometry, apply_lens=apply_lens, apply_crop=True)

    @pytest.mark.parametrize("rotation", [0, 1])
    def test_bounding_box_holds_the_corrected_crop(self, tmp_path: str, rotation: int) -> None:
        geometry = GeometryConfig(rotation=rotation, lens_distortion_from_metadata=True, crop_rect=(0.1, 0.15, 0.4, 0.6))
        engine = _apply_user_geometry(_ids()[:, ::-1], geometry)
        y1, y2, x1, x2 = get_manual_rect_coords(engine.shape[:2], geometry.crop_rect)
        wanted = set(np.unique(engine[y1:y2, x1:x2, 0]))

        rgb, _, _, meta = self._decode(tmp_path, geometry, apply_lens=False)
        assert wanted <= set(np.unique(rgb[..., 0]))
        assert rgb.shape[0] <= (y2 - y1) + 6 and rgb.shape[1] <= (x2 - x1) + 6
        assert meta.crop == "cropped"

    def test_applied_warp_crops_directly(self, tmp_path: str) -> None:
        geometry = GeometryConfig(lens_distortion_from_metadata=True, crop_rect=(0.1, 0.15, 0.4, 0.6))
        engine = _ids()[:, ::-1]
        y1, y2, x1, x2 = get_manual_rect_coords(engine.shape[:2], geometry.crop_rect)
        rgb, _, _, _ = self._decode(tmp_path, geometry, apply_lens=True)
        np.testing.assert_array_equal(rgb, engine[y1:y2, x1:x2])


class TestCropPlumbing:
    def test_off_by_default(self, tmp_path: str) -> None:
        path = _write_tiff(tmp_path, _ids())
        rgb, _, _, meta = _decode_linear(path, GeometryConfig(crop_rect=(0.1, 0.1, 0.5, 0.5)))
        assert rgb.shape[:2] == (_H, _W)
        assert meta.crop == ""

    def test_armed_auto_crop_is_not_detected(self, tmp_path: str) -> None:
        path = _write_tiff(tmp_path, _ids())
        out = os.path.join(str(tmp_path), "out.tiff")
        geometry = GeometryConfig(crop_from_auto=True, crop_rect=None)
        export_linear_output(path, out, geometry=geometry, apply_crop=True)
        with tifffile.TiffFile(out) as tf:
            assert tf.pages[0].shape[:2] == (_H, _W)
            assert "crop: not resolved" in tf.pages[0].description

    def test_ir_is_cropped_with_rgb(self, tmp_path: str) -> None:
        from tests.test_linear_output import _make_linearraw_dng_4ch

        path = _make_linearraw_dng_4ch(str(tmp_path))
        geometry = GeometryConfig(rotation=1, crop_rect=(0.1, 0.2, 0.6, 0.7))
        rgb, ir, _, meta = _decode_linear(path, geometry, apply_crop=True)
        full, full_ir, _, _ = _decode_linear(path, geometry)
        y1, y2, x1, x2 = get_manual_rect_coords(full.shape[:2], geometry.crop_rect)
        np.testing.assert_array_equal(rgb, full[y1:y2, x1:x2])
        np.testing.assert_array_equal(ir, full_ir[y1:y2, x1:x2])
        assert meta.crop == "cropped"

    def test_description_names_the_crop_and_half(self, tmp_path: str) -> None:
        from negpy.services.assets.half_frame import HalfGeometry

        path = _write_tiff(tmp_path, _ids())
        out = os.path.join(str(tmp_path), "out.tiff")
        export_linear_output(
            path, out, geometry=GeometryConfig(crop_rect=(0.1, 0.1, 0.9, 0.9)), apply_crop=True, half=2, half_geometry=HalfGeometry()
        )
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
        assert "half-frame 2" in desc
        assert "corrections: crop" in desc
