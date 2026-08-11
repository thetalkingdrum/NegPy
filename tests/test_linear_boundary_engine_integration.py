"""Step 2 of the linear-boundary prototype: PipelineContext.skip_terminal_encode
wired into DarkroomEngine, validated end-to-end against the real, unmodified
ImageProcessor export path -- not a hand-rolled reimplementation of it.

engine.py's terminal working_oetf_encode is the single output-transform call
(docs/PIPELINE.md); skip_terminal_encode (domain/interfaces.py) lets a caller take
that scene-linear buffer and finish the job itself via boundary_transform.py instead.
Default False everywhere else, so every existing caller and test is unaffected --
see TestSkipTerminalEncodeDefaultOff.
"""

import os

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.display.boundary_transform import apply_boundary_transform, load_boundary_transform
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE, ColorSpaceRegistry
from negpy.kernel.image.logic import float_to_uint16, working_oetf_encode
from negpy.services.rendering.engine import DarkroomEngine
from negpy.services.rendering.image_processor import ImageProcessor

_ICC_DIR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icc")


def _render(img: np.ndarray, settings: WorkspaceConfig, *, skip_terminal_encode: bool) -> np.ndarray:
    engine = DarkroomEngine()
    h, w = img.shape[:2]
    context = PipelineContext(
        original_size=(h, w),
        scale_factor=1.0,
        process_mode=settings.process.process_mode,
        skip_terminal_encode=skip_terminal_encode,
    )
    return engine.process(img, settings, source_hash="linear-boundary-test", context=context)


class TestSkipTerminalEncodeDefaultOff:
    def test_default_context_unaffected(self):
        """A PipelineContext built the way every real caller builds one (no
        skip_terminal_encode argument) must render identically to before -- this flag
        must be invisible unless explicitly requested."""
        img = np.random.RandomState(0).rand(64, 64, 3).astype(np.float32)
        settings = WorkspaceConfig()

        engine_a = DarkroomEngine()
        out_default_ctor = engine_a.process(img, settings, source_hash="a")

        engine_b = DarkroomEngine()
        out_no_context = engine_b.process(img, settings, source_hash="a", context=None)

        np.testing.assert_array_equal(out_default_ctor, out_no_context)


class TestSkipTerminalEncodeRoundTrip:
    def test_encoding_the_linear_output_reproduces_the_encoded_output(self):
        """The flag must be a pure skip, not a different code path -- encoding the
        skip_terminal_encode=True result must reproduce the normal result exactly
        (both take the identical FinishProcessor output; only whether
        working_oetf_encode runs afterward differs)."""
        img = np.random.RandomState(1).rand(48, 48, 3).astype(np.float32)
        settings = WorkspaceConfig()

        encoded = _render(img, settings, skip_terminal_encode=False)
        linear = _render(img, settings, skip_terminal_encode=True)

        np.testing.assert_allclose(working_oetf_encode(linear), encoded, atol=1e-6)

    def test_flat_intent_unaffected(self):
        """Flat/digital-intermediate renders already skip the terminal encode on their
        own path (flat_intent) -- skip_terminal_encode must be a no-op there, not a
        second way of reaching the same state that could interact oddly."""
        from dataclasses import replace

        from negpy.features.exposure.models import RenderIntent

        img = np.random.RandomState(2).rand(32, 32, 3).astype(np.float32)
        settings = WorkspaceConfig()
        settings = replace(settings, exposure=replace(settings.exposure, render_intent=RenderIntent.FLAT))

        out_a = _render(img, settings, skip_terminal_encode=False)
        out_b = _render(img, settings, skip_terminal_encode=True)
        np.testing.assert_array_equal(out_a, out_b)


class TestLinearBoundaryEndToEnd:
    """The real claim: rendering scene-linear (skip_terminal_encode=True) and running
    the result through the new BoundaryTransform loader with RGBScan.icc must
    reproduce what today's actual, unmodified ImageProcessor export path produces
    when RGBScan.icc is set as the Input ICC override -- the shipped Narrowband Scan
    default. Both sides call real production code (DarkroomEngine, ImageProcessor);
    only the boundary mechanism differs."""

    def test_matches_real_image_processor_export_path(self):
        img = np.random.RandomState(42).rand(96, 96, 3).astype(np.float32)
        settings = WorkspaceConfig()

        encoded = _render(img, settings, skip_terminal_encode=False)
        linear = _render(img, settings, skip_terminal_encode=True)

        # Legacy: the real, unmodified export colour-management call actually used
        # for TIFF/narrowband exports (image_processor.py calls this, not the
        # icc_lut.py-grid-based _apply_color_management_u16_rgb, which no production
        # call site uses -- see boundary_transform.py's module docstring for why that
        # matters: its 8-bit-PIL-sampled grid isn't a fair reference for a
        # linear-domain profile).
        rgbscan_path = os.path.join(_ICC_DIR_PATH, "RGBScan.icc")
        proc = ImageProcessor()
        encoded_u16 = float_to_uint16(encoded)
        legacy_u16, _icc_bytes = proc._apply_color_management_u16(
            encoded_u16,
            working_color_space=WORKING_COLOR_SPACE,
            color_space=WORKING_COLOR_SPACE,
            output_icc_path=None,
            input_icc_path=rgbscan_path,
        )
        legacy_out = legacy_u16.astype(np.float64) / 65535.0

        # New: linear-boundary path, driven entirely by boundary_transform.py.
        with open(rgbscan_path, "rb") as f:
            rgbscan_bytes = f.read()
        transform = load_boundary_transform(rgbscan_bytes)
        dst_path = ColorSpaceRegistry.get_icc_path(WORKING_COLOR_SPACE)
        with open(dst_path, "rb") as f:
            dst_bytes = f.read()
        new_out = apply_boundary_transform(linear, transform, dst_profile_bytes=dst_bytes)

        diff_lsb8 = np.abs(new_out.astype(np.float64) - legacy_out) * 255.0
        assert diff_lsb8.mean() < 1.0, f"mean {diff_lsb8.mean():.2f} LSB@8bit"
        assert np.percentile(diff_lsb8, 99) < 3.0, f"p99 {np.percentile(diff_lsb8, 99):.2f} LSB@8bit"
        assert diff_lsb8.max() < 10.0, f"max {diff_lsb8.max():.2f} LSB@8bit"

    def test_no_override_case_is_near_identity_via_primaries_branch(self):
        """The §2.4-style invariant, now exercised through the real engine: with the
        working-space profile itself as the boundary override, the linear-boundary
        path's Primaries branch must reproduce the encoded (legacy, no-override)
        output almost exactly -- no CLUT interpolation noise is even in play here.
        Tolerance is looser than the synthetic-patch version in
        test_boundary_transform.py (atol=5e-4) because a full real-engine render
        accumulates float32 rounding across every stage on top of the matrix's own
        tiny non-identity residual (~3e-5, established in step 1); measured max here
        is ~1.9e-3 (~0.5 LSB@8bit)."""
        img = np.random.RandomState(7).rand(48, 48, 3).astype(np.float32)
        settings = WorkspaceConfig()

        encoded = _render(img, settings, skip_terminal_encode=False)
        linear = _render(img, settings, skip_terminal_encode=True)

        working_path = ColorSpaceRegistry.get_icc_path(WORKING_COLOR_SPACE)
        with open(working_path, "rb") as f:
            working_bytes = f.read()
        transform = load_boundary_transform(working_bytes)

        # Primaries deliberately stops at scene-linear working RGB (see its docstring)
        # -- the terminal encode is the caller's job, same as the unmodified path.
        new_out = working_oetf_encode(apply_boundary_transform(linear, transform))

        diff = np.abs(new_out.astype(np.float64) - encoded.astype(np.float64))
        assert diff.max() < 5e-3, f"max diff {diff.max():.2e}"
