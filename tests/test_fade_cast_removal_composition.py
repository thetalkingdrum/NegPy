"""Fade Restoration and Cast Removal on a transparency: composition, not competition.

Per PLAN_after_cast_removal.md §1: Cast Removal's neutral-axis meter reads the film
*after* the fade matrix (features/exposure/processor.py), so it fits whatever per-channel
residual the fade correction leaves rather than colliding with it. No conflict guard
exists between the two -- this file pins that as a deliberate invariant, not an oversight,
so nobody adds one later on the assumption they collide (mirrors fade_delta_conflict_reason,
which *does* guard fade against Crosstalk, for the genuinely different reason that both
describe the same physical side absorption).
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from negpy.features.process.models import ProcessMode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG

_GENERIC_E6_DELTA = (0.0689, 0.0111, 0.2246, 0.0486, 0.0854, 0.1815)


def _config(fade_ratio_g=1.0, fade_ratio_b=1.0, fade_delta=None, cast_strength=0.0):
    cfg = DEFAULT_WORKSPACE_CONFIG
    return replace(
        cfg,
        process=replace(
            cfg.process,
            process_mode=ProcessMode.E6,
            fade_strength=1.0,
            fade_ratio_g=fade_ratio_g,
            fade_ratio_b=fade_ratio_b,
            fade_delta=fade_delta,
            fade_process=ProcessMode.E6,
        ),
        exposure=replace(cfg.exposure, cast_removal_strength=cast_strength),
    )


def _render(image, cfg):
    h, w = image.shape[:2]
    ctx = PipelineContext(
        original_size=(h, w),
        scale_factor=1.0,
        process_mode=cfg.process.process_mode,
        cam_xyz=None,
        camera_wb=None,
        wants_uv_grid=False,
    )
    norm = NormalizationProcessor(cfg.process, cfg.exposure.cast_removal_strength).process(image, ctx)
    return np.asarray(PhotometricProcessor(cfg.exposure, cfg.local, cfg.process).process(norm, ctx)), ctx


def _faded_neutral_wedge(ratio_g=1.0, ratio_b=1.0, delta=(0.0,) * 6, seed=5):
    """A neutral wedge (spans the meter's three luma bands) with a known differential fade
    baked in via the forward model, so Cast Removal has a real per-channel residual to meet
    the fade correction partway on."""
    rng = np.random.default_rng(seed)
    t = np.geomspace(5e-4, 0.9, 64 * 64).astype(np.float64).reshape(64, 64)
    d_gr, d_br, d_rg, d_bg, d_rb, d_gb = delta
    s_matrix = np.array([[1.0, d_gr, d_br], [d_rg, 1.0, d_bg], [d_rb, d_gb, 1.0]])
    log_t = np.log10(t)
    concentration_log = np.stack([log_t, log_t + np.log10(ratio_g), log_t + np.log10(ratio_b)], axis=-1)
    measured_log = concentration_log @ s_matrix.T
    img = np.power(10.0, measured_log).astype(np.float32)
    return np.ascontiguousarray(img + rng.uniform(0, 1e-4, img.shape).astype(np.float32))


class TestFadeAndCastRemovalCompose(unittest.TestCase):
    def test_both_active_meters_an_axis_and_renders(self):
        """Neither feature disables the other: a non-identity fade and a non-zero Cast
        Removal strength together still meter a neutral axis and produce a finite render."""
        img = _faded_neutral_wedge(0.7, 1.2, _GENERIC_E6_DELTA)
        cfg = _config(fade_ratio_g=0.7, fade_ratio_b=1.2, fade_delta=_GENERIC_E6_DELTA, cast_strength=0.8)
        out, ctx = _render(img, cfg)
        self.assertIn("neutral_axis_refs", ctx.metrics)
        self.assertIsNotNone(ctx.metrics["neutral_axis_refs"])
        self.assertTrue(np.all(np.isfinite(out)))

    def test_cast_removal_solve_changes_when_fade_changes(self):
        """Cast Removal meters downstream of the fade matrix, so its own solved (gain,
        offset) depends on what fade did -- documenting the coupling, not a bug. Same
        capture, same Cast Removal strength, different fade ratios: different refs."""
        img = _faded_neutral_wedge(0.7, 1.2, _GENERIC_E6_DELTA)
        cfg_a = _config(fade_ratio_g=0.7, fade_ratio_b=1.2, fade_delta=_GENERIC_E6_DELTA, cast_strength=0.8)
        cfg_b = _config(fade_ratio_g=1.0, fade_ratio_b=1.0, fade_delta=_GENERIC_E6_DELTA, cast_strength=0.8)
        _out_a, ctx_a = _render(img, cfg_a)
        _out_b, ctx_b = _render(img, cfg_b)
        refs_a, refs_b = ctx_a.metrics["neutral_axis_refs"], ctx_b.metrics["neutral_axis_refs"]
        self.assertIsNotNone(refs_a)
        self.assertIsNotNone(refs_b)
        self.assertNotEqual(refs_a[0], refs_b[0])  # midtone ref moved

    def test_cross_channel_error_is_not_recoverable_by_cast_removal(self):
        """The test that justifies Fade Restoration continuing to exist alongside Cast
        Removal: a *cross-channel* fade error (wrong delta, not just wrong ratios) is not
        something a per-channel affine can fix. Render the same faded wedge once with the
        correct delta and once with a badly wrong one (same ratios, same Cast Removal
        strength); Cast Removal's own gain/offset cannot make the two converge, because a
        diagonal operator cannot undo an off-diagonal error."""
        ratio_g, ratio_b = 0.7, 1.2
        img = _faded_neutral_wedge(ratio_g, ratio_b, _GENERIC_E6_DELTA)

        cfg_correct = _config(fade_ratio_g=ratio_g, fade_ratio_b=ratio_b, fade_delta=_GENERIC_E6_DELTA, cast_strength=1.0)
        wrong_delta = (0.25, 0.02, 0.25, 0.02, 0.25, 0.02)  # a different dye set's cross-channel shape
        cfg_wrong = _config(fade_ratio_g=ratio_g, fade_ratio_b=ratio_b, fade_delta=wrong_delta, cast_strength=1.0)

        out_correct, _ctx_correct = _render(img, cfg_correct)
        out_wrong, _ctx_wrong = _render(img, cfg_wrong)

        # Even at full Cast Removal strength, the wrong-delta render is not pulled back to
        # match the correct one -- a real, visible difference remains.
        self.assertGreater(float(np.abs(out_correct.astype(np.float64) - out_wrong.astype(np.float64)).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
