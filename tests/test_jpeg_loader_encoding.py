"""A positive-source JPEG's ingestion must land in the working space's own primaries,
not just its own TRC decoded and left mislabeled as Adobe RGB (the bug behind an
oversaturated E-6 Positive render — see docs/PIPELINE.md's transfer-curve section)."""

import os
import tempfile

import numpy as np
from PIL import Image

from negpy.domain.models import ColorSpace
from negpy.infrastructure.loaders.helpers import resolve_srgb_to_xyz
from negpy.infrastructure.loaders.jpeg_loader import JpegLoader
from negpy.kernel.image.logic import SRGB_TO_XYZ, apply_linear_primaries_transform, srgb_to_linear


def _load(path: str) -> tuple[np.ndarray, dict]:
    ctx, metadata = JpegLoader().load(path)
    with ctx as raw:
        return raw.data, metadata


def test_untagged_jpeg_gets_srgb_decode_and_working_primaries() -> None:
    data = np.linspace(0, 255, 32 * 48 * 3).reshape(32, 48, 3).astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "scan.jpg")
        Image.fromarray(data, mode="RGB").save(path, quality=100, subsampling=0)
        f32, metadata = _load(path)
        expected = apply_linear_primaries_transform(srgb_to_linear(data.astype(np.float32) / 255.0), SRGB_TO_XYZ)
        # JPEG re-compression rounds pixel values; a generous tolerance isolates the
        # decode/primaries math from lossy-codec noise.
        np.testing.assert_allclose(f32, expected, atol=2e-2)
        assert metadata["color_space"] == ColorSpace.SRGB.value


def test_srgb_primaries_are_not_identity_into_working_space() -> None:
    """The regression this exists to catch: skipping the primaries step (leaving sRGB
    numbers mislabeled as Adobe RGB) is exactly Photoshop's Assign-Profile artifact.
    A fully saturated sRGB linear red must come out attenuated in the working space,
    not pass through unchanged."""
    red = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    out = apply_linear_primaries_transform(red, resolve_srgb_to_xyz(None))
    assert out[0, 0, 0] < 0.99
