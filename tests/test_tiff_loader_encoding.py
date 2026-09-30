"""How a scanner TIFF is encoded, and what the loader makes of it."""

import base64
import os
import tempfile

import numpy as np
import tifffile
import logging
import struct

import pytest
from PIL import ImageCms

from negpy.domain.models import ColorSpace
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper
from negpy.infrastructure.loaders.tiff_loader import TiffLoader
from negpy.kernel.image.logic import srgb_to_linear, working_oetf_decode
from negpy.features.process.logic import effective_linear_raw
from negpy.features.process.models import ProcessConfig, ProcessMode

# The standard Adobe RGB (1998) ICC profile — real-world scanner/export software tags
# TIFFs with exactly this, so the loader's Adobe RGB branch is tested against what
# actually reaches it, not a synthetic stand-in.
_ADOBE_RGB_ICC = base64.b64decode(
    "AAACMEFEQkUCEAAAbW50clJHQiBYWVogB88ABgADAAAAAAAAYWNzcEFQUEwAAAAAbm9uZQAAAAAAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1B"
    "REJFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKY3BydAAAAPwAAAAyZGVzYwAAATAAAABrd3RwdAAA"
    "AZwAAAAUYmtwdAAAAbAAAAAUclRSQwAAAcQAAAAOZ1RSQwAAAdQAAAAOYlRSQwAAAeQAAAAOclhZWgAAAfQAAAAUZ1hZWgAAAggAAAAUYlhZ"
    "WgAAAhwAAAAUdGV4dAAAAABDb3B5cmlnaHQgMTk5OSBBZG9iZSBTeXN0ZW1zIEluY29ycG9yYXRlZAAAAGRlc2MAAAAAAAAAEUFkb2JlIFJH"
    "QiAoMTk5OCkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAFhZWiAAAAAAAADzUQABAAAAARbMWFlaIAAAAAAAAAAAAAAAAAAAAABjdXJ2AAAAAAAAAAECMwAAY3VydgAAAAAAAAABAjMAAGN1"
    "cnYAAAAAAAAAAQIzAABYWVogAAAAAAAAnBgAAE+lAAAE/FhZWiAAAAAAAAA0jQAAoCwAAA+VWFlaIAAAAAAAACYxAAAQLwAAvpw="
)


# The ICC para parameters are s15Fixed16, so a tagged sRGB curve is not bit-exact analytic sRGB.
_TRC_ATOL = 1e-5


def _rgb16(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # dark linear-scan range, where the sRGB toe distorts most
    return rng.integers(0, 30000, (32, 48, 3), dtype=np.uint16)


def _s15(v: float) -> bytes:
    return struct.pack(">i", round(v * 65536))


def _xyz_tag(x: float, y: float, z: float) -> bytes:
    return b"XYZ \0\0\0\0" + _s15(x) + _s15(y) + _s15(z)


def _desc_tag(text: str) -> bytes:
    ascii_ = text.encode("ascii") + b"\0"
    return b"desc\0\0\0\0" + struct.pack(">I", len(ascii_)) + ascii_ + b"\0" * 8 + b"\0" * 3 + b"\0" * 67


def _u8f8(gamma: float) -> float:
    """The gamma a count-1 curv tag stores: u8Fixed8, so 2.2 is 563/256."""
    return round(gamma * 256) / 256


def _curv_gamma(gamma: float) -> bytes:
    return b"curv\0\0\0\0" + struct.pack(">IH", 1, round(gamma * 256)) + b"\0\0"


_CURV_IDENTITY = b"curv\0\0\0\0" + struct.pack(">I", 0)


def _para(ftype: int, *params: float) -> bytes:
    return b"para\0\0\0\0" + struct.pack(">HH", ftype, 0) + b"".join(_s15(p) for p in params)


def _icc(description: str, trc: bytes, *, space: bytes = b"RGB ", extra: tuple[tuple[bytes, bytes], ...] = ()) -> bytes:
    """A minimal v2 display profile built from parts, so the description and the curve
    vary independently. `space=b"GRAY"` writes a kTRC instead of colorants and r/g/bTRC."""
    tags: list[tuple[bytes, bytes]] = [(b"desc", _desc_tag(description)), (b"wtpt", _xyz_tag(0.9642, 1.0, 0.8249))]
    if space == b"GRAY":
        tags.append((b"kTRC", trc))
    else:
        tags += [
            (b"rXYZ", _xyz_tag(0.4361, 0.2225, 0.0139)),
            (b"gXYZ", _xyz_tag(0.3851, 0.7169, 0.0971)),
            (b"bXYZ", _xyz_tag(0.1431, 0.0606, 0.7141)),
            (b"rTRC", trc),
            (b"gTRC", trc),
            (b"bTRC", trc),
        ]
    tags += list(extra)
    offset = 128 + 4 + 12 * len(tags)
    table, body = b"", b""
    for sig, payload in tags:
        payload += b"\0" * (-len(payload) % 4)
        table += sig + struct.pack(">II", offset + len(body), len(payload))
        body += payload
    size = offset + len(body)
    header = struct.pack(">I", size) + b"\0" * 4 + struct.pack(">I", 0x02100000) + b"mntr" + space + b"XYZ "
    header += b"\0" * 12 + b"acsp" + b"\0" * 24 + _s15(0.9642) + _s15(1.0) + _s15(0.8249)
    header += b"\0" * (128 - len(header))
    return header + struct.pack(">I", len(tags)) + table + body


def _write_tagged(tmpdir: str, data: np.ndarray, icc: bytes) -> str:
    path = os.path.join(tmpdir, "tagged.tif")
    photometric = "minisblack" if data.ndim == 2 else "rgb"
    tifffile.imwrite(path, data, photometric=photometric, extratags=[(34675, 7, len(icc), icc, True)])
    return path


def _load(path: str, linear_raw: bool = False, positive_source: bool = False) -> tuple[np.ndarray, dict]:
    ctx, metadata = TiffLoader().load(path, linear_raw=linear_raw, positive_source=positive_source)
    with ctx as raw:
        return raw.data, metadata


class TestTiffEncodingAssumptions:
    def test_untagged_uint16_reads_linear(self) -> None:
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scan.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path)
            np.testing.assert_allclose(f32, data.astype(np.float32) / 65535.0, atol=1e-7)
            # Scanner-raw linear: no ColorSpace names it, so it must not claim one.
            assert metadata["color_space"] is None

    def test_untagged_uint16_positive_source_gets_srgb_decode(self) -> None:
        """The ambiguity an untagged 16-bit file usually carries — scanner-linear or a
        finished positive — is already resolved once the caller says positive_source."""
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "positivized.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path, positive_source=True)
            np.testing.assert_allclose(f32, srgb_to_linear(data.astype(np.float32) / 65535.0), atol=1e-6)
            assert metadata["color_space"] == ColorSpace.SRGB.value

    def test_positive_source_does_not_override_an_actual_tag(self) -> None:
        """A real profile still wins — positive_source only fills the gap an untagged
        file leaves, it does not second-guess a tag that is already there."""
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tagged.tif")
            tifffile.imwrite(path, data, photometric="rgb", extratags=[(34675, 7, len(icc), icc, True)])
            f32, metadata = _load(path, positive_source=True)
            np.testing.assert_allclose(f32, srgb_to_linear(data.astype(np.float32) / 65535.0), atol=_TRC_ATOL)
            assert metadata["color_space"] == ColorSpace.SRGB.value

    def test_positive_source_off_keeps_the_untagged_default(self) -> None:
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scan.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path, positive_source=False)
            np.testing.assert_allclose(f32, data.astype(np.float32) / 65535.0, atol=1e-7)
            assert metadata["color_space"] is None

    def test_untagged_uint8_gets_srgb_decode(self) -> None:
        data = np.linspace(0, 255, 32 * 48 * 3).reshape(32, 48, 3).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "photo.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path)
            np.testing.assert_allclose(f32, srgb_to_linear(data.astype(np.float32) / 255.0), atol=1e-6)
            assert metadata["color_space"] == ColorSpace.SRGB.value

    @pytest.mark.parametrize(
        "icc",
        [
            ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes(),
            _ADOBE_RGB_ICC,
            _icc("Custom scanner profile", _curv_gamma(2.2)),
        ],
        ids=["srgb", "adobe", "custom"],
    )
    def test_tagged_uint16_negative_ignores_its_profile(self, icc: bytes) -> None:
        """A 16-bit negative is scanner-raw linear: its profile neither decodes nor labels it."""
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            f32, metadata = _load(_write_tagged(tmpdir, data, icc))
        np.testing.assert_array_equal(f32, data.astype(np.float32) / 65535.0)
        assert metadata["color_space"] is None

    def test_tagged_float_negative_ignores_its_profile(self) -> None:
        data = (_rgb16().astype(np.float32) / 65535.0).astype(np.float32)
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        with tempfile.TemporaryDirectory() as tmpdir:
            f32, metadata = _load(_write_tagged(tmpdir, data, icc))
        np.testing.assert_array_equal(f32, data)
        assert metadata["color_space"] is None

    def test_srgb_icc_uint16_positive_gets_srgb_decode(self) -> None:
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            f32, metadata = _load(_write_tagged(tmpdir, data, icc), positive_source=True)
        np.testing.assert_allclose(f32, srgb_to_linear(data.astype(np.float32) / 65535.0), atol=_TRC_ATOL)
        assert metadata["color_space"] == ColorSpace.SRGB.value

    def test_adobe_rgb_icc_uint16_positive_gets_the_working_oetf_decode(self) -> None:
        """Adobe RGB's curv tag is the working space's own gamma, so its decode matches
        working_oetf_decode."""
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            f32, metadata = _load(_write_tagged(tmpdir, data, _ADOBE_RGB_ICC), positive_source=True)
        np.testing.assert_allclose(f32, working_oetf_decode(data.astype(np.float32) / 65535.0), atol=1e-6)
        assert metadata["color_space"] == ColorSpace.ADOBE_RGB.value

    def test_linear_raw_ignores_srgb_icc_tag(self) -> None:
        """Linear RAW couples to the loader: a stitched/scanner TIFF that lies about
        being sRGB (see #588) must decode as literal linear data when it's on."""
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tagged.tif")
            tifffile.imwrite(path, data, photometric="rgb", extratags=[(34675, 7, len(icc), icc, True)])
            f32, metadata = _load(path, linear_raw=True)
            np.testing.assert_allclose(f32, data.astype(np.float32) / 65535.0, atol=1e-7)
            assert metadata["color_space"] is None

    def test_linear_raw_still_wins_over_positive_source(self) -> None:
        """An explicit request for literal linear data overrides positive_source too,
        same precedence as effective_linear_raw's own stored-flag check."""
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scan.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path, linear_raw=True, positive_source=True)
            np.testing.assert_allclose(f32, data.astype(np.float32) / 65535.0, atol=1e-7)
            assert metadata["color_space"] is None

    def test_linear_raw_ignores_untagged_uint8_srgb_assumption(self) -> None:
        data = np.linspace(0, 255, 32 * 48 * 3).reshape(32, 48, 3).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "photo.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path, linear_raw=True)
            np.testing.assert_allclose(f32, data.astype(np.float32) / 255.0, atol=1e-7)
            assert metadata["color_space"] is None


class TestEmbeddedTrcDecode:
    """Where a profile is read (Positive, 8-bit), the decode follows its own TRC curves;
    the description is a label only."""

    def _load_tagged(self, icc: bytes, data: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        data = _rgb16() if data is None else data
        with tempfile.TemporaryDirectory() as tmpdir:
            f32, metadata = _load(_write_tagged(tmpdir, data, icc), positive_source=True)
        return data.astype(np.float32) / 65535.0, f32, metadata

    @pytest.mark.parametrize("trc", [_CURV_IDENTITY, _curv_gamma(1.0), _para(0, 1.0)])
    def test_linear_profile_named_srgb_loads_like_untagged(self, trc: bytes) -> None:
        x, f32, metadata = self._load_tagged(_icc("sRGB linear", trc))
        np.testing.assert_array_equal(f32, x)
        assert metadata["color_space"] == ColorSpace.SRGB.value

    def test_custom_profile_decodes_its_gamma(self) -> None:
        x, f32, _ = self._load_tagged(_icc("Custom scanner profile", _curv_gamma(2.2)))
        np.testing.assert_allclose(f32, np.power(x, _u8f8(2.2)), atol=_TRC_ATOL)

    def test_display_p3_para_type_3_matches_srgb(self) -> None:
        trc = _para(3, 2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045)
        x, f32, metadata = self._load_tagged(_icc("Display P3", trc))
        np.testing.assert_allclose(f32, srgb_to_linear(x), atol=_TRC_ATOL)
        assert metadata["color_space"] == ColorSpace.P3_D65.value

    def test_prophoto_decodes_gamma_1_8(self) -> None:
        x, f32, metadata = self._load_tagged(_icc("ProPhoto RGB", _curv_gamma(1.8)))
        np.testing.assert_allclose(f32, np.power(x, _u8f8(1.8)), atol=_TRC_ATOL)
        assert metadata["color_space"] == ColorSpace.PROPHOTO.value

    def test_uint8_is_decoded_by_its_tag_not_the_srgb_assumption(self) -> None:
        data = np.linspace(0, 255, 32 * 48 * 3).reshape(32, 48, 3).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            f32, _ = _load(_write_tagged(tmpdir, data, _icc("Custom scanner profile", _curv_gamma(1.8))))
        np.testing.assert_allclose(f32, np.power(data.astype(np.float32) / 255.0, _u8f8(1.8)), atol=_TRC_ATOL)

    def test_gray_profile_decodes_its_ktrc(self) -> None:
        data = _rgb16()[:, :, 0]
        x, f32, _ = self._load_tagged(_icc("Gray Gamma 2.2", _curv_gamma(2.2), space=b"GRAY"), data)
        np.testing.assert_allclose(f32, np.power(np.stack([x] * 3, axis=-1), _u8f8(2.2)), atol=_TRC_ATOL)

    def test_lut_profile_takes_the_untagged_srgb_decode(self, caplog: pytest.LogCaptureFixture) -> None:
        icc = _icc("sRGB LUT", _curv_gamma(2.2), extra=((b"A2B0", b"mft2" + b"\0" * 48),))
        with caplog.at_level(logging.WARNING, logger="negpy"):
            x, f32, _ = self._load_tagged(icc)
        np.testing.assert_allclose(f32, srgb_to_linear(x), atol=1e-6)
        assert "is not decoded" in caplog.text

    def test_unsupported_curve_type_takes_the_untagged_srgb_decode(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="negpy"):
            x, f32, _ = self._load_tagged(_icc("sRGB IEC61966-2.1", b"xxxx\0\0\0\0\0\0\0\0"))
        np.testing.assert_allclose(f32, srgb_to_linear(x), atol=1e-6)
        assert "unsupported TRC curve type" in caplog.text


class TestPositiveSourceOnTheTransferPath:
    """An already-positive TIFF loaded as Transparency is not a raw
    scanner capture: Positive must reach the loader through effective_linear_raw
    so its sRGB tag decodes instead of being read as literal linear data."""

    def test_positive_source_reaches_the_srgb_decode(self) -> None:
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "positivized.tif")
            tifffile.imwrite(path, data, photometric="rgb", extratags=[(34675, 7, len(icc), icc, True)])

            process = ProcessConfig(process_mode=ProcessMode.E6, positive_source=True)
            f32, metadata = _load(path, linear_raw=effective_linear_raw(process), positive_source=process.positive_source)
            np.testing.assert_allclose(f32, srgb_to_linear(data.astype(np.float32) / 65535.0), atol=_TRC_ATOL)
            assert metadata["color_space"] == ColorSpace.SRGB.value

    def test_without_positive_source_the_tag_is_still_ignored(self) -> None:
        """The default: the same frame, minus the toggle, keeps today's forced-linear read."""
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "positivized.tif")
            tifffile.imwrite(path, data, photometric="rgb", extratags=[(34675, 7, len(icc), icc, True)])

            process = ProcessConfig(process_mode=ProcessMode.E6, positive_source=False)
            f32, metadata = _load(path, linear_raw=effective_linear_raw(process), positive_source=process.positive_source)
            np.testing.assert_allclose(f32, data.astype(np.float32) / 65535.0, atol=1e-7)
            assert metadata["color_space"] is None


class TestUncharacterisedSourceResolvesToWorkingSpace:
    """A source with no embedded profile is already in the working space: "Same as
    Source" must export it without a needless conversion into a narrower gamut."""

    def test_none_resolves_to_working_space(self) -> None:
        source_cs = str({"color_space": None}.get("color_space") or WORKING_COLOR_SPACE)
        assert source_cs == WORKING_COLOR_SPACE

    def test_embedded_profile_still_wins(self) -> None:
        source_cs = str({"color_space": ColorSpace.SRGB.value}.get("color_space") or WORKING_COLOR_SPACE)
        assert source_cs == ColorSpace.SRGB.value

    def test_same_as_source_export_does_not_convert_uncharacterised_source(self) -> None:
        """The payoff: working == target, so the transform short-circuits."""
        from negpy.services.rendering.image_processor import ImageProcessor

        rng = np.random.default_rng(0)
        u16 = rng.integers(0, 65535, (16, 16, 3), dtype=np.uint16)
        proc = ImageProcessor()
        # source_cs for an untagged scan, as resolved above
        out, icc = proc._apply_color_management_u16_rgb(u16, WORKING_COLOR_SPACE, WORKING_COLOR_SPACE, None, None)
        np.testing.assert_array_equal(out, u16)
        assert icc is not None, "the file must still carry the working-space profile"


class TestWrapperGamma:
    def test_gamma_1_1_is_linear_passthrough(self) -> None:
        data = np.linspace(0.0, 1.0, 300, dtype=np.float32).reshape(10, 10, 3)
        out = NonStandardFileWrapper(data).postprocess(gamma=(1, 1), output_bps=16)
        np.testing.assert_array_equal(out, (data * 65535.0).astype(np.uint16))

    def test_default_gamma_applies_bt709_encode(self) -> None:
        data = np.linspace(0.0, 1.0, 300, dtype=np.float32).reshape(10, 10, 3)
        out = NonStandardFileWrapper(data).postprocess(output_bps=16)
        expected = np.where(data < 0.018, data * 4.5, 1.099 * np.power(data, 1.0 / 2.222) - 0.099)
        np.testing.assert_allclose(out.astype(np.float32) / 65535.0, expected, atol=1.5 / 65535.0)
