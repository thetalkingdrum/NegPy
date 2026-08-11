"""Tests for the linear-boundary prototype's BoundaryTransform loader
(negpy/infrastructure/display/boundary_transform.py).

Validates:
- Profile-type detection (matrix/TRC vs LUT-based)
- Primaries matrix extraction (PCS-D50 tag values -> D65-referenced result)
- The load-bearing identity: the working-space profile as boundary override == no override
- The composed A2B0 input curve (LUT branch) against the app's own shipped RGBScan.icc
"""

import os
import struct
from typing import Optional

import imagecodecs
import numpy as np
import pytest

from negpy.infrastructure.display.boundary_transform import (
    Lut,
    Primaries,
    apply_boundary_transform,
    extract_primaries_matrix,
    is_matrix_trc_profile,
    load_boundary_transform,
)
from negpy.kernel.image.logic import working_oetf_decode, working_oetf_encode

_RELATIVE_COLORIMETRIC = 1
_BLACKPOINTCOMPENSATION = 0x2000

_ICC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icc")


def _s15fixed16(val: float) -> bytes:
    return struct.pack(">i", int(round(val * 65536.0)))


def _xyz_tag(x: float, y: float, z: float) -> bytes:
    return b"XYZ " + b"\x00" * 4 + _s15fixed16(x) + _s15fixed16(y) + _s15fixed16(z)


def _trc_tag_gamma(gamma: float) -> bytes:
    return b"curv" + b"\x00" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(gamma * 256.0))) + b"\x00\x00"


def _chad_tag(m: np.ndarray) -> bytes:
    return b"sf32" + b"\x00" * 4 + b"".join(_s15fixed16(v) for v in m.flatten())


def _build_icc(tag_data: list[tuple[bytes, bytes]], color_space: bytes = b"RGB ") -> bytes:
    tag_count = len(tag_data)
    header_size = 128 + 4 + tag_count * 12
    offset = header_size
    offsets: list[tuple[int, int]] = []
    for _, payload in tag_data:
        padded = len(payload) + (-len(payload) % 4)
        offsets.append((offset, len(payload)))
        offset += padded
    total_size = offset

    header = bytearray(128)
    struct.pack_into(">I", header, 0, total_size)
    header[36:40] = b"acsp"
    header[12:16] = b"mntr"
    header[16:20] = color_space
    header[40:44] = b"APPL"

    tag_table = struct.pack(">I", tag_count)
    for i, (sig, _) in enumerate(tag_data):
        tag_table += sig + struct.pack(">II", offsets[i][0], offsets[i][1])

    body = b""
    for _, payload in tag_data:
        padded = payload + b"\x00" * (-len(payload) % 4)
        body += padded

    return bytes(header) + tag_table + body


def _build_matrix_trc_icc(
    r_xyz: tuple[float, float, float],
    g_xyz: tuple[float, float, float],
    b_xyz: tuple[float, float, float],
    wtpt: tuple[float, float, float] = (0.9642, 1.0000, 0.8249),
    gamma: float = 2.2,
    add_a2b0: bool = False,
    chad: Optional[np.ndarray] = None,
) -> bytes:
    tag_data: list[tuple[bytes, bytes]] = [
        (b"rXYZ", _xyz_tag(*r_xyz)),
        (b"gXYZ", _xyz_tag(*g_xyz)),
        (b"bXYZ", _xyz_tag(*b_xyz)),
        (b"wtpt", _xyz_tag(*wtpt)),
    ]
    trc = _trc_tag_gamma(gamma)
    tag_data += [(b"rTRC", trc), (b"gTRC", trc), (b"bTRC", trc)]
    if chad is not None:
        tag_data.append((b"chad", _chad_tag(chad)))
    if add_a2b0:
        tag_data.append((b"A2B0", b"mft2" + b"\x00" * 40))
    return _build_icc(tag_data)


def _corrupt_tag_field(data: bytes, sig: bytes, *, offset: Optional[int] = None, size: Optional[int] = None) -> bytes:
    """Overwrite a tag table entry's declared offset/size in place, without touching
    the tag's actual payload -- builds fixtures for "littleCMS loads it fine, this
    module's lightweight parser must not trust it" and "truncated/out-of-range tag,
    must degrade gracefully rather than crash"."""
    tag_count = struct.unpack_from(">I", data, 128)[0]
    out = bytearray(data)
    for i in range(tag_count):
        base = 132 + i * 12
        if out[base : base + 4] == sig:
            cur_off, cur_size = struct.unpack_from(">II", out, base + 4)
            struct.pack_into(">II", out, base + 4, offset if offset is not None else cur_off, size if size is not None else cur_size)
            return bytes(out)
    raise KeyError(sig)


def _build_gray_icc() -> bytes:
    tag_data: list[tuple[bytes, bytes]] = [
        (b"wtpt", _xyz_tag(0.9642, 1.0, 0.8249)),
        (b"bkpt", _xyz_tag(0.0, 0.0, 0.0)),
        (b"kTRC", _trc_tag_gamma(2.2)),
    ]
    return _build_icc(tag_data, color_space=b"GRAY")


# PCS-D50-relative sRGB primaries as a real v4 profile stores them.
_SRGB_R = (0.43603516, 0.22248840, 0.01391602)
_SRGB_G = (0.38511658, 0.71690369, 0.09706116)
_SRGB_B = (0.14305115, 0.06060791, 0.71392822)
_SRGB_CHAD = np.array(
    [
        [1.04788208, 0.0229187, -0.05021667],
        [0.02958679, 0.99047852, -0.01707458],
        [-0.00924683, 0.01507568, 0.75167847],
    ]
)
_SRGB_R_D65 = (0.4124564, 0.2126729, 0.0193339)
_SRGB_G_D65 = (0.3575761, 0.7151522, 0.1191920)
_SRGB_B_D65 = (0.1804375, 0.0721750, 0.9503041)

# D60-native (ACES AP0) — inv(chad) alone lands on D60, not D65.
_ACES_R = (0.9908905, 0.3618927, -0.00271606)
_ACES_G = (0.01223755, 0.72251892, 0.008255)
_ACES_B = (-0.03892517, -0.08441162, 0.81936646)
_ACES_CHAD = np.array(
    [
        [1.03416443, 0.01681519, -0.03747559],
        [0.0216217, 0.99223328, -0.01272583],
        [-0.00694275, 0.01132202, 0.8130188],
    ]
)

# D50-native (ProPhoto) — chad is near-identity, stays at D50.
_PROPHOTO_R = (0.79771423, 0.28805542, 0.0)
_PROPHOTO_G = (0.13516235, 0.71186829, 1.526e-05)
_PROPHOTO_B = (0.03132629, 7.629e-05, 0.82489014)
_PROPHOTO_CHAD = np.array(
    [
        [9.99954220e-01, -3.05200000e-05, -3.05200000e-05],
        [-4.57800000e-05, 1.00004578e00, -1.52600000e-05],
        [-1.52600000e-05, 1.52600000e-05, 9.99740600e-01],
    ]
)

_D65_XYZ = np.array([0.95047, 1.0, 1.08883])


class TestProfileDetection:
    def test_matrix_trc_detected(self):
        assert is_matrix_trc_profile(_build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B))

    def test_lut_not_detected_as_matrix(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, add_a2b0=True)
        assert not is_matrix_trc_profile(icc)

    def test_gray_profile_not_matrix_and_extraction_is_none(self):
        icc = _build_gray_icc()
        assert not is_matrix_trc_profile(icc)
        assert extract_primaries_matrix(icc) is None

    def test_rgbscan_icc_is_lut_not_matrix(self):
        with open(os.path.join(_ICC_DIR, "RGBScan.icc"), "rb") as f:
            data = f.read()
        assert not is_matrix_trc_profile(data)

    def test_adobe_compat_v4_is_matrix(self):
        with open(os.path.join(_ICC_DIR, "AdobeCompat-v4.icc"), "rb") as f:
            data = f.read()
        assert is_matrix_trc_profile(data)


class TestPrimariesExtraction:
    def test_extract_srgb_via_chad(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, chad=_SRGB_CHAD)
        m = extract_primaries_matrix(icc)
        expected = np.array([_SRGB_R_D65, _SRGB_G_D65, _SRGB_B_D65], dtype=np.float64).T
        np.testing.assert_allclose(m, expected, atol=5e-4)

    def test_extract_srgb_bradford_fallback_without_chad(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B)
        m = extract_primaries_matrix(icc)
        expected = np.array([_SRGB_R_D65, _SRGB_G_D65, _SRGB_B_D65], dtype=np.float64).T
        np.testing.assert_allclose(m, expected, atol=2e-3)

    def test_returns_none_without_colorant_tags(self):
        header = bytearray(128)
        struct.pack_into(">I", header, 0, 132)
        header[36:40] = b"acsp"
        data = bytes(header) + struct.pack(">I", 0)
        assert extract_primaries_matrix(data) is None

    @pytest.mark.parametrize(
        "r,g,b,chad,label",
        [
            (_SRGB_R, _SRGB_G, _SRGB_B, _SRGB_CHAD, "sRGB (D65-native)"),
            (_ACES_R, _ACES_G, _ACES_B, _ACES_CHAD, "ACES AP0 (D60-native)"),
            (_PROPHOTO_R, _PROPHOTO_G, _PROPHOTO_B, _PROPHOTO_CHAD, "ProPhoto (D50-native)"),
        ],
    )
    def test_extracted_primaries_are_d65_referenced(self, r, g, b, chad, label):
        """Standing invariant (independent of the profile's own native white):
        extract_primaries_matrix(p) @ (1,1,1) ~= D65."""
        icc = _build_matrix_trc_icc(r, g, b, chad=chad)
        m = extract_primaries_matrix(icc)
        white = m.sum(axis=1)
        np.testing.assert_allclose(white, _D65_XYZ, atol=1e-3, err_msg=f"{label}: white did not land on D65")


class TestTrcIndependence:
    def test_different_trcs_extract_identical_primaries(self):
        """The matrix branch discards TRC entirely -- two profiles differing only in
        declared gamma must produce the exact same extracted matrix."""
        icc_g18 = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, gamma=1.8)
        icc_g24 = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, gamma=2.4)
        m1 = extract_primaries_matrix(icc_g18)
        m2 = extract_primaries_matrix(icc_g24)
        np.testing.assert_array_equal(m1, m2)


class TestLoadBoundaryTransform:
    def test_matrix_profile_loads_as_primaries(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, chad=_SRGB_CHAD)
        transform = load_boundary_transform(icc)
        assert isinstance(transform, Primaries)

    def test_lut_profile_loads_as_lut(self):
        with open(os.path.join(_ICC_DIR, "RGBScan.icc"), "rb") as f:
            data = f.read()
        transform = load_boundary_transform(data)
        assert isinstance(transform, Lut)

    def test_working_space_profile_is_near_identity_matrix(self):
        """The single-line invariant the whole design collapses to: selecting the
        working-space profile as the boundary override must be identical to selecting
        nothing, i.e. its Primaries matrix must be I to within a few 16-bit LSB."""
        with open(os.path.join(_ICC_DIR, "AdobeCompat-v4.icc"), "rb") as f:
            data = f.read()
        transform = load_boundary_transform(data)
        assert isinstance(transform, Primaries)
        np.testing.assert_allclose(transform.matrix, np.eye(3), atol=5e-4)


class TestApplyBoundaryTransformPrimaries:
    def test_identity_matrix_is_a_noop(self):
        img = np.random.RandomState(7).rand(32, 32, 3).astype(np.float32) * 0.8 + 0.1
        out = apply_boundary_transform(img, Primaries(matrix=np.eye(3, dtype=np.float64)))
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_property_working_space_override_matches_no_override(self):
        """Property test over random chromatic inputs standing in for §2.4: rendering
        through the working-space profile's Primaries transform must reproduce the
        input to within a few 16-bit LSB, for arbitrary chromatic (non-gray) patches."""
        with open(os.path.join(_ICC_DIR, "AdobeCompat-v4.icc"), "rb") as f:
            data = f.read()
        transform = load_boundary_transform(data)
        rng = np.random.RandomState(123)
        img = rng.rand(64, 64, 3).astype(np.float32)
        out = apply_boundary_transform(img, transform)
        np.testing.assert_allclose(out, img, atol=5e-4)

    def test_shape_and_dtype_preserved(self):
        img = np.random.RandomState(0).rand(50, 20, 3).astype(np.float32)
        out = apply_boundary_transform(img, Primaries(matrix=np.eye(3, dtype=np.float64)))
        assert out.shape == img.shape
        assert out.dtype == np.float32

    def test_output_never_negative(self):
        img = np.random.RandomState(0).rand(8, 8, 3).astype(np.float32) - 0.5
        m = -np.eye(3, dtype=np.float64)
        out = apply_boundary_transform(img, Primaries(matrix=m))
        assert out.min() >= 0.0


class TestApplyBoundaryTransformLut:
    def test_requires_dst_profile_bytes(self):
        transform = Lut(profile_bytes=b"not a real profile", composed=False)
        img = np.zeros((2, 2, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            apply_boundary_transform(img, transform)

    def test_rgbscan_composed_curve_matches_srgb_encode_of_linear(self):
        """Step 0's validated claim, pinned as a permanent test: composing
        working_oetf_encode into RGBScan.icc's A2B0 input curve collapses to
        sRGB_encode(x) directly on the *linear* value, within the known
        563/256-vs-2.2 curve-fit residual (~0.037 LSB @ 8-bit, ~1.5e-4 normalized).
        This is the numerical foundation the whole prototype stands on -- if a future
        change to the finish-step OETF breaks this, it must fail loudly here, not
        silently in the shadows of a scan.
        """
        with open(os.path.join(_ICC_DIR, "RGBScan.icc"), "rb") as f:
            data = f.read()
        transform = load_boundary_transform(data)
        assert isinstance(transform, Lut)

        tags_off = struct.unpack_from(">I", transform.profile_bytes, 128)[0]
        tag_count = tags_off
        tags: dict[bytes, tuple[int, int]] = {}
        for i in range(tag_count):
            base = 132 + i * 12
            sig = transform.profile_bytes[base : base + 4]
            off, sz = struct.unpack_from(">II", transform.profile_bytes, base + 4)
            tags[sig] = (off, sz)
        off, _ = tags[b"A2B0"]
        assert transform.profile_bytes[off : off + 4] == b"mft2"
        ni = struct.unpack_from(">H", transform.profile_bytes, off + 48)[0]
        curve = np.array(struct.unpack_from(f">{ni}H", transform.profile_bytes, off + 52), dtype=np.float64) / 65535.0

        x = np.linspace(0.0, 1.0, ni)
        srgb_encoded = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
        assert np.max(np.abs(curve - srgb_encoded)) < 2e-4

    def test_rgbscan_original_curve_unchanged_before_compose(self):
        """Sanity check on the rewrite itself: before composing, the curve matches the
        old contract (decode pure-power ~2.2, re-encode sRGB) -- confirms the rewrite
        changed the domain, not just accidentally converged on the same numbers."""
        with open(os.path.join(_ICC_DIR, "RGBScan.icc"), "rb") as f:
            data = f.read()
        tag_count = struct.unpack_from(">I", data, 128)[0]
        tags: dict[bytes, tuple[int, int]] = {}
        for i in range(tag_count):
            base = 132 + i * 12
            sig = data[base : base + 4]
            o, sz = struct.unpack_from(">II", data, base + 4)
            tags[sig] = (o, sz)
        off, _ = tags[b"A2B0"]
        ni = struct.unpack_from(">H", data, off + 48)[0]
        curve = np.array(struct.unpack_from(f">{ni}H", data, off + 52), dtype=np.float64) / 65535.0

        x = np.linspace(0.0, 1.0, ni)
        linear = working_oetf_decode(x.astype(np.float32)).astype(np.float64)
        srgb_of_linear = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
        assert np.max(np.abs(curve - srgb_of_linear)) < 2e-4

    def test_equivalence_against_legacy_path_on_chromatic_patches(self):
        """Step-3-style equivalence proof, scoped to this module: for the same
        scene-linear buffer, the new linear-boundary path (composed-curve profile,
        fed linear pixels directly) must agree with today's shipped path
        (working_oetf_encode then the original profile through the unmodified
        full-CMS pipeline) to within the curve-composition math's own known residual
        -- both sides go through imagecodecs.cms_transform (real lcms2, full 16-bit
        precision, no LUT-grid approximation), matching
        ImageProcessor._apply_color_management_u16, the app's actual production path.

        Tolerances below are set from measurement (seed=99, 64x64 random chromatic
        patch) and independently confirmed against real DarkroomEngine-rendered
        content in tests/test_linear_boundary_engine_integration.py.
        """
        with open(os.path.join(_ICC_DIR, "RGBScan.icc"), "rb") as f:
            rgbscan_bytes = f.read()
        with open(os.path.join(_ICC_DIR, "AdobeCompat-v4.icc"), "rb") as f:
            dst_bytes = f.read()
        transform = load_boundary_transform(rgbscan_bytes)

        rng = np.random.RandomState(99)
        linear = rng.rand(64, 64, 3).astype(np.float32)

        new_out = apply_boundary_transform(linear, transform, dst_profile_bytes=dst_bytes)

        encoded_u16 = np.clip(working_oetf_encode(linear).astype(np.float64) * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        legacy_u16 = imagecodecs.cms_transform(
            np.ascontiguousarray(encoded_u16),
            rgbscan_bytes,
            dst_bytes,
            colorspace="RGB",
            outcolorspace="RGB",
            intent=_RELATIVE_COLORIMETRIC,
            flags=_BLACKPOINTCOMPENSATION,
        )
        legacy_out = legacy_u16.astype(np.float64) / 65535.0

        diff_lsb8 = np.abs(new_out.astype(np.float64) - legacy_out) * 255.0
        assert diff_lsb8.mean() < 1.0, f"mean {diff_lsb8.mean():.2f} LSB@8bit"
        assert np.percentile(diff_lsb8, 99) < 3.0, f"p99 {np.percentile(diff_lsb8, 99):.2f} LSB@8bit"
        assert diff_lsb8.max() < 10.0, f"max {diff_lsb8.max():.2f} LSB@8bit"


class TestMisclassifiedProfileFallback:
    """A profile that looks matrix/TRC but whose colorant tags are unreadable (e.g.
    a declared tag size too small for the payload -- littleCMS itself is lenient
    about this and loads such a profile fine) must fall back to Lut(composed=False)
    and be sampled exactly like today's legacy full-CMS path -- i.e. the query
    buffer must be working_oetf_encode'd first, matching an uncomposed (still
    TRC-encoded) profile's actual device domain. composed=True and a raw, unencoded
    linear query are only valid for a profile this module actually rewrote; getting
    that pairing wrong for an uncomposed profile silently produces badly wrong
    colours (regression: previously ~48 LSB@8bit mean error, no error or warning)."""

    def test_malformed_colorant_tag_falls_back_to_uncomposed_lut(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, chad=_SRGB_CHAD)
        # rXYZ's declared tag size says 4 bytes, too small for a 20-byte XYZ payload.
        # The tag is otherwise intact -- only the declared size is wrong.
        corrupted = _corrupt_tag_field(icc, b"rXYZ", size=4)
        assert extract_primaries_matrix(corrupted) is None

        transform = load_boundary_transform(corrupted)
        assert isinstance(transform, Lut)
        assert transform.composed is False
        # No A2B0 tag on a matrix/TRC-shaped profile, so the rewrite is a no-op --
        # profile_bytes must be the original, untouched (still TRC-encoded) bytes.
        assert transform.profile_bytes == corrupted

    def test_uncomposed_lut_matches_legacy_path_not_linear_domain_query(self):
        """End-to-end regression for the bug itself: an uncomposed Lut (composed=False)
        must have its query buffer working_oetf_encode'd before the CMS call, exactly
        matching today's legacy full-CMS path -- feeding it a raw linear query (valid
        only for a composed profile) would silently sample the wrong region of a
        still-TRC-encoded profile's domain. Constructs Lut directly (rather than via a
        corrupted real profile, which littleCMS may or may not tolerate depending on
        exactly what's corrupted) so this pins apply_boundary_transform's own
        contract regardless of that."""
        with open(os.path.join(_ICC_DIR, "RGBScan.icc"), "rb") as f:
            rgbscan = f.read()
        with open(os.path.join(_ICC_DIR, "AdobeCompat-v4.icc"), "rb") as f:
            dst_bytes = f.read()

        rng = np.random.RandomState(11)
        linear = rng.rand(32, 32, 3).astype(np.float32)

        uncomposed = Lut(profile_bytes=rgbscan, composed=False)
        new_out = apply_boundary_transform(linear, uncomposed, dst_profile_bytes=dst_bytes)

        encoded_u16 = np.clip(working_oetf_encode(linear).astype(np.float64) * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        legacy_u16 = imagecodecs.cms_transform(
            np.ascontiguousarray(encoded_u16),
            rgbscan,
            dst_bytes,
            colorspace="RGB",
            outcolorspace="RGB",
            intent=_RELATIVE_COLORIMETRIC,
            flags=_BLACKPOINTCOMPENSATION,
        )
        legacy_out = legacy_u16.astype(np.float64) / 65535.0

        diff_lsb8 = np.abs(new_out.astype(np.float64) - legacy_out) * 255.0
        assert diff_lsb8.max() < 1.0, f"max {diff_lsb8.max():.2f} LSB@8bit -- unencoded linear query leaked in?"


class TestBoundsSafety:
    """A truncated or corrupted ICC file must degrade gracefully (return None /
    composed=False), not crash with an uncaught struct.error -- matches this
    codebase's convention elsewhere for parsing untrusted binary metadata
    (infrastructure/loaders/helpers.py)."""

    def test_out_of_range_colorant_offset_does_not_crash(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, chad=_SRGB_CHAD)
        corrupted = _corrupt_tag_field(icc, b"rXYZ", offset=999_999)
        assert extract_primaries_matrix(corrupted) is None
        transform = load_boundary_transform(corrupted)
        assert isinstance(transform, Lut)
        assert transform.composed is False

    def test_out_of_range_chad_offset_falls_back_to_bradford(self):
        """A chad tag that can't be read must fall back to the no-chad Bradford path,
        not crash -- mirrors the existing np.linalg.LinAlgError handling a few lines
        below it in extract_primaries_matrix."""
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, chad=_SRGB_CHAD)
        corrupted = _corrupt_tag_field(icc, b"chad", offset=999_999)
        m = extract_primaries_matrix(corrupted)
        assert m is not None

    def test_out_of_range_a2b0_offset_does_not_crash(self):
        with open(os.path.join(_ICC_DIR, "RGBScan.icc"), "rb") as f:
            data = f.read()
        corrupted = _corrupt_tag_field(data, b"A2B0", offset=999_999)
        transform = load_boundary_transform(corrupted)
        assert isinstance(transform, Lut)
        assert transform.composed is False
        assert transform.profile_bytes == corrupted
