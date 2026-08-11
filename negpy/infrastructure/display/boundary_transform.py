"""Loader for the linear-boundary colour pipeline prototype.

Normalises an ICC profile at *load* time into a ``BoundaryTransform`` that consumes
scene-linear working-space RGB directly. No apply-time function here ever receives a
raw profile object or a TRC parameter -- TRC reconciliation happens once, here, not
at the call site.

Two profile shapes, two mechanisms (see docs background: an ICC profile bundles
{TRC, primaries, whitepoint} and a whole-bundle API forces every caller to re-derive
which parts to trust):

- **Matrix/TRC** (rXYZ/gXYZ/bXYZ + rTRC/gTRC/bTRC, no A2B0): the buffer's TRC is a
  pipeline fact (``working_oetf_encode``, applied once at the engine's terminal
  step), not the profile's declaration, so the profile's own TRC is discarded here,
  deliberately. Only the primaries survive, as a linear-light 3x3 matrix, adapted
  PCS(D50) -> profile's native white -> D65 (Bradford; ``chad`` gives the exact
  native white when present, else D50 is the only assumption available -- see
  ``extract_primaries_matrix``).
- **LUT** (A2B0 CLUT): the input curve *is* the TRC. Composing
  ``working_oetf_encode`` into it at load makes the resulting profile's input curve
  expect linear working RGB instead of TRC-encoded RGB, so the existing full-CMS
  path can consume it unchanged downstream.

``load_boundary_transform`` picks the mechanism; ``apply_boundary_transform`` runs
it. The two mechanisms are not symmetric in what they need at apply time -- an A2B0
tag maps device RGB to PCS (Lab/XYZ), not RGB to RGB, so unlike the matrix branch,
which fully resolves in working-space linear RGB, the LUT branch cannot stop there
and still needs a real destination profile to produce pixels. See
``apply_boundary_transform``'s docstring.

The LUT branch calls ``imagecodecs.cms_transform`` (lcms2 at full 16-bit precision,
already a dependency here -- ``ImageProcessor._apply_color_management_u16`` uses the
same call for the app's real TIFF/narrowband export path) rather than this codebase's
own ``icc_lut.py`` 3D-LUT approximation. That module's ``build_3d_lut`` samples the
profile transform via an 8-bit PIL image, which is fine when the device domain is
TRC-encoded (perceptually spread, so 8-bit levels stay meaningfully distinct in the
shadows) but not when it's linear: 1/255 in linear light is already ~19% gray, so
nearly all shadow detail collapses into a single interpolation cell no matter how the
grid points are chosen -- measured at up to ~15 LSB@8bit disagreement on real
rendered content, persisting even at grid sizes far beyond what's practical to keep
fast. Calling lcms2 directly at full precision sidesteps the 8-bit sampling ceiling
entirely and measures ~0.3 mean / ~1.3 p99 / ~6 max LSB@8bit against the unmodified
profile -- consistent with the curve-composition math's own known residual (see
``test_boundary_transform.py``), not a new source of error.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Union

import imagecodecs
import numpy as np

from negpy.kernel.image.logic import _XYZ_TO_WORKING, working_oetf_encode

# ICC's PCS is D50-relative by spec (s15Fixed16, ICC.1:2010 §7.2.16), independent of
# a profile's actual native illuminant -- use the ICC-encoded value, not the CIE one;
# `chad` is computed against the former (a 3e-5 vs 3.5e-4 difference downstream, see
# reference notes this module is built from).
_D50_PCS = np.array([0.96420, 1.0, 0.82491], dtype=np.float64)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

_BRADFORD_CONE = np.array(
    [
        [0.8951000, 0.2664000, -0.1614000],
        [-0.7502000, 1.7135000, 0.0367000],
        [0.0389000, -0.0685000, 1.0296000],
    ],
    dtype=np.float64,
)


def _bradford_adaptation(src_white: np.ndarray, dst_white: np.ndarray) -> np.ndarray:
    lms_src = _BRADFORD_CONE @ src_white
    lms_dst = _BRADFORD_CONE @ dst_white
    return np.linalg.inv(_BRADFORD_CONE) @ np.diag(lms_dst / lms_src) @ _BRADFORD_CONE


def _read_tag_table(data: bytes) -> dict[bytes, tuple[int, int]]:
    if len(data) < 132:
        return {}
    n = struct.unpack_from(">I", data, 128)[0]
    tags: dict[bytes, tuple[int, int]] = {}
    for i in range(n):
        base = 132 + i * 12
        if base + 12 > len(data):
            break
        sig = data[base : base + 4]
        off, size = struct.unpack_from(">II", data, base + 4)
        tags[sig] = (off, size)
    return tags


def _read_xyz_tag(data: bytes, off: int, size: int) -> Optional[np.ndarray]:
    if size < 20 or off + 20 > len(data):
        return None
    x, y, z = struct.unpack_from(">3i", data, off + 8)
    return np.array([x, y, z], dtype=np.float64) / 65536.0


def _read_chad_tag(data: bytes, off: int, size: int) -> Optional[np.ndarray]:
    needed = 8 + 9 * 4
    if size < needed or off + needed > len(data):
        return None
    vals = struct.unpack_from(">9i", data, off + 8)
    return np.array(vals, dtype=np.float64).reshape(3, 3) / 65536.0


def is_matrix_trc_profile(data: bytes) -> bool:
    """True for a shaper-matrix profile: colorant + TRC tags, no CLUT."""
    tags = _read_tag_table(data)
    has_colorants = all(sig in tags for sig in (b"rXYZ", b"gXYZ", b"bXYZ"))
    has_trc = any(sig in tags for sig in (b"rTRC", b"gTRC", b"bTRC"))
    has_lut = any(sig in tags for sig in (b"A2B0", b"B2A0"))
    return has_colorants and has_trc and not has_lut


def extract_primaries_matrix(data: bytes) -> Optional[np.ndarray]:
    """D65-referenced RGB->XYZ matrix from rXYZ/gXYZ/bXYZ, or None if absent/malformed
    (e.g. a GRAY-space profile, which has no colorant tags to extract)."""
    tags = _read_tag_table(data)
    cols = []
    for sig in (b"rXYZ", b"gXYZ", b"bXYZ"):
        entry = tags.get(sig)
        if entry is None:
            return None
        xyz = _read_xyz_tag(data, *entry)
        if xyz is None:
            return None
        cols.append(xyz)
    m_pcs = np.column_stack(cols)

    m_native, native_white = m_pcs, _D50_PCS
    chad_entry = tags.get(b"chad")
    if chad_entry is not None:
        chad = _read_chad_tag(data, *chad_entry)
        if chad is not None:
            try:
                inv = np.linalg.inv(chad)
                m_native, native_white = inv @ m_pcs, inv @ _D50_PCS
            except np.linalg.LinAlgError:
                pass
    return _bradford_adaptation(native_white, _D65) @ m_native


def _compose_oetf_into_a2b0_input_curve(data: bytes) -> tuple[bytes, bool]:
    """Rewrite an mft2 A2B0 tag's input curve c(b) into c'(x) = c(working_oetf_encode(x)),
    so the profile this returns expects scene-linear x on its device-RGB input instead
    of working-TRC-encoded b. Returns (data, False) unchanged -- deliberately, not an
    error -- whenever there's nothing to rewrite: no A2B0 tag (e.g. a matrix profile),
    a non-mft2 LUT structure (out of scope for this prototype), or a malformed/
    truncated tag whose declared sizes don't fit the buffer. Callers must not treat
    the returned bytes as composed unless the bool says so -- see Lut.composed and
    apply_boundary_transform's Lut branch, which sample a linear-domain source
    differently from an as-shipped (TRC-encoded) one.
    """
    tags = _read_tag_table(data)
    entry = tags.get(b"A2B0")
    if entry is None:
        return data, False
    off, _size = entry
    if off + 52 > len(data) or data[off : off + 4] != b"mft2":
        return data, False
    ic = data[off + 8]
    ni, _no = struct.unpack_from(">HH", data, off + 48)
    curve_off = off + 52
    curve_bytes = ic * ni * 2
    if curve_off + curve_bytes > len(data):
        return data, False
    curves = np.frombuffer(data, dtype=">u2", count=ic * ni, offset=curve_off).astype(np.float64).reshape(ic, ni) / 65535.0
    grid = np.linspace(0.0, 1.0, ni)
    b = np.asarray(working_oetf_encode(grid.astype(np.float32)), dtype=np.float64)

    composed = np.empty_like(curves)
    for ch in range(ic):
        composed[ch] = np.interp(b, grid, curves[ch])

    packed = np.clip(composed * 65535.0 + 0.5, 0.0, 65535.0).astype(">u2").tobytes()
    out = bytearray(data)
    out[curve_off : curve_off + len(packed)] = packed
    return bytes(out), True


@dataclass(frozen=True)
class Primaries:
    """Linear-light working-RGB -> working-RGB matrix (a source's primaries, already
    concatenated with XYZ->working). Consumes and produces scene-linear working RGB;
    no TRC parameter exists because there is nothing left to reconcile."""

    matrix: np.ndarray


@dataclass(frozen=True)
class Lut:
    """A full ICC profile, optionally with its A2B0 input curve composed with the
    working OETF so it expects scene-linear working RGB on input. Still PCS-bound
    like any A2B0 profile -- applying it requires a real destination profile, unlike
    Primaries (see apply_boundary_transform).

    ``composed`` is False for a profile load_boundary_transform couldn't confidently
    classify into either mechanism (a matrix/TRC-shaped profile with malformed
    colorant tags, a non-mft2 LUT, ...): ``profile_bytes`` is then the untouched
    original, still TRC-encoded, and apply_boundary_transform must sample and
    pre-warp it exactly like today's legacy full-CMS path -- not like a composed
    linear-domain profile, which would silently produce wrong colours (see
    ``apply_boundary_transform``).
    """

    profile_bytes: bytes
    composed: bool


BoundaryTransform = Union[Primaries, Lut]


def load_boundary_transform(icc_bytes: bytes) -> BoundaryTransform:
    """Normalise ICC profile bytes into a BoundaryTransform. Never returns a bare
    profile object with an unresolved TRC -- that resolution happens here, once."""
    if is_matrix_trc_profile(icc_bytes):
        src_to_xyz = extract_primaries_matrix(icc_bytes)
        if src_to_xyz is not None:
            matrix = _XYZ_TO_WORKING.astype(np.float64) @ src_to_xyz
            return Primaries(matrix=matrix)
    profile_bytes, composed = _compose_oetf_into_a2b0_input_curve(icc_bytes)
    return Lut(profile_bytes=profile_bytes, composed=composed)


_RELATIVE_COLORIMETRIC = 1
_BLACKPOINTCOMPENSATION = 0x2000


def apply_boundary_transform(
    linear_buf: np.ndarray,
    transform: BoundaryTransform,
    dst_profile_bytes: Optional[bytes] = None,
) -> np.ndarray:
    """Apply a normalised boundary transform to scene-linear working RGB.

    Primaries: returns scene-linear working RGB with the primaries correction
    applied -- nothing more. ``dst_profile_bytes`` is ignored. This is deliberate:
    after this call the buffer is *as if* the working-space profile were the source,
    which the caller's existing, unchanged working->destination step already knows
    how to finish (the §2.4-style invariant: selecting the working space as the
    boundary override must be identical to selecting nothing).

    Lut: an A2B0 tag maps device RGB to PCS (Lab/XYZ), not RGB to RGB, so this branch
    cannot stop at "linear working RGB" the way Primaries does -- it still runs a
    real src->dst CMS transform, same shape as today's profileToProfile call, via
    ``imagecodecs.cms_transform`` (see module docstring for why not icc_lut.py's own
    approximation). ``dst_profile_bytes`` is required here.

    Which domain the query buffer is expressed in depends on ``transform.composed``:
    a composed profile's device domain is linear, so ``linear_buf`` is fed directly.
    An uncomposed ``Lut`` (see its docstring) is still TRC-encoded, so it needs
    ``working_oetf_encode`` first -- exactly matching today's legacy full-CMS path,
    since that *is* what it falls back to.
    """
    if isinstance(transform, Primaries):
        h, w = linear_buf.shape[:2]
        flat = np.ascontiguousarray(linear_buf, dtype=np.float64).reshape(-1, 3)
        out = flat @ transform.matrix.T
        return np.clip(out, 0.0, None).reshape(h, w, 3).astype(np.float32)

    if dst_profile_bytes is None:
        raise ValueError("Lut boundary transform requires dst_profile_bytes")
    query = linear_buf if transform.composed else working_oetf_encode(linear_buf)
    query = np.clip(np.asarray(query, dtype=np.float64), 0.0, 1.0)
    img_u16 = np.clip(query * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)
    result_u16 = imagecodecs.cms_transform(
        np.ascontiguousarray(img_u16),
        transform.profile_bytes,
        dst_profile_bytes,
        colorspace="RGB",
        outcolorspace="RGB",
        intent=_RELATIVE_COLORIMETRIC,
        flags=_BLACKPOINTCOMPENSATION,
    )
    return (result_u16.astype(np.float32)) / 65535.0
