"""The film strip flags a thumbnail whose fingerprint differs from its saved edit's."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QStyleOptionViewItem

from negpy.desktop.view.sidebar.files import _ThumbnailDelegate
from negpy.desktop.view.styles.theme import THEME
from negpy.services.assets.thumbnails import asset_thumbnail_key


def _paint(delegate, file_info: dict) -> QPixmap:
    thumb = QPixmap(60, 40)
    thumb.fill(QColor(120, 120, 120))
    index = MagicMock()
    index.data.side_effect = lambda role: file_info if role == Qt.ItemDataRole.UserRole else QIcon(thumb)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 66, 46)
    target = QPixmap(66, 46)
    target.fill(QColor(0, 0, 0))
    painter = QPainter(target)
    delegate.paint(painter, option, index)
    painter.end()
    return target


def _top_left_colour(pix: QPixmap) -> QColor:
    # Dot center: image rect starts at margin 3, dot radius 4 offset by 4px from each edge.
    return QColor(pix.toImage().pixel(3 + 4 + 4, 3 + 4 + 4))


KEY = asset_thumbnail_key({"hash": "h1"})


def _state(stored=None, expected=None, current_hash=None):
    return SimpleNamespace(
        is_dirty=False,
        current_file_path=None,
        current_file_hash=current_hash,
        thumbnail_fingerprints={KEY: stored} if stored else {},
        expected_thumbnail_fingerprints={KEY: expected},
    )


def _is_amber(state, file_hash="h1") -> bool:
    dot = _top_left_colour(_paint(_ThumbnailDelegate(state=state), {"path": "/a.nef", "hash": file_hash}))
    return dot.name().upper() == THEME.warn_amber.upper()


def test_mismatched_fingerprint_gets_a_dot(qapp):
    assert _is_amber(_state(stored="a" * 32, expected="b" * 32))


def test_placeholder_of_an_edited_frame_gets_a_dot(qapp):
    assert _is_amber(_state(stored=None, expected="b" * 32))


def test_matching_fingerprint_does_not(qapp):
    assert not _is_amber(_state(stored="b" * 32, expected="b" * 32))


def test_acceptable_placeholder_does_not(qapp):
    assert not _is_amber(_state(stored=None, expected=None))


def test_active_frame_never_stale(qapp):
    assert not _is_amber(_state(stored="a" * 32, expected="b" * 32, current_hash="h1"))


def test_other_frame_does_not(qapp):
    assert not _is_amber(_state(stored="a" * 32, expected="b" * 32), file_hash="h2")


def test_no_state_never_stale(qapp):
    clean = _top_left_colour(_paint(_ThumbnailDelegate(state=None), {"path": "/a.nef", "hash": "h1"}))
    assert clean.name().upper() != THEME.warn_amber.upper()
