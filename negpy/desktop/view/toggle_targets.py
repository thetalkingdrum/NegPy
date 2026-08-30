"""Toggle id -> live widget, the checkable-button counterpart to slider_targets.py's
registry. Favourites resolves toggle entries through here. Kept separate from
SLIDER_ATTRS: a toggle has no step or inc/dec pair, so it has no keyboard-shortcut
group and no place in SLIDER_GROUPS."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter

#: id -> attribute path on the controls panel.
TOGGLE_ATTRS: dict[str, str] = {
    "e6_normalize": "process_sidebar.normalize_e6_btn",
}

#: id -> (category, label), for the Favourites picker. A checkable QPushButton has no
#: on-panel label widget to read back, unlike CompactSlider, so this is spelled out here.
TOGGLE_LABELS: dict[str, tuple[str, str]] = {
    "e6_normalize": ("Process", "Normalize"),
}


def toggle_widget_map(controls) -> dict[str, Callable[[], object]]:
    """Resolve lazily: sidebars rebuild their widgets, so a getter must re-read the
    attribute rather than capture the instance."""
    return {toggle_id: _bind(controls, path) for toggle_id, path in TOGGLE_ATTRS.items()}


def _bind(controls, path: str) -> Callable[[], object]:
    getter = attrgetter(path)
    return lambda: getter(controls)
