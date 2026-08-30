"""Combo-box id -> live widget, the dropdown counterpart to slider_targets.py's and
toggle_targets.py's registries. Favourites resolves combo entries through here. Kept
separate from the other two: a combo has no step, no checked state, and its own item
list can be rebuilt by its owner (e.g. on a process-mode change), which the other kinds
never need to account for."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter

#: id -> attribute path on the controls panel.
COMBO_ATTRS: dict[str, str] = {
    "fade_profile": "sensor_sidebar.fade_combo",
}

#: id -> (category, label), for the Favourites picker.
COMBO_LABELS: dict[str, tuple[str, str]] = {
    "fade_profile": ("Process", "Fade Profile"),
}


def combo_widget_map(controls) -> dict[str, Callable[[], object]]:
    """Resolve lazily: sidebars rebuild their widgets, so a getter must re-read the
    attribute rather than capture the instance."""
    return {combo_id: _bind(controls, path) for combo_id, path in COMBO_ATTRS.items()}


def _bind(controls, path: str) -> Callable[[], object]:
    getter = attrgetter(path)
    return lambda: getter(controls)
