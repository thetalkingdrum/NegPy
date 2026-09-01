"""Action id -> live widget, the one-shot-button counterpart to slider_targets.py,
toggle_targets.py and combo_targets.py. Favourites resolves action entries through
here. A one-shot action has no value or checked state to mirror and, being icon-only,
no on-panel text either -- forwarding a click is the whole job, and the label comes
from ACTION_LABELS rather than the widget."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter

#: id -> attribute path on the controls panel.
ACTION_ATTRS: dict[str, str] = {
    "estimate_fade": "sensor_sidebar.estimate_fade_btn",
}

#: id -> (category, label), for the Favourites picker and panel row.
ACTION_LABELS: dict[str, tuple[str, str]] = {
    "estimate_fade": ("Process", "Estimate"),
}

#: id -> attribute path of the hint label under the action's own control, for the
#: subset of actions that report a result (e.g. Estimate's survival-ratio readout).
#: Not every action has one.
ACTION_HINT_ATTRS: dict[str, str] = {
    "estimate_fade": "sensor_sidebar.fade_estimate_hint",
}


def action_widget_map(controls) -> dict[str, Callable[[], object]]:
    """Resolve lazily: sidebars rebuild their widgets, so a getter must re-read the
    attribute rather than capture the instance."""
    return {action_id: _bind(controls, path) for action_id, path in ACTION_ATTRS.items()}


def action_hint_widget_map(controls) -> dict[str, Callable[[], object]]:
    """Same lazy-resolution rationale as action_widget_map, for the hint labels."""
    return {action_id: _bind(controls, path) for action_id, path in ACTION_HINT_ATTRS.items()}


def _bind(controls, path: str) -> Callable[[], object]:
    getter = attrgetter(path)
    return lambda: getter(controls)
