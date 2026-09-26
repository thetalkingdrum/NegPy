from typing import Callable, Optional, Sequence

from PyQt6.QtWidgets import QHBoxLayout, QInputDialog, QMessageBox, QVBoxLayout, QWidget

from negpy.desktop.view.styles.templates import disclosure_subheader, hint_label, icon_button, wrap_tooltip
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.services.assets.presets import is_valid_preset_name

#: A 3x3 matrix's off-diagonal terms in row-major order, as (row, column) channel indices. The
#: column is the source channel, the row the channel that reads it.
OFF_DIAGONAL = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
_CHANNELS = ("Red", "Green", "Blue")
#: Below the sliders' three-decimal display, so a value read back off a slider is not an edit.
_EDIT_TOLERANCE = 5e-4


def terms_differ(a: Sequence[float], b: Sequence[float]) -> bool:
    return any(abs(float(x) - float(y)) > _EDIT_TOLERANCE for x, y in zip(a, b))


class ProfileTermsGroup:
    """A profile's six off-diagonal terms as sliders, hidden until the header is clicked,
    with a note when they differ from the profile, Reset, and Save as Profile….

    The owner writes the config: `on_change(index, value, persist)` fires per slider, and
    writes only that term, since the sliders round and reading all six back would round
    the rest."""

    def __init__(
        self,
        layout: QVBoxLayout,
        title: str,
        header_tip: str,
        value_range: tuple[float, float],
        term_tip: Callable[[str, str], str],
        reset_tip: str,
        save_tip: str,
        on_change: Callable[[int, float, bool], None],
        on_reset: Callable[[], None],
        on_save: Callable[[], None],
    ) -> None:
        self.available = False
        self.header = disclosure_subheader(title)
        self.header.setToolTip(wrap_tooltip(header_tip))
        self.header.toggled.connect(lambda _on: self.sync_visibility())
        layout.addWidget(self.header)
        # Why the sliders are grayed: a state where they cannot move the render.
        self.gate_hint = hint_label("")
        layout.addWidget(self.gate_hint)

        self.sliders: list[CompactSlider] = []
        for i, (row, col) in enumerate(OFF_DIAGONAL):
            source, target = _CHANNELS[col], _CHANNELS[row]
            sld = CompactSlider(f"{source} in {target}", *value_range, 0.0, step=0.001, precision=1000, has_neutral=True)
            sld.spin.setDecimals(3)
            sld.setToolTip(wrap_tooltip(term_tip(source.lower(), target.lower())))
            sld.valueChanged.connect(lambda v, i=i: on_change(i, v, False))
            sld.valueCommitted.connect(lambda v, i=i: on_change(i, v, True))
            self.sliders.append(sld)
            layout.addWidget(sld)

        actions = QHBoxLayout()
        self.note = hint_label("")
        self.reset_btn = icon_button("fa5s.undo", reset_tip)
        self.save_btn = icon_button("fa5s.save", save_tip)
        self.reset_btn.clicked.connect(on_reset)
        self.save_btn.clicked.connect(on_save)
        actions.addWidget(self.note, 1)
        actions.addWidget(self.reset_btn)
        actions.addWidget(self.save_btn)
        layout.addLayout(actions)

    def sync(self, values: Sequence[float], note: str, inert: str, available: bool) -> None:
        """Show *values*. *note* is the edited-from-profile line ("" when unedited), *inert*
        why the terms cannot move the render ("" when they can), *available* whether the
        group belongs on the panel at all."""
        self.available = available
        for sld, v in zip(self.sliders, values):
            sld.blockSignals(True)
            sld.setValue(float(v))
            sld.blockSignals(False)
            sld.setEnabled(not inert)
        self.note.setText(note)
        self.reset_btn.setEnabled(bool(note))
        self.gate_hint.setText(inert)
        self.header.setVisible(available)
        self.sync_visibility()

    def sync_visibility(self) -> None:
        shown = self.available and self.header.isChecked()
        for w in (*self.sliders, self.reset_btn, self.save_btn):
            w.setVisible(shown)
        self.gate_hint.setVisible(shown and bool(self.gate_hint.text()))
        # Visible with the sliders hidden, so an edit is never out of sight.
        self.note.setVisible(self.available and bool(self.note.text()))


def ask_profile_name(parent: QWidget, title: str, is_reserved: Callable[[str], bool], existing: Sequence[str]) -> Optional[str]:
    """A name for a new user profile, or None. Refuses a reserved (bundled or built-in) name
    and asks before replacing one of the user's own."""
    name, ok = QInputDialog.getText(parent, title, "Profile name:")
    name = name.strip()
    if not ok or not name:
        return None
    if not is_valid_preset_name(name) or is_reserved(name):
        QMessageBox.warning(parent, title, f"“{name}” cannot be used: it names a bundled profile or holds characters a file name cannot.")
        return None
    if name in existing and QMessageBox.question(parent, title, f"Replace your profile “{name}”?") != QMessageBox.StandardButton.Yes:
        return None
    return name
