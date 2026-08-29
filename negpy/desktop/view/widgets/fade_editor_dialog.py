from typing import List, Optional

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.tone import _CH_COLORS
from negpy.desktop.view.styles.templates import dialog_pane_qss, hint_label, pane_header_qss
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.crosstalk_editor_dialog import _MatrixGridWidget, unique_copy_name
from negpy.desktop.view.widgets.floating_panel import float_over_app
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.services.assets.crosstalk import CrosstalkType
from negpy.services.assets.fade import FadeProfiles

#: Selectable provenances, in dropdown group order. "Other" is not offered: it exists to
#: keep a hand-written type loadable, not as something to choose.
_TYPE_CHOICES: tuple[tuple[str, str], ...] = (
    (str(CrosstalkType.TUNED), "Tuned on a rig"),
    (str(CrosstalkType.MEASURED), "Measured"),
    (str(CrosstalkType.SPECSHEET), "From spec sheets (approx)"),
)

#: Off-diagonal grid position -> delta index, matching ProcessConfig.fade_delta's
#: (gr, br, rg, bg, rb, gb) order: row-major, diagonal skipped.
_DELTA_POSITIONS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


class FadeEditorDialog(QDialog):
    """Modeless editor for dye-fade side-absorption profiles (`delta`).

    The diagonal is fixed at 1.0: a profile is only the dye set's side absorptions.
    The two surviving-dye ratios that vary per slide are not profile data -- they are
    sidebar sliders, not edited here (see IMPLEMENT_FADE_AUTO.md §1-2).

    Bundled profiles and "None" are read-only (view + copy); user profiles live as
    TOMLs in the docs folder. Emits live previews as sliders move; the sidebar
    renders them and decides whether to apply or restore on close. E-6 only: every
    profile saved here describes a transparency dye set.
    """

    delta_previewed = pyqtSignal(object, float)  # (delta 6-list, strength)
    profiles_changed = pyqtSignal()

    def __init__(self, current_profile: str, current_strength: float, parent=None):
        super().__init__(parent)
        self._selected_name: Optional[str] = None
        self._updating = False

        self.setWindowTitle("Fade Restoration Profiles")
        float_over_app(self)
        self.resize(680, 620)
        self.setMinimumSize(520, 560)
        self._init_ui()

        self._reload_list(select=current_profile if current_profile in FadeProfiles.list_profiles() else FadeProfiles.NONE_NAME)
        self.preview_strength_slider.setValue(current_strength if current_strength > 0 else 1.0)

    # ------------------------------------------------------------------ UI

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left.setMinimumWidth(180)
        left.setStyleSheet(dialog_pane_qss())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        header = QLabel("PROFILES")
        header.setStyleSheet(pane_header_qss())
        left_layout.addWidget(header)

        self.profile_list = QListWidget()
        self.profile_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.profile_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.profile_list.currentRowChanged.connect(self._on_row_changed)
        left_layout.addWidget(self.profile_list)

        btns = QHBoxLayout()
        self.new_btn = self._tool_btn("fa5s.plus", "New profile (starts from identity — no side absorption)", self._on_new)
        self.copy_btn = self._tool_btn("fa5s.copy", "Make an editable copy of the selected profile", self._on_copy)
        self.delete_btn = self._tool_btn("fa5s.trash-alt", "Delete the selected profile", self._on_delete)
        btns.addWidget(self.new_btn)
        btns.addWidget(self.copy_btn)
        btns.addWidget(self.delete_btn)
        btns.addStretch()
        left_layout.addLayout(btns)

        splitter.addWidget(left)

        right = QWidget()
        right.setStyleSheet(f"background: {THEME.bg_dark};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 16, 16, 16)
        rl.setSpacing(12)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Profile name")
        self.name_edit.textChanged.connect(self._on_name_changed)
        name_row.addWidget(self.name_edit, 1)
        rl.addLayout(name_row)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type"))
        self.type_combo = QComboBox()
        for value, label in _TYPE_CHOICES:
            self.type_combo.addItem(label, value)
        self.type_combo.setToolTip(
            "<table width='300'><tr><td>"
            "Where these numbers came from — it groups the profile in the dropdown and tells the "
            "next person how far to trust it.<br><br>"
            "<b>Measured</b>: fitted against real faded and unfaded scans of the same stock.<br>"
            "<b>Tuned on a rig</b>: dialled in by eye. The default for anything you edit here.<br>"
            "<b>From spec sheets</b>: read off published dye-density curves."
            "</td></tr></table>"
        )
        type_row.addWidget(self.type_combo, 1)
        rl.addLayout(type_row)

        info = QLabel(
            "<b>Dye-fade restoration — side absorptions</b><br>"
            "A faded dye set leaks a little of one layer's density into the channels it "
            "shouldn't. That leak (δ) is a property of the dye set, not of any one frame.<br>"
            "<br>"
            "• The diagonal is fixed: this profile is <b>δ only</b>.<br>"
            "• Each off-diagonal slider is a side-absorption ratio — column is the source "
            "layer, row the one it leaks into.<br>"
            "• The two per-frame survival ratios (how much this slide itself has faded) are "
            "<b>not</b> edited here — they're the Green/Blue Survival sliders in the sidebar, "
            "or the Estimate action next to them."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            f"background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); "
            f"border-radius: 6px; padding: 8px; color: {THEME.text_secondary};"
        )
        rl.addWidget(info)

        self.readonly_hint = hint_label("Bundled profile — read-only. Make an editable copy to change it.")
        rl.addWidget(self.readonly_hint)

        rl.addWidget(self._build_grid())

        self.preview_strength_slider = CompactSlider("Preview strength", 0.0, 1.0, 1.0, has_neutral=False)
        self.preview_strength_slider.setToolTip(
            "How strongly the profile previews here (view-only — set Fade Strength in the sidebar to apply)"
        )
        self.preview_strength_slider.valueChanged.connect(lambda _v: self._emit_preview())
        rl.addWidget(self.preview_strength_slider)

        rl.addStretch()

        save_row = QHBoxLayout()
        save_row.addStretch()
        self.save_btn = QPushButton(" Save to disk")
        self.save_btn.setIcon(qta.icon("fa5s.save", color=THEME.text_primary))
        self.save_btn.setToolTip("Write this profile as a .toml in the NegPy/fade folder so it's reusable")
        self.save_btn.clicked.connect(self._on_save)
        save_row.addWidget(self.save_btn)
        rl.addLayout(save_row)

        close_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply and close")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self.accept)
        close_row.addStretch()
        close_row.addWidget(cancel_btn)
        close_row.addWidget(apply_btn)
        rl.addLayout(close_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([210, 450])
        root.addWidget(splitter)

    def _build_grid(self) -> QWidget:
        # The diagonal is fixed (a profile is delta only), so only off-diagonal cells
        # are sliders. self._cells is 3x3 with None on the diagonal, mirroring the
        # crosstalk grid exactly.
        self._cells: List[List[Optional[CompactSlider]]] = []
        container = _MatrixGridWidget(self._cells)
        grid = QGridLayout(container)
        grid.setSpacing(10)
        grid.setContentsMargins(2, 4, 2, 4)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)
        for j in (2, 3, 4):
            grid.setColumnStretch(j, 1)

        in_title = QLabel("LAYER")
        in_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        in_title.setStyleSheet(f"color: {THEME.text_secondary}; font-weight: bold; letter-spacing: 3px;")
        in_title.setToolTip("Columns are the source layer a slider mixes in; each row is the layer it leaks into.")
        grid.addWidget(in_title, 0, 2, 1, 3)

        for c in range(3):
            col = QLabel()
            col.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            col.setFixedHeight(22)
            col.setStyleSheet(f"background: {_CH_COLORS[c]}; border-radius: 4px;")
            grid.addWidget(col, 1, c + 2)

        for r in range(3):
            row_lbl = QLabel()
            row_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            row_lbl.setFixedWidth(22)
            row_lbl.setStyleSheet(f"background: {_CH_COLORS[r]}; border-radius: 4px;")
            grid.addWidget(row_lbl, r + 2, 1)
            row_cells: List[Optional[CompactSlider]] = []
            for c in range(3):
                if r == c:
                    dash = QLabel("—")
                    dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    dash.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    dash.setStyleSheet(f"color: {THEME.text_muted};")
                    dash.setToolTip("Diagonal is fixed — a profile is side absorptions only. Survival ratios are the sidebar sliders.")
                    grid.addWidget(dash, r + 2, c + 2)
                    row_cells.append(None)
                    continue
                sld = CompactSlider("", -0.3, 0.3, 0.0, step=0.001, precision=1000, has_neutral=True)
                sld.setToolTip("Side-absorption ratio (δ) — a property of the dye set, small in magnitude.")
                sld.spin.setDecimals(3)
                sld.valueChanged.connect(lambda _v: self._emit_preview())
                grid.addWidget(sld, r + 2, c + 2)
                row_cells.append(sld)
            self._cells.append(row_cells)
        return container

    def _tool_btn(self, icon: str, tooltip: str, slot) -> QPushButton:
        btn = QPushButton()
        btn.setIcon(qta.icon(icon, color=THEME.text_primary, color_disabled=THEME.text_muted))
        btn.setToolTip(tooltip)
        btn.setFixedWidth(34)
        btn.clicked.connect(slot)
        return btn

    # ------------------------------------------------------------- helpers

    def working_delta(self) -> List[float]:
        return [
            self._cells[r][c].value()  # type: ignore[union-attr]
            for r, c in _DELTA_POSITIONS
        ]

    def preview_strength(self) -> float:
        return self.preview_strength_slider.value()

    def selected_name(self) -> Optional[str]:
        return self._selected_name

    def _delta_for(self, name: str) -> List[float]:
        found = FadeProfiles.get_delta(name)
        return list(found) if found is not None else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _all_names(self) -> list:
        return FadeProfiles.list_profiles()

    def selected_type(self) -> str:
        return self.type_combo.currentData() or CrosstalkType.TUNED

    def _set_type(self, value: str) -> None:
        """Select `value`, falling back to Tuned for a built-in or hand-written type.

        Not the first entry: saving must not relabel an unknown type as a spec-sheet claim."""
        idx = self.type_combo.findData(str(value))
        self.type_combo.setCurrentIndex(idx if idx >= 0 else self.type_combo.findData(str(CrosstalkType.TUNED)))

    def _set_grid(self, delta: List[float]) -> None:
        for (r, c), v in zip(_DELTA_POSITIONS, delta):
            self._cells[r][c].setValue(v)  # type: ignore[union-attr]

    def _set_grid_enabled(self, enabled: bool) -> None:
        for row in self._cells:
            for cell in row:
                if cell is not None:
                    cell.setEnabled(enabled)

    def _emit_preview(self) -> None:
        if self._updating:
            return
        self.delta_previewed.emit(self.working_delta(), self.preview_strength())

    # ------------------------------------------------------------- list

    def _reload_list(self, select: Optional[str] = None) -> None:
        self._updating = True
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        names = [*sorted(FadeProfiles.scan_user()), FadeProfiles.NONE_NAME, *sorted(FadeProfiles.scan_bundled())]
        for name in names:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            if FadeProfiles.is_bundled(name):
                item.setForeground(QColor(THEME.text_muted))
                item.setIcon(qta.icon("fa5s.lock", color=THEME.text_muted))
            self.profile_list.addItem(item)
        self.profile_list.blockSignals(False)
        self._updating = False

        target = select if select in names else (names[0] if names else None)
        if target is not None:
            self.profile_list.setCurrentRow(names.index(target))

    def _on_row_changed(self, row: int) -> None:
        item = self.profile_list.item(row)
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self._selected_name = name
        editable = not FadeProfiles.is_bundled(name)

        self._updating = True
        self._set_grid(self._delta_for(name))
        self.name_edit.setText(name)
        self._set_type(FadeProfiles.get_type(name))
        self._updating = False

        self.name_edit.setEnabled(editable)
        self.type_combo.setEnabled(editable)
        self._set_grid_enabled(editable)
        self.save_btn.setEnabled(editable)
        self.delete_btn.setEnabled(editable)
        self.readonly_hint.setVisible(not editable)
        self._emit_preview()

    # ------------------------------------------------------------- actions

    def _on_name_changed(self, _text: str) -> None:
        if self._updating:
            return
        name = self.name_edit.text().strip()
        self.save_btn.setEnabled(bool(name) and not FadeProfiles.is_bundled(name))

    def _on_new(self) -> None:
        existing = set(self._all_names())
        name, i = "New Profile", 2
        while name in existing:
            name = f"New Profile {i}"
            i += 1
        FadeProfiles.save(name, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.profiles_changed.emit()
        self._reload_list(select=name)

    def _on_copy(self) -> None:
        if self._selected_name is None:
            return
        new_name = unique_copy_name(self._selected_name, self._all_names())
        FadeProfiles.save(new_name, self.working_delta())
        self.profiles_changed.emit()
        self._reload_list(select=new_name)

    def _on_save(self) -> None:
        name = self.name_edit.text().strip()
        if not name or FadeProfiles.is_bundled(name):
            return
        old = self._selected_name
        if old and old != name and not FadeProfiles.is_bundled(old):
            FadeProfiles.delete(old)
        FadeProfiles.save(name, self.working_delta(), self.selected_type())
        self.profiles_changed.emit()
        self._reload_list(select=name)

    def accept(self) -> None:
        # Apply-and-close persists the edited profile too (bundled/None are read-only).
        if self._selected_name is not None and not FadeProfiles.is_bundled(self._selected_name):
            self._on_save()
        super().accept()

    def _on_delete(self) -> None:
        if self._selected_name is None or FadeProfiles.is_bundled(self._selected_name):
            return
        FadeProfiles.delete(self._selected_name)
        self.profiles_changed.emit()
        self._reload_list(select=FadeProfiles.NONE_NAME)
