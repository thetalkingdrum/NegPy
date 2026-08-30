import qtawesome as qta
from PyQt6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from negpy.desktop.controller import AppController
from negpy.desktop.view.combo_targets import COMBO_ATTRS, COMBO_LABELS, combo_widget_map
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.slider_shortcut_groups import SLIDER_GROUPS
from negpy.desktop.view.slider_targets import SLIDER_ATTRS, slider_widget_map
from negpy.desktop.view.styles.templates import field_label, hint_label
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.toggle_targets import TOGGLE_ATTRS, TOGGLE_LABELS, toggle_widget_map
from negpy.desktop.view.widgets.collapsible import hidden_by_gating
from negpy.desktop.view.widgets.favourites_dialog import FavouritesDialog
from negpy.desktop.view.widgets.sliders import clone_slider

_SETTING_KEY = "favourite_sliders"


def load_favourites(repo) -> list[str]:
    """Drop ids that no longer exist so a retired slider, toggle or combo degrades quietly."""
    stored = repo.get_global_setting(_SETTING_KEY)
    if not isinstance(stored, list):
        return []
    return [item_id for item_id in stored if item_id in SLIDER_ATTRS or item_id in TOGGLE_ATTRS or item_id in COMBO_ATTRS]


def _clone_toggle(src: QPushButton) -> QPushButton:
    """A second checkable button onto the same control, mirroring clone_slider's role
    for sliders: forwards to the original via a real click rather than duplicating its
    binding, so persistence, mode-gating and rendering all stay with the source."""
    clone = QPushButton(src.text())
    clone.setCheckable(True)
    clone.setChecked(src.isChecked())
    clone.setIcon(src.icon())
    clone.setToolTip(src.toolTip())
    return clone


def _sync_combo(clone: QComboBox, src: QComboBox) -> None:
    """Copies src's rows -- including any non-selectable heading row, disabled through
    the model the same way _fill_fade_combo builds them -- and its current selection.
    The source can rebuild its own item list (e.g. on a process-mode change), which
    nothing announces, so this runs on every sync rather than once at construction."""
    rows = [(src.itemText(i), src.model().item(i).isEnabled()) for i in range(src.count())]
    current_rows = [(clone.itemText(i), clone.model().item(i).isEnabled()) for i in range(clone.count())]
    clone.blockSignals(True)
    if rows != current_rows:
        clone.clear()
        for text, enabled in rows:
            clone.addItem(text)
            if not enabled:
                clone.model().item(clone.count() - 1).setEnabled(False)
    clone.setCurrentText(src.currentText())
    clone.blockSignals(False)


class FavouritesSidebar(BaseSidebar):
    """User-chosen sliders and toggles gathered in one tab. Each favourite is a *mirror*
    of the real control, not the control itself — a QWidget has one parent, so
    re-parenting would tear it out of its home section. The mirror forwards to the
    original, which keeps its existing binding, mode-gating and channel-retargeting
    intact."""

    SIDE_MARGIN = 5

    def __init__(self, controller: AppController, controls):
        self.controls = controls
        # (container, clone, src, "slider" | "toggle" | "combo") -- container is what was
        # added to the layout (a combo's is a label+combo row, so its label hides and
        # disables along with it; a slider or toggle's container is the clone itself).
        # Each kind needs different value and forwarding calls, so sync_ui and _rebuild
        # branch on the tag rather than probing the widget type.
        self._mirrors: list[tuple[object, object, object, str]] = []
        super().__init__(controller)

    def _init_ui(self) -> None:
        row = QHBoxLayout()
        self.edit_btn = QPushButton("  Edit Favourites")
        self.edit_btn.setIcon(qta.icon("fa5s.sliders-h", color=THEME.text_primary))
        self.edit_btn.setToolTip("Choose which controls appear here, and in what order")
        self.edit_btn.clicked.connect(self._open_editor)
        row.addWidget(self.edit_btn)
        row.addStretch()
        self.layout.addLayout(row)

        self.empty_hint = hint_label("No favourites yet — use Edit Favourites to pick the controls you reach for most.")
        self.empty_hint.setWordWrap(True)
        self.layout.addWidget(self.empty_hint)

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(THEME.space_sm)
        self.layout.addWidget(self._container)
        self.layout.addStretch(1)

        self._rebuild()

    def _connect_signals(self) -> None:
        # Not controller.config_updated: sidebar syncing is debounced and this signal fires at the
        # end of it, so the originals already hold the fresh values.
        self.controls.modified_synced.connect(self.sync_ui)

    def _choices(self) -> list[tuple[str, str, str]]:
        """Sliders, then toggles, then combos, stably grouped by category so each lands
        inside its matching category block (e.g. Process) rather than opening a duplicate
        header of its own at the end."""
        widgets = slider_widget_map(self.controls)
        sliders = [(group.id, group.category, widgets[group.id]().label.text()) for group in SLIDER_GROUPS]
        toggles = [(toggle_id, category, label) for toggle_id, (category, label) in TOGGLE_LABELS.items()]
        combos = [(combo_id, category, label) for combo_id, (category, label) in COMBO_LABELS.items()]
        combined = sliders + toggles + combos
        category_order: dict[str, int] = {}
        for _id, category, _label in combined:
            category_order.setdefault(category, len(category_order))
        return sorted(combined, key=lambda choice: category_order[choice[1]])

    def _open_editor(self) -> None:
        repo = self.controller.session.repo
        dlg = FavouritesDialog(self, self._choices(), load_favourites(repo))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            repo.save_global_setting(_SETTING_KEY, dlg.selected_ids())
            self._rebuild()

    def _rebuild(self) -> None:
        for container, _, _, _ in self._mirrors:
            container.setParent(None)
            container.deleteLater()
        self._mirrors.clear()

        slider_widgets = slider_widget_map(self.controls)
        toggle_widgets = toggle_widget_map(self.controls)
        combo_widgets = combo_widget_map(self.controls)
        for item_id in load_favourites(self.controller.session.repo):
            if item_id in SLIDER_ATTRS:
                src = slider_widgets[item_id]()
                clone = clone_slider(src)
                clone.valueChanged.connect(lambda v, s=src: s.mirror_value(v, commit=False))
                clone.valueCommitted.connect(lambda v, s=src: s.mirror_value(v, commit=True))
                self._container_layout.addWidget(clone)
                self._mirrors.append((clone, clone, src, "slider"))
            elif item_id in TOGGLE_ATTRS:
                src = toggle_widgets[item_id]()
                clone = _clone_toggle(src)
                clone.clicked.connect(lambda _checked, s=src: s.click())
                self._container_layout.addWidget(clone)
                self._mirrors.append((clone, clone, src, "toggle"))
            else:
                src = combo_widgets[item_id]()
                _category, item_label = COMBO_LABELS[item_id]
                clone = QComboBox()
                clone.setToolTip(src.toolTip())
                clone.currentTextChanged.connect(lambda text, s=src: s.setCurrentText(text))
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.addWidget(field_label(item_label))
                row_layout.addWidget(clone, 1)
                self._container_layout.addWidget(row)
                self._mirrors.append((row, clone, src, "combo"))

        self.empty_hint.setVisible(not self._mirrors)
        self.sync_ui()

    def sync_ui(self) -> None:
        for container, clone, src, kind in self._mirrors:
            if kind == "slider":
                clone.setValue(src.value())
            elif kind == "toggle":
                clone.setChecked(src.isChecked())
            else:
                _sync_combo(clone, src)
            # Only mode gating should hide a mirror. A collapsed section or an off-screen tab must
            # not. B&W hides the whole Colour section, so isHidden() alone misses it.
            container.setVisible(not hidden_by_gating(src))
            container.setEnabled(src.isEnabled())
