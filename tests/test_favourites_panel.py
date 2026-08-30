import pytest

from conftest import FakeController as _Controller, FakeRepo as _Repo
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.action_targets import ACTION_ATTRS, action_widget_map
from negpy.desktop.view.sidebar.favourites import FavouritesSidebar, load_favourites
from negpy.desktop.view.slider_shortcut_groups import SLIDER_GROUP_BY_ID
from negpy.desktop.view.slider_targets import SLIDER_ATTRS, slider_widget_map
from negpy.desktop.view.combo_targets import COMBO_ATTRS, combo_widget_map
from negpy.desktop.view.toggle_targets import TOGGLE_ATTRS, toggle_widget_map
from negpy.desktop.view.widgets.favourites_dialog import FavouritesDialog
from negpy.desktop.view.widgets.sliders import CompactSlider, HueSlider, KelvinSlider, PowerWarpSlider, clone_slider


@pytest.fixture(scope="module")
def controls(qapp):
    controller = _Controller(_Repo())
    return ControlsPanel(controller)


def test_widget_map_covers_every_shortcut_group(controls):
    assert set(slider_widget_map(controls)) == set(SLIDER_GROUP_BY_ID)


def test_every_favouritable_slider_resolves_and_is_clonable(controls):
    """Guards drift in both directions: a renamed sidebar attribute breaks resolution, and a
    slider switched to an unhandled class (e.g. PowerWarpSlider) makes clone_slider raise."""
    widgets = slider_widget_map(controls)
    for slider_id in SLIDER_ATTRS:
        src = widgets[slider_id]()
        clone = clone_slider(src)
        assert clone.label.text() == src.label.text()


def test_clone_round_trips_compact_slider_construction(qapp):
    src = CompactSlider("Density", -2.0, 2.0, 0.25, step=0.05, precision=1000, has_neutral=True, unit=" EV", inverted=True)
    clone = clone_slider(src)
    assert (clone._min, clone._max, clone._default, clone._precision) == (-2.0, 2.0, 0.25, 1000)
    assert clone.spin.singleStep() == pytest.approx(0.05)
    assert clone.spin.suffix() == " EV"
    assert clone.slider.objectName() == "neutral_slider"
    assert clone.slider.invertedAppearance()


def test_clone_round_trips_hue_and_kelvin(qapp):
    hue = clone_slider(HueSlider("Shadow hue", 210.0))
    assert isinstance(hue, HueSlider)
    assert (hue._min, hue._max, hue._default) == (0.0, 360.0, 210.0)

    kelvin = clone_slider(KelvinSlider("Temperature"))
    assert isinstance(kelvin, KelvinSlider)
    assert (kelvin._min, kelvin._max) == (3000.0, 12000.0)


def test_every_favouritable_toggle_resolves(controls):
    widgets = toggle_widget_map(controls)
    for toggle_id in TOGGLE_ATTRS:
        src = widgets[toggle_id]()
        assert src.isCheckable()


def test_every_favouritable_combo_resolves(controls):
    widgets = combo_widget_map(controls)
    for combo_id in COMBO_ATTRS:
        src = widgets[combo_id]()
        assert src.count() > 0


def test_every_favouritable_action_resolves(controls):
    widgets = action_widget_map(controls)
    for action_id in ACTION_ATTRS:
        src = widgets[action_id]()
        assert not src.isCheckable()


def test_clone_refuses_unhandled_class(qapp):
    """A PowerWarpSlider flattened into a CompactSlider would keep its range but lose its
    nonlinear travel — that must fail loudly, not ship a subtly wrong control."""
    with pytest.raises(TypeError, match="PowerWarpSlider"):
        clone_slider(PowerWarpSlider("Warp", 0.0, 4.0, 1.0, center=1.0))


def test_mirror_value_drives_the_original(qapp):
    src = CompactSlider("Chroma", 0.0, 2.0, 1.0)
    live, committed = [], []
    src.valueChanged.connect(live.append)
    src.valueCommitted.connect(committed.append)

    src.mirror_value(1.5, commit=False)
    assert live == [1.5] and committed == []

    src.mirror_value(1.75, commit=True)
    assert live == [1.5, 1.75] and committed == [1.75]


def test_mirror_commits_even_when_value_matches_a_previous_load(qapp):
    """_rebase_commit=False is load-bearing: setValue's rebase would move the commit baseline
    onto the new value and _on_committed would then see no change and stay silent."""
    src = CompactSlider("Chroma", 0.0, 2.0, 1.0)
    src.setValue(1.5)
    committed = []
    src.valueCommitted.connect(committed.append)
    src.mirror_value(1.5, commit=True)
    assert committed == []

    src.mirror_value(1.6, commit=True)
    assert committed == [1.6]


def _favourites(controls, repo):
    controls.controller.session.repo = repo
    return FavouritesSidebar(controls.controller, controls)


def test_panel_is_empty_by_default(controls, qapp):
    panel = _favourites(controls, _Repo())
    assert panel._mirrors == []
    assert panel.empty_hint.isVisible() or not panel.isVisible()


def test_panel_mirrors_stored_favourites_in_order(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["saturation", "density"]))
    labels = [clone.label.text() for _container, clone, _src, _kind in panel._mirrors]
    assert labels == [controls.lab_sidebar.saturation_slider.label.text(), controls.tone_sidebar.density_slider.label.text()]


def test_moving_a_mirror_moves_the_original(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["saturation"]))
    _container, clone, src, kind = panel._mirrors[0]
    assert kind == "slider"
    committed = []
    src.valueCommitted.connect(committed.append)

    clone.mirror_value(1.4, commit=True)
    assert committed == [1.4]
    assert src.value() == pytest.approx(1.4)


def test_sync_hides_a_mirror_only_when_the_original_is_explicitly_hidden(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["saturation"]))
    container, _clone, src, _kind = panel._mirrors[0]

    src.setVisible(False)
    panel.sync_ui()
    assert container.isHidden()

    src.setVisible(True)
    panel.sync_ui()
    assert not container.isHidden()


def test_choices_includes_a_toggle_grouped_with_its_category(controls, qapp):
    ids_in_order = [choice_id for choice_id, _category, _label in FavouritesSidebar(controls.controller, controls)._choices()]
    assert "e6_normalize" in ids_in_order
    process_ids = {group_id for group_id, group in SLIDER_GROUP_BY_ID.items() if group.category == "Process"}
    process_run = [i for i in ids_in_order if i in process_ids or i == "e6_normalize"]
    # e6_normalize's category is "Process": it must land inside that contiguous run, not
    # split off into a second, duplicate "Process" block at the end of the list.
    assert process_run[-1] == "e6_normalize"
    assert ids_in_order.index("e6_normalize") == ids_in_order.index(process_run[0]) + len(process_run) - 1


def test_clicking_a_toggle_mirror_clicks_the_original(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["e6_normalize"]))
    _container, clone, src, kind = panel._mirrors[0]
    assert kind == "toggle"
    before = src.isChecked()

    clone.click()
    assert src.isChecked() != before


def test_sync_reflects_a_toggle_mirror_checked_state(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["e6_normalize"]))
    _container, clone, src, _kind = panel._mirrors[0]

    src.setChecked(not src.isChecked())
    panel.sync_ui()
    assert clone.isChecked() == src.isChecked()


def test_choices_includes_a_combo_grouped_with_its_category(controls, qapp):
    ids_in_order = [choice_id for choice_id, _category, _label in FavouritesSidebar(controls.controller, controls)._choices()]
    assert "fade_profile" in ids_in_order
    process_ids = {group_id for group_id, group in SLIDER_GROUP_BY_ID.items() if group.category == "Process"}
    process_run = [i for i in ids_in_order if i in process_ids or i in ("e6_normalize", "fade_profile")]
    # fade_profile's category is "Process": it must land inside that contiguous run, after
    # the toggle that's already grouped there, not open a duplicate header of its own.
    assert process_run[-1] == "fade_profile"


def test_combo_mirror_starts_synced_to_the_original(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["fade_profile"]))
    _container, clone, src, kind = panel._mirrors[0]
    assert kind == "combo"
    assert clone.currentText() == src.currentText()
    assert [clone.itemText(i) for i in range(clone.count())] == [src.itemText(i) for i in range(src.count())]


def test_picking_a_combo_mirror_item_drives_the_original(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["fade_profile"]))
    _container, clone, src, _kind = panel._mirrors[0]
    selectable = [clone.itemText(i) for i in range(clone.count()) if clone.model().item(i).isEnabled()]
    other = next(text for text in selectable if text != src.currentText())

    clone.setCurrentText(other)
    assert src.currentText() == other


def test_combo_mirror_disabled_heading_rows_are_not_selectable(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["fade_profile"]))
    _container, clone, src, _kind = panel._mirrors[0]
    src_disabled = {src.itemText(i) for i in range(src.count()) if not src.model().item(i).isEnabled()}
    clone_disabled = {clone.itemText(i) for i in range(clone.count()) if not clone.model().item(i).isEnabled()}
    assert clone_disabled == src_disabled


def test_choices_includes_an_action_grouped_with_its_category(controls, qapp):
    ids_in_order = [choice_id for choice_id, _category, _label in FavouritesSidebar(controls.controller, controls)._choices()]
    assert "estimate_fade" in ids_in_order
    process_ids = {group_id for group_id, group in SLIDER_GROUP_BY_ID.items() if group.category == "Process"}
    process_tag = ("e6_normalize", "fade_profile", "estimate_fade")
    process_run = [i for i in ids_in_order if i in process_ids or i in process_tag]
    # estimate_fade's category is "Process": it must land inside that contiguous run,
    # after the toggle and combo already grouped there, not open a duplicate header.
    assert process_run[-1] == "estimate_fade"


def test_clicking_an_action_mirror_clicks_the_original(controls, qapp):
    panel = _favourites(controls, _Repo(favourite_sliders=["estimate_fade"]))
    _container, clone, src, kind = panel._mirrors[0]
    assert kind == "action"

    clicked = []
    src.clicked.connect(lambda: clicked.append(True))
    clone.click()
    assert clicked == [True]


def test_load_drops_unknown_ids(qapp):
    repo = _Repo(favourite_sliders=["density", "retired_slider", "saturation"])
    assert load_favourites(repo) == ["density", "saturation"]


def test_load_tolerates_missing_and_malformed_settings(qapp):
    assert load_favourites(_Repo()) == []
    assert load_favourites(_Repo(favourite_sliders="density")) == []


def _dialog(qapp, selected):
    choices = [("density", "Exposure", "Density"), ("grade", "Exposure", "Grade"), ("saturation", "Lab", "Chroma")]
    return FavouritesDialog(None, choices, selected)


def test_dialog_reads_the_dropped_order_back(qapp):
    """InternalMove takes the row out and re-inserts it, so the list is the truth after a drop."""
    dlg = _dialog(qapp, ["density", "grade", "saturation"])

    dlg.chosen_list.insertItem(0, dlg.chosen_list.takeItem(2))
    dlg.chosen_list.reordered.emit()

    assert dlg.selected_ids() == ["saturation", "density", "grade"]


def test_chosen_list_takes_internal_drags(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QAbstractItemView

    dlg = _dialog(qapp, ["density", "grade"])

    assert dlg.chosen_list.dragDropMode() == QAbstractItemView.DragDropMode.InternalMove
    assert dlg.chosen_list.defaultDropAction() == Qt.DropAction.MoveAction


def test_dialog_ticking_appends_and_unticking_removes(qapp):
    dlg = _dialog(qapp, ["grade"])
    assert dlg.selected_ids() == ["grade"]

    items = {dlg.available_list.item(i).data(256): dlg.available_list.item(i) for i in range(dlg.available_list.count())}
    from PyQt6.QtCore import Qt

    items["density"].setCheckState(Qt.CheckState.Checked)
    assert dlg.selected_ids() == ["grade", "density"]

    items["grade"].setCheckState(Qt.CheckState.Unchecked)
    assert dlg.selected_ids() == ["density"]


def test_dialog_drops_stored_ids_it_was_not_offered(qapp):
    dlg = _dialog(qapp, ["density", "retired_slider"])
    assert dlg.selected_ids() == ["density"]


def test_dialog_without_defaults_has_no_restore_button(qapp):
    from PyQt6.QtWidgets import QPushButton

    dlg = _dialog(qapp, ["density"])
    assert [b.text() for b in dlg.findChildren(QPushButton) if b.text() == "Restore Defaults"] == []


def test_dialog_restore_defaults_resets_the_chosen_list_and_the_ticks(qapp):
    """The toolbar editor passes defaults so the stock row is one click away."""
    choices = [("density", "Exposure", "Density"), ("grade", "Exposure", "Grade"), ("saturation", "Lab", "Chroma")]
    from PyQt6.QtCore import Qt

    dlg = FavouritesDialog(None, choices, ["saturation"], defaults=["grade", "density"])

    dlg._restore_defaults()

    assert dlg.selected_ids() == ["grade", "density"]
    states = {
        dlg.available_list.item(i).data(Qt.ItemDataRole.UserRole): dlg.available_list.item(i).checkState()
        for i in range(dlg.available_list.count())
    }
    assert states["grade"] == Qt.CheckState.Checked
    assert states["density"] == Qt.CheckState.Checked
    assert states["saturation"] == Qt.CheckState.Unchecked
