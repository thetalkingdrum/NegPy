from PyQt6.QtWidgets import QComboBox, QDialog, QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, hint_label, section_subheader, wrap_tooltip
from negpy.desktop.view.widgets.file_dialogs import last_open_folder
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.exposure.normalization import fade_delta_conflict_reason, fade_reject_reason
from negpy.features.process.fade import RATIO_BOUNDS, RED_SURVIVAL_BOUNDS
from negpy.features.process.models import ProcessMode, invalidate_local_bounds
from negpy.features.process.sensor import unmix_block_reason
from negpy.services.assets.crosstalk import CrosstalkProfiles
from negpy.services.assets.fade import FadeProfiles
from negpy.services.assets.sensor import SensorProfiles

#: The manual sliders share the estimator's sane bounds: a hand-set value the estimator
#: itself could never produce (and vice versa) is exactly the inconsistency to avoid.
_RATIO_SLIDER_RANGE = RATIO_BOUNDS


class SensorSidebar(BaseSidebar):
    """
    Capture-side color corrections, one per cause and not interchangeable: the
    camera's cross-channel response (linear capture), the film's dye absorptions
    (negative density), and an odd light spectrum's hue rotation (the print).
    """

    def _init_ui(self) -> None:
        conf = self.state.config.process

        self.capture_header = section_subheader("CAPTURE")
        self.layout.addWidget(self.capture_header)

        self.linear_raw_btn = self._small_toggle(
            "fa5s.sliders-h",
            "Linear RAW",
            conf.linear_raw,
            "Decode RAW with neutral multipliers (1,1,1,1) — bypasses as-shot camera white balance for a clean starting point",
        )
        self.narrowband_scan_btn = self._small_toggle(
            "mdi6.led-strip-variant",
            "Narrowband",
            conf.narrowband_scan,
            "Correct narrowband capture oversaturation with the bundled input profile. "
            "An explicit Input ICC in Export settings overrides it. Not applied to transparencies: "
            "the profile describes narrowband capture of negative dyes",
        )
        self.scan_setup_btn = self._icon_action(
            "mdi6.lightbulb-on-outline",
            "Scanning setup — set Linear RAW and Narrowband from your camera/scanner and its light source",
            width=28,
        )
        capture_row = QHBoxLayout()
        capture_row.addWidget(self.linear_raw_btn, 1)
        capture_row.addWidget(self.narrowband_scan_btn, 1)
        capture_row.addWidget(self.scan_setup_btn)
        self.layout.addLayout(capture_row)

        # Greyed rather than hidden: these are sticky settings, so a hidden one is a setting the
        # user cannot see the state of. Hiding them is what let a rig's narrowband pair follow a
        # frame into Transparency unnoticed.
        self.capture_hint = hint_label("")
        self.capture_hint.setVisible(False)  # text and tooltip are set per film process in sync_ui
        self.layout.addWidget(self.capture_hint)

        self.layout.addWidget(section_subheader("SINGLE-SHOT NARROWBAND CALIBRATION"))

        row = QHBoxLayout()
        self.sensor_label = field_label("Profile")
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(SensorProfiles.list_profiles())
        self.sensor_combo.setToolTip(
            "<table width='280'><tr><td>"
            "Sensor crosstalk correction for single-shot narrowband scans: un-mixes the camera's "
            "cross-channel response in the LINEAR capture, before inversion — a fixed property of "
            "your sensor + light, independent of film. Calibrate it from three bare-light R/G/B "
            "exposures; custom .toml matrices live in the NegPy/sensor folder. Skipped automatically "
            "for RGB-triplet assets, when Linear RAW is off, and on transparencies — which are not "
            "scanned with narrowband light. Re-run Batch Analysis after changing this."
            "</td></tr></table>"
        )
        self.calibrate_sensor_btn = self._icon_action("fa5s.vials", "Calibrate the sensor from three bare-light R/G/B exposures", width=32)
        row.addWidget(self.sensor_label)
        row.addWidget(self.sensor_combo, 1)
        row.addWidget(self.calibrate_sensor_btn)
        self.layout.addLayout(row)

        # Muted, not warning: this is the normal state for anyone not using Linear RAW, so it
        # explains the greyed controls rather than flagging a problem. Text and tooltip are set
        # per reason in _apply_gate.
        self.sensor_hint = hint_label("Requires Linear RAW.")
        self.layout.addWidget(self.sensor_hint)

        self.crosstalk_header = section_subheader("CROSSTALK")
        self.layout.addWidget(self.crosstalk_header)

        matrix_row = QHBoxLayout()
        self.crosstalk_label = field_label("Matrix")
        self.crosstalk_combo = QComboBox()
        self._fill_crosstalk_combo()
        self.crosstalk_combo.setCurrentText(conf.crosstalk_profile)
        # Wrap the long tooltip in a fixed-width table, so Qt word-wraps it to the panel width
        # instead of rendering one line that runs off the screen. Qt auto-wraps rich text only.
        self.crosstalk_combo.setToolTip(
            "<table width='280'><tr><td>"
            "Channel unmix on the raw NEGATIVE densities, before analysis and inversion — the domain "
            "where every cause of channel mixing is linear. The film's dyes absorb outside their own "
            "band, but so do your light's spectrum and your sensor's color filters, and here they all "
            "arrive as the same kind of error. So read a profile as <b>your whole scanning setup</b>, "
            "not just the stock.<br><br>"
            "<b>The bundled film matrices are read off published spec sheets, not measured</b> — they "
            "are marked (approx) for that reason. They describe the film's dyes alone, so they are only "
            "the whole story where the capture reads each dye cleanly: a Narrowband Scanner (a Coolscan's mono "
            "sensor reads one LED at a time, fully clean; a Pakon's trilinear array comes close, with slight "
            "residual bleed) or a Trichrome capture, or a Single-Shot Narrowband rig with Single-Shot Narrowband "
            "Calibration applied. Under a broadband light "
            "and a Bayer sensor your capture adds its own mixing on top, and a dyes-only matrix will not "
            "describe it.<br><br>"
            "So treat them as starting points and expect to tune: raise Strength until colors separate "
            "without going garish, and if a stock or a light gives you trouble, open the editor, nudge "
            "the six off-diagonal terms and save your own profile — name it after the combination "
            "('Gold 200 + Spectracolor'). A profile measured on your own rig beats any datasheet. "
            "Custom .toml matrices live in the NegPy/crosstalk folder (see docs/CROSSTALK.md).<br><br>"
            "Re-run Batch Analysis after changing this."
            "</td></tr></table>"
        )
        self.manage_crosstalk_btn = self._icon_action(
            "fa5s.sliders-h", "Open the crosstalk matrix editor — view, copy and edit density-unmix profiles", width=32
        )
        matrix_row.addWidget(self.crosstalk_label)
        matrix_row.addWidget(self.crosstalk_combo, 1)
        matrix_row.addWidget(self.manage_crosstalk_btn)
        self.layout.addLayout(matrix_row)

        # Shown when the film process has no matrices yet. Muted, not a warning: it is the normal
        # state for any process NegPy ships nothing for.
        self.crosstalk_hint = hint_label("No matrices for this film process — build one in the editor.")
        self.crosstalk_hint.setToolTip(
            wrap_tooltip(
                "A matrix describes one film's dye set, so it only appears here in the process it was "
                "saved for. Open the editor to start one from identity, set its Process, and save it."
            )
        )
        self.layout.addWidget(self.crosstalk_hint)

        self.crosstalk_strength_slider = CompactSlider("Strength", 0.0, 1.0, conf.crosstalk_strength, has_neutral=True)
        self.layout.addWidget(self.crosstalk_strength_slider)

        # E-6 only: a fade operator is fitted to one dye set, and a color negative's differs.
        self.fade_header = section_subheader("FADE RESTORATION")
        self.layout.addWidget(self.fade_header)

        fade_row = QHBoxLayout()
        self.fade_label = field_label("Profile")
        self.fade_combo = QComboBox()
        self._fill_fade_combo()
        self.fade_combo.setCurrentText(conf.fade_profile)
        self.fade_combo.setToolTip(
            "<table width='280'><tr><td>"
            "Restores a faded transparency's original per-layer densities: inverts a fade operator "
            "built from the dye set's six side-absorption ratios (this profile) and the Green/Blue "
            "Survival sliders below (how much this particular slide has faded), composed with the "
            "crosstalk unmix. Labelled restoration rather than correction, because it undoes fading "
            "rather than the ordinary channel bleed a fresh scan already has.<br><br>"
            "<b>The bundled profiles are computed from published spectral dye-density curves at "
            "450/550/650 nm</b> (Gschwind's narrowband bands) — they describe a Narrowband Scanner or "
            "Trichrome capture, not a broadband scan, where the real side absorption is much larger. "
            "Custom .toml profiles live in the NegPy/fade folder."
            "</td></tr></table>"
        )
        self.manage_fade_btn = self._icon_action(
            "fa5s.sliders-h", "Open the fade restoration editor — view, copy and edit dye-set side-absorption profiles", width=32
        )
        fade_row.addWidget(self.fade_label)
        fade_row.addWidget(self.fade_combo, 1)
        fade_row.addWidget(self.manage_fade_btn)
        self.layout.addLayout(fade_row)

        self.fade_strength_slider = CompactSlider("Strength", 0.0, 1.0, conf.fade_strength, has_neutral=True)
        self.layout.addWidget(self.fade_strength_slider)

        # Muted, not a warning: reports why the composition dropped or declined a factor
        # (a same-mode crosstalk profile already active, or the restoration matrix too
        # ill-conditioned to invert), so a silent no-op is never mistaken for a dead slider.
        self.fade_reject_hint = hint_label("")
        self.fade_reject_hint.setVisible(False)
        self.layout.addWidget(self.fade_reject_hint)

        # The dye set's own six side absorptions live in the profile above; these three are
        # the per-image unknowns -- how much this particular slide's three layers have
        # faded. Green and blue are ratios to red, since that is all a neutral in the image
        # can constrain; red's own absolute survival has no such reference and needs its
        # own control.
        self.fade_ratio_r_slider = CompactSlider(
            "Red Survival", *RED_SURVIVAL_BOUNDS, conf.fade_ratio_r, step=0.01, precision=1000, has_neutral=True
        )
        self.fade_ratio_r_slider.setToolTip(
            "Red layer's own surviving dye fraction (absolute, not relative to another "
            "channel -- there is nothing else to measure it against in the frame). 1.0 = "
            "red has not faded. E-6's fastest-fading dye is read on the red channel, so a "
            "heavily faded slide often needs this well below 1.0 even after Green/Blue "
            "Survival have fixed the colour balance -- without it the correction is "
            "colour-accurate but under-restored in density, which reads as washed out."
        )
        self.layout.addWidget(self.fade_ratio_r_slider)

        self.fade_ratio_g_slider = CompactSlider(
            "Green Survival", *_RATIO_SLIDER_RANGE, conf.fade_ratio_g, step=0.01, precision=1000, has_neutral=True
        )
        self.fade_ratio_g_slider.setToolTip(
            "Green layer's surviving dye fraction, relative to red. 1.0 = green and red have faded "
            "equally. Below 1.0, green has faded more than red; above, less"
        )
        self.layout.addWidget(self.fade_ratio_g_slider)

        self.fade_ratio_b_slider = CompactSlider(
            "Blue Survival", *_RATIO_SLIDER_RANGE, conf.fade_ratio_b, step=0.01, precision=1000, has_neutral=True
        )
        self.fade_ratio_b_slider.setToolTip(
            "Blue layer's surviving dye fraction, relative to red. 1.0 = blue and red have faded "
            "equally. Below 1.0, blue has faded more than red; above, less"
        )
        self.layout.addWidget(self.fade_ratio_b_slider)

        estimate_row = QHBoxLayout()
        self.estimate_fade_btn = self._icon_action(
            "fa5s.magic",
            "Estimate Green/Blue Survival using Cast Removal's own two-point neutral detection "
            "(midtone and shadow), unmixed by the selected Profile's side absorption first so the "
            "read is in dye concentration rather than measured density. A suggestion, not a lock: "
            "it populates the sliders above and can be overridden, and re-running it overwrites "
            "rather than accumulates. Changing Profile clears a previous estimate, since it was "
            "read against the old profile's delta",
            width=28,
        )
        self.estimate_fade_label = field_label("Estimate")
        estimate_row.addWidget(self.estimate_fade_label)
        estimate_row.addWidget(self.estimate_fade_btn)
        estimate_row.addStretch()
        self.layout.addLayout(estimate_row)

        # Muted, not a warning: reports which fail-closed condition fired, or the estimate
        # itself, so a silent identity is never mistaken for a broken feature.
        self.fade_estimate_hint = hint_label("")
        self.fade_estimate_hint.setVisible(False)
        self.layout.addWidget(self.fade_estimate_hint)

        self.layout.addWidget(section_subheader("LIGHT SOURCE"))

        self.hue_trim_slider = CompactSlider("Hue Trim", -30.0, 30.0, conf.hue_trim, step=0.5, precision=10, has_neutral=True, unit="°")
        self.hue_trim_slider.setToolTip(
            "<table width='280'><tr><td>"
            "Hue Trim — rotates every hue by a fixed angle (degrees) to undo the rotation an unusual "
            "scanning light imposes. Narrowband LED and odd-phosphor sources shift hues by a near-constant "
            "angle (yellows reading orange, greens olive) that white balance cannot fix, because it is a "
            "rotation rather than a cast. Neutrals are unaffected, so it does not disturb cast removal. "
            "Leave at 0 for a standard broadband light."
            "</td></tr></table>"
        )
        self.layout.addWidget(self.hue_trim_slider)

        self._apply_gate(conf)

    @staticmethod
    def _heading_row(heading: str) -> str:
        return f"— {heading} —"

    def _expected_crosstalk_rows(self, process_mode=None) -> list:
        """The rows _fill_crosstalk_combo would produce, for change detection."""
        rows: list = []
        for heading, names in CrosstalkProfiles.grouped_profiles(process_mode):
            rows.append(self._heading_row(heading))
            rows.extend(names)
        return rows

    def _fill_crosstalk_combo(self, process_mode=None) -> None:
        """Rebuild the matrix dropdown with a non-selectable heading per provenance group.

        Qt has no group concept, so headings are combo rows disabled through the model.
        Bracketing them keeps a heading from colliding with a profile name, which would let
        setCurrentText land on one.
        """
        self.crosstalk_combo.clear()
        for heading, names in CrosstalkProfiles.grouped_profiles(process_mode):
            self.crosstalk_combo.addItem(self._heading_row(heading))
            item = self.crosstalk_combo.model().item(self.crosstalk_combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
            for name in names:
                self.crosstalk_combo.addItem(name)

    def _expected_fade_rows(self, process_mode=None) -> list:
        """The rows _fill_fade_combo would produce, for change detection."""
        rows: list = [FadeProfiles.NONE_NAME]
        for heading, names in FadeProfiles.grouped_profiles(process_mode):
            rows.append(self._heading_row(heading))
            rows.extend(names)
        return rows

    def _fill_fade_combo(self, process_mode=None) -> None:
        self.fade_combo.clear()
        # Ungrouped: "None" is the absence of a profile, not a provenance bucket like the others.
        self.fade_combo.addItem(FadeProfiles.NONE_NAME)
        for heading, names in FadeProfiles.grouped_profiles(process_mode):
            self.fade_combo.addItem(self._heading_row(heading))
            item = self.fade_combo.model().item(self.fade_combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
            for name in names:
                self.fade_combo.addItem(name)

    def _crosstalk_names(self) -> list:
        """Selectable profile names currently in the combo, headings excluded."""
        model = self.crosstalk_combo.model()
        return [
            self.crosstalk_combo.itemText(i)
            for i in range(self.crosstalk_combo.count())
            if model.item(i) is None or model.item(i).isEnabled()
        ]

    # Disabled widgets do not receive the hover that raises a tooltip, so the detail hangs off
    # the hint label rather than the combo it describes.
    _SENSOR_BLOCKED = {
        "linear_raw": (
            "Requires Linear RAW.",
            "Sensor profiles are calibrated against neutral white balance. With Linear RAW "
            "off, RAW decodes carry the camera's as-shot gains instead, which would misapply "
            "the matrix — so it is skipped. Your selection is remembered.",
        ),
        "transparency": (
            "Not applied to a transparency.",
            "The unmix corrects a narrowband light and your sensor's filters against each other, "
            "so it only means anything for a capture made under narrowband light — and narrowband "
            "is not used for slides. A profile carried over from your negative rig would correct "
            "for a light this frame was not shot under. Your selection is remembered, and applies "
            "again on a negative.",
        ),
    }

    def _apply_gate(self, conf) -> None:
        """Grey the sensor unmix and show "None" while it cannot be applied, saying why.

        Display-only: conf.sensor_profile is left alone, so the selection comes back intact
        after a Linear RAW or film-process round-trip. Crosstalk and Hue Trim depend on
        neither the decode basis nor the light, so they stay enabled.
        """
        reason = unmix_block_reason(conf)
        available = not reason
        self.sensor_combo.setCurrentText(conf.sensor_profile if available else SensorProfiles.NONE_NAME)
        self.sensor_combo.setEnabled(available)
        self.calibrate_sensor_btn.setEnabled(available)
        self.sensor_hint.setVisible(bool(reason))
        if reason:
            text, tip = self._SENSOR_BLOCKED[reason]
            self.sensor_hint.setText(text)
            self.sensor_hint.setToolTip(wrap_tooltip(tip))

    def _connect_signals(self) -> None:
        self.linear_raw_btn.toggled.connect(self._on_linear_raw_toggled)
        self.narrowband_scan_btn.toggled.connect(self._on_narrowband_scan_toggled)
        self.scan_setup_btn.clicked.connect(self._open_scan_setup)

        self.sensor_combo.currentTextChanged.connect(self._on_sensor_profile_changed)
        self.calibrate_sensor_btn.clicked.connect(self._open_sensor_calibration)

        self.crosstalk_combo.currentTextChanged.connect(self._on_crosstalk_profile_changed)
        self.manage_crosstalk_btn.clicked.connect(self._open_crosstalk_editor)
        self.crosstalk_strength_slider.valueChanged.connect(lambda v: self._on_crosstalk_strength_changed(v, persist=False))
        self.crosstalk_strength_slider.valueCommitted.connect(lambda v: self._on_crosstalk_strength_changed(v, persist=True))

        self.fade_combo.currentTextChanged.connect(self._on_fade_profile_changed)
        self.manage_fade_btn.clicked.connect(self._open_fade_editor)
        self.fade_strength_slider.valueChanged.connect(lambda v: self._on_fade_strength_changed(v, persist=False))
        self.fade_strength_slider.valueCommitted.connect(lambda v: self._on_fade_strength_changed(v, persist=True))
        self.fade_ratio_r_slider.valueChanged.connect(lambda v: self._on_fade_ratio_r_changed(v, persist=False))
        self.fade_ratio_r_slider.valueCommitted.connect(lambda v: self._on_fade_ratio_r_changed(v, persist=True))
        self.fade_ratio_g_slider.valueChanged.connect(
            lambda v: self._on_fade_ratio_changed(v, self.fade_ratio_b_slider.value(), persist=False)
        )
        self.fade_ratio_g_slider.valueCommitted.connect(
            lambda v: self._on_fade_ratio_changed(v, self.fade_ratio_b_slider.value(), persist=True)
        )
        self.fade_ratio_b_slider.valueChanged.connect(
            lambda v: self._on_fade_ratio_changed(self.fade_ratio_g_slider.value(), v, persist=False)
        )
        self.fade_ratio_b_slider.valueCommitted.connect(
            lambda v: self._on_fade_ratio_changed(self.fade_ratio_g_slider.value(), v, persist=True)
        )
        self.estimate_fade_btn.clicked.connect(self._on_estimate_fade)

        self.hue_trim_slider.valueChanged.connect(lambda v: self._on_hue_trim_changed(v, persist=False))
        self.hue_trim_slider.valueCommitted.connect(lambda v: self._on_hue_trim_changed(v, persist=True))

    def _on_linear_raw_toggled(self, checked: bool) -> None:
        from dataclasses import replace

        new_config = replace(
            self.state.config,
            process=replace(
                self.state.config.process,
                linear_raw=checked,
                **invalidate_local_bounds(self.state.config.process),
            ),
        )
        # linear_raw switches use_camera_wb, so it is a source change: apply_config re-decodes and
        # suppresses the bounds analysis over the stale buffer.
        self.controller.apply_config(new_config, persist=True)

    def _on_narrowband_scan_toggled(self, checked: bool) -> None:
        self.update_config_section("process", narrowband_scan=checked, persist=True, render=True)

    def _open_scan_setup(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.show_scan_setup()

    def _on_sensor_profile_changed(self, name: str) -> None:
        # Bake the matrix like crosstalk does. The per-frame bounds were analyzed under the
        # previous mix, so clear them.
        matrix = SensorProfiles.get_matrix(name)
        self.update_config_section(
            "process",
            persist=True,
            render=True,
            sensor_profile=name,
            sensor_matrix=tuple(matrix) if matrix is not None else None,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _open_sensor_calibration(self) -> None:
        from negpy.desktop.view.widgets.sensor_calibration_dialog import SensorCalibrationDialog

        dlg = SensorCalibrationDialog(parent=self, start_dir=last_open_folder(self.controller.session.repo))
        dlg.profile_saved.connect(self._on_sensor_profile_saved)
        dlg.exec()

    def _on_sensor_profile_saved(self, name: str) -> None:
        self._on_sensor_profile_changed(name)
        self.sync_ui()  # rebuild the combo (now includes the new profile) and select it

    def _on_crosstalk_profile_changed(self, name: str) -> None:
        # Bake the matrix into the config, so saved edits stay reproducible if the profile file is
        # later moved or deleted. The persisted per-frame bounds were analyzed under the previous
        # matrix, so clear them and the stretch re-derives from the unmixed data. Otherwise the
        # mask redistribution leaks through.
        matrix = CrosstalkProfiles.get_matrix(name)
        self.update_config_section(
            "process",
            persist=True,
            render=True,
            crosstalk_profile=name,
            crosstalk_matrix=matrix,
            # Baked with the matrix so the render can gate on it without disk I/O.
            crosstalk_process=CrosstalkProfiles.get_process(name),
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_crosstalk_strength_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section(
            "process",
            persist=persist,
            render=True,
            readback_metrics=persist,
            crosstalk_strength=val,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _open_crosstalk_editor(self) -> None:
        from negpy.desktop.view.widgets.crosstalk_editor_dialog import CrosstalkEditorDialog

        conf = self.state.config.process
        self._crosstalk_snapshot = (conf.crosstalk_profile, conf.crosstalk_matrix, conf.crosstalk_strength, conf.crosstalk_process)
        dlg = CrosstalkEditorDialog(conf.crosstalk_profile, conf.crosstalk_strength, conf.process_mode, parent=self)
        dlg.matrix_previewed.connect(self._on_crosstalk_preview)
        dlg.profiles_changed.connect(self.sync_ui)
        dlg.finished.connect(lambda result: self._on_crosstalk_editor_finished(dlg, result))
        self._crosstalk_dialog = dlg  # keep a reference so the modeless dialog isn't GC'd
        dlg.show()

    def _on_crosstalk_preview(self, matrix: object, strength: float, process: str) -> None:
        # The process rides along: the render gates the unmix on it, so a preview without it shows
        # nothing whenever the edited profile is for another film.
        self.update_config_section(
            "process",
            persist=False,
            render=True,
            crosstalk_matrix=tuple(matrix) if matrix is not None else None,
            crosstalk_strength=strength,
            crosstalk_process=process,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_crosstalk_editor_finished(self, dlg, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            name = dlg.selected_name() or CrosstalkProfiles.DEFAULT_NAME
            snap_strength = self._crosstalk_snapshot[2]
            self.update_config_section(
                "process",
                persist=True,
                render=True,
                crosstalk_profile=name,
                # Default stores no matrix (falls back to the built-in) by convention.
                crosstalk_matrix=None if name == CrosstalkProfiles.DEFAULT_NAME else tuple(dlg.working_matrix()),
                # Preview strength is view-only; only adopt it if the edit had crosstalk off.
                crosstalk_strength=dlg.preview_strength() if snap_strength == 0 else snap_strength,
                crosstalk_process=dlg.selected_process(),
                **invalidate_local_bounds(self.state.config.process),
            )
        else:
            profile, matrix, strength, process = self._crosstalk_snapshot
            self.update_config_section(
                "process",
                persist=True,
                render=True,
                crosstalk_profile=profile,
                crosstalk_matrix=matrix,
                crosstalk_strength=strength,
                crosstalk_process=process,
                **invalidate_local_bounds(self.state.config.process),
            )
        self.sync_ui()

    def _on_fade_profile_changed(self, name: str) -> None:
        # Bake delta like crosstalk bakes its matrix, and clear the per-frame bounds
        # analyzed under the previous fade state. The survival ratios are per-image, not
        # part of the profile, so they are untouched here -- but a prior Estimate result
        # was computed against the old delta (the estimator unmixes by it), so the hint
        # is now stale and is cleared rather than left showing a number for a profile
        # that no longer applies.
        self.update_config_section(
            "process",
            persist=True,
            render=True,
            fade_profile=name,
            fade_delta=FadeProfiles.get_delta(name),
            fade_process=FadeProfiles.get_process(name),
            **invalidate_local_bounds(self.state.config.process),
        )
        self.fade_estimate_hint.setText("")
        self.fade_estimate_hint.setVisible(False)

    def _on_fade_strength_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section(
            "process",
            persist=persist,
            render=True,
            readback_metrics=persist,
            fade_strength=val,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_fade_ratio_r_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section(
            "process",
            persist=persist,
            render=True,
            readback_metrics=persist,
            fade_ratio_r=val,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_fade_ratio_changed(self, ratio_g: float, ratio_b: float, persist: bool = True) -> None:
        self.update_config_section(
            "process",
            persist=persist,
            render=True,
            readback_metrics=persist,
            fade_ratio_g=ratio_g,
            fade_ratio_b=ratio_b,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_estimate_fade(self) -> None:
        from negpy.features.exposure.normalization import resolve_analysis_region
        from negpy.features.process.fade import estimate_fade_ratios

        image = self.state.preview_raw
        if image is None:
            self.fade_estimate_hint.setText("No image loaded to estimate from.")
            self.fade_estimate_hint.setVisible(True)
            return
        conf = self.state.config.process
        roi, buffer = resolve_analysis_region(image.shape, None, conf.analysis_buffer, conf.analysis_rect)
        ratio_g, ratio_b, reason = estimate_fade_ratios(image, conf.process_mode, roi, buffer, conf.fade_delta)
        self.fade_ratio_g_slider.setValue(ratio_g)
        self.fade_ratio_b_slider.setValue(ratio_b)
        self._on_fade_ratio_changed(ratio_g, ratio_b, persist=True)
        self.fade_estimate_hint.setText(reason or f"Estimated: green {ratio_g:.2f}, blue {ratio_b:.2f}")
        self.fade_estimate_hint.setVisible(True)

    def _open_fade_editor(self) -> None:
        from negpy.desktop.view.widgets.fade_editor_dialog import FadeEditorDialog

        conf = self.state.config.process
        self._fade_snapshot = (conf.fade_profile, conf.fade_delta, conf.fade_strength, conf.fade_process)
        dlg = FadeEditorDialog(conf.fade_profile, conf.fade_strength, parent=self)
        dlg.delta_previewed.connect(self._on_fade_preview)
        dlg.profiles_changed.connect(self.sync_ui)
        dlg.finished.connect(lambda result: self._on_fade_editor_finished(dlg, result))
        self._fade_dialog = dlg  # keep a reference so the modeless dialog isn't GC'd
        dlg.show()

    def _on_fade_preview(self, delta: object, strength: float) -> None:
        self.update_config_section(
            "process",
            persist=False,
            render=True,
            fade_delta=tuple(delta),
            fade_strength=strength,
            fade_process=ProcessMode.E6,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_fade_editor_finished(self, dlg, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            name = dlg.selected_name() or FadeProfiles.NONE_NAME
            snap_strength = self._fade_snapshot[2]
            self.update_config_section(
                "process",
                persist=True,
                render=True,
                fade_profile=name,
                fade_delta=FadeProfiles.get_delta(name),
                # Preview strength is view-only; only adopt it if the edit had fade off.
                fade_strength=dlg.preview_strength() if snap_strength == 0 else snap_strength,
                fade_process=FadeProfiles.get_process(name),
                **invalidate_local_bounds(self.state.config.process),
            )
        else:
            profile, delta, strength, process = self._fade_snapshot
            self.update_config_section(
                "process",
                persist=True,
                render=True,
                fade_profile=profile,
                fade_delta=delta,
                fade_strength=strength,
                fade_process=process,
                **invalidate_local_bounds(self.state.config.process),
            )
        self.sync_ui()

    def _on_hue_trim_changed(self, val: float, persist: bool = True) -> None:
        # Sticky on commit only, so a drag doesn't write every intermediate value.
        self.update_config_section("process", hue_trim=val, persist=persist, readback_metrics=persist)

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            self.linear_raw_btn.setChecked(conf.linear_raw)
            self.narrowband_scan_btn.setChecked(conf.narrowband_scan)
            # Three reasons, three gates. Narrowband is refused for any transparency, because the
            # bundled profile describes narrowband capture of negative dyes. Linear RAW is inert
            # on the *transfer*, where the camera matrix folds the as-shot multipliers back in
            # (with Normalize on it still decides the decode, so it stays live there), and on an
            # RGB-scan triplet, where a narrowband exposure has no full-spectrum scene for a WB
            # gain to describe in the first place — every exposure decodes neutral regardless.
            from negpy.features.exposure.transfer import is_transparency_transfer
            from negpy.features.rgbscan.models import is_rgb_triplet

            e6 = conf.process_mode == ProcessMode.E6
            transfer = is_transparency_transfer(conf.process_mode, conf.e6_normalize)
            triplet = is_rgb_triplet(self.state.config.rgbscan)
            self.narrowband_scan_btn.setEnabled(not e6)
            self.linear_raw_btn.setEnabled(not transfer and not triplet)
            self.scan_setup_btn.setEnabled(not e6)
            self.capture_hint.setVisible(e6 or triplet)
            if e6:
                self.capture_hint.setText(
                    "Narrowband is not used for slides." if not transfer else "Not applied to an as-captured transparency."
                )
                self.capture_hint.setToolTip(
                    wrap_tooltip(
                        "Narrowband's bundled input profile describes narrowband capture of *negative* "
                        "dyes, and a slide has a different dye set, so on a transparency it would correct "
                        "for film that is not there. Its real payoffs — defeating the orange mask, clean "
                        "separation before a high-gain inversion — belong to negatives."
                        + (
                            " Linear RAW is inert here too: the camera matrix folds the as-shot multipliers "
                            "back in, so the render is the same either way."
                            if transfer
                            else " Linear RAW still applies, and stays live."
                        )
                        + (" A triplet locks Linear RAW off as well, for the same reason as on a plain frame." if triplet else "")
                        + " Both settings are remembered, and apply again on a negative."
                    )
                )
            elif triplet:
                self.capture_hint.setText("Linear RAW is locked for a Trichrome triplet.")
                self.capture_hint.setToolTip(
                    wrap_tooltip(
                        "A triplet exposure is a single narrowband channel: only one raw channel carries "
                        "real signal, so a white-balance gain corrects nothing — there is no full-spectrum "
                        "scene for it to describe. Every exposure decodes neutral regardless of this "
                        "toggle, so it is locked rather than left live with no effect. Remembered, and "
                        "applies again once the frame is no longer a triplet."
                    )
                )

            profiles = SensorProfiles.list_profiles()
            if profiles != [self.sensor_combo.itemText(i) for i in range(self.sensor_combo.count())]:
                self.sensor_combo.clear()
                self.sensor_combo.addItems(profiles)
            self._apply_gate(conf)

            # Headings included, so a changed `type` rebuilds too (the name set alone would not).
            if self._expected_crosstalk_rows(conf.process_mode) != [
                self.crosstalk_combo.itemText(i) for i in range(self.crosstalk_combo.count())
            ]:
                self._fill_crosstalk_combo(conf.process_mode)
            self.crosstalk_combo.setCurrentText(conf.crosstalk_profile)
            self.crosstalk_strength_slider.setValue(conf.crosstalk_strength)
            # Nothing to unmix on one B&W emulsion. Every other process keeps the section even with
            # no matrices of its own, because the editor is the only way to make one and hiding it
            # would leave no route in. The empty dropdown and the Strength slider it feeds are
            # disabled instead, and a hint says why.
            is_bw = conf.process_mode == ProcessMode.BW
            has_profiles = bool(CrosstalkProfiles.grouped_profiles(conf.process_mode))
            for w in (self.crosstalk_header, self.crosstalk_label, self.crosstalk_combo, self.manage_crosstalk_btn):
                w.setVisible(not is_bw)
            self.crosstalk_strength_slider.setVisible(not is_bw)
            self.crosstalk_hint.setVisible(not is_bw and not has_profiles)
            self.crosstalk_combo.setEnabled(has_profiles)
            self.crosstalk_strength_slider.setEnabled(has_profiles)

            # E-6 only: a fade operator is fitted to one dye set, and every other process
            # either has no dye layers to fade (B&W) or no fade profile yet (C-41).
            if self._expected_fade_rows(conf.process_mode) != [self.fade_combo.itemText(i) for i in range(self.fade_combo.count())]:
                self._fill_fade_combo(conf.process_mode)
            self.fade_combo.setCurrentText(conf.fade_profile)
            self.fade_strength_slider.setValue(conf.fade_strength)
            self.fade_ratio_r_slider.setValue(conf.fade_ratio_r)
            self.fade_ratio_g_slider.setValue(conf.fade_ratio_g)
            self.fade_ratio_b_slider.setValue(conf.fade_ratio_b)
            for w in (
                self.fade_header,
                self.fade_label,
                self.fade_combo,
                self.manage_fade_btn,
                self.fade_strength_slider,
                self.fade_ratio_r_slider,
                self.fade_ratio_g_slider,
                self.fade_ratio_b_slider,
                self.estimate_fade_label,
                self.estimate_fade_btn,
            ):
                w.setVisible(e6)
            self.fade_estimate_hint.setVisible(e6 and bool(self.fade_estimate_hint.text()))
            reject_reason = fade_delta_conflict_reason(conf, conf.process_mode) or fade_reject_reason(
                conf.fade_strength, conf.fade_ratio_r, conf.fade_ratio_g, conf.fade_ratio_b, conf.fade_delta
            )
            self.fade_reject_hint.setText(reject_reason)
            self.fade_reject_hint.setVisible(e6 and bool(reject_reason))

            self.hue_trim_slider.setValue(conf.hue_trim)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.linear_raw_btn,
            self.narrowband_scan_btn,
            self.sensor_combo,
            self.crosstalk_combo,
            self.crosstalk_strength_slider,
            self.fade_combo,
            self.fade_strength_slider,
            self.fade_ratio_r_slider,
            self.fade_ratio_g_slider,
            self.fade_ratio_b_slider,
            self.hue_trim_slider,
        ):
            w.blockSignals(blocked)
