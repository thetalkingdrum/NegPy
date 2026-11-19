"""A virtual roll's frame takes each card from its folder roll; Save as Virtual Roll… keeps what a
frame in no folder roll showed, in its own edit, and a rig change reaches the folder rolls."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.domain.models import WorkspaceConfig
from negpy.services.assets import rolls


def _controller():
    ctrl = AppController.__new__(AppController)
    ctrl.session = MagicMock()
    store: dict = {}
    repo = ctrl.session.repo
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    repo.read_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    ctrl.state = MagicMock()
    ctrl.session.state = ctrl.state
    return ctrl, repo


def _lens(k1: float) -> WorkspaceConfig:
    config = WorkspaceConfig()
    return replace(config, geometry=replace(config.geometry, distortion_k1=k1))


def test_save_as_virtual_roll_forks_only_a_frame_in_no_folder_roll():
    ctrl, repo = _controller()
    source = rolls.recognize_folder(repo, "/scans/portra")
    other = rolls.recognize_folder(repo, "/scans/hp5")
    rolls.set_roll_defaults(repo, source, distortion_k1=0.1)
    inside = {"path": "/scans/portra/1.tif", "hash": "h1"}
    neighbor = {"path": "/scans/hp5/2.tif", "hash": "h2"}
    loose = {"path": "/elsewhere/3.tif", "hash": "h3"}
    ctrl.state.uploaded_files = [inside, neighbor, loose]
    ctrl.state.current_file_hash = None
    picks = rolls.create_virtual_roll(repo, "Picks", [a["path"] for a in (inside, neighbor, loose)])

    def config_for(asset):
        # Every frame shows the open roll's lens there, and its own edit elsewhere.
        shown = ctrl.state.active_roll_id == source or asset is inside
        return _lens(0.1 if shown else 0.0)

    ctrl.session.config_for_asset.side_effect = config_for
    ctrl.state.active_roll_id = source

    moved = AppController._keep_loose_frames(ctrl, source, picks)

    assert ctrl.state.active_roll_id == source
    assert moved == (True, False)
    fork = rolls.roll_edit_hash("h3", picks)
    repo.save_file_settings.assert_called_once_with(fork, _lens(0.1), file_path="/elsewhere/3.tif")
    assert loose["hash"] == fork
    assert (inside["hash"], neighbor["hash"]) == ("h1", "h2")
    assert rolls.frame_override_cards(repo, picks, "h3") == set()
    assert rolls.fallback_roll(repo, picks, neighbor["path"]) == other


def test_a_rig_change_reaches_the_folder_roll_a_virtual_roll_falls_back_to():
    ctrl, repo = _controller()
    folder = rolls.recognize_folder(repo, "/scans/portra")
    picks = rolls.create_virtual_roll(repo, "Picks", ["/scans/portra/1.tif", "/loose/2.tif"])
    ctrl.state.active_roll_id = picks
    ctrl.state.uploaded_files = [{"path": "/scans/portra/1.tif", "hash": "h1"}, {"path": "/loose/2.tif", "hash": "h2"}]

    assert AppController._rolls_loaded_frames_follow(ctrl) == {picks, folder}
