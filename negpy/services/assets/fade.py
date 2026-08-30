import os
import tomllib
from typing import List, Optional

from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.paths import get_resource_path
from negpy.services.assets.crosstalk import GROUP_ORDER, CrosstalkType
from negpy.services.assets.naming import escape_toml_string, slugify

logger = get_logger(__name__)

#: No profile selected: no correction, same as fade_strength = 0. Distinct from a
#: bundled profile, whose delta is real spec-sheet-derived data (see fade/README.md).
NONE_NAME = "None"


class FadeProfiles:
    """
    TOML I/O for dye-fade restoration profiles: the six side-absorption ratios `delta`
    (gr/br/rg/bg/rb/gb order) `resolve_fade_matrix` builds a restoration operator from,
    and the R/G/B measurement wavelengths (`bands`, nm) delta was read at.

    Delta is a property of the dye set *and the scanner's bands* -- moving the red band
    from a narrowband LED to a colorimetric sensor's response peak changes delta by an
    order of magnitude on stocks checked (see fade/README.md), so a profile without its
    bands is not just incomplete, it invites silently applying the wrong numbers. `bands`
    is required; a profile missing it is rejected the same way a malformed delta is.

    The three surviving-dye ratios are a property of one faded slide instead, and live on
    ProcessConfig directly (`fade_ratio_r`/`fade_ratio_g`/`fade_ratio_b`), not in a
    profile — see IMPLEMENT_FADE_AUTO.md §1-2.

    Files live in APP_CONFIG.fade_dir; bundled read-only profiles in the packaged
    `fade/` resource dir. "None" means no profile — no built-in fallback numbers, unlike
    crosstalk's "Generic C41": a fade profile only exists once someone supplies real
    dye-spectral data. Disk I/O only happens on dropdown build and on selection -- never
    per render (delta is baked into ProcessConfig; bands is display-only).
    """

    NONE_NAME = NONE_NAME

    @staticmethod
    def _scan_dir(directory: str) -> dict:
        """Maps display-name -> (delta 6-float tuple, bands 3-float tuple) for valid
        .toml files in a directory."""
        result: dict = {}
        if not os.path.isdir(directory):
            return result
        for fname in os.listdir(directory):
            if not fname.endswith(".toml"):
                continue
            parsed = FadeProfiles._parse_file(os.path.join(directory, fname))
            if parsed is None:
                continue
            name, delta, bands = parsed
            name = name or fname[:-5]
            if name != NONE_NAME:
                result[name] = (delta, bands)
        return result

    @staticmethod
    def scan_bundled() -> dict:
        """Read-only profiles shipped with the app, keyed by display name."""
        return FadeProfiles._scan_dir(get_resource_path("fade"))

    @staticmethod
    def scan_user() -> dict:
        """User-editable profiles in the docs folder, keyed by display name."""
        return FadeProfiles._scan_dir(APP_CONFIG.fade_dir)

    @staticmethod
    def _scan() -> dict:
        """Bundled ∪ user custom profiles, keyed by display name; bundled wins."""
        return {**FadeProfiles.scan_user(), **FadeProfiles.scan_bundled()}

    @staticmethod
    def _parse_file(path: str) -> Optional[tuple]:
        """Parses a .toml file to (name, delta 6-float list, bands 3-float list), or
        None if invalid -- including when `bands` is missing or malformed. `type`/
        `process` are read separately: callers unpack this tuple positionally."""
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            delta = data.get("delta")
            if not isinstance(delta, list) or len(delta) != 6:
                return None
            for v in delta:
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    return None
            bands = data.get("bands")
            if not isinstance(bands, list) or len(bands) != 3 or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in bands):
                # The likelier real-world trigger than a malformed delta: a profile saved
                # before `bands` existed. Logged (unlike a malformed delta, silent like
                # crosstalk's own precedent) because it otherwise vanishes with no sign why.
                logger.warning("Fade profile %s has no valid `bands` (R/G/B measurement wavelengths) -- skipping", path)
                return None
            raw_name = data.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
            return name, [float(v) for v in delta], [float(v) for v in bands]
        except Exception:
            return None

    @staticmethod
    def _parse_type(path: str) -> str:
        """The profile's `type`, lowercased; "" when absent or unreadable."""
        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f).get("type")
        except Exception:
            return ""
        return raw.strip().lower() if isinstance(raw, str) else ""

    @staticmethod
    def _scan_types() -> dict:
        """display-name -> type for every valid profile; bundled wins, like _scan."""
        types: dict = {}
        for directory in (APP_CONFIG.fade_dir, get_resource_path("fade")):
            if not os.path.isdir(directory):
                continue
            for fname in os.listdir(directory):
                if not fname.endswith(".toml"):
                    continue
                path = os.path.join(directory, fname)
                parsed = FadeProfiles._parse_file(path)
                if parsed is None:
                    continue
                name = parsed[0] or fname[:-5]
                if name != NONE_NAME:
                    types[name] = FadeProfiles._parse_type(path)
        return types

    @staticmethod
    def _scan_processes() -> dict:
        """display-name -> film process the profile describes; bundled wins, like _scan.

        Absent `process` means transparency: fade restoration is E-6-only until a
        negative fade profile exists, so that is the honest default rather than a
        guess. Values are coerced through ProcessMode, so a file written with the
        pre-rename names still matches."""
        from negpy.features.process.models import ProcessMode

        out: dict = {}
        for directory in (APP_CONFIG.fade_dir, get_resource_path("fade")):
            if not os.path.isdir(directory):
                continue
            for fname in os.listdir(directory):
                if not fname.endswith(".toml"):
                    continue
                try:
                    with open(os.path.join(directory, fname), "rb") as f:
                        data = tomllib.load(f)
                except Exception:
                    continue
                raw_name = data.get("name")
                name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else fname[:-5]
                value = data.get("process")
                out[name] = str(ProcessMode(value.strip() if isinstance(value, str) else ProcessMode.E6))
        return out

    @staticmethod
    def get_process(name: str) -> str:
        """The film process a profile was derived for; E-6 when unknown."""
        from negpy.features.process.models import ProcessMode

        return FadeProfiles._scan_processes().get(name, str(ProcessMode.E6))

    @staticmethod
    def grouped_profiles(process_mode: Optional[str] = None) -> List[tuple]:
        """[(heading, [profile names])] in GROUP_ORDER, skipping empty groups.

        `process_mode` keeps the dropdown to profiles derived for the film being
        processed, mirroring CrosstalkProfiles.grouped_profiles."""
        types = FadeProfiles._scan_types()
        if process_mode is not None:
            processes = FadeProfiles._scan_processes()
            types = {n: t for n, t in types.items() if processes.get(n, "") == str(process_mode)}
        known = {t for t, _ in GROUP_ORDER if t}
        buckets: dict = {t: [] for t, _ in GROUP_ORDER}
        for name in sorted(types):
            bucket = types[name] if types[name] in known else CrosstalkType.OTHER
            buckets[bucket].append(name)
        return [(heading, buckets[t]) for t, heading in GROUP_ORDER if buckets[t]]

    @staticmethod
    def get_type(name: str) -> str:
        """A profile's type, or "" when unknown ("None" has no type of its own)."""
        return FadeProfiles._scan_types().get(name, CrosstalkType.OTHER)

    @staticmethod
    def list_profiles() -> List[str]:
        """["None", *sorted custom+bundled display-names]."""
        return [NONE_NAME, *sorted(FadeProfiles._scan().keys())]

    @staticmethod
    def get_delta(name: str) -> Optional[tuple]:
        """Delta 6-tuple for a profile, or None for "None" / missing / invalid (=
        resolve_fade_matrix treats it as zero side-absorption)."""
        if name == NONE_NAME:
            return None
        found = FadeProfiles._scan().get(name)
        if found is None:
            return None
        return tuple(found[0])

    @staticmethod
    def get_bands(name: str) -> Optional[tuple]:
        """(R, G, B) measurement wavelengths in nm a profile's delta was read at, or None
        for "None" / missing / invalid."""
        if name == NONE_NAME:
            return None
        found = FadeProfiles._scan().get(name)
        if found is None:
            return None
        return tuple(found[1])

    @staticmethod
    def is_bundled(name: str) -> bool:
        """True for read-only profiles: "None" or any bundled profile."""
        return name == NONE_NAME or name in FadeProfiles.scan_bundled()

    @staticmethod
    def path_for_name(name: str) -> str:
        """Filesystem path a user profile with this display name would use."""
        return os.path.join(APP_CONFIG.fade_dir, f"{slugify(name, 'fade')}.toml")

    @staticmethod
    def save(
        name: str,
        delta: List[float],
        bands: List[float],
        profile_type: str = CrosstalkType.TUNED,
        process: Optional[str] = None,
    ) -> str:
        """Write a user profile TOML and return its path.

        Defaults to `tuned` so editor saves are not grouped with the spec-sheet
        estimates. `process` is always written: a profile only reaches the render in
        the film process it declares. `bands` is required: delta means nothing without
        the R/G/B wavelengths it was measured at."""
        from negpy.features.process.models import ProcessMode

        os.makedirs(APP_CONFIG.fade_dir, exist_ok=True)
        delta_row = "[{:.6g}, {:.6g}, {:.6g}, {:.6g}, {:.6g}, {:.6g}]".format(*delta)
        bands_row = "[{:.6g}, {:.6g}, {:.6g}]".format(*bands)
        content = (
            f'name = "{escape_toml_string(name)}"\n'
            f'type = "{escape_toml_string(profile_type)}"\n'
            f'process = "{escape_toml_string(str(ProcessMode(process or ProcessMode.E6)))}"\n'
            f"bands = {bands_row}\n"
            f"delta = {delta_row}\n"
        )
        path = FadeProfiles.path_for_name(name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    @staticmethod
    def delete(name: str) -> None:
        """Remove the user profile file whose display name matches (no-op if absent)."""
        directory = APP_CONFIG.fade_dir
        if not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            if not fname.endswith(".toml"):
                continue
            path = os.path.join(directory, fname)
            parsed = FadeProfiles._parse_file(path)
            if parsed is None:
                continue
            parsed_name = parsed[0] or fname[:-5]
            if parsed_name == name:
                os.remove(path)
                return

    @staticmethod
    def ensure_user_dir() -> None:
        """Make sure the user's fade directory exists; no seeding."""
        os.makedirs(APP_CONFIG.fade_dir, exist_ok=True)
