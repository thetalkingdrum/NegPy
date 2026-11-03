"""Run NegativeConverter's expiredRescue.js under Node on the port-check inputs and store its output as
tests/fixtures/nc_rescue_port.json, which tests/test_nc_rescue.py compares the Python port against. With
a node_modules folder that holds @techstark/opencv-js (their package.json pins 5.0.0-release.1), it also
runs their uneven-fog stage into tests/fixtures/nc_fog_port.json.

    uv run python experiments/fade_crossover/port_check/port_check.py <NegativeConverter checkout> [<node_modules>]
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))

from nc_rescue_fixtures import CASES, FOG_CASES, FOG_PROBES, FOG_VARIANTS, PROBES  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "nc_rescue_port.json"
FOG_OUT = ROOT / "tests" / "fixtures" / "nc_fog_port.json"


def _encoded(cases: dict) -> dict:
    out = {}
    for name, make in cases.items():
        img = make()
        out[name] = {"width": img.shape[1], "height": img.shape[0], "data": base64.b64encode(img.astype("<u2").tobytes()).decode()}
    return out


def fog_check(checkout: str, modules: str, commit: str) -> None:
    src = Path(checkout) / "negative2positive" / "src"
    with tempfile.TemporaryDirectory() as tmp:
        inputs = Path(tmp) / "inputs.json"
        inputs.write_text(json.dumps({"cases": _encoded(FOG_CASES), "probes": FOG_PROBES, "variants": list(FOG_VARIANTS)}))
        harness = Path(__file__).with_name("run_nc_fog.mjs")
        result = subprocess.run(["node", str(harness), str(src), modules, str(inputs)], check=True, capture_output=True, text=True)
    FOG_OUT.write_text(json.dumps({"source_commit": commit, "cases": json.loads(result.stdout)}, indent=1) + "\n")
    print(f"wrote {FOG_OUT} from NegativeConverter {commit}")


def main(checkout: str, modules: str = "") -> int:
    module = Path(checkout) / "negative2positive" / "src" / "pipeline" / "expiredRescue.js"
    cases = {}
    for name, make in CASES.items():
        img = make()
        cases[name] = {"width": img.shape[1], "height": img.shape[0], "data": base64.b64encode(img.astype("<u2").tobytes()).decode()}
    with tempfile.TemporaryDirectory() as tmp:
        inputs = Path(tmp) / "inputs.json"
        inputs.write_text(json.dumps({"cases": cases, "probes": PROBES}))
        harness = Path(__file__).with_name("run_nc.mjs")
        result = subprocess.run(["node", str(harness), str(module), str(inputs)], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    commit = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    OUT.write_text(json.dumps({"source_commit": commit, "cases": data}, indent=1) + "\n")
    print(f"wrote {OUT} from NegativeConverter {commit}")
    if modules:
        fog_check(checkout, modules, commit)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
