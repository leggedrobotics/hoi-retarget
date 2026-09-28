# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every asset the code asks for is present, and nothing unshippable is."""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets"

DESCRIPTIONS = sorted(ASSETS.rglob("*.xml")) + sorted(ASSETS.rglob("*.urdf"))

#: The 13 rigid OMOMO object categories the paper evaluates on. OMOMO ships 15;
#: `mop` and `vacuum` are articulated and are not part of this release.
OMOMO_OBJECTS = [
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "monitor",
    "plasticbox",
    "smallbox",
    "smalltable",
    "suitcase",
    "trashcan",
    "tripod",
    "whitechair",
    "woodchair",
]


def _mesh_references(desc: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """Mesh paths as the loader resolves them: MJCF against ``meshdir``, URDF against its own dir."""
    text = desc.read_text(errors="ignore")
    meshdir = re.search(r'meshdir="([^"]*)"', text)
    base = desc.parent / (meshdir.group(1) if meshdir and desc.suffix == ".xml" else "")
    out = []
    for m in re.finditer(r'(?:file|filename)="([^"]+)"', text):
        ref = m.group(1)
        out.append((ref, (base if desc.suffix == ".xml" else desc.parent) / ref))
    return out


@pytest.mark.parametrize("desc", DESCRIPTIONS, ids=lambda p: str(p.relative_to(ASSETS)))
def test_every_mesh_reference_resolves(desc: pathlib.Path) -> None:
    missing = [ref for ref, path in _mesh_references(desc) if not path.exists()]
    assert not missing, f"{desc.relative_to(ASSETS)} references missing meshes: {missing}"


def test_registered_robots_have_their_descriptions() -> None:
    """Both registries point at files that exist, for every supported robot."""
    # Reaching the registry imports the gmr package, which pulls in the solver
    # stack. On a bare runner this skips; the reference-resolution tests above
    # are the ones that must always run.
    pytest.importorskip("torch")
    from hoi_retarget.gmr.params import ROBOT_URDF_DICT, ROBOT_XML_DICT

    for robot, path in ROBOT_XML_DICT.items():
        assert pathlib.Path(path).is_file(), f"{robot}: MJCF missing at {path}"
    for robot, path in ROBOT_URDF_DICT.items():
        assert pathlib.Path(path).is_file(), f"{robot}: URDF missing at {path}"


@pytest.mark.parametrize("obj", OMOMO_OBJECTS)
def test_object_ships_with_urdf_and_surface_samples(obj: str) -> None:
    urdf = ASSETS / "objects" / f"{obj}.urdf"
    assert urdf.is_file(), f"{obj}: no URDF"
    samples = ASSETS / "objects" / "objects" / obj / "sample_points.npy"
    assert samples.is_file(), f"{obj}: no sample_points.npy"


def test_robot_licences_travel_with_the_meshes() -> None:
    """Unitree's BSD-3 notice must sit beside the meshes it covers."""
    for robot in ("g1", "h2"):
        lic = ASSETS / "robots" / robot / "LICENSE"
        assert lic.is_file(), f"assets/robots/{robot}/LICENSE is missing"
        assert "BSD 3-Clause" in lic.read_text()
        assert "Unitree" in lic.read_text()


def test_no_body_models_are_committed() -> None:
    """SMPL-X/SMPL-H models are not redistributable; nothing under ``assets/body_models`` is tracked."""
    bm = ASSETS / "body_models"
    if not bm.exists():
        return
    # A local symlink for development is fine; committed files are not.
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "assets/body_models"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert not tracked, f"SMPL-X files are tracked by git:\n{tracked}"
