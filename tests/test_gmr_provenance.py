# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The provenance claims in ``hoi_retarget/gmr/README.md``, as tests.

Hashes are from upstream GMR ``5bac4bd`` (2025-10-16). A failure means a verbatim file was
edited (update the README table and give it a header) or the pin is stale (re-derive it).
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

GMR = pathlib.Path(__file__).resolve().parents[1] / "hoi_retarget" / "gmr"

#: sha256 of each file the project claims is byte-identical to upstream GMR.
VERBATIM = {
    "data_loader.py": "48d2857fe856124509295a57678655c6301e1e36d0109bf15962b648ff61c635",
    "kinematics_model.py": "f4b69a6876cfc967383555cc82220077f768a8246eac5ae6d29e680ae9bdb1b1",
    "torch_utils.py": "0a3f0714d7c70e25d9938f6c933e914b8e41b9487a956665e97e56861be32b88",
    "utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "LICENSE": "c5a85c0b0012230739a0ab8f30eafba7f8ee7a1b29e025d01e27fa330298e522",
    "ik_configs/DOC.md": "6b00962039b79bcfb81a8625daadb235e0a8ec4a569e070f87330af025b89224",
}

#: Files we modified. Each must carry the provenance header, because for these
#: byte-identity is no longer available as the attribution.
MODIFIED = [
    "__init__.py",
    "motion_retarget.py",
    "params.py",
    "rot_utils.py",
    "robot_motion_viewer.py",
    "utils/smpl.py",
]


@pytest.mark.parametrize("name", sorted(VERBATIM))
def test_verbatim_files_are_unchanged(name: str) -> None:
    """A file we call verbatim must still hash to upstream's bytes."""
    got = hashlib.sha256((GMR / name).read_bytes()).hexdigest()
    assert got == VERBATIM[name], (
        f"hoi_retarget/gmr/{name} is no longer byte-identical to upstream GMR.\n"
        f"  expected {VERBATIM[name]}\n  got      {got}\n"
        "Either revert the edit, or move the file to the MODIFIED list, give it a "
        "provenance header, and update gmr/README.md and THIRD_PARTY_NOTICES.md."
    )


@pytest.mark.parametrize("name", MODIFIED)
def test_modified_files_carry_provenance(name: str) -> None:
    """Every modified GMR file states where it came from and under what terms."""
    head = (GMR / name).read_text().splitlines()[:8]
    text = "\n".join(head)
    assert "Derived from GMR" in text, f"gmr/{name}: missing the provenance header"
    assert "5bac4bd" in text, f"gmr/{name}: header does not name the fork baseline"
    assert "SPDX-License-Identifier: MIT" in text, f"gmr/{name}: missing the MIT SPDX tag"


def test_nvidia_header_survives_in_torch_utils() -> None:
    """``torch_utils.py`` keeps NVIDIA's BSD-3 notice (GMR's MIT does not supersede it)."""
    head = (GMR / "torch_utils.py").read_text()[:400]
    assert "Copyright (c) 2018-2022, NVIDIA Corporation" in head
    assert "Redistribution and use in source and binary forms" in head


def test_no_noncommercial_code_is_vendored() -> None:
    """Upstream's CC BY-NC-ND LAFAN1 helpers cannot ship in a BSD-3 release; keep them out."""
    assert not (GMR / "utils" / "lafan_vendor").exists(), "lafan_vendor/ is back"
    assert not (GMR / "utils" / "lafan1.py").exists(), "lafan1.py is back"
    offenders = [p.relative_to(GMR.parents[1]) for p in GMR.rglob("*.py") if "lafan_vendor" in p.read_text()]
    assert not offenders, f"code still reaches into lafan_vendor: {offenders}"


def test_quat_mul_replacement_is_the_hamilton_product() -> None:
    """``rot_utils.quat_mul_wxyz`` (used by the optimization) matches SciPy, in the right order."""
    numpy = pytest.importorskip("numpy")
    scipy_rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    # rot_utils imports CasADi; on a bare CI runner this test skips.
    pytest.importorskip("torch")
    pytest.importorskip("casadi")
    from hoi_retarget.gmr.rot_utils import quat_mul_wxyz

    WXYZ_TO_XYZW = [1, 2, 3, 0]
    XYZW_TO_WXYZ = [3, 0, 1, 2]

    def reference(a_wxyz, b_wxyz):
        """``a (x) b`` via SciPy. Composition order is checked below, not assumed."""
        a = scipy_rotation.from_quat(numpy.asarray(a_wxyz)[WXYZ_TO_XYZW])
        b = scipy_rotation.from_quat(numpy.asarray(b_wxyz)[WXYZ_TO_XYZW])
        return (a * b).as_quat()[XYZW_TO_WXYZ]

    rng = numpy.random.default_rng(0)
    pairs = []
    for _ in range(20_000):
        a = rng.normal(size=4)
        b = rng.normal(size=4)
        pairs.append((a / numpy.linalg.norm(a), b / numpy.linalg.norm(b)))

    worst = max(
        float(numpy.abs(reference(a, b) - numpy.asarray(quat_mul_wxyz(a, b)).reshape(-1)).max()) for a, b in pairs
    )
    assert worst < 1e-12, f"quat_mul_wxyz is not the Hamilton product: max deviation {worst:.2e}"

    # Swapped operands must fail, or the test could not see a wrong multiplication order.
    swapped = max(
        float(numpy.abs(reference(b, a) - numpy.asarray(quat_mul_wxyz(a, b)).reshape(-1)).max()) for a, b in pairs
    )
    assert swapped > 1e-3, (
        "swapping the operands changed nothing, so this test cannot see composition "
        f"order at all (max deviation {swapped:.2e})"
    )
