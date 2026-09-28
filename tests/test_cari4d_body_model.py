# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The CARI4D SMPL-H -> SMPL-X conversion reproduces CARI4D's own body.

Needs SMPL-X, SMPL-H and a CARI4D ``.pth`` (``HOI_RETARGET_TEST_CARI4D``); skips otherwise.
"""

from __future__ import annotations

import os
import pathlib

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
smplx = pytest.importorskip("smplx")

from hoi_retarget.paths import BODY_MODEL_ROOT  # noqa: E402

#: Body joints compared; the feet are excluded because the converter flattens and smooths them.
KEEP = [i for i in range(22) if i not in (10, 11)]

#: Within the reconstruction's own uncertainty; the unconverted body is ~10 cm off.
MAX_JOINT_ERROR_CM = 3.0

CARI4D_PTH = os.environ.get("HOI_RETARGET_TEST_CARI4D", "")

pytestmark = pytest.mark.skipif(
    not (
        CARI4D_PTH
        and pathlib.Path(CARI4D_PTH).is_file()
        and (BODY_MODEL_ROOT / "smplh").is_dir()
        and (BODY_MODEL_ROOT / "smplx").is_dir()
    ),
    reason="needs SMPL-X, SMPL-H and a CARI4D .pth (set HOI_RETARGET_TEST_CARI4D)",
)


def _smplh_reference_joints(pth: str, gender: str) -> np.ndarray:
    """Joints of the SMPL-H body CARI4D optimised."""
    from hoi_retarget.datasets.cari4d import load_cari4d_pth

    pr = load_cari4d_pth(pth)["pr"]
    pose = np.asarray(pr["smpl_pose"], np.float32)
    betas = np.asarray(pr["betas"], np.float32)
    betas_10 = (betas[0] if betas.ndim == 2 else betas)[:10]
    transl = np.asarray(pr["smpl_t"], np.float32)
    model = smplx.create(
        str(BODY_MODEL_ROOT),
        model_type="smplh",
        gender=gender,
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=len(transl),
    )
    with torch.no_grad():
        out = model(
            betas=torch.tensor(betas_10).view(1, -1).repeat(len(transl), 1),
            global_orient=torch.tensor(pose[:, 0:3]),
            body_pose=torch.tensor(pose[:, 3:66]),
            transl=torch.tensor(transl),
        )
    return out.joints[:, :22].numpy()


def _converted_joints(converted: dict) -> np.ndarray:
    n = len(converted["trans"])
    model = smplx.create(
        str(BODY_MODEL_ROOT),
        "smplx",
        gender=str(converted["gender"]),
        use_pca=False,
        batch_size=n,
    )
    with torch.no_grad():
        out = model(
            betas=torch.tensor(converted["betas"]).view(1, -1).repeat(n, 1),
            global_orient=torch.tensor(converted["root_orient"]),
            body_pose=torch.tensor(converted["pose_body"]),
            transl=torch.tensor(converted["trans"]),
        )
    return out.joints[:, :22].numpy()


#: Rigid post-transforms off, so the comparison stays in CARI4D's camera frame.
_CAMERA_FRAME = dict(
    apply_axis_fix=False,
    apply_floor_offset=False,
    apply_xy_recenter=False,
    flatten_toes=False,
    smooth_lowerleg_win=0,
)


def test_gender_is_read_from_the_subject_code() -> None:
    from hoi_retarget.datasets.cari4d import infer_cari4d_gender

    assert infer_cari4d_gender("Date03_Sub01_OrangeTable9-wild.pth") == "male"
    assert infer_cari4d_gender("Date03_Sub06_whatever.pth") == "female"
    assert infer_cari4d_gender("no_subject_here.pth") == "neutral"


def test_converted_body_matches_cari4d_smplh() -> None:
    """Within 3 cm of CARI4D's body at the joints; pelvis matched exactly."""
    from hoi_retarget.datasets.cari4d import convert_cari4d_to_smplx, infer_cari4d_gender

    gender = infer_cari4d_gender(CARI4D_PTH)
    reference = _smplh_reference_joints(CARI4D_PTH, gender)
    joints = _converted_joints(convert_cari4d_to_smplx(CARI4D_PTH, **_CAMERA_FRAME))

    error_cm = float(np.linalg.norm(joints[:, KEEP] - reference[:, KEEP], axis=-1).mean() * 100)
    pelvis_cm = float(np.linalg.norm(joints[:, 0] - reference[:, 0], axis=-1).mean() * 100)
    assert error_cm <= MAX_JOINT_ERROR_CM, f"body joints are {error_cm:.2f} cm from CARI4D's SMPL-H"
    assert pelvis_cm < 0.1, f"pelvis is {pelvis_cm:.2f} cm off; the pelvis match is not applied"
