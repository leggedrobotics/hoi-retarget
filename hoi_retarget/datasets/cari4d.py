# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read CARI4D (https://github.com/NVlabs/CARI4D) ``.pth`` files into the SMPL-X dict ``load_smplx_data`` expects.

Uses the ``pr`` (post-optimization) branch: ``smpl_pose`` -> root_orient + pose_body, ``smpl_t`` -> trans,
``betas`` -> 16 SMPL-X betas, ``pose_abs`` -> obj_state (pos + rotvec), ``contact_logits`` -> obj_contact
and a hand-only ``contact_human`` (T, 52).
CARI4D fits SMPL-H in BEHAVE's y-down camera frame; the converter moves it to a z-up SMPL-X body.
A stub for CARI4D's ``TrainState`` class lets ``torch.load`` unpickle the file without the upstream repo.
"""

from __future__ import annotations

import numpy as np
import os
import pathlib
import re
import sys
import types
from typing import Any

import torch
from scipy.spatial.transform import Rotation as R

from hoi_retarget.datasets.intermimic import INTERMIMIC_OMOMO_BODY_NAMES, OMOMO_TO_G1_CONTACT_LINK
from hoi_retarget.paths import BODY_MODEL_ROOT, smplx_ext
from hoi_retarget.robots import get_active_profile

_FRAME_TIMESTAMP_RE = re.compile(r"t(\d+)\.(\d+)$")

# BEHAVE/CARI4D frame: +x right, +y down, +z forward. To z-up: +90° about +x
# (new = (x, z, -y)); a per-sequence z-offset then puts the floor at z = 0.
_BEHAVE_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


def _register_trainstate_stub() -> None:
    """Register a stand-in ``learning.training.training_utils.TrainState`` for unpickling. Idempotent."""

    if "learning.training.training_utils" in sys.modules and hasattr(
        sys.modules["learning.training.training_utils"], "TrainState"
    ):
        return

    class _TrainStateStub:
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

        def __repr__(self) -> str:
            keys = list(self.__dict__.keys())[:10]
            return f"<TrainStateStub dict_keys={keys}>"

    for mod_path in ("learning", "learning.training", "learning.training.training_utils"):
        sys.modules.setdefault(mod_path, types.ModuleType(mod_path))

    sys.modules["learning.training.training_utils"].TrainState = _TrainStateStub


def _infer_fps_from_frame_ids(frames: list | None, default: float = 30.0) -> float:
    """Source FPS from BEHAVE-style frame IDs (``.../t0003.033``); ``default`` for IDs without timestamps."""
    if not frames or len(frames) < 2:
        return default

    def _parse(fid: str) -> float | None:
        m = _FRAME_TIMESTAMP_RE.search(str(fid))
        if not m:
            return None
        whole = int(m.group(1))
        frac_str = m.group(2)
        frac = int(frac_str) / (10 ** len(frac_str))
        return float(whole) + frac

    times = []
    for f in frames[:10]:
        t = _parse(f)
        if t is not None:
            times.append(t)

    if len(times) < 2:
        return default

    deltas = np.diff(times)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return default

    dt = float(np.median(deltas))
    if dt <= 0:
        return default
    return float(round(1.0 / dt))


def _contact_probs_from_logits(x: np.ndarray) -> np.ndarray:
    """Contact probabilities in [0, 1]; ``contact_logits`` hold raw logits (opt) or probabilities (gt)."""
    if x.size == 0:
        return x
    if x.max() > 1.1 or x.min() < -0.1:
        return 1.0 / (1.0 + np.exp(-x))
    return np.clip(x, 0.0, 1.0)


def _synthesise_contact_human(probs: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Per-body ``contact_human`` (T, 52) from per-hand contact: each hand's bit is copied to its finger bodies."""
    T = probs.shape[0]
    contact_human = np.zeros((T, len(INTERMIMIC_OMOMO_BODY_NAMES)), dtype=np.int32)

    if T == 0:
        return contact_human

    l_mask = (probs[:, 0] > thresh).astype(np.int32)
    r_mask = (probs[:, 1] > thresh).astype(np.int32)

    # OMOMO_TO_G1_CONTACT_LINK holds the active robot's links, so G1 literals would miss.
    left_hand, right_hand = get_active_profile().hand_links
    for i, body_name in enumerate(INTERMIMIC_OMOMO_BODY_NAMES):
        link = OMOMO_TO_G1_CONTACT_LINK.get(body_name)
        if link == left_hand:
            contact_human[:, i] = l_mask
        elif link == right_hand:
            contact_human[:, i] = r_mask

    return contact_human


def _compose_world_rotation_on_axisangle(axisangle_tNx3: np.ndarray, R_world: np.ndarray) -> np.ndarray:
    """Axis-angle of ``R_world @ R(axisangle)``, same shape."""
    if axisangle_tNx3.size == 0:
        return axisangle_tNx3
    rot_old = R.from_rotvec(axisangle_tNx3)
    rot_new = R.from_matrix(R_world) * rot_old
    return rot_new.as_rotvec().astype(np.float32)


# ---------------------------------------------------------------------------
# CARI4D is SMPL-H; the pipeline is SMPL-X.
# ---------------------------------------------------------------------------
# The rest joints and shape spaces differ, so CARI4D's parameters fed to SMPL-X put the body ~10 cm
# off its SMPL-H fit. Corrections: gender, pelvis matched to SMPL-H's rest pelvis, betas refitted to
# SMPL-H's rest joints. Gender alone can be worse than neutral; it helps only with the other two.

#: Subject code -> gender, from CARI4D's ``behave_data/const.py::_sub_gender`` (the SMPL-H fit is gendered).
CARI4D_SUBJECT_GENDER = {
    "Sub01": "male",
    "Sub02": "male",
    "Sub03": "male",
    "Sub04": "male",
    "Sub05": "male",
    "Sub06": "female",
    "Sub07": "female",
    "Sub08": "female",
}

_BETA_FIT_CACHE: dict[tuple, np.ndarray] = {}


def infer_cari4d_gender(input_path: str, fallback: str | None = "neutral") -> str | None:
    """Gender from the ``SubNN`` code in the filename (``Date03_Sub01_...`` -> male), else ``fallback``."""
    m = re.search(r"(Sub\d{2})", pathlib.Path(input_path).stem)
    if m and m.group(1) in CARI4D_SUBJECT_GENDER:
        return CARI4D_SUBJECT_GENDER[m.group(1)]
    return fallback


def _smplh_model_dir(gender: str | None = None) -> pathlib.Path:
    """Body-model root holding ``smplh/SMPLH_{MALE,FEMALE}.pkl`` (user-supplied, needed only for CARI4D)."""
    root = BODY_MODEL_ROOT
    if gender is not None:
        expected = root / "smplh" / f"SMPLH_{str(gender).upper()}.pkl"
        if not expected.is_file():
            raise FileNotFoundError(
                f"SMPL-H model not found: {expected}\n"
                "CARI4D fits a gendered SMPL-H body, so the specific file is needed, not just "
                "the directory. Register at https://mano.is.tue.mpg.de/download.php, take the "
                '"Extended SMPL+H model used in the AMASS project", and place '
                f"SMPLH_{{MALE,FEMALE}}.pkl in {root / 'smplh'}.\n"
                "Note CARI4D subjects are male or female; there is no neutral SMPL-H in that "
                "release, which is why the gender must be resolved before this point."
            )
    if not (root / "smplh").is_dir():
        raise FileNotFoundError(
            f"SMPL-H body models not found at {root / 'smplh'}.\n"
            "CARI4D fits an SMPL-H body, so reading it correctly needs the SMPL-H "
            "models as well as SMPL-X. They are not redistributable and are not "
            "part of this repository. Register at https://mano.is.tue.mpg.de/ and "
            f"place SMPLH_{{MALE,FEMALE}}.pkl in {root / 'smplh'}."
        )
    return root


def _smplh_rest_pelvis(betas_10: np.ndarray, gender: str) -> np.ndarray:
    """SMPL-H rest pelvis ``J_c[0]^H`` for these betas: where the SMPL-X pelvis must be moved to."""
    import smplx

    model = smplx.create(
        str(_smplh_model_dir(gender)),
        model_type="smplh",
        gender=str(gender),
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
    )
    with torch.no_grad():
        out = model(betas=torch.tensor(betas_10, dtype=torch.float32).view(1, -1))
    return out.joints.detach().cpu().numpy()[0, 0].astype(np.float32)


def fit_smplx_betas_to_smplh(betas_10: np.ndarray, gender: str) -> np.ndarray:
    """16 SMPL-X betas whose rest body joints match SMPL-H's for these 10 betas.

    Zero-padding leaves ~4 cm joint error after pelvis matching; the fit gives ~2.5 cm.
    Pelvis-relative, so translation stays with the pelvis shift.
    """
    key = (tuple(np.round(np.asarray(betas_10, dtype=np.float64), 6)), str(gender))
    if key in _BETA_FIT_CACHE:
        return _BETA_FIT_CACHE[key]

    import smplx

    smplh = smplx.create(
        str(_smplh_model_dir(gender)),
        model_type="smplh",
        gender=str(gender),
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
    )
    with torch.no_grad():
        rest_h = smplh(betas=torch.tensor(betas_10, dtype=torch.float32).view(1, -1)).joints[0, :22]
    target = (rest_h - rest_h[0]).detach()

    smplx_model = smplx.create(str(BODY_MODEL_ROOT), "smplx", gender=str(gender), ext=smplx_ext(), use_pca=False)
    betas = torch.zeros(1, 16, requires_grad=True)
    opt = torch.optim.Adam([betas], lr=0.05)
    for _ in range(600):
        joints = smplx_model(betas=betas).joints[0, :22]
        # Small L2 keeps the betas in the plausible shape space.
        loss = ((joints - joints[0] - target) ** 2).sum() + 1e-4 * (betas**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    fitted = betas.detach().numpy()[0].astype(np.float32)
    _BETA_FIT_CACHE[key] = fitted
    return fitted


def _canonical_pelvis_offset_via_smplx(
    betas_16: np.ndarray,
    gender: str,
    smplx_body_models_path: pathlib.Path | None = None,
) -> np.ndarray:
    """SMPL-X rest pelvis ``J_c[0]`` for these betas.

    The posed pelvis is always ``transl + J_c[0]``; ``global_orient`` does not rotate ``J_c[0]``
    (≈ (0.003, -0.336, 0.013) m for neutral betas).
    """
    import smplx

    if smplx_body_models_path is None:
        smplx_body_models_path = BODY_MODEL_ROOT
    body_model = smplx.create(
        str(smplx_body_models_path),
        "smplx",
        gender=str(gender),
        ext=smplx_ext(),
        use_pca=False,
    )
    with torch.no_grad():
        out = body_model(
            betas=torch.tensor(betas_16, dtype=torch.float32).view(1, -1),
            global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, 63),
            transl=torch.zeros(1, 3),
            left_hand_pose=torch.zeros(1, 45),
            right_hand_pose=torch.zeros(1, 45),
            jaw_pose=torch.zeros(1, 3),
            leye_pose=torch.zeros(1, 3),
            reye_pose=torch.zeros(1, 3),
        )
    return out.joints.detach().cpu().numpy()[0, 0].astype(np.float32)


def _axis_fix_pelvis_shift(
    betas_16: np.ndarray,
    gender: str,
    R_axis: np.ndarray,
    smplx_body_models_path: pathlib.Path | None = None,
    betas_10_smplh: np.ndarray | None = None,
) -> np.ndarray:
    """Offset to add to ``R_axis @ trans`` so the SMPL-X body lands where it was before the rotation.

    Because ``J_c[0]`` is added unrotated, rotating ``transl`` and ``global_orient`` alone misplaces the
    pelvis by ``R_axis @ J_c[0] - J_c[0]`` (≈ (0, 0.35, 0.32) m for neutral betas, BEHAVE -> z-up).
    With ``betas_10_smplh`` the target is SMPL-H's pelvis: ``R_axis @ J_c^H[0] - J_c^X[0]``.
    """
    Jc0_x = _canonical_pelvis_offset_via_smplx(
        betas_16=betas_16,
        gender=gender,
        smplx_body_models_path=smplx_body_models_path,
    )
    if betas_10_smplh is None:
        # No SMPL-H reference: frame-change compensation only.
        return (R_axis @ Jc0_x - Jc0_x).astype(np.float32)
    # R_axis @ (smpl_t + Jc0_H) == (R_axis @ smpl_t + shift) + Jc0_X; at R_axis = I this is the pelvis match.
    Jc0_h = _smplh_rest_pelvis(betas_10_smplh, gender)
    return (R_axis @ Jc0_h - Jc0_x).astype(np.float32)


def _compute_floor_offset_via_smplx(
    betas_16: np.ndarray,
    trans: np.ndarray,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    gender: str,
    smplx_body_models_path: pathlib.Path | None = None,
    object_pos: np.ndarray | None = None,
    object_rvec: np.ndarray | None = None,
    object_sample_points_local: np.ndarray | None = None,
) -> float:
    """z offset that puts the lowest SMPL-X mesh vertex (or object sample point, if given) 1 cm above z = 0."""
    import smplx

    if smplx_body_models_path is None:
        smplx_body_models_path = BODY_MODEL_ROOT

    body_model = smplx.create(
        str(smplx_body_models_path),
        "smplx",
        gender=str(gender),
        ext=smplx_ext(),
        use_pca=False,
    )
    T = trans.shape[0]
    with torch.no_grad():
        out = body_model(
            betas=torch.tensor(betas_16).float().view(1, -1),
            global_orient=torch.tensor(root_orient).float(),
            body_pose=torch.tensor(pose_body).float(),
            transl=torch.tensor(trans).float(),
            left_hand_pose=torch.zeros(T, 45).float(),
            right_hand_pose=torch.zeros(T, 45).float(),
            jaw_pose=torch.zeros(T, 3).float(),
            leye_pose=torch.zeros(T, 3).float(),
            reye_pose=torch.zeros(T, 3).float(),
            return_verts=True,
        )

    # Mesh vertices, not joints: the foot joint sits ~3 cm above the sole.
    body_min_z = float(out.vertices.detach().cpu().numpy()[..., 2].min())
    min_z = body_min_z

    if object_pos is not None and object_rvec is not None and object_sample_points_local is not None:
        sp = np.asarray(object_sample_points_local, dtype=np.float32)
        rot_mats = R.from_rotvec(np.asarray(object_rvec, dtype=np.float32)).as_matrix()  # (T, 3, 3)
        obj_pos = np.asarray(object_pos, dtype=np.float32)  # (T, 3)
        obj_min_z = float(np.min([((rot_mats[t] @ sp.T).T + obj_pos[t])[:, 2].min() for t in range(T)]))
        if obj_min_z < min_z:
            min_z = obj_min_z

    floor_clearance = 0.01
    return -min_z + floor_clearance


def load_cari4d_pth(input_path: str) -> dict[str, Any]:
    """Load a CARI4D ``.pth`` as a dict with ``pr`` / ``gt`` / ``in`` entries, untransformed."""
    _register_trainstate_stub()
    return torch.load(input_path, map_location="cpu", weights_only=False)


def convert_cari4d_to_smplx(
    input_path: str,
    gender: str = "auto",
    *,
    smplh_shape_fit: bool = True,
    apply_axis_fix: bool = True,
    apply_pelvis_correction: bool = True,
    apply_floor_offset: bool = True,
    apply_xy_recenter: bool = True,
    object_sample_points_path: str | None = None,
    floor_use_object: bool = False,
    flatten_toes: bool = True,
    smooth_lowerleg_win: int = 5,
) -> dict[str, Any]:
    """Convert a CARI4D ``.pth`` (``pr`` branch) to the SMPL-X dict used by ``load_smplx_data``.

    Returns ``gender``, ``betas`` (16,), ``mocap_frame_rate``, ``trans`` (T, 3), ``root_orient`` (T, 3),
    ``pose_body`` (T, 63), ``obj_state`` (T, 6) = [pos, rotvec], ``obj_contact`` (T, 1) and hand-only
    ``contact_human`` (T, 52).

    Args:
        gender: ``"auto"`` reads the ``SubNN`` code from the filename; CARI4D's SMPL-H fit is gendered.
        smplh_shape_fit: fit SMPL-X betas to SMPL-H's rest joints instead of zero-padding
            (~2.5 cm vs ~4.2 cm joint error to CARI4D's body).
        apply_axis_fix: rotate from BEHAVE's y-down frame to z-up.
        apply_pelvis_correction: shift ``trans`` so the SMPL-X pelvis lands on the SMPL-H one after the axis
            fix (see ``_axis_fix_pelvis_shift``); without it the body is off by ~35 cm.
        apply_floor_offset: shift z so the lowest body vertex (see ``floor_use_object``) sits 1 cm above z = 0.
            Needs ``apply_axis_fix``.
        apply_xy_recenter: centre the x/y range of ``trans`` (and the object with it) on the origin; GMR scales
            root positions about the origin, which would otherwise open a body-object gap.
        object_sample_points_path: object ``sample_points.npy`` for ``floor_use_object``; defaults to
            ``data/CARI4D/assets/objects/<stem>/sample_points.npy``.
        floor_use_object: also let the object define the floor. Off by default: reconstructed objects
            penetrate the floor and would float the body.
        flatten_toes: zero the foot joints' rotation; CARI4D's monocular foot fits fold and jitter.
        smooth_lowerleg_win: moving-average window (frames) on ankle and foot rotations; 0/1 disables.
            Also removes the IK branch flips ``flatten_toes`` alone can cause.
    """
    payload = load_cari4d_pth(input_path)

    if not isinstance(payload, dict) or "pr" not in payload:
        raise ValueError(f"Expected a CARI4D `.pth` with a 'pr' key; got {type(payload).__name__} for {input_path}")
    pr = payload["pr"]
    missing = [k for k in ("smpl_pose", "smpl_t", "betas", "pose_abs", "contact_logits") if k not in pr]
    if missing:
        raise ValueError(
            f"{input_path}: CARI4D 'pr' lacks {missing}. Only optimisation-stage .pth files "
            "(the opt/ directory) carry contact predictions; ground-truth .pth files are not supported."
        )

    # ── Body pose ────────────────────────────────────────────────────
    if str(gender) == "auto":
        gender = infer_cari4d_gender(input_path, fallback=None)
        if gender is None:
            raise ValueError(
                f"Cannot infer the subject's gender from {pathlib.Path(input_path).stem!r} "
                "(no known SubNN code); pass --gender male or --gender female."
            )

    smpl_pose = _to_np(pr["smpl_pose"])
    if smpl_pose.ndim != 2 or smpl_pose.shape[1] not in (72, 156, 165):
        raise ValueError(f"Unexpected smpl_pose shape {smpl_pose.shape}; expected (T, 72|156|165).")
    root_orient = smpl_pose[:, 0:3].astype(np.float32)
    pose_body = smpl_pose[:, 3:66].astype(np.float32)  # 21 SMPL-X body joints

    # ── Lower-leg temporal smoothing ──
    if smooth_lowerleg_win and smooth_lowerleg_win > 1:
        from scipy.ndimage import uniform_filter1d

        for c0 in (18, 21, 27, 30):  # ankles (j7,8) + feet (j10,11): cols [(j-1)*3:..]
            pose_body[:, c0 : c0 + 3] = uniform_filter1d(
                pose_body[:, c0 : c0 + 3],
                size=int(smooth_lowerleg_win),
                axis=0,
                mode="nearest",
            )

    # ── Toe-fold suppression (feet j10, j11) ────────────────────────
    if flatten_toes:
        pose_body[:, 27:33] = 0.0

    trans = _to_np(pr["smpl_t"]).astype(np.float32)
    if trans.shape[1] != 3:
        raise ValueError(f"Unexpected smpl_t shape {trans.shape}; expected (T, 3).")

    T = trans.shape[0]

    # ── Shape ────────────────────────────────────────────────────────
    betas_all = _to_np(pr["betas"])
    if betas_all.ndim == 2:
        betas_row = betas_all[0]
    elif betas_all.ndim == 1:
        betas_row = betas_all
    else:
        raise ValueError(f"Unexpected betas shape {betas_all.shape}.")
    betas_row = betas_row.astype(np.float32).reshape(-1)
    if betas_row.shape[0] < 10:
        betas_row = np.concatenate([betas_row, np.zeros(10 - betas_row.shape[0], dtype=np.float32)])
    betas_10 = betas_row[:10]
    if smplh_shape_fit:
        # CARI4D's betas are SMPL-H's; refitting removes ~1.6 cm of joint error vs zero-padding.
        betas_16 = fit_smplx_betas_to_smplh(betas_10, gender)
    else:
        betas_16 = np.concatenate([betas_10, np.zeros(6, dtype=np.float32)])

    # ── Object pose ──────────────────────────────────────────────────
    pose_abs = _to_np(pr["pose_abs"])
    if pose_abs.shape[-2:] != (4, 4):
        raise ValueError(f"Unexpected pose_abs shape {pose_abs.shape}; expected (T, 4, 4).")
    obj_pos = pose_abs[:, :3, 3].astype(np.float32)
    obj_rot_mat = pose_abs[:, :3, :3]
    obj_rvec = R.from_matrix(obj_rot_mat).as_rotvec().astype(np.float32)
    obj_state = np.concatenate([obj_pos, obj_rvec], axis=1)  # (T, 6)

    # ── Contact ──────────────────────────────────────────────────────
    contact_logits = _to_np(pr["contact_logits"])
    if contact_logits.ndim != 2 or contact_logits.shape[1] != 2:
        raise ValueError(f"Unexpected contact_logits shape {contact_logits.shape}; expected (T, 2).")
    probs = _contact_probs_from_logits(contact_logits.astype(np.float32))
    obj_contact = ((probs[:, 0] > 0.5) | (probs[:, 1] > 0.5)).astype(np.int32).reshape(-1, 1)
    contact_human = _synthesise_contact_human(probs)

    # ── FPS ──────────────────────────────────────────────────────────
    fps = _infer_fps_from_frame_ids(pr.get("frames"), default=30.0)

    if smpl_pose.shape[0] != T or pose_abs.shape[0] != T or obj_contact.shape[0] != T:
        raise ValueError(
            f"Inconsistent frame counts: trans={T}, smpl_pose={smpl_pose.shape[0]}, "
            f"pose_abs={pose_abs.shape[0]}, obj_contact={obj_contact.shape[0]}."
        )

    # ── Axis fix (BEHAVE y-down → pipeline z-up) ─────────────────────
    if apply_axis_fix:
        trans = (trans @ _BEHAVE_TO_ZUP.T).astype(np.float32)
        obj_state[:, 0:3] = (obj_state[:, 0:3] @ _BEHAVE_TO_ZUP.T).astype(np.float32)
        root_orient = _compose_world_rotation_on_axisangle(root_orient, _BEHAVE_TO_ZUP)
        obj_state[:, 3:6] = _compose_world_rotation_on_axisangle(obj_state[:, 3:6], _BEHAVE_TO_ZUP)

    # ── Pelvis compensation ───────────────────────────────────────────
    # Frame change (BEHAVE y-down -> z-up) plus the SMPL-H -> SMPL-X rest-pelvis offset;
    # without the axis fix R_axis is the identity.
    if apply_pelvis_correction:
        trans = (
            trans
            + _axis_fix_pelvis_shift(
                betas_16=betas_16,
                gender=gender,
                R_axis=_BEHAVE_TO_ZUP if apply_axis_fix else np.eye(3, dtype=np.float32),
                betas_10_smplh=betas_10 if smplh_shape_fit else None,
            )[None, :]
        )

    # ── Floor offset ──────────────────────────────────────────────────
    if apply_axis_fix and apply_floor_offset:
        object_sample_points = None
        if floor_use_object:
            try:
                sp_path = object_sample_points_path
                if not sp_path:
                    stem = os.path.splitext(os.path.basename(input_path))[0]
                    sp_path = os.path.join(
                        "data",
                        "CARI4D",
                        "assets",
                        "objects",
                        stem,
                        "sample_points.npy",
                    )
                if sp_path and os.path.isfile(sp_path):
                    object_sample_points = np.load(sp_path).astype(np.float32)
                else:
                    print(
                        f"[cari4d] WARN: object sample_points not found "
                        f"(tried {sp_path!r}); floor offset will use body sole only."
                    )
            except Exception:
                object_sample_points = None

        z_offset = _compute_floor_offset_via_smplx(
            betas_16=betas_16,
            trans=trans,
            root_orient=root_orient,
            pose_body=pose_body,
            gender=gender,
            object_pos=obj_state[:, 0:3] if object_sample_points is not None else None,
            object_rvec=obj_state[:, 3:6] if object_sample_points is not None else None,
            object_sample_points_local=object_sample_points,
        )
        trans[:, 2] += z_offset
        obj_state[:, 2] += z_offset

    # ── XY recenter ──────────────────────────────────────────────────
    if apply_xy_recenter:
        xy_center = 0.5 * (trans[:, :2].max(axis=0) + trans[:, :2].min(axis=0))
        trans[:, :2] -= xy_center.astype(np.float32)
        obj_state[:, :2] -= xy_center.astype(np.float32)

    return {
        "gender": np.array(gender),
        "betas": betas_16,
        "mocap_frame_rate": np.array(fps, dtype=np.float32),
        "trans": trans,
        "root_orient": root_orient,
        "pose_body": pose_body,
        "obj_state": obj_state,
        "obj_contact": obj_contact,
        "contact_human": contact_human,
    }


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
