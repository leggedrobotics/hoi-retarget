# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 1: source clip -> GMR IK robot reference, contact flags and targets, object scaling.

Entry point: :func:`build_projection_motion_data_with_gmr`. Quaternions are wxyz.
"""

from __future__ import annotations

import numpy as np
import os
import pathlib
import pickle
from typing import Any

import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from hoi_retarget.datasets.intermimic import aggregate_contact_human_to_g1
from hoi_retarget.gmr.kinematics_model import KinematicsModel
from hoi_retarget.gmr.motion_retarget import GeneralMotionRetargeting
from hoi_retarget.gmr.rot_utils import (
    pad_or_truncate,
    pad_or_truncate_1d,
    rot_from_quat_wxyz,
)
from hoi_retarget.gmr.utils.smpl import (
    convert_intermimic_to_smplx,
    get_smplx_data_offline_fast,
    load_smplx_data,
)
from hoi_retarget.paths import BODY_MODEL_ROOT
from hoi_retarget.retargeting.object_model import (
    G1_LINK_TO_SMPLX_ANCHOR_STATIC,
    compute_min_object_height_per_frame,
)
from hoi_retarget.retargeting.object_scaling import _GEOM_SCALED_FIELDS, apply_object_scaling, resolve_object_scale


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _find_contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end_exclusive)`` per contiguous run of ``True``: the grab /
    release / regrab events in a per-link contact flag sequence."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size == 0:
        return []
    # Pad with False at both ends so every run has a matched rising/falling edge pair.
    padded = np.concatenate(([False], m, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _projection_keys_present(payload: dict[str, Any]) -> bool:
    required = {
        "fps",
        "root_pos",
        "root_rot",
        "dof_pos",
        "world_body_pos",
        "world_body_orient",
        "local_body_pos",
        "local_body_orient",
        "link_body_list",
        "dof_names",
        "object_pos",
        "object_rot",
    }
    return isinstance(payload, dict) and required.issubset(set(payload.keys()))


def _convert_projection_dict_to_numpy(payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in payload.items():
        out[key] = _to_numpy(value) if isinstance(value, torch.Tensor) else value
    return out


def _load_projection_motion_data(input_path: str) -> dict[str, Any] | None:
    ext = os.path.splitext(input_path)[1].lower()

    if ext in (".pkl", ".pickle"):
        try:
            with open(input_path, "rb") as f:
                payload = pickle.load(f)
            if _projection_keys_present(payload):
                return _convert_projection_dict_to_numpy(payload)
        except Exception:
            pass

    if ext in (".pt", ".pth"):
        try:
            payload = torch.load(input_path, map_location="cpu")
            if _projection_keys_present(payload):
                return _convert_projection_dict_to_numpy(payload)
        except Exception:
            pass

    try:
        payload = np.load(input_path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.shape == ():
            payload = payload.item()
        if _projection_keys_present(payload):
            return _convert_projection_dict_to_numpy(payload)
    except Exception:
        pass

    return None


def _extract_source_data(
    input_path: str,
    gender: str,
    tgt_fps: int,
    min_object_height: float,
    object_model_path: str | None = None,
) -> tuple[dict[str, Any], str]:
    smplx_folder = BODY_MODEL_ROOT

    # Dispatch on the file suffix first; InterAct output paths contain neither dataset name.
    suffix = pathlib.Path(input_path).suffix.lower()
    is_cari4d = suffix == ".pth" or "cari4d" in input_path.lower()

    if is_cari4d:
        from hoi_retarget.datasets.cari4d import convert_cari4d_to_smplx
        from hoi_retarget.retargeting.object_model import get_sample_points_path

        object_sp_path = get_sample_points_path(object_model_path) if object_model_path is not None else None
        smplx_data = convert_cari4d_to_smplx(
            input_path,
            gender,
            object_sample_points_path=object_sp_path,
        )
        source_name = "cari4d_gmr"
    elif suffix == ".pt" or "intermimic" in input_path.lower():
        # "auto" is resolved here, not earlier: only CARI4D infers gender from its filename.
        smplx_data = convert_intermimic_to_smplx(input_path, "neutral" if str(gender) == "auto" else gender)
        source_name = "intermimic_omomo_gmr"
    else:
        raise ValueError(
            f"Unrecognized source file: {input_path}\n"
            "Expected an InterMimic/InterAct clip (.pt) or a CARI4D reconstruction (.pth). "
            "See docs/DATA.md for where each comes from."
        )

    body_model, smplx_output, actual_human_height, obj_data = load_smplx_data(smplx_data, smplx_folder)
    smplx_frames, aligned_fps, object_frames = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=tgt_fps,
        object_data=obj_data,
    )

    if object_frames is None:
        raise ValueError("Input data has no object trajectory.")

    object_pos = np.asarray([frame[0] for frame in object_frames], dtype=np.float32)
    object_rot = np.asarray([frame[1] for frame in object_frames], dtype=np.float32)
    object_contact = np.asarray([frame[2] for frame in object_frames], dtype=np.float32).reshape(-1, 1)

    # InterMimic/OMOMO store the object relative to ``transl``; shift by (pelvis - transl).
    # CARI4D is already absolute (its converter pre-corrects ``trans``).
    if source_name != "cari4d_gmr":
        object_pos += (smplx_frames[-1]["pelvis"][0] - smplx_output.transl[-1].numpy()).astype(np.float32)

    # CARI4D has no per-body contact flag extraction.
    contact_human = None
    contact_human_raw = smplx_data.get("contact_human")
    if contact_human_raw is not None:
        contact_human_raw = np.asarray(contact_human_raw, dtype=np.int32)
        T_resampled = len(smplx_frames)
        T_orig = contact_human_raw.shape[0]
        if T_orig == T_resampled:
            contact_human = contact_human_raw
        else:
            indices = np.round(np.linspace(0, T_orig - 1, T_resampled)).astype(int)
            contact_human = contact_human_raw[indices]

    return (
        {
            "smplx_frames": smplx_frames,
            "fps": float(aligned_fps),
            "actual_human_height": actual_human_height,
            "object_pos": object_pos,
            "object_rot": object_rot,
            "object_contact": object_contact,
            "object_ground_contact": (object_pos[:, 2:3] <= min_object_height).astype(np.int8),
            "contact_human": contact_human,
        },
        source_name,
    )


def _build_motion_data_from_source_frames(
    *,
    smplx_frames: list,
    fps: float,
    object_pos: np.ndarray,
    object_rot_wxyz: np.ndarray,
    object_contact: np.ndarray,
    object_ground_contact: np.ndarray,
    robot: str,
    actual_human_height: float | None,
    device: str | None,
    contact_links: list[str],
    contact_human: np.ndarray | None = None,
    object_model_path: str | None = None,
    object_scale: float | None = None,
    scale_object_trajectory: bool = False,
    ground_contact_tol: float = 0.025,
    foot_contact_enabled: bool = True,
    foot_contact_z_thresh: float = 0.10,
    foot_contact_vel_thresh: float = 0.30,
    foot_contact_smplx_joints: tuple[str, str] = ("left_foot", "right_foot"),
    contact_points_per_frame_mode: bool = False,
    contact_points_smooth_window: int = 0,
) -> dict[str, Any]:
    if not smplx_frames:
        raise ValueError("Empty motion sequence.")

    T = len(smplx_frames)
    object_pos = pad_or_truncate(np.asarray(object_pos, dtype=np.float32), T)
    object_rot_wxyz = pad_or_truncate(np.asarray(object_rot_wxyz, dtype=np.float32), T)
    object_contact = pad_or_truncate(np.asarray(object_contact, dtype=np.float32).reshape(-1, 1), T)
    contact_mask = object_contact.reshape(-1) > 0.5

    object_min_height_per_frame: np.ndarray | None = None
    if object_model_path is not None:
        object_min_height_per_frame = compute_min_object_height_per_frame(
            object_model_path,
            object_rot_wxyz,
        )
    if object_min_height_per_frame is not None:
        reclassified = (object_pos[:, 2:3] <= (object_min_height_per_frame.reshape(-1, 1) + ground_contact_tol)).astype(
            np.int8
        )
        if not np.array_equal(reclassified, object_ground_contact.astype(np.int8)):
            flipped = int(np.abs(reclassified.astype(int) - object_ground_contact.astype(int)).sum())
            print(
                f"[INFO] Reclassified ground_contact with per-frame floor "
                f"({flipped} frames changed, tol={ground_contact_tol:.3f} m)."
            )
        object_ground_contact = reclassified

    retarget = GeneralMotionRetargeting(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
    )
    required_joint_keys = (
        set(retarget.human_body_to_task1.keys()) | set(retarget.human_body_to_task2.keys()) | {retarget.human_root_name}
    )
    missing = sorted(k for k in required_joint_keys if k not in smplx_frames[0])
    if missing:
        raise ValueError("Input motion is missing SMPL-X joints required by IK config: " + ", ".join(missing))

    # offset_to_ground re-plants the lowest foot; otherwise a source fit high floats.
    # HOI_RETARGET_IK_SETTLE=n pre-solves frame 0 n times: it alone starts from the home
    # pose rather than a warm start, so it can still be converging.
    settle = int(os.environ.get("HOI_RETARGET_IK_SETTLE", "0"))
    for _ in range(settle):
        retarget.retarget(smplx_frames[0], offset_to_ground=True)

    qpos = np.asarray(
        [retarget.retarget(frame, offset_to_ground=True) for frame in smplx_frames],
        dtype=np.float32,
    )
    root_pos = qpos[:, :3]
    root_rot = qpos[:, 3:7]
    dof_pos = qpos[:, 7:]

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    kinematics = KinematicsModel(retarget.xml_file, device=device)
    world_body_pos, world_body_orient = kinematics.forward_kinematics(
        root_pos=torch.from_numpy(root_pos).to(device=device, dtype=torch.float32),
        root_rot=torch.from_numpy(root_rot)[..., [1, 2, 3, 0]].to(device=device, dtype=torch.float32),
        dof_pos=torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
    )
    local_body_pos, local_body_orient = kinematics.forward_kinematics(
        root_pos=torch.zeros(root_pos.shape).to(device=device, dtype=torch.float32),
        root_rot=(
            torch.zeros(root_rot.shape).to(device=device, dtype=torch.float32)
            + torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        ),
        dof_pos=torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
    )

    num_frames = root_pos.shape[0]
    object_pos = pad_or_truncate(np.asarray(object_pos, dtype=np.float32), num_frames)
    object_rot_wxyz = pad_or_truncate(np.asarray(object_rot_wxyz, dtype=np.float32), num_frames)
    object_contact = pad_or_truncate(np.asarray(object_contact, dtype=np.float32).reshape(-1, 1), num_frames)
    object_ground_contact = pad_or_truncate(
        np.asarray(object_ground_contact, dtype=np.float32).reshape(-1, 1), num_frames
    )

    if object_min_height_per_frame is not None:
        object_min_height_per_frame = pad_or_truncate_1d(
            object_min_height_per_frame,
            num_frames,
        )

    motion_data = {
        "fps": float(fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "world_body_pos": world_body_pos.cpu().numpy(),
        "world_body_orient": world_body_orient[..., [3, 0, 1, 2]].cpu().numpy(),
        "local_body_pos": local_body_pos.cpu().numpy(),
        "local_body_orient": local_body_orient[..., [3, 0, 1, 2]].cpu().numpy(),
        "link_body_list": kinematics.body_names,
        "dof_names": retarget.robot_motor_names,
        "object_pos": object_pos,
        "object_rot": object_rot_wxyz,
        "_s_root": float(retarget.human_scale_table[retarget.human_root_name]),
        "object_contact_sequence": np.round(object_contact).astype(np.int8),
        "object_ground_contact_sequence": np.round(object_ground_contact).astype(np.int8),
    }
    if object_min_height_per_frame is not None:
        motion_data["object_min_height_per_frame"] = object_min_height_per_frame

    # ── Per-foot ground-contact detection from SMPLX toe kinematics ──
    if foot_contact_enabled and all(j in smplx_frames[0] for j in foot_contact_smplx_joints):
        l_joint, r_joint = foot_contact_smplx_joints
        l_toe = np.asarray([f[l_joint][0] for f in smplx_frames], dtype=np.float32)
        r_toe = np.asarray([f[r_joint][0] for f in smplx_frames], dtype=np.float32)
        # Pad to num_frames to match the other motion_data arrays.
        l_toe = pad_or_truncate(l_toe, num_frames)
        r_toe = pad_or_truncate(r_toe, num_frames)
        dt_frame = 1.0 / float(fps)
        l_vel = np.linalg.norm(np.gradient(l_toe, dt_frame, axis=0), axis=1)
        r_vel = np.linalg.norm(np.gradient(r_toe, dt_frame, axis=0), axis=1)
        # Gate: max(z_thresh, per-clip toe minimum + margin), so sources whose SMPL-X feet
        # sit high still get stance frames.
        rel_margin = 0.04
        l_gate = max(foot_contact_z_thresh, float(np.min(l_toe[:, 2])) + rel_margin)
        r_gate = max(foot_contact_z_thresh, float(np.min(r_toe[:, 2])) + rel_margin)
        l_stance = (l_toe[:, 2] < l_gate) & (l_vel < foot_contact_vel_thresh)
        r_stance = (r_toe[:, 2] < r_gate) & (r_vel < foot_contact_vel_thresh)
        foot_ground_contact_flags = np.stack([l_stance, r_stance], axis=1).astype(bool)
        motion_data["robot_foot_ground_contact_flags"] = foot_ground_contact_flags
        n_l = int(l_stance.sum())
        n_r = int(r_stance.sum())
        print(
            f"[INFO] Foot-ground contact detection (SMPLX toe z below "
            f"max({foot_contact_z_thresh:.3f}, per-foot-floor+{rel_margin:.3f}) m, "
            f"|v|<{foot_contact_vel_thresh:.3f} m/s): "
            f"L={n_l}/{num_frames}, R={n_r}/{num_frames} stance frames."
        )

    contact_link_names = list(contact_links)

    per_link_contact_flags = None
    if contact_human is not None and contact_link_names:
        contact_human_padded = pad_or_truncate(np.asarray(contact_human, dtype=np.int32), num_frames)
        per_link_contact_flags = aggregate_contact_human_to_g1(contact_human_padded, contact_link_names)
        motion_data["per_link_contact_flags"] = per_link_contact_flags

        T_total = per_link_contact_flags.shape[0]
        global_contact_pct = contact_mask.sum() / max(T_total, 1) * 100
        print(
            f"[INFO] Per-link contact flags from InterMimic contact_human "
            f"({T_total} frames, global contact_obj: {global_contact_pct:.1f}%):"
        )
        for c, name in enumerate(contact_link_names):
            link_count = int(per_link_contact_flags[:, c].sum())
            link_pct = link_count / max(T_total, 1) * 100
            print(f"  {name:>35s}: {link_count:4d} frames ({link_pct:5.1f}%)")

    if not contact_link_names:
        return motion_data

    contact_world_by_link: dict[str, np.ndarray] = {}
    for link_name in contact_link_names:
        if link_name in G1_LINK_TO_SMPLX_ANCHOR_STATIC:
            smplx_joint = G1_LINK_TO_SMPLX_ANCHOR_STATIC[link_name]
            contact_world_by_link[link_name] = pad_or_truncate(
                np.asarray([f[smplx_joint][0] for f in smplx_frames], dtype=np.float32),
                num_frames,
            )
        else:
            contact_world_by_link[link_name] = object_pos  # fallback for unmapped links

    object_rot_inv = R.from_quat(object_rot_wxyz[:, [1, 2, 3, 0]]).inv()

    # (T, C, 3) object-frame contact points, NaN where the link is not in contact.
    contact_points_per_frame = np.full((num_frames, len(contact_link_names), 3), np.nan, dtype=np.float32)
    for c, link_name in enumerate(contact_link_names):
        contact_world = contact_world_by_link[link_name]
        per_frame_points_object = object_rot_inv.apply(contact_world - object_pos)

        # Fall back to the global object-contact mask only when there is no per-link info.
        if per_link_contact_flags is not None:
            link_mask = per_link_contact_flags[:, c].astype(bool)
        else:
            link_mask = contact_mask.astype(bool)

        if per_frame_points_object.shape[0] == 0 or not np.any(link_mask):
            continue

        # Per-frame mode: the (optionally smoothed) per-frame point; else one mean per run.
        for seg_start, seg_end in _find_contiguous_segments(link_mask):
            if contact_points_per_frame_mode:
                seg_vals = per_frame_points_object[seg_start:seg_end]
                if contact_points_smooth_window >= 3 and seg_vals.shape[0] > contact_points_smooth_window:
                    w = contact_points_smooth_window | 1
                    seg_vals = savgol_filter(seg_vals, w, 2, axis=0)
                contact_points_per_frame[seg_start:seg_end, c, :] = seg_vals.astype(np.float32)
            else:
                seg_mean = per_frame_points_object[seg_start:seg_end].mean(axis=0)
                contact_points_per_frame[seg_start:seg_end, c, :] = seg_mean.astype(np.float32)

    motion_data["contact_link_names"] = contact_link_names
    motion_data["fixed_contact_points_per_frame_in_object_frame"] = contact_points_per_frame

    s_root = float(motion_data.get("_s_root", 1.0))  # robot-fit scale (about world origin)
    scale_object_mesh, aug = resolve_object_scale(object_scale, s_root, robot)  # aug: about object origin
    if scale_object_mesh:
        geom_s = s_root * aug  # object geometry scale
        if s_root != 1.0 or aug != 1.0:
            # Trajectory: robot-fit scale only, about the world origin.
            op = np.asarray(motion_data["object_pos"], dtype=np.float32) * s_root
            # Geometry (object frame): full geom_s, about the object's origin.
            for k in _GEOM_SCALED_FIELDS:  # contact points, object_min_height
                if motion_data.get(k) is not None:
                    motion_data[k] = np.asarray(motion_data[k], dtype=np.float32) * geom_s
            if object_min_height_per_frame is not None:
                object_min_height_per_frame = np.asarray(object_min_height_per_frame, dtype=np.float32) * geom_s
            # Grounded frames rest the resized object on the floor; airborne keep the s_root path.
            if object_min_height_per_frame is not None:
                grounded = object_ground_contact.reshape(-1).astype(bool)
                if grounded.any():
                    op[grounded, 2] = object_min_height_per_frame[grounded]
                motion_data["object_pos"] = op
            else:
                motion_data["object_pos"] = op

    return motion_data


def build_projection_motion_data_with_gmr(
    *,
    input_path: str,
    robot: str,
    gender: str,
    tgt_fps: int = 60,
    device: str,
    min_object_height: float,
    contact_links: list[str] | None,
    object_model_path: str | None = None,
    object_scale: float | None = None,
    scale_object_trajectory: bool = False,
    object_traj_smooth_window: int = 0,
    ground_contact_tol: float = 0.025,
    foot_contact_enabled: bool = True,
    foot_contact_z_thresh: float = 0.10,
    foot_contact_vel_thresh: float = 0.30,
    foot_contact_smplx_joints: tuple[str, str] = ("left_foot", "right_foot"),
    contact_points_per_frame_mode: bool = False,
    contact_points_smooth_window: int = 0,
) -> tuple[dict[str, Any], str]:

    source_data, source_name = _extract_source_data(
        input_path=input_path,
        gender=gender,
        tgt_fps=tgt_fps,
        min_object_height=min_object_height,
        object_model_path=object_model_path,
    )

    motion_data = _build_motion_data_from_source_frames(
        smplx_frames=source_data["smplx_frames"],
        fps=source_data["fps"],
        object_pos=source_data["object_pos"],
        object_rot_wxyz=source_data["object_rot"],
        object_contact=source_data["object_contact"],
        object_ground_contact=source_data["object_ground_contact"],
        robot=robot,
        actual_human_height=source_data["actual_human_height"],
        device=device,
        contact_links=contact_links or [],
        contact_human=source_data.get("contact_human"),
        object_model_path=object_model_path,
        object_scale=object_scale,
        scale_object_trajectory=scale_object_trajectory,
        ground_contact_tol=ground_contact_tol,
        foot_contact_enabled=foot_contact_enabled,
        foot_contact_z_thresh=foot_contact_z_thresh,
        foot_contact_vel_thresh=foot_contact_vel_thresh,
        foot_contact_smplx_joints=foot_contact_smplx_joints,
        contact_points_per_frame_mode=contact_points_per_frame_mode,
        contact_points_smooth_window=contact_points_smooth_window,
    )

    s_root = float(motion_data.pop("_s_root", 1.0))
    scale_object_mesh, aug = resolve_object_scale(object_scale, s_root, robot)
    if scale_object_trajectory and not scale_object_mesh:
        # apply_object_scaling mutates object_pos; keep the real-size trajectory.
        motion_data["object_pos_realsize_ref"] = np.asarray(motion_data["object_pos"], dtype=np.float32).copy()
    geom_s = (s_root * aug) if scale_object_mesh else s_root
    motion_data, object_geom_scale = apply_object_scaling(
        motion_data,
        scale_object_mesh,
        scale_object_trajectory,
        geom_s,
        ground_contact_tol=ground_contact_tol,
        geometry_prescaled=scale_object_mesh,
        smooth_window=object_traj_smooth_window,
    )
    if scale_object_mesh:
        # object_pos is x s_root about the world origin; recover the real-size placement.
        motion_data["object_pos_realsize_ref"] = (
            np.asarray(motion_data["object_pos"], dtype=np.float32) / s_root
        ).astype(np.float32)
    motion_data["object_scale"] = float(object_geom_scale)
    motion_data["object_aug_scale"] = float(aug)
    motion_data["scale_object_mesh"] = bool(scale_object_mesh)
    motion_data["scale_object_trajectory"] = bool(scale_object_trajectory)
    return motion_data, source_name


def compute_object_contact_points_world_by_frame(
    motion_data: dict,
    num_frames: int,
) -> list[dict[str, np.ndarray]]:
    """Per-frame ``{link: world point}`` from the object-frame contact points.

    A link appears in frame ``t`` only when its point is finite (in contact).
    """
    contact_link_names = list(motion_data.get("contact_link_names", []))
    contact_points_per_frame = motion_data.get("fixed_contact_points_per_frame_in_object_frame")
    if contact_points_per_frame is None or not contact_link_names:
        return []

    contact_points_per_frame = np.asarray(contact_points_per_frame, dtype=np.float32)
    if contact_points_per_frame.size == 0:
        return []
    if contact_points_per_frame.ndim != 3 or contact_points_per_frame.shape[1] != len(contact_link_names):
        return []
    contact_points_per_frame = pad_or_truncate(contact_points_per_frame, num_frames)

    if "object_pos" not in motion_data or "object_rot" not in motion_data:
        return []
    object_pos = pad_or_truncate(np.asarray(motion_data["object_pos"], dtype=np.float32), num_frames)
    object_rot = pad_or_truncate(np.asarray(motion_data["object_rot"], dtype=np.float32), num_frames)

    contact_points_world_by_frame = []
    for t in range(num_frames):
        R_world_object = rot_from_quat_wxyz(object_rot[t])
        frame_contact_points = {}
        for c, link_name in enumerate(contact_link_names):
            p_local = contact_points_per_frame[t, c]
            if not np.all(np.isfinite(p_local)):
                continue  # NaN = link not in contact at this frame
            frame_contact_points[link_name] = object_pos[t] + R_world_object @ p_local
        contact_points_world_by_frame.append(frame_contact_points)
    return contact_points_world_by_frame
