# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo video recording of a stage pickle.

``record_motion_video`` runs forward kinematics on the pickle and renders the
robot + object through ``RobotMotionViewer`` in a :mod:`render_style` preset
(default ``square``) or a ``Style`` instance.
"""

from __future__ import annotations

import numpy as np
import os

import imageio.v2 as imageio
import torch

from hoi_retarget.gmr.kinematics_model import KinematicsModel
from hoi_retarget.gmr.params import ROBOT_BASE_DICT, ROBOT_XML_DICT
from hoi_retarget.gmr.robot_motion_viewer import RobotMotionViewer
from hoi_retarget.rendering import render_style
from hoi_retarget.rendering.render_style import DEFAULT_STYLE, Style
from hoi_retarget.retargeting.object_model import compute_min_object_height_per_frame
from hoi_retarget.retargeting.source_data import compute_object_contact_points_world_by_frame


def _object_corners_world(viewer, obj_pos, obj_rot) -> np.ndarray | None:
    """``(T, 8, 3)`` world corners of the object's AABB (already scaled), or None if no geom."""
    from scipy.spatial.transform import Rotation as R

    gid = viewer.object_geom_id
    if gid is None or obj_pos is None or obj_rot is None:
        return None
    c, half = viewer.model.geom_aabb[gid][:3], viewer.model.geom_aabb[gid][3:]
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    local = c + signs * half  # (8, 3) geom frame
    g_rot = R.from_quat(np.roll(viewer.model.geom_quat[gid], -1))
    local = g_rot.apply(local) + viewer.model.geom_pos[gid]  # -> body frame
    mats = R.from_quat(np.roll(np.asarray(obj_rot, dtype=float), -1, axis=-1)).as_matrix()  # (T, 3, 3)
    return np.einsum("tij,kj->tki", mats, local) + np.asarray(obj_pos, dtype=float)[:, None, :]


def _fit(viewer, style, world_body_pos, obj_pos, obj_rot, track, fps):
    """``fit_camera`` over this clip's robot links + object bound."""
    content = [np.asarray(world_body_pos, dtype=float)]
    corners = _object_corners_world(viewer, obj_pos, obj_rot)
    if corners is not None:
        content.append(corners)
    n = min(c.shape[0] for c in content)
    return render_style.fit_camera(
        np.concatenate([c[:n] for c in content], axis=1),
        np.asarray(track, dtype=float)[:n],
        style,
        aspect=style.width / style.height,
        fovy_deg=float(viewer.model.vis.global_.fovy),
        fps=fps,
    )


def solve_camera(
    motion_data: dict,
    robot: str,
    style: str | Style | None = DEFAULT_STYLE,
    object_model_path: str | None = None,
    object_scale: float = 1.0,
):
    """The ``(distance, lookat)`` :func:`record_motion_video` would fit, or None for a fixed-camera style.

    Pass it as ``camera=`` to another render (e.g. the SMPL source) to share the framing.
    """
    style = render_style.get_style(style)
    if not style.cam_fit:
        return None
    fps = float(motion_data["fps"])
    world_body_pos, _orient, base_idx, _names = forward_kinematics(motion_data, robot)
    track = render_style.smooth_track(world_body_pos[:, base_idx, :].astype(float), style)
    viewer = RobotMotionViewer(
        robot_type=robot,
        motion_fps=fps,
        record_video=False,
        object_model_path=object_model_path,
        headless=True,
        object_scale=float(object_scale),
        style=style,
    )
    try:
        return _fit(
            viewer, style, world_body_pos, motion_data.get("object_pos"), motion_data.get("object_rot"), track, fps
        )
    finally:
        viewer.close()


def forward_kinematics(motion_data: dict, robot: str):
    """``(world_body_pos, world_body_orient, base_idx, body_names)`` for a clip."""
    root_pos = np.asarray(motion_data["root_pos"])
    root_rot = np.asarray(motion_data["root_rot"])
    dof_pos = np.asarray(motion_data["dof_pos"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kinematics_model = KinematicsModel(str(ROBOT_XML_DICT[robot]), device=device)
    pos, orient = kinematics_model.forward_kinematics(
        root_pos=torch.from_numpy(root_pos).to(device=device, dtype=torch.float32),
        root_rot=torch.from_numpy(root_rot)[..., [1, 2, 3, 0]].to(device=device, dtype=torch.float32),
        dof_pos=torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
    )
    body_names = kinematics_model.body_names
    base_idx = {n: i for i, n in enumerate(body_names)}.get(ROBOT_BASE_DICT[robot])
    return (pos.cpu().numpy(), orient[..., [3, 0, 1, 2]].cpu().numpy(), base_idx, body_names)


def record_motion_video(
    motion_data: dict,
    robot: str,
    video_path: str,
    object_model_path: str | None = None,
    headless: bool = True,
    object_scale: float = 1.0,
    style: str | Style | None = DEFAULT_STYLE,
    single_frame: int | None = None,
    camera: tuple | None = None,
) -> None:
    """Render ``motion_data`` to ``video_path``.

    ``single_frame`` writes that frame as a PNG instead of an mp4. ``camera`` is
    ``(distance, lookat)`` (scalar or ``(T,)`` distance, ``(T, 3)`` target) and
    overrides the style's framing.
    """
    object_scale = float(object_scale)
    style = render_style.get_style(style)

    fps = float(motion_data["fps"])
    root_pos = np.asarray(motion_data["root_pos"])
    root_rot = np.asarray(motion_data["root_rot"])
    dof_pos = np.asarray(motion_data["dof_pos"])
    world_body_pos, world_body_orient, base_idx, body_names = forward_kinematics(motion_data, robot)
    obj_pos = np.asarray(motion_data.get("object_pos")) if "object_pos" in motion_data else None
    obj_rot = np.asarray(motion_data.get("object_rot")) if "object_rot" in motion_data else None
    contact_seq = (
        np.asarray(motion_data.get("object_contact_sequence")).reshape(-1)
        if "object_contact_sequence" in motion_data
        else None
    )
    ground_contact_seq = (
        np.asarray(motion_data.get("object_ground_contact_sequence")).reshape(-1)
        if "object_ground_contact_sequence" in motion_data
        else None
    )
    per_frame_heights = motion_data.get("object_min_height_per_frame")
    if per_frame_heights is None and object_model_path is not None and obj_rot is not None:
        per_frame_heights = compute_min_object_height_per_frame(object_model_path, obj_rot)
        if per_frame_heights is not None and object_scale != 1.0:
            per_frame_heights = np.asarray(per_frame_heights, dtype=float) * object_scale
    if per_frame_heights is not None:
        per_frame_heights = np.asarray(per_frame_heights, dtype=float).reshape(-1)
    contact_points_world_by_frame = compute_object_contact_points_world_by_frame(
        motion_data=motion_data,
        num_frames=root_pos.shape[0],
    )

    contact_link_names = list(motion_data.get("contact_link_names", []))
    per_link_flags = motion_data.get("per_link_contact_flags")
    body_name_to_idx = {name: i for i, name in enumerate(body_names)}

    # Smoothed pelvis lookat; None lets the viewer use the raw pelvis position.
    lookat_track = None
    if base_idx is not None and style.styled and style.smooth_camera:
        lookat_track = render_style.smooth_track(world_body_pos[:, base_idx, :].astype(float), style)

    still = single_frame is not None
    frames = [int(single_frame)] if still else range(root_pos.shape[0])

    viewer = RobotMotionViewer(
        robot_type=robot,
        motion_fps=fps,
        record_video=not still,
        video_path=None if still else video_path,
        object_model_path=object_model_path,
        headless=headless or still,
        object_scale=object_scale,
        style=style,
    )
    cam_distance_track = None
    if camera is not None:
        distance, lookat_track = camera
        lookat_track = np.asarray(lookat_track, dtype=float)
        viewer.cam_lookat_dz = 0.0
        if np.ndim(distance) == 0:
            viewer.viewer_cam_distance = float(distance)
        else:
            cam_distance_track = np.asarray(distance, dtype=float)
            viewer.viewer_cam_distance = float(cam_distance_track[0])
        print("[viewer] framing supplied by caller")
    elif style.cam_fit:
        # Fit on the same lookat track the render uses.
        track = (
            world_body_pos[:, base_idx, :].astype(float)
            if lookat_track is None and base_idx is not None
            else lookat_track
        )
        if track is not None:
            distance, lookat_track = _fit(viewer, style, world_body_pos, obj_pos, obj_rot, track, fps)
            viewer.cam_lookat_dz = 0.0  # the fitted track already carries it
            if np.ndim(distance) == 0:
                viewer.viewer_cam_distance = float(distance)
                print(f"[viewer] auto-framed ({style.cam_fit_track}): distance={distance:.2f} m")
            else:
                cam_distance_track = np.asarray(distance, dtype=float)
                viewer.viewer_cam_distance = float(cam_distance_track[0])
                print(
                    f"[viewer] auto-framed ({style.cam_fit_track}): distance "
                    f"{cam_distance_track.min():.2f}-{cam_distance_track.max():.2f} m "
                    f"(dolly)"
                )
    try:
        img = None
        for t in frames:
            if cam_distance_track is not None and t < cam_distance_track.shape[0]:
                viewer.viewer_cam_distance = float(cam_distance_track[t])
            contact = float(contact_seq[t]) if (contact_seq is not None and t < contact_seq.shape[0]) else 0.0
            object_data = None
            if obj_pos is not None and obj_rot is not None:
                ground_contact = (
                    bool(ground_contact_seq[t])
                    if (ground_contact_seq is not None and t < ground_contact_seq.shape[0])
                    else False
                )
                min_height_t = (
                    float(per_frame_heights[t])
                    if (per_frame_heights is not None and t < per_frame_heights.shape[0])
                    else None
                )
                object_data = (obj_pos[t], obj_rot[t], contact, ground_contact, min_height_t)
            human_motion_data = {
                name: (world_body_pos[t, i], world_body_orient[t, i]) for i, name in enumerate(body_names)
            }
            object_contact_points_world = None
            if t < len(contact_points_world_by_frame):
                object_contact_points_world = contact_points_world_by_frame[t]

            per_link_contact_data = None
            if contact_link_names:
                per_link_contact_data = []
                frame_cp = contact_points_world_by_frame[t] if t < len(contact_points_world_by_frame) else {}
                for c, link_name in enumerate(contact_link_names):
                    if per_link_flags is not None and t < per_link_flags.shape[0]:
                        is_active = bool(per_link_flags[t, c]) if c < per_link_flags.shape[1] else False
                    else:
                        is_active = bool(contact)
                    if link_name in body_name_to_idx:
                        link_pos = world_body_pos[t, body_name_to_idx[link_name]]
                    else:
                        continue
                    per_link_contact_data.append(
                        {
                            "link_pos": link_pos,
                            "contact_point_world": frame_cp.get(link_name),
                            "active": is_active,
                        }
                    )

            img = viewer.step(
                root_pos=root_pos[t],
                root_rot=root_rot[t],
                dof_pos=dof_pos[t],
                human_motion_data=human_motion_data,
                show_human_body_name=True,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                object_data=object_data,
                object_contact_points_world=object_contact_points_world,
                per_link_contact_data=per_link_contact_data,
                rate_limit=(not headless) and not still,
                camera_lookat=None if lookat_track is None else lookat_track[t],
                capture=still,
            )
        if still:
            out_dir = os.path.dirname(video_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            imageio.imwrite(video_path, img)
    finally:
        viewer.close()
