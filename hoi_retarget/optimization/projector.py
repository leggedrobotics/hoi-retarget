# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre- and post-processing around the windowed NLP.

``_preprocess_motion`` packs the Stage-1 output into a :class:`PreprocessedMotion`;
``_postprocess_result`` runs FK on the solution and builds the output ``motion_data``.
The object pose is the fixed (scaled) reference, passed through unchanged.
"""

from __future__ import annotations

import numpy as np
from typing import Any

import casadi as ca
import pinocchio as pin

from hoi_retarget.config import PreprocessedMotion
from hoi_retarget.gmr.rot_utils import (
    finite_difference_velocities,
    pad_or_truncate,
    pad_or_truncate_1d,
    rot_from_quat_wxyz,
)


class BasePinocchioProjector:
    def _preprocess_motion(self, motion_data: dict) -> PreprocessedMotion:
        """Pad per-frame arrays to N and build index maps; IK and object scaling happen upstream."""
        fps = float(motion_data["fps"])
        dt = 1.0 / fps

        root_pos = np.asarray(motion_data["root_pos"])
        root_rot = np.asarray(motion_data["root_rot"])
        dof_pos = np.asarray(motion_data["dof_pos"])
        obj_pos = np.asarray(motion_data["object_pos"])
        obj_rot = np.asarray(motion_data["object_rot"])

        dof_names = list(motion_data["dof_names"])
        body_names = list(motion_data["link_body_list"])
        ref_world_body_pos = np.asarray(motion_data["world_body_pos"])
        ref_world_body_orient = np.asarray(motion_data["world_body_orient"])
        ref_local_body_pos = np.asarray(motion_data["local_body_pos"])
        ref_local_body_orient = np.asarray(motion_data["local_body_orient"])
        contact_seq = np.asarray(motion_data["object_contact_sequence"]).reshape(-1)
        ground_contact_seq = np.asarray(
            motion_data.get(
                "object_ground_contact_sequence",
                (obj_pos[:, 2] <= self.min_object_height).astype(np.int8),
            )
        ).reshape(-1)

        N, _ = dof_pos.shape
        nq = self.robot.nq
        nv = self.robot.nv
        dof_idx_map = self.robot.dof_name_to_index

        obj_pos = pad_or_truncate(obj_pos, N)
        obj_rot = pad_or_truncate(obj_rot, N)
        contact_seq = pad_or_truncate_1d(contact_seq, N)
        ground_contact_seq = pad_or_truncate_1d(ground_contact_seq, N)

        # Base-relative rotations, for the torso orientation anchor.
        R_local_ref = np.zeros(ref_local_body_orient.shape[:-1] + (3, 3))
        for t in range(N):
            for i in range(len(body_names)):
                R_local_ref[t, i] = rot_from_quat_wxyz(ref_local_body_orient[t, i])

        body_contact_flags = contact_seq != 0
        ground_contact_flags = ground_contact_seq != 0

        per_link_flags_raw = motion_data.get("per_link_contact_flags")
        per_link_flags = None
        if per_link_flags_raw is not None:
            per_link_flags = pad_or_truncate(
                np.asarray(per_link_flags_raw, dtype=bool),
                N,
            )
            contact_link_names_dbg = list(motion_data.get("contact_link_names", []))
            global_contact_frames = int(body_contact_flags.sum())
            print(f"[INFO] Projector per-link contact gating ({N} frames, global contact: {global_contact_frames}):")
            for c, name in enumerate(contact_link_names_dbg):
                if c < per_link_flags.shape[1]:
                    active = int((body_contact_flags & per_link_flags[:, c]).sum())
                    skipped = global_contact_frames - active
                    print(f"  {name:>35s}: {active:4d} active, {skipped:4d} skipped")

        # Object velocity, for the output pickle only.
        v_ref = finite_difference_velocities(obj_pos, dt)

        contact_link_names = list(motion_data.get("contact_link_names", []))
        contact_points_per_frame_np = np.asarray(
            motion_data.get("fixed_contact_points_per_frame_in_object_frame", []),
            dtype=np.float32,
        )
        contact_points_object_dm_per_frame: list[dict[str, Any]] = [dict() for _ in range(N)]
        if (
            contact_points_per_frame_np.ndim == 3
            and contact_points_per_frame_np.shape[1] == len(contact_link_names)
            and len(contact_link_names) > 0
        ):
            cp = pad_or_truncate(contact_points_per_frame_np, N)
            for t in range(N):
                frame_dm = contact_points_object_dm_per_frame[t]
                for c, name in enumerate(contact_link_names):
                    p_local = cp[t, c]
                    if np.all(np.isfinite(p_local)):
                        frame_dm[name] = ca.DM(p_local.reshape(3, 1))
        R_world_object_dm = [ca.DM(rot_from_quat_wxyz(obj_rot[t])) for t in range(N)]

        foot_ground_contact_flags = motion_data.get("robot_foot_ground_contact_flags")
        if foot_ground_contact_flags is not None:
            foot_ground_contact_flags = np.asarray(foot_ground_contact_flags, dtype=bool)
            if foot_ground_contact_flags.shape[0] != N:
                foot_ground_contact_flags = pad_or_truncate(foot_ground_contact_flags, N)

        # Stance-foot orientation references (foot pin_orient, on by default).
        R_world_foot_ref = None
        fc = self.cfg.foot_contact
        if fc.enabled and fc.pin_orient and fc.pin_orient_w > 0.0:
            R_world_foot_ref = np.zeros((N, len(fc.foot_link_names), 3, 3))
            for fi, foot_link in enumerate(fc.foot_link_names):
                if foot_link not in body_names:
                    continue
                bi = body_names.index(foot_link)
                for t in range(N):
                    R_world_foot_ref[t, fi] = rot_from_quat_wxyz(ref_world_body_orient[t, bi])

        # Contact-link orientation references (object pin_orient, on by default).
        R_world_contact_ref = None
        occ = self.cfg.object
        if getattr(occ, "pin_orient", False) and occ.pin_orient_w > 0.0 and contact_link_names:
            R_world_contact_ref = np.zeros((N, len(contact_link_names), 3, 3))
            for ci, link in enumerate(contact_link_names):
                if link not in body_names:
                    continue
                bi = body_names.index(link)
                for t in range(N):
                    R_world_contact_ref[t, ci] = rot_from_quat_wxyz(ref_world_body_orient[t, bi])

        return PreprocessedMotion(
            N=N,
            dt=dt,
            nq=nq,
            nv=nv,
            fps=fps,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            obj_pos=obj_pos,
            obj_rot=obj_rot,
            dof_names=dof_names,
            body_names=body_names,
            dof_idx_map=dof_idx_map,
            ref_world_body_pos=ref_world_body_pos,
            ref_world_body_orient=ref_world_body_orient,
            ref_local_body_pos=ref_local_body_pos,
            ref_local_body_orient=ref_local_body_orient,
            R_local_ref=R_local_ref,
            body_contact_flags=body_contact_flags,
            ground_contact_flags=ground_contact_flags,
            per_link_flags=per_link_flags,
            contact_points_object_dm_per_frame=contact_points_object_dm_per_frame,
            contact_link_names=contact_link_names,
            R_world_object_dm=R_world_object_dm,
            v_ref=v_ref,
            foot_ground_contact_flags=foot_ground_contact_flags,
            R_world_foot_ref=R_world_foot_ref,
            R_world_contact_ref=R_world_contact_ref,
            motion_data=motion_data,
        )

    def _postprocess_result(
        self,
        result: dict[str, np.ndarray],
        prep: PreprocessedMotion,
    ) -> dict:
        """Run FK on solved trajectory and build output motion_data dict."""
        N = prep.N
        dof_pos = prep.dof_pos
        dof_names = prep.dof_names
        body_names = prep.body_names
        dof_idx_map = prep.dof_idx_map
        root_pos = prep.root_pos
        root_rot = prep.root_rot
        ref_world_body_pos = prep.ref_world_body_pos
        ref_world_body_orient = prep.ref_world_body_orient
        ref_local_body_pos = prep.ref_local_body_pos
        ref_local_body_orient = prep.ref_local_body_orient
        per_link_flags = prep.per_link_flags
        contact_link_names = prep.contact_link_names
        motion_data = prep.motion_data

        contact_seq = prep.body_contact_flags.astype(np.int8)
        ground_contact_seq = prep.ground_contact_flags.astype(np.int8)

        q_sol = result["q"]
        # Object pose/velocity are the fixed reference.
        obj_pos_sol = result["obj_pos"]
        obj_vel_sol = result["obj_vel"]
        obj_rot_sol = result["obj_rot"]

        new_dof_pos = np.zeros_like(dof_pos)
        new_world_body_pos = np.zeros_like(ref_world_body_pos)
        new_world_body_orient = np.zeros_like(ref_world_body_orient)
        new_local_body_pos = np.zeros_like(ref_local_body_pos)
        new_local_body_orient = np.zeros_like(ref_local_body_orient)
        new_root_pos = np.zeros_like(root_pos)
        new_root_rot = np.zeros_like(root_rot)

        base_frame_id = self.robot.model.getFrameId(self.base_body_name)
        for t in range(N):
            q_t = q_sol[t, :]
            self.robot.compute_kinematics(q_t)

            oMf_base = self.robot.data.oMf[base_frame_id]
            new_root_pos[t] = np.array(oMf_base.translation).reshape(
                3,
            )
            R_WBase = oMf_base.rotation
            quat_base = pin.Quaternion(np.array(R_WBase))
            quat_base.normalize()
            new_root_rot[t] = np.array([quat_base.w, quat_base.x, quat_base.y, quat_base.z])

            for i, j_name in enumerate(dof_names):
                if j_name in dof_idx_map:
                    idx_q = dof_idx_map[j_name]
                    new_dof_pos[t, i] = q_t[idx_q]
                else:
                    new_dof_pos[t, i] = dof_pos[t, i]

            for i, b_name in enumerate(body_names):
                if b_name not in self.robot.body_names:
                    # MuJoCo-only bodies (left_toe_link, head_mocap, ...) have no Pinocchio
                    # frame; copy the Stage-1 reference.
                    new_world_body_pos[t, i] = ref_world_body_pos[t, i]
                    new_world_body_orient[t, i] = ref_world_body_orient[t, i]
                    new_local_body_pos[t, i] = ref_local_body_pos[t, i]
                    new_local_body_orient[t, i] = ref_local_body_orient[t, i]
                    continue
                f_id = self.robot.model.getFrameId(b_name)
                oMf_body = self.robot.data.oMf[f_id]

                new_world_body_pos[t, i] = np.array(oMf_body.translation).reshape(
                    3,
                )
                R_WB = oMf_body.rotation
                quat_WB = pin.Quaternion(np.array(R_WB))
                quat_WB.normalize()
                new_world_body_orient[t, i] = np.array([quat_WB.w, quat_WB.x, quat_WB.y, quat_WB.z])

                base_to_body = oMf_base.inverse() * oMf_body
                new_local_body_pos[t, i] = np.array(base_to_body.translation).reshape(
                    3,
                )
                R_BB = base_to_body.rotation
                quat_BB = pin.Quaternion(np.array(R_BB))
                quat_BB.normalize()
                new_local_body_orient[t, i] = np.array([quat_BB.w, quat_BB.x, quat_BB.y, quat_BB.z])

        fps = prep.fps
        new_motion_data = dict(motion_data)
        new_motion_data.update(
            {
                "fps": fps,
                "root_pos": new_root_pos,
                "root_rot": new_root_rot,
                "dof_pos": new_dof_pos,
                "world_body_pos": new_world_body_pos,
                "world_body_orient": new_world_body_orient,
                "local_body_pos": new_local_body_pos,
                "local_body_orient": new_local_body_orient,
                "object_pos": obj_pos_sol,
                "object_vel": obj_vel_sol,
                "object_rot": np.asarray(obj_rot_sol, dtype=np.float32),
                "object_contact_sequence": contact_seq.reshape(-1, 1),
                "per_link_contact_flags": per_link_flags,
                "contact_link_names": contact_link_names,
                "object_ground_contact_sequence": ground_contact_seq.reshape(-1, 1),
            }
        )
        if prep.foot_ground_contact_flags is not None:
            new_motion_data["robot_foot_ground_contact_flags"] = prep.foot_ground_contact_flags.astype(bool)

        return new_motion_data
