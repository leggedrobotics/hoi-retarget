# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin


class PinocchioCasadiRobot:
    """Pinocchio free-flyer model with CasADi FK functions for every frame (world and base-relative)."""

    def __init__(
        self,
        urdf_path: str,
        object_model_path: str | None = None,
        base_body_name: str = "pelvis",
        package_dirs: list[str] | None = None,
    ):
        if package_dirs is None:
            package_dirs = [os.path.dirname(urdf_path)]

        root_joint = pin.JointModelFreeFlyer()
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, package_dirs, root_joint
        )
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.q_lower = self.model.lowerPositionLimit.copy()
        self.q_upper = self.model.upperPositionLimit.copy()
        self.v_abs = self.model.velocityLimit.copy()

        self.joint_q_indices = []
        self.joint_v_indices = []
        for jid, joint in enumerate(self.model.joints):
            if jid <= 1:
                continue
            if joint.nq == 1 and joint.nv == 1:
                self.joint_q_indices.append(joint.idx_q)
                self.joint_v_indices.append(joint.idx_v)
        self.joint_q_indices = np.array(self.joint_q_indices, dtype=int)
        self.joint_v_indices = np.array(self.joint_v_indices, dtype=int)

        ff_joint = self.model.joints[1]
        self.base_q_start = ff_joint.idx_q
        self.base_q_size = ff_joint.nq

        self.base_body_name = base_body_name
        self.body_names = [f.name for f in self.model.frames]

        self.dof_name_to_index = {}
        for jid, joint in enumerate(self.model.joints):
            name = self.model.names[jid]
            if jid <= 1:
                continue
            if joint.nq == 1:
                self.dof_name_to_index[name] = joint.idx_q

        self.cmodel = cpin.Model(self.model)
        q_sym = ca.SX.sym("q", self.cmodel.nq, 1)
        data_sym = self.cmodel.createData()
        # FK normalises the base quaternion; otherwise its norm scales the robot.
        b = self.base_q_start + 3
        quat = q_sym[b : b + 4]
        q_unit = ca.vertcat(q_sym[:b], quat / ca.norm_2(quat), q_sym[b + 4 :])
        cpin.forwardKinematics(self.cmodel, data_sym, q_unit)
        cpin.updateFramePlacements(self.cmodel, data_sym)

        self.fk_world_pos = {}
        self.fk_world_rot = {}
        self.fk_rel_pos = {}
        self.fk_rel_rot = {}

        base_frame_id = self.cmodel.getFrameId(self.base_body_name)
        for frame_id in range(self.cmodel.nframes):
            frame = self.cmodel.frames[frame_id]
            name = frame.name
            oMf = data_sym.oMf[frame_id]
            p_WB = oMf.translation
            R_WB = oMf.rotation
            self.fk_world_pos[name] = ca.Function(f"fk_wp_{name}", [q_sym], [p_WB])
            self.fk_world_rot[name] = ca.Function(f"fk_wr_{name}", [q_sym], [R_WB])

            oMf_base = data_sym.oMf[base_frame_id]
            base_to_body = oMf_base.inverse() * oMf
            p_BB = base_to_body.translation
            R_BB = base_to_body.rotation
            self.fk_rel_pos[(self.base_body_name, name)] = ca.Function(f"fk_bp_{name}", [q_sym], [p_BB])
            self.fk_rel_rot[(self.base_body_name, name)] = ca.Function(f"fk_br_{name}", [q_sym], [R_BB])

        self.q0 = pin.neutral(self.model)

        self.object_model_path = os.path.abspath(object_model_path) if object_model_path else None

    def world_pos(self, body_name, q):
        return self.fk_world_pos[body_name](q)

    def world_rotmat(self, body_name, q):
        return self.fk_world_rot[body_name](q)

    def rel_pos(self, base_name, body_name, q):
        return self.fk_rel_pos[(base_name, body_name)](q)

    def rel_rotmat(self, base_name, body_name, q):
        return self.fk_rel_rot[(base_name, body_name)](q)

    def compute_kinematics(self, q):
        pin.forwardKinematics(self.model, self.data, pin.normalize(self.model, q))
        pin.updateFramePlacements(self.model, self.data)
