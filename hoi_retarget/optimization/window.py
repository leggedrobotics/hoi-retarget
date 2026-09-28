# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Windowed trajectory optimization (Sec. III-B).

Refines the Stage-1 IK reference over short overlapping windows. The only decision
variable is ``q``; velocities are its finite differences and the object pose is fixed.
Costs: tracking, contact, smoothness. Weights and window sizes come from
:mod:`hoi_retarget.config` (defaults = published results).
"""

import numpy as np
import time

import casadi as ca

from hoi_retarget.config import (
    PreprocessedMotion,
    ProjectionConfig,
)
from hoi_retarget.gmr.rot_utils import (
    axis_angle_from_quat_wxyz,
    quat_conjugate_wxyz,
    quat_mul_wxyz,
)
from hoi_retarget.optimization.analysis import NLPAnalysisRecorder
from hoi_retarget.optimization.pinocchio_model import PinocchioCasadiRobot
from hoi_retarget.optimization.projector import BasePinocchioProjector


class PinocchioMotionProjectorWindow(BasePinocchioProjector):
    """Sliding-window Stage-2 projector.

    Each window tracks the Stage-1 reference and pulls the contact links onto the
    object; its first ``pin_frames`` frames are hard-pinned to the previous output.
    """

    def __init__(
        self,
        cfg: ProjectionConfig,
        min_object_height: float,
        contact_links: list[str] | None = None,
        compute_analysis: bool = False,
    ):
        self.cfg = cfg
        self.robot = PinocchioCasadiRobot(
            cfg.urdf_path,
            object_model_path=cfg.object_model_path,
            base_body_name=cfg.base_body_name,
        )
        self.base_body_name = cfg.base_body_name
        self.object_model_path = cfg.object_model_path
        self.min_object_height = float(min_object_height)
        self.contact_links = contact_links or []
        self.projection_mode = cfg.projection_mode
        self._analysis = NLPAnalysisRecorder(enabled=bool(compute_analysis), config=cfg)
        # Always collected: tells solved, wall-time-capped and failed windows apart.
        self.window_status: list[dict[str, object]] = []

        w = cfg.window
        print(f"[INFO] Window projection (mode={cfg.projection_mode}): q-only contact NLP.")
        print(
            f"[INFO]   window={w.window_size}, overlap={w.overlap}, "
            f"pin={w.pin_frames}, ik_tracking_weight={w.ik_tracking_weight}"
        )

    def _build_initial_guess(self, prep: PreprocessedMotion) -> dict[str, np.ndarray]:
        """Seed ``q`` from the Stage-1 reference, joint angles clipped to limits."""
        N, nq = prep.N, prep.nq
        dof_idx_map = prep.dof_idx_map
        dof_names = prep.dof_names
        root_pos, root_rot, dof_pos = prep.root_pos, prep.root_rot, prep.dof_pos
        base_idx = self.robot.base_q_start

        q_init = np.zeros((N, nq))
        for t in range(N):
            x, y, z = root_pos[t]
            w_ref, qx_ref, qy_ref, qz_ref = root_rot[t]
            q_init[t, base_idx : base_idx + 3] = np.array([x, y, z])
            q_init[t, base_idx + 3 : base_idx + 7] = np.array([qx_ref, qy_ref, qz_ref, w_ref])
            for i, j_name in enumerate(dof_names):
                if j_name in dof_idx_map:
                    idx_q = dof_idx_map[j_name]
                    q_init[t, idx_q] = np.clip(
                        dof_pos[t, i],
                        self.robot.q_lower[idx_q],
                        self.robot.q_upper[idx_q],
                    )
        return {"q": q_init}

    def _solve_window(
        self,
        prep: PreprocessedMotion,
        ws: int,
        we: int,
        accumulated: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray] | None:
        """Build and solve the q-only NLP on frames [ws, we); returns None on failure."""
        W = we - ws
        dt = prep.dt
        nq = prep.nq
        body_names = prep.body_names

        cfg = self.cfg
        tw = cfg.tracking
        sm = cfg.smoothness
        oc = cfg.object
        wc = cfg.window

        self._analysis.start_pass(f"window [{ws}:{we})", W)

        opti = ca.Opti()
        q_traj = opti.variable(W, nq)
        self._analysis.add_variable("q_traj", (W, nq))

        total_cost = 0
        base_pos_idx = self.robot.base_q_start
        base_quat_idx = self.robot.base_q_start + 3

        # The first window has nothing to stitch to, so none of its frames are pinned.
        pin_frames = 0 if ws == 0 else wc.pin_frames
        terminal_frames = wc.terminal_frames
        ik_w = wc.ik_tracking_weight
        term_w = wc.terminal_weight

        for t in range(W):
            tg = ws + t  # global frame index
            q_t = q_traj[t, :].T

            # ---- Boundary pinning (hard constraints to accumulated output) ----
            if t < pin_frames:
                opti.subject_to(q_traj[t, :] == ca.DM(accumulated["q"][tg, :]).T)
                self._analysis.add_constraint("pinned")
                continue

            q_base_pos = q_t[base_pos_idx : base_pos_idx + 3]
            q_joints = q_t[7:]

            # ---- Joint position limits ----
            for idx_q in self.robot.joint_q_indices:
                q_lo = float(self.robot.q_lower[idx_q])
                q_hi = float(self.robot.q_upper[idx_q])
                if np.isfinite(q_lo) and np.isfinite(q_hi):
                    opti.subject_to(q_t[idx_q] >= q_lo)
                    opti.subject_to(q_t[idx_q] <= q_hi)
                    self._analysis.add_constraint("joint_limits", n=2)

            # ---- Joint velocity limits: -v_lim*dt <= q_j[t] - q_j[t-1] <= v_lim*dt ----
            if t > 0:
                q_prev = q_traj[t - 1, :].T
                for idx_q, idx_v in zip(self.robot.joint_q_indices, self.robot.joint_v_indices):
                    v_lim = float(self.robot.v_abs[idx_v])
                    if np.isfinite(v_lim) and v_lim > 0:
                        dqj = q_t[idx_q] - q_prev[idx_q]
                        opti.subject_to(dqj <= v_lim * dt)
                        opti.subject_to(dqj >= -v_lim * dt)
                        self._analysis.add_constraint("velocity_limits", n=2)

                # ---- Base velocity limits (default off) ----
                # The free-flyer has no URDF limit: box the translation delta and the
                # axis-angle of the relative quaternion.
                if sm.base_velocity_limits:
                    d_base = q_base_pos - q_prev[base_pos_idx : base_pos_idx + 3]
                    opti.subject_to(d_base <= sm.base_lin_vel_max * dt)
                    opti.subject_to(d_base >= -sm.base_lin_vel_max * dt)

                    # q stores the base quaternion xyzw; reorder to wxyz for the helpers.
                    def _wxyz(qv):
                        return ca.vertcat(
                            qv[base_quat_idx + 3], qv[base_quat_idx], qv[base_quat_idx + 1], qv[base_quat_idx + 2]
                        )

                    dq_base = quat_mul_wxyz(quat_conjugate_wxyz(_wxyz(q_prev)), _wxyz(q_t))
                    omega = axis_angle_from_quat_wxyz(dq_base)
                    opti.subject_to(omega <= sm.base_ang_vel_max * dt)
                    opti.subject_to(omega >= -sm.base_ang_vel_max * dt)
                    self._analysis.add_constraint("base_velocity_limits", n=4)

            # ---- Jerk regularization: 3rd difference of q ----
            # On the base quaternion this approximates angular jerk (the pelvis rotates slowly).
            # In the first window the stencil is clamped at frame 0, starting at rest.
            if t > 2 or ws == 0:
                q_tm1 = q_traj[max(t - 1, 0), :].T
                q_tm2 = q_traj[max(t - 2, 0), :].T
                q_tm3 = q_traj[max(t - 3, 0), :].T
                da = (q_t - 3.0 * q_tm1 + 3.0 * q_tm2 - q_tm3) / dt**2
                total_cost += sm.velocity_reg * ca.sumsqr(da)
                self._analysis.add_cost("jerk_reg")

            # ---- First-order (velocity) regularization: 1st difference of q ----
            # Off by default; for windows under four frames, where the jerk term is inert.
            if t > 0 and sm.first_order_reg > 0.0:
                dv = (q_t - q_traj[t - 1, :].T) / dt
                total_cost += sm.first_order_reg * ca.sumsqr(dv)
                self._analysis.add_cost("first_order_reg")

            # ---- IK reference tracking (base pos + joints) ----
            ik_base_pos = ca.DM(prep.root_pos[tg].reshape(3, 1))
            total_cost += ik_w * ca.sumsqr(q_base_pos - ik_base_pos)
            # The kinematics normalise the base quaternion; this pins its otherwise free norm.
            total_cost += ik_w * (ca.sumsqr(q_t[base_quat_idx : base_quat_idx + 4]) - 1) ** 2
            self._analysis.add_cost("ik_tracking")

            ik_joints_ref = np.zeros(nq - 7)
            for i, j_name in enumerate(prep.dof_names):
                if j_name in prep.dof_idx_map:
                    idx_q = prep.dof_idx_map[j_name]
                    ik_joints_ref[idx_q - 7] = np.clip(
                        prep.dof_pos[tg, i],
                        self.robot.q_lower[idx_q],
                        self.robot.q_upper[idx_q],
                    )
            total_cost += ik_w * ca.sumsqr(q_joints - ca.DM(ik_joints_ref.reshape(-1, 1)))
            self._analysis.add_cost("ik_tracking")

            # ---- Per-link world-position tracking (head + ankles) ----
            for link_name, w in tw.link_weights_world.items():
                if link_name in body_names and link_name in self.robot.fk_world_pos:
                    idx = body_names.index(link_name)
                    p_WB = self.robot.world_pos(link_name, q_t)
                    total_cost += w * tw.link_pos * ca.sumsqr(p_WB - prep.ref_world_body_pos[tg, idx])
                    self._analysis.add_cost("link_pos_tracking")

            # ---- Per-foot ground pin ----
            fc = cfg.foot_contact
            if fc.enabled and prep.foot_ground_contact_flags is not None:
                for fi, foot_link in enumerate(fc.foot_link_names):
                    if not prep.foot_ground_contact_flags[tg, fi]:
                        continue
                    if foot_link not in body_names or foot_link not in self.robot.fk_world_pos:
                        continue
                    f_idx = body_names.index(foot_link)
                    p_foot = self.robot.world_pos(foot_link, q_t)
                    ref_p = prep.ref_world_body_pos[tg, f_idx]
                    if fc.pin_xy:
                        total_cost += fc.pin_w * ca.sumsqr(p_foot - ca.DM(np.asarray(ref_p, float).reshape(3, 1)))
                    else:
                        # Default: z only. x/y keep the always-on link_weights_world
                        # ankle cost, just not the heavier stance pin.
                        total_cost += fc.pin_w * (p_foot[2] - float(ref_p[2])) ** 2
                    self._analysis.add_cost("foot_ground_pin")

                    # SO(3) pin on the ankle orientation: ||R - R_ref||_F^2 = 8 sin^2(theta/2),
                    # smooth everywhere (no acos/atan2).
                    if fc.pin_orient and fc.pin_orient_w > 0.0 and prep.R_world_foot_ref is not None:
                        R_WB = self.robot.world_rotmat(foot_link, q_t)
                        R_ref = ca.DM(prep.R_world_foot_ref[tg, fi])
                        total_cost += fc.pin_orient_w * ca.sumsqr(R_WB - R_ref)
                        self._analysis.add_cost("foot_ground_pin_orient")

            # ---- Base-relative orientation anchor (torso_link) ----
            # IK tracks base position + joints only; without this the pelvis orientation is
            # a null space and snaps on single frames.
            for link_name, w in tw.link_weights_base.items():
                key = (self.base_body_name, link_name)
                if link_name in body_names and key in self.robot.fk_rel_pos:
                    idx = body_names.index(link_name)
                    R_BB = self.robot.rel_rotmat(self.base_body_name, link_name, q_t)
                    total_cost += w * tw.link_orient_base * ca.sumsqr(R_BB - ca.DM(prep.R_local_ref[tg, idx]))
                    self._analysis.add_cost("link_orient_b")

            # ---- Terminal cost (last `terminal_frames` frames, default off) ----
            if terminal_frames > 0 and t >= W - terminal_frames and term_w > 0:
                total_cost += term_w * ca.sumsqr(q_joints - ca.DM(ik_joints_ref.reshape(-1, 1)))
                total_cost += term_w * ca.sumsqr(q_base_pos - ik_base_pos)
                self._analysis.add_cost("terminal_tracking", n=2)

            # ---- Contact cost (robot link → fixed object-frame contact point) ----
            if prep.body_contact_flags[tg]:
                frame_dm = prep.contact_points_object_dm_per_frame[tg]
                p_object_t = ca.DM(prep.obj_pos[tg, :].reshape(3, 1))
                R_obj_t = prep.R_world_object_dm[tg]
                for c, name in enumerate(prep.contact_link_names):
                    if name not in self.robot.fk_world_pos:
                        continue
                    if (
                        prep.per_link_flags is not None
                        and c < prep.per_link_flags.shape[1]
                        and not prep.per_link_flags[tg, c]
                    ):
                        continue
                    p_OC_DM = frame_dm.get(name)
                    if p_OC_DM is None:
                        continue  # per_link_flags disagrees with the producer's mask
                    p_WB = self.robot.world_pos(name, q_t)
                    p_WC_from_object = p_object_t + R_obj_t @ p_OC_DM
                    total_cost += oc.contact_weight * ca.sumsqr(p_WC_from_object - p_WB)
                    self._analysis.add_cost("object_contact")

                    # Position alone leaves wrist/forearm roll free; same residual as the foot pin.
                    if (
                        oc.pin_orient
                        and oc.pin_orient_w > 0.0
                        and prep.R_world_contact_ref is not None
                        and name in oc.pin_orient_links
                    ):
                        R_WB = self.robot.world_rotmat(name, q_t)
                        R_ref_c = ca.DM(prep.R_world_contact_ref[tg, c])
                        total_cost += oc.pin_orient_w * ca.sumsqr(R_WB - R_ref_c)
                        self._analysis.add_cost("object_contact_orient")

        opti.minimize(total_cost)

        opti.set_initial(q_traj, accumulated["q"][ws:we])

        ipopt_opts = {
            "ipopt.print_level": 0,
            "print_time": True,
            "ipopt.max_iter": 1000,
            "ipopt.acceptable_iter": 10,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_constr_viol_tol": 1e-3,
        }
        if wc.max_wall_time > 0.0:
            ipopt_opts["ipopt.max_wall_time"] = float(wc.max_wall_time)
        opti.solver("ipopt", ipopt_opts)

        print(f"    Window [{ws}:{we}) — {W} frames")
        t0 = time.time()
        status = "solved"
        try:
            q_out = opti.solve().value(q_traj)
        except RuntimeError as e:
            # IPOPT raises the same error for divergence and wall-time cap; use elapsed time.
            capped = wc.max_wall_time > 0.0 and time.time() - t0 >= 0.95 * wc.max_wall_time
            status = "capped" if capped else "failed"
            q_out = None
            if wc.keep_last_iterate_on_failure:
                try:
                    q_out = opti.debug.value(q_traj)
                    status += "_last_iterate"
                except Exception:  # no iterate to recover (e.g. failure at setup)
                    q_out = None
            print(f"    Window [{ws}:{we}) {status}: {e}")
        solve_time = time.time() - t0
        if status == "solved":
            print(f"    Window [{ws}:{we}) solved in {solve_time:.2f}s")

        self._analysis.end_pass(solve_time)
        self.window_status.append(
            dict(
                start=ws,
                end=we,
                n_frames=W,
                solve_time=solve_time,
                status=status,
            )
        )

        return None if q_out is None else {"q": q_out}

    def _unified_pass(
        self,
        prep: PreprocessedMotion,
    ) -> dict[str, np.ndarray] | None:
        """Solve all windows; each writes frames [pin_frames, W) back into the buffer."""
        N = prep.N
        wc = self.cfg.window

        print("[INFO] === Unified-window pass ===")
        print(
            f"[INFO]   window={wc.window_size}, overlap={wc.overlap}, "
            f"pin={wc.pin_frames}, ik_tracking={wc.ik_tracking_weight}"
        )

        accumulated = self._build_initial_guess(prep)
        stride = max(1, wc.window_size - wc.overlap)

        n_windows = 0
        n_failed = 0
        ws = 0
        while ws < N:
            we = min(ws + wc.window_size, N)
            if we - ws < wc.pin_frames + wc.tail_extend_below:
                # Tail too short for a useful solve — extend backward to cover.
                ws = max(0, N - wc.window_size)
                we = N
                if we - ws < wc.pin_frames + wc.tail_extend_below:
                    break

            win = self._solve_window(prep, ws, we, accumulated)

            if win is not None:
                W = we - ws
                pin = 0 if ws == 0 else wc.pin_frames
                accumulated["q"][ws + pin : we] = win["q"][pin:W]
            else:
                n_failed += 1

            n_windows += 1
            if we >= N:
                break
            ws += stride

        print(f"[INFO] Unified-window pass complete: {n_windows} windows, {n_failed} failed")

        if n_windows == 0:
            raise ValueError(
                f"No window to solve: {N} frames with window_size={wc.window_size}; need at least "
                f"pin_frames + tail_extend_below = {wc.pin_frames + wc.tail_extend_below} frames."
            )
        if n_failed == n_windows:
            return None
        return {
            "q": accumulated["q"],
            "obj_pos": prep.obj_pos,
            "obj_vel": prep.v_ref,
            "obj_rot": prep.obj_rot,
        }

    def project_motion(self, motion_data: dict) -> dict:
        """Run the unified-window projection pipeline end-to-end."""
        self._analysis.reset()
        self.window_status = []
        _pipeline_t0 = time.time()

        prep = self._preprocess_motion(motion_data)

        result = self._unified_pass(prep)
        if result is None:
            raise RuntimeError("Every contact-stage window failed to solve; no contact result was written.")

        self._analysis.set_pipeline_time(time.time() - _pipeline_t0)
        if self._analysis.enabled and self._analysis.log:
            print(self._analysis.format())

        return self._postprocess_result(result, prep)
