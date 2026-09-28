# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate every registered robot's assets against what the pipeline assumes.

Stage 1 is MuJoCo (MJCF), Stage 2 is Pinocchio (URDF). Each check below fails silently in the
pipeline, producing a wrong trajectory and a plausible-looking video:

  1. ``nq`` agreement.
  2. Joint order agreement: Stage 2 feeds ``dof_pos = qpos[7:]`` straight into the URDF model.
  2b. Actuator names equal joint names; a miss keeps the Stage-1 value for that joint
     (tell: Stage-2 ``dof_pos`` bit-identical to Stage 1 while ``root_pos`` differs).
  2c. Actuator order equals joint order.
  3. Every profile link exists as a MuJoCo body and a Pinocchio frame (H2 has both
     ``*_ankle_roll_link`` and ``*_ankle_pitch_link``, so a wrong name resolves).
  4. Every IK-config target frame exists.
  5. MuJoCo and Pinocchio FK agree at a random pose.

Run:  hoi-retarget-check-robot [robot ...]
"""

from __future__ import annotations

import json
import numpy as np
import sys
from pathlib import Path

from hoi_retarget.gmr.params import (
    IK_CONFIG_DICT,
    ROBOT_BASE_DICT,
    ROBOT_URDF_DICT,
    ROBOT_XML_DICT,
)
from hoi_retarget.robots import (
    ROBOT_PROFILES,
    SUPPORTED_ROBOTS,
)

FK_TOL_M = 1e-4


def _ik_target_frames(robot: str) -> set:
    """Frame names the SMPL-X IK config asks Stage 1 to track."""
    path = IK_CONFIG_DICT.get("smplx", {}).get(robot)
    if path is None or not Path(path).is_file():
        return set()
    cfg = json.loads(Path(path).read_text())
    frames = set()
    for table in ("ik_match_table1", "ik_match_table2"):
        frames |= set(cfg.get(table, {}))
    return frames


def check(robot: str) -> list:
    import mujoco as mj
    import pinocchio as pin

    errs: list = []
    profile = ROBOT_PROFILES[robot]
    xml, urdf = ROBOT_XML_DICT.get(robot), ROBOT_URDF_DICT.get(robot)
    base_body = ROBOT_BASE_DICT.get(robot, "pelvis")
    if xml is None or not Path(xml).is_file():
        return [f"no MJCF for {robot!r} (ROBOT_XML_DICT -> {xml})"]
    if urdf is None or not Path(urdf).is_file():
        return [f"no URDF for {robot!r} (ROBOT_URDF_DICT -> {urdf})"]

    m = mj.MjModel.from_xml_path(str(xml))
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())

    # 1. DoF count agreement
    if m.nq != model.nq:
        errs.append(f"nq mismatch: mujoco {m.nq} vs pinocchio {model.nq}")

    # 2. Joint order agreement
    mj_joints = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    mj_joints = [j for j in mj_joints if j != base_body]
    pin_joints = [model.names[i] for i in range(2, len(model.names))]
    if mj_joints != pin_joints:
        for i, (a, b) in enumerate(zip(mj_joints, pin_joints)):
            if a != b:
                errs.append(f"joint order diverges at index {i}: mujoco {a!r} vs pinocchio {b!r}")
                break
        else:
            errs.append(f"joint count differs: mujoco {len(mj_joints)} vs pinocchio {len(pin_joints)}")

    # 2b. Actuator names must equal joint names.
    actuators = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    act_joints = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]
    for a, joint in zip(actuators, act_joints):
        if a != joint:
            errs.append(
                f"actuator {a!r} drives joint {joint!r}: names must match or Stage 2 silently discards this joint"
            )
    missing_act = [j for j in pin_joints if j not in set(actuators)]
    if missing_act:
        errs.append(
            f"URDF joints with no matching actuator name (Stage 2 would silently keep the Stage-1 value): {missing_act}"
        )

    # 2c. Actuator ORDER must equal joint order.
    if act_joints != mj_joints:
        for i, (a, j) in enumerate(zip(act_joints, mj_joints)):
            if a != j:
                errs.append(
                    f"actuator order diverges from joint order at index {i}: "
                    f"actuator drives {a!r} but qpos column {i} is {j!r} -- "
                    f"Stage 2 would write solutions into the wrong columns"
                )
                break
        else:
            errs.append(f"actuator/joint count differs: {len(act_joints)} vs {len(mj_joints)}")

    # 3. Every link the profile names must exist in BOTH models
    mj_bodies = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
    pin_frames = {f.name for f in model.frames}
    required = (
        set(profile.contact_links)
        | set(profile.hand_links)
        | set(profile.foot_links)
        | set(profile.track_world_links)
        | set(profile.track_base_links)
        | {base_body}
    )
    for name in sorted(required):
        if name not in mj_bodies:
            errs.append(f"body {name!r} missing from MuJoCo model")
        if name not in pin_frames:
            errs.append(f"frame {name!r} missing from URDF/Pinocchio model")

    # 3b. The anchor/OMOMO tables must only mention links this robot has.
    for name in sorted(set(profile.smplx_anchor) | set(profile.omomo_contact.values())):
        if name not in mj_bodies:
            errs.append(f"contact table names {name!r}, absent from MuJoCo model")

    # 4. Every IK-config target frame must exist in the MuJoCo model
    targets = _ik_target_frames(robot)
    if not targets:
        errs.append(f"no readable SMPL-X IK config for {robot!r}")
    for frame in sorted(targets):
        if frame not in mj_bodies:
            errs.append(f"IK config targets body {frame!r}, absent from MuJoCo model")

    # 5. The two models must agree geometrically, not just nominally.
    rng = np.random.default_rng(0)
    q_mj = np.zeros(m.nq)
    q_mj[3] = 1.0  # identity quaternion (w, x, y, z)
    q_mj[7:] = rng.uniform(-0.3, 0.3, m.nq - 7)
    d.qpos[:] = q_mj
    mj.mj_forward(m, d)

    q_pin = pin.neutral(model)
    q_pin[:3] = q_mj[:3]
    q_pin[3:7] = [q_mj[4], q_mj[5], q_mj[6], q_mj[3]]  # wxyz -> xyzw
    q_pin[7:] = q_mj[7:]
    data = model.createData()
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)

    for name in sorted(set(profile.contact_links) | set(profile.track_world_links)):
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0 or not model.existFrame(name):
            continue
        err = float(np.linalg.norm(d.xpos[bid] - data.oMf[model.getFrameId(name)].translation))
        if err > FK_TOL_M:
            errs.append(f"{name}: MuJoCo/Pinocchio FK disagree by {err * 1000:.2f} mm")

    return errs


def main(argv=None) -> int:
    # None when called as the console script.
    argv = sys.argv[1:] if argv is None else argv
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return 0
    robots = argv or [r for r in SUPPORTED_ROBOTS if r in ROBOT_URDF_DICT]
    failed = False
    for robot in robots:
        if robot not in ROBOT_PROFILES:
            print(f"[FAIL] {robot}: no profile; known: {', '.join(SUPPORTED_ROBOTS)}")
            failed = True
            continue
        errs = check(robot)
        print(f"[{'FAIL' if errs else '  ok':>4}] {robot}")
        for e in errs:
            print(f"         - {e}")
        failed |= bool(errs)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
