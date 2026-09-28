# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-robot link roles (:class:`RobotProfile`) and :func:`activate_robot`, which installs them.

Much of the package names G1 links in module-level tables; ``activate_robot("unitree_h2")``
rewrites those tables before they are read (``unitree_g1`` leaves them unchanged).
``DEFAULT_CONTACT_LINKS``, ``G1_LINK_TO_SMPLX_ANCHOR_STATIC`` and ``OMOMO_TO_G1_CONTACT_LINK``
are aliased or ``from``-imported elsewhere, so they are mutated in place, never rebound.

A new robot needs: an MJCF and a URDF with matching joint order under ``assets/robots/<robot>/``
(check with ``hoi-retarget-check-robot``), an IK config in ``gmr/ik_configs/``, entries in
``gmr/params.py``, and a profile below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hoi_retarget.gmr.params import ROBOT_URDF_DICT as _ROBOT_URDF_DICT

__all__ = [
    "ROBOT_PROFILES",
    "SUPPORTED_ROBOTS",
    "RobotProfile",
    "activate_robot",
    "active_robot_name",
    "get_active_profile",
    "get_profile",
]


# ── SMPL-X / OMOMO source-side names (robot-independent) ───────────────────
_FINGER_BODIES = tuple(
    f"{side}_{digit}{i}"
    for side in ("L", "R")
    for digit in ("Index", "Middle", "Pinky", "Ring", "Thumb")
    for i in (1, 2, 3)
)


def _standard_maps(left_palm: str, right_palm: str, left_foot: str, right_foot: str):
    """Source→robot contact tables: OMOMO fingers map to one palm pad per side, ankle+toe to one foot link."""
    anchor = {
        left_palm: "left_palm",
        right_palm: "right_palm",
        left_foot: "left_ankle",
        right_foot: "right_ankle",
    }
    omomo = {
        "L_Ankle": left_foot,
        "L_Toe": left_foot,
        "R_Ankle": right_foot,
        "R_Toe": right_foot,
    }
    for body in _FINGER_BODIES:
        omomo[body] = left_palm if body.startswith("L_") else right_palm
    return anchor, omomo


@dataclass(frozen=True)
class RobotProfile:
    """Which of a robot's links play the roles the pipeline cares about."""

    name: str

    # Links that may touch the object; column order of ``per_link_contact_flags`` in the output pkl.
    contact_links: tuple[str, ...]
    # Subset of the above that gets the SO(3) contact pin (ObjectConfig).
    hand_links: tuple[str, ...]
    # Ankle links used for stance detection and the foot pin (FootContactConfig).
    foot_links: tuple[str, str]

    # Stage-2 tracking: world position / base-relative orientation (TrackingWeights).
    track_world_links: dict[str, float]
    track_base_links: dict[str, float]

    # Source-side maps, derived by ``_standard_maps`` unless overridden.
    smplx_anchor: dict[str, str] = field(default_factory=dict)
    omomo_contact: dict[str, str] = field(default_factory=dict)

    height_m: float = 0.0
    dof: int = 0
    #: Default for ``--object_scale``: shrink the mesh to the fit scale (G1, 0.83) or keep it at 1.0 (H2).
    scales_object_mesh: bool = True
    remarks: str = ""


def _profile(name, left_palm, right_palm, left_foot, right_foot, upright_link, **kw) -> RobotProfile:
    anchor, omomo = _standard_maps(left_palm, right_palm, left_foot, right_foot)
    return RobotProfile(
        name=name,
        contact_links=(left_palm, right_palm, left_foot, right_foot),
        hand_links=(left_palm, right_palm),
        foot_links=(left_foot, right_foot),
        # Upright link at 1 keeps the upper body upright, ankles at 100 plant the feet; G1-tuned, same for all robots.
        track_world_links={upright_link: 1.0, left_foot: 100.0, right_foot: 100.0},
        track_base_links={"torso_link": 1.0},
        smplx_anchor=anchor,
        omomo_contact=omomo,
        **kw,
    )


_G1 = dict(
    left_palm="left_palm_pad",
    right_palm="right_palm_pad",
    left_foot="left_ankle_roll_link",
    right_foot="right_ankle_roll_link",
    upright_link="head_link",
)

ROBOT_PROFILES: dict[str, RobotProfile] = {
    "unitree_g1": _profile(
        "unitree_g1",
        **_G1,
        height_m=1.323,
        dof=29,
        remarks="Reference platform: every weight in config.py was tuned here.",
    ),
    "unitree_h2": _profile(
        "unitree_h2",
        left_palm="left_palm_pad",
        right_palm="right_palm_pad",
        # H2's ankle roll is the parent of pitch, so the sole hangs off ``*_ankle_pitch_link``.
        # ``*_ankle_roll_link`` also exists, so the wrong name resolves silently.
        left_foot="left_ankle_pitch_link",
        right_foot="right_ankle_pitch_link",
        # H2 has no ``head_link``.
        upright_link="torso_link",
        scales_object_mesh=False,  # ships at object_scale 1.0
        height_m=1.80,
        dof=31,
        remarks=(
            "31 DoF, 1.80 m, 3-DoF waist (yaw+roll+pitch) -- waist pitch is what "
            "lets a full-size humanoid reach floor-level objects at all. Assets "
            "are the SERIAL URDF round-tripped through MuJoCo: Unitree's own MJCF "
            "models the parallel ankle/knee linkages (nq=68 vs the URDF's 38) and "
            "breaks the joint-order invariant. The palm pad hangs off "
            "`*_wrist_yaw_link` because MuJoCo collapses the fixed `*_hand_joint`, "
            "so `*_hand_link` exists only in the URDF."
        ),
    ),
}

#: Robots the CLI offers: those with a profile and a URDF (the optimization stage needs one).
SUPPORTED_ROBOTS: tuple[str, ...] = tuple(sorted(r for r in ROBOT_PROFILES if r in _ROBOT_URDF_DICT))

_ACTIVE: RobotProfile | None = None


def get_profile(robot: str) -> RobotProfile:
    try:
        return ROBOT_PROFILES[robot]
    except KeyError:
        raise KeyError(f"unknown robot {robot!r}; known: {', '.join(SUPPORTED_ROBOTS)}") from None


def robot_for_pkl(motion_data: dict, requested: str | None = None) -> str:
    """The robot a retargeted pkl was solved for; ``requested`` must agree with it."""
    recorded = (motion_data.get("meta") or {}).get("robot")
    if recorded is None and len(motion_data.get("dof_pos", ())):
        n = len(motion_data["dof_pos"][0])
        recorded = next((r for r in SUPPORTED_ROBOTS if get_profile(r).dof == n), None)
    if requested and recorded and requested != recorded:
        raise ValueError(f"this pkl was retargeted to {recorded}, not {requested}; pass --robot {recorded}")
    return requested or recorded or "unitree_g1"


def active_robot_name() -> str:
    """Name of the activated robot, defaulting to the G1."""
    return _ACTIVE.name if _ACTIVE is not None else "unitree_g1"


def get_active_profile() -> RobotProfile:
    """The activated profile, defaulting to the G1's (the shipped tables)."""
    return _ACTIVE if _ACTIVE is not None else ROBOT_PROFILES["unitree_g1"]


def activate_robot(robot: str) -> RobotProfile:
    """Install ``robot``'s link roles into the G1-named tables. Idempotent; call once, early."""
    global _ACTIVE
    profile = get_profile(robot)

    from hoi_retarget.datasets import intermimic
    from hoi_retarget.retargeting import object_model as omu

    # In place: every OBJECT_MOTION_DEFAULTS entry aliases this list.
    omu.DEFAULT_CONTACT_LINKS[:] = list(profile.contact_links)
    for cfg in omu.OBJECT_MOTION_DEFAULTS.values():
        cfg["contact_links"] = omu.DEFAULT_CONTACT_LINKS

    omu.G1_LINK_TO_SMPLX_ANCHOR_STATIC.clear()
    omu.G1_LINK_TO_SMPLX_ANCHOR_STATIC.update(profile.smplx_anchor)

    # Unmapped bodies (torso/head/limbs) stay None.
    table = intermimic.OMOMO_TO_G1_CONTACT_LINK
    for body in list(table):
        table[body] = profile.omomo_contact.get(body)

    _ACTIVE = profile
    return profile
