# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen dataclass configuration for the Stage-2 contact NLP, composed into :class:`ProjectionConfig`.

The only decision variable is the robot configuration trajectory ``q``; the object trajectory
is a fixed (optionally scaled) reference. :class:`PreprocessedMotion` carries derived
per-frame data through the solver.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, replace
from typing import Any

from hoi_retarget.gmr.params import ROBOT_BASE_DICT
from hoi_retarget.robots import get_active_profile

# ── Tuning dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrackingWeights:
    """Per-link tracking costs.

    ``link_weights_world``: world position only (head keeps the upper body upright, ankles at 100 plant the feet).
    ``link_weights_base``: base-relative orientation anchor, ``torso_link`` only; without it the pelvis snaps
    on single frames.
    """

    # G1 link names; ``from_args`` substitutes the active robot's (weights unchanged).
    link_weights_world: dict[str, float] = field(
        default_factory=lambda: {
            "head_link": 1.0,
            "left_ankle_roll_link": 100.0,
            "right_ankle_roll_link": 100.0,
        }
    )
    link_weights_base: dict[str, float] = field(
        default_factory=lambda: {
            "torso_link": 1.0,
        }
    )
    link_pos: float = 1.0
    link_orient_base: float = 1.0


@dataclass(frozen=True)
class SmoothnessWeights:
    """Temporal regularisation of ``q``.

    ``velocity_reg``: jerk penalty on the third difference of ``q`` (quaternion third difference for base
    rotation). At 0 the motion is visibly jittery.
    ``base_velocity_limits`` (off): hard box on the floating base's FD velocity; the free-flyer has no URDF limit.
    ``first_order_reg`` (ablation, default 0): ``w * ||(q[t] - q[t-1]) / dt||^2``, the only smoothness cost on
    windows under four frames, where the jerk term is inert.
    """

    velocity_reg: float = 1e-3
    first_order_reg: float = 0.0
    base_velocity_limits: bool = False
    base_lin_vel_max: float = 3.0  # m/s, applied to the base translation FD
    base_ang_vel_max: float = 6.0  # rad/s, applied to the base orientation FD


@dataclass(frozen=True)
class ObjectConfig:
    """Object preprocessing and contact costs (the object pose is not optimised).

    Preprocessing:
      * ``scale``: object geometry scale; None = robot convention (0.83 fit scale on G1, 1.0 on H2).
        See ``resolve_object_scale``.
      * ``scale_trajectory``: reposition the trajectory by the GMR robot-fit scale.
      * ``traj_smooth_window``: Savitzky-Golay window (odd frames, <3 disables); removes floor noise and the
        lift-off/set-down steps that scaling introduces.
      * ``ground_contact_tol``: object-ground contact tolerance (m).

    NLP cost:
      * ``contact_weight``: pulls each active contact link onto its point on the object.
      * ``pin_orient`` / ``pin_orient_w``: SO(3) pin on ``pin_orient_links`` (hands by default), which fixes
        the otherwise free wrist/forearm roll. Unit-matching suggests ~165; the useful weight is ~30x lower.
      * ``contact_points_per_frame``: track the per-frame source contact point instead of each contact
        segment's mean (the mean is static in the object frame).
    """

    scale: float | None = None
    scale_trajectory: bool = True
    traj_smooth_window: int = 11
    ground_contact_tol: float = 0.045
    contact_weight: float = 1000.0
    pin_orient: bool = True
    pin_orient_w: float = 5.0
    pin_orient_links: tuple[str, ...] = ("left_palm_pad", "right_palm_pad")
    contact_points_per_frame: bool = True
    contact_points_smooth_window: int = 0  # Savitzky-Golay window (odd, <3 disables)


@dataclass(frozen=True)
class FootContactConfig:
    """Per-foot stance detection from the SMPL-X toe joints, and the stance pins.

    A stance foot gets a soft z-pin to the Stage-1 reference, so Stage 2 cannot lift it to chase an
    unreachable contact.
      * ``pin_xy``: also pin ankle x/y at ``pin_w`` during stance (x/y are already tracked at 100).
      * ``pin_orient``: Frobenius SO(3) residual ``||R_q - R_ref||_F^2`` (smooth, no ``acos``). Keep the weight
        small: ankle orientation is a null space of the other costs; unit-matching (~330) overshoots the useful 10.
    """

    enabled: bool = True
    z_thresh: float = 0.10  # m: SMPLX toe z below this is candidate for stance
    vel_thresh: float = 0.30  # m/s: SMPLX toe speed below this is candidate for stance
    pin_w: float = 2000.0  # z-pin weight during detected stance (2x contact_weight)
    pin_xy: bool = False  # also pin ankle world x/y at pin_w during stance
    pin_orient: bool = True  # also pin ankle world orientation during stance
    pin_orient_w: float = 10.0  # weight of the SO(3) residual (only used when pin_orient)
    # G1 names; ``from_args`` substitutes the active robot's (H2 uses ``*_ankle_pitch_link``).
    foot_link_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
    smplx_joint_names: tuple[str, str] = ("left_foot", "right_foot")  # SMPLX toe joints used for detection


@dataclass(frozen=True)
class WindowConfig:
    """Sizing and stitching for the sequential pass of short-horizon NLPs.

    Each window tracks the IK reference directly and hard-pins its first
    ``pin_frames`` frames to the accumulated output.
    """

    window_size: int = 60
    overlap: int = 3
    pin_frames: int = 3
    terminal_frames: int = 0
    ik_tracking_weight: float = 1.0
    terminal_weight: float = 0.0

    # ── Horizon-ablation knobs (defaults reproduce the released behaviour) ──
    # A tail window shorter than ``pin_frames + tail_extend_below`` is replaced by a backward-extended
    # final window. Set 1 for one-frame horizons (window_size=4, pin_frames=3).
    tail_extend_below: int = 2
    # Per-window IPOPT wall-clock cap (s); 0 = unlimited.
    max_wall_time: float = 0.0
    # On a failed or capped solve, keep IPOPT's last iterate instead of the Stage-1 guess.
    keep_last_iterate_on_failure: bool = False


@dataclass(frozen=True)
class ProjectionConfig:
    """Top-level config for the projection script — composes every sub-config."""

    # Paths / identity
    urdf_path: str = "assets/robots/g1/g1_29dof_w_hands_capsule.urdf"
    object_model_path: str | None = None
    auto_object_from_path: bool = True
    base_body_name: str = "pelvis"

    # Mode flags — one of {"kinematic", "contact"}.
    #   kinematic — stop after Stage-1 IK retargeting.
    #   contact   — run the windowed q-only contact NLP.
    projection_mode: str = "contact"

    # Composed sub-configs
    tracking: TrackingWeights = field(default_factory=TrackingWeights)
    smoothness: SmoothnessWeights = field(default_factory=SmoothnessWeights)
    object: ObjectConfig = field(default_factory=ObjectConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    foot_contact: FootContactConfig = field(default_factory=FootContactConfig)

    @classmethod
    def from_args(cls, args: Any) -> ProjectionConfig:
        """Default config with argparse overrides; missing flags keep their defaults."""
        base = cls()
        window = replace(
            base.window,
            window_size=getattr(args, "window_size", base.window.window_size),
            overlap=getattr(args, "overlap", base.window.overlap),
            pin_frames=getattr(args, "pin_frames", base.window.pin_frames),
            terminal_frames=getattr(args, "terminal_frames", base.window.terminal_frames),
            ik_tracking_weight=getattr(args, "ik_tracking_weight", base.window.ik_tracking_weight),
            terminal_weight=getattr(args, "terminal_weight", base.window.terminal_weight),
            tail_extend_below=getattr(args, "tail_extend_below", base.window.tail_extend_below),
            max_wall_time=getattr(args, "max_wall_time", base.window.max_wall_time),
            keep_last_iterate_on_failure=getattr(
                args, "keep_last_iterate_on_failure", base.window.keep_last_iterate_on_failure
            ),
        )
        obj = replace(
            base.object,
            scale=getattr(args, "object_scale", base.object.scale),
            scale_trajectory=getattr(args, "scale_object_trajectory", base.object.scale_trajectory),
            traj_smooth_window=getattr(args, "object_traj_smooth_window", base.object.traj_smooth_window),
            contact_weight=getattr(args, "contact_weight", base.object.contact_weight),
            pin_orient=getattr(args, "contact_pin_orient", base.object.pin_orient),
            pin_orient_w=getattr(args, "contact_pin_orient_weight", base.object.pin_orient_w),
            contact_points_per_frame=getattr(args, "contact_points_per_frame", base.object.contact_points_per_frame),
            contact_points_smooth_window=getattr(
                args, "contact_points_smooth_window", base.object.contact_points_smooth_window
            ),
        )
        smoothness = replace(
            base.smoothness,
            velocity_reg=getattr(args, "velocity_reg", base.smoothness.velocity_reg),
            first_order_reg=getattr(args, "first_order_reg", base.smoothness.first_order_reg),
            base_velocity_limits=getattr(args, "base_velocity_limits", base.smoothness.base_velocity_limits),
            base_lin_vel_max=getattr(args, "base_lin_vel_max", base.smoothness.base_lin_vel_max),
            base_ang_vel_max=getattr(args, "base_ang_vel_max", base.smoothness.base_ang_vel_max),
        )
        # Link names come from the active robot profile, weights stay G1-tuned. Replace, never merge: H2 has
        # both ``*_ankle_roll_link`` and ``*_ankle_pitch_link``, so a merged G1 map adds extra targets silently.
        prof = get_active_profile()
        lww = dict(prof.track_world_links)
        tracking = replace(
            base.tracking,
            link_orient_base=getattr(args, "link_orient_base_weight", base.tracking.link_orient_base),
            link_weights_world=lww,
            link_weights_base=dict(prof.track_base_links),
        )
        obj = replace(obj, pin_orient_links=tuple(prof.hand_links))
        foot_contact = replace(
            base.foot_contact,
            pin_w=getattr(args, "foot_pin_weight", base.foot_contact.pin_w),
            pin_xy=getattr(args, "foot_pin_xy", base.foot_contact.pin_xy),
            pin_orient=getattr(args, "foot_pin_orient", base.foot_contact.pin_orient),
            pin_orient_w=getattr(args, "foot_pin_orient_weight", base.foot_contact.pin_orient_w),
            foot_link_names=tuple(prof.foot_links),
        )
        return replace(
            base,
            urdf_path=args.urdf_path,
            object_model_path=args.object_model_path,
            projection_mode=args.mode,
            base_body_name=ROBOT_BASE_DICT.get(getattr(args, "robot", None), base.base_body_name),
            object=obj,
            smoothness=smoothness,
            tracking=tracking,
            window=window,
            foot_contact=foot_contact,
        )


# ── Preprocessed motion container ──────────────────────────────────────────


@dataclass
class PreprocessedMotion:
    """Per-frame solver inputs built by ``BasePinocchioProjector._preprocess_motion``."""

    # Trajectory shape & timing
    N: int
    dt: float
    nq: int
    nv: int
    fps: float

    # Reference trajectories (Stage-1 IK output; object pose is the scaled reference)
    root_pos: np.ndarray
    root_rot: np.ndarray
    dof_pos: np.ndarray
    obj_pos: np.ndarray
    obj_rot: np.ndarray

    # Names / index maps
    dof_names: list[str]
    body_names: list[str]
    dof_idx_map: dict[str, int]

    # Reference kinematics
    ref_world_body_pos: np.ndarray
    ref_world_body_orient: np.ndarray
    ref_local_body_pos: np.ndarray
    ref_local_body_orient: np.ndarray
    # Base-relative reference rotations (the torso anchor).
    R_local_ref: np.ndarray

    # Contact flags / data
    body_contact_flags: np.ndarray
    ground_contact_flags: np.ndarray
    per_link_flags: np.ndarray | None
    # Per-frame, per-link contact-point CasADi DM (in object-local frame).
    contact_points_object_dm_per_frame: list[dict[str, Any]]  # List[Dict[str, casadi.DM]]
    contact_link_names: list[str]
    R_world_object_dm: list[Any]  # List[casadi.DM]

    # FD velocity of the fixed object_pos, carried only for the output pickle's ``object_vel``.
    v_ref: np.ndarray

    # (N, 2) bool stance flags (left, right) from SMPL-X toe kinematics; None if disabled or toes missing.
    foot_ground_contact_flags: np.ndarray | None = None

    # Stage-1 world rotations of the foot links, (N, 2, 3, 3); built only when ``pin_orient`` is on.
    R_world_foot_ref: np.ndarray | None = None

    # Same for the object-contact links, (N, C, 3, 3), ordered like ``contact_link_names``.
    R_world_contact_ref: np.ndarray | None = None

    # Raw input
    motion_data: dict[str, Any] = field(default_factory=dict)
