# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object scaling to the robot's size.

GMR shrinks the human by the root scale ``s_root`` (G1 about 0.83) about the world
origin; the object arrives at human size and is rescaled to stay with the robot.
Trajectory-only path (captured-size mesh): XY x ``s`` about the origin, z about the
per-frame lowest point ``d``: ``z' = s*z + (1-s)*d``. Mesh path: geometry resized, the
trajectory x ``s_root``, grounded frames rest on the floor (see ``source_data``).
:func:`resolve_object_scale` maps ``--object_scale`` onto the two paths.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from hoi_retarget.robots import get_profile

# Fields that scale with the object geometry (mesh path only).
_GEOM_SCALED_FIELDS = (
    "fixed_contact_points_per_frame_in_object_frame",
    "object_min_height_per_frame",
)


def _fd_velocity(p: np.ndarray, fps: float) -> np.ndarray:
    if p.shape[0] < 2:
        return np.zeros_like(p, dtype=np.float32)
    dt = 1.0 / float(fps)
    return np.gradient(p, dt, axis=0).astype(np.float32)


def smooth_object_positions(
    object_pos: np.ndarray,
    window: int,
    poly: int = 3,
    floor: np.ndarray | None = None,
    floor_slack: float = 0.005,
) -> np.ndarray:
    """Savitzky-Golay smooth a ``(T, 3)`` object-origin trajectory; ``window < 3`` is a no-op.

    The object pose is not optimised, so this is the only place to remove floor-snap
    jitter, liftoff/set-down steps and reconstruction noise. With ``floor`` (per-frame
    min-height), ``z`` stays at most ``floor_slack`` below the smoothed floor.
    """
    from scipy.signal import savgol_filter

    op = np.asarray(object_pos, dtype=np.float32)
    T = op.shape[0]
    w = int(window)
    if w < 3 or T < 5:
        return op
    if w % 2 == 0:
        w += 1
    # window must be odd and <= T
    if w > T:
        w = T if T % 2 == 1 else T - 1
    if w < 3:
        return op
    p = min(int(poly), w - 1)
    sm = savgol_filter(op, w, p, axis=0).astype(np.float32)
    if floor is not None:
        f = np.asarray(floor, dtype=np.float32).reshape(-1)
        if f.shape[0] == T:
            fs = savgol_filter(f, w, p) if w <= T else f
            sm[:, 2] = np.maximum(sm[:, 2], fs.astype(np.float32) - float(floor_slack))
    return sm


def scale_object_geometry_fields(motion_data: dict[str, Any], s: float) -> dict[str, Any]:
    """Scale ``object_pos`` and the object-frame geometry fields by ``s``, in place."""
    s = float(s)
    motion_data["object_pos"] = np.asarray(motion_data["object_pos"], dtype=np.float32) * s
    for k in _GEOM_SCALED_FIELDS:
        if motion_data.get(k) is not None:
            motion_data[k] = np.asarray(motion_data[k], dtype=np.float32) * s
    return motion_data


def resolve_object_scale(object_scale: float | None, s_root: float, robot: str) -> tuple[bool, float]:
    """``(scale_object_mesh, aug)`` giving object geometry scale ``object_scale``.

    None = robot convention: ``s_root`` (about 0.83) on the G1, captured size (1.0) on the H2.
    1.0 takes the trajectory-only path; any other value the mesh path, which resizes the
    geometry to ``object_scale`` (``aug = object_scale / s_root``) and always fits the
    trajectory by ``s_root``. The two paths move airborne frames differently.
    """
    if object_scale is None:
        return get_profile(robot).scales_object_mesh, 1.0
    if object_scale == 1.0:
        return False, 1.0
    return True, float(object_scale) / s_root


def apply_object_scaling(
    motion_data: dict[str, Any],
    scale_object_mesh: bool,
    scale_object_trajectory: bool,
    s: float,
    *,
    ground_contact_tol: float = 0.025,
    geometry_prescaled: bool = False,
    smooth_window: int = 0,
) -> tuple[dict[str, Any], float]:
    """Scale the object; returns ``(motion_data, object_geom_scale)`` on a shallow copy.

    ``s``: ``s_root`` on the trajectory-only path, the geometry scale on the mesh path;
    ``object_geom_scale`` is ``s`` on the mesh path, else 1.0.
    ``geometry_prescaled`` (mesh path): geometry and ``object_pos`` were already scaled
    upstream; only derived fields are refreshed. ``smooth_window``: see
    :func:`smooth_object_positions`.
    """
    if not scale_object_mesh and not scale_object_trajectory:
        return motion_data, 1.0

    md = dict(motion_data)
    s = float(s)
    op = np.asarray(md["object_pos"], dtype=np.float32).copy()
    d = md.get("object_min_height_per_frame")
    d = np.asarray(d, dtype=np.float32) if d is not None else None

    if not scale_object_mesh:  # trajectory only, real-size object
        op[:, 0] *= s
        op[:, 1] *= s
        if d is not None:
            op[:, 2] = s * op[:, 2] + (1.0 - s) * d  # lowest point scales by s
        else:  # no sample points: scale the lift about the resting z0
            z0 = float(op[:, 2].min())
            op[:, 2] = z0 + s * (op[:, 2] - z0)
        geom_scale = 1.0
    else:  # mesh + trajectory
        if not geometry_prescaled:
            # Standalone use; the pipeline pre-scales and passes geometry_prescaled=True.
            op *= s
            for k in _GEOM_SCALED_FIELDS:
                if md.get(k) is not None:
                    md[k] = np.asarray(md[k], dtype=np.float32) * s
        else:
            op = np.asarray(md["object_pos"], dtype=np.float32).copy()
        d = (
            np.asarray(md["object_min_height_per_frame"], dtype=np.float32)
            if md.get("object_min_height_per_frame") is not None
            else None
        )
        geom_scale = s

    if smooth_window and int(smooth_window) >= 3:
        op = smooth_object_positions(op, int(smooth_window), floor=d)

    md["object_pos"] = op

    # Refresh the object_pos-derived fields.
    if "fps" in md:
        md["object_vel"] = _fd_velocity(op, md["fps"])
    if d is not None:
        md["object_near_floor_flags"] = op[:, 2] <= d + float(ground_contact_tol)

    return md, geom_scale
