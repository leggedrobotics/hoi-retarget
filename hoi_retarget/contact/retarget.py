# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-mode retarget from edited contact points.

Same build -> solve -> save -> video flow as ``hoi_retarget/cli.py``, with the edited
strips injected into the contact inputs before the window NLP. Edits are real-size
object-frame metres, scaled by ``object_scale`` on injection; only xyz reaches the
solve. Writes ``kinematic_window.pkl``, ``contact_window.pkl``,
``contact_window_<robot>.mp4`` and ``shifts.json`` to ``out_dir``.
"""

from __future__ import annotations

import contextlib
import numpy as np
import os
from types import SimpleNamespace

import torch

from hoi_retarget.config import ProjectionConfig
from hoi_retarget.contact.shifts import ShiftSet
from hoi_retarget.gmr.params import ROBOT_URDF_DICT
from hoi_retarget.io import build_pkl_meta, save_motion_pickle
from hoi_retarget.optimization.window import PinocchioMotionProjectorWindow
from hoi_retarget.paths import CONTACT_EDITOR_OUTPUT
from hoi_retarget.rendering.video import record_motion_video
from hoi_retarget.retargeting.object_model import get_object_motion_defaults
from hoi_retarget.retargeting.source_data import build_projection_motion_data_with_gmr
from hoi_retarget.robots import activate_robot

OUTPUT_ROOT = str(CONTACT_EDITOR_OUTPUT)


def _inject_shifts(md: dict, shiftset: ShiftSet):
    """Rebuild the solver's contact inputs from the edited strips, in place.

    Each enabled strip sets its ``[start, end)`` flags and points (the built point
    ``+ delta * object_scale``), and sets ``object_contact_sequence`` there so the NLP's
    per-frame gate admits it; disabled strips set nothing. An unedited ShiftSet
    reproduces the original flags and points. Returns ``(moved, disabled, retimed)``.
    """
    cp_old = np.asarray(md["fixed_contact_points_per_frame_in_object_frame"], dtype=np.float32)
    link_names = list(md["contact_link_names"])
    object_scale = float(md.get("object_scale", 1.0))
    n, C = cp_old.shape[0], cp_old.shape[1]

    per_link = np.zeros((n, C), dtype=bool)
    cp = np.full((n, C, 3), np.nan, dtype=np.float32)
    obj_contact = np.asarray(md.get("object_contact_sequence", np.zeros((n, 1)))).reshape(-1).astype(np.int8)
    if obj_contact.shape[0] != n:
        obj_contact = np.zeros(n, dtype=np.int8)

    moved = disabled = retimed = 0
    for sh in shiftset.shifts.values():
        if sh.link_name not in link_names:
            continue
        c = link_names.index(sh.link_name)
        if not sh.enabled:
            disabled += 1
            continue
        s = max(0, int(sh.start))
        e = min(n, int(sh.end))
        if e <= s:
            continue
        per_link[s:e, c] = True
        obj_contact[s:e] = 1
        base = cp_old[s:e, c, :].copy()
        missing = ~np.isfinite(base).all(axis=-1)
        if missing.any():
            base[missing] = np.asarray(sh.auto_real, dtype=np.float32) * object_scale
        cp[s:e, c, :] = base + sh.delta.astype(np.float32) * object_scale
        if sh.moved:
            moved += 1
        if sh.retimed:
            retimed += 1

    md["fixed_contact_points_per_frame_in_object_frame"] = cp
    md["per_link_contact_flags"] = per_link
    md["object_contact_sequence"] = obj_contact.reshape(-1, 1)
    return moved, disabled, retimed


def run_contact_retarget_with_shifts(
    source_path: str,
    object_model_path: str,
    shiftset: ShiftSet,
    *,
    out_dir: str | None = None,
    robot: str = "unitree_g1",
    gender: str = "neutral",
    tgt_fps: int = 30,
    record_video: bool = True,
    headless: bool = True,
    verbose: bool = True,
    scale_object_trajectory: bool = True,
    strip_mean_targets: bool = False,
) -> str:
    """Run the contact-mode solve from shifted points; returns ``out_dir``.

    The object is solved at ``object_scale * object_aug_scale`` (the editor's size).
    """
    # Selects the per-robot link names used downstream (e.g. the H2's feet).
    activate_robot(robot)
    source_path = os.path.abspath(source_path)
    stem = os.path.splitext(os.path.basename(source_path))[0]
    if out_dir is None:
        out_dir = os.path.join(OUTPUT_ROOT, stem)
    os.makedirs(out_dir, exist_ok=True)
    # Saved before the solve so a failed Go keeps the edits.
    shiftset.save(os.path.join(out_dir, "shifts.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Same config as the CLI: the robot's own URDF and link profile.
    cfg = ProjectionConfig.from_args(
        SimpleNamespace(
            robot=robot,
            urdf_path=str(ROBOT_URDF_DICT[robot]),
            object_model_path=object_model_path,
            mode="contact",
            object_scale=float(shiftset.object_scale * shiftset.object_aug_scale),
            scale_object_trajectory=scale_object_trajectory,
            contact_points_per_frame=not strip_mean_targets,
        )
    )
    obj_def = get_object_motion_defaults(object_model_path)
    projector = PinocchioMotionProjectorWindow(
        cfg=cfg,
        min_object_height=obj_def["min_object_height"],
        contact_links=obj_def["contact_links"],
        compute_analysis=False,
    )
    oc = cfg.object
    fc = cfg.foot_contact

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── Build (silent unless verbose) ────────────────────────────────────
    _build_ctx = contextlib.nullcontext()
    if not verbose:
        _null = open(os.devnull, "w")
        _build_ctx = contextlib.ExitStack()
        _build_ctx.enter_context(_null)
        _build_ctx.enter_context(contextlib.redirect_stdout(_null))
        _build_ctx.enter_context(contextlib.redirect_stderr(_null))
    with _build_ctx:
        md, _src = build_projection_motion_data_with_gmr(
            input_path=source_path,
            robot=robot,
            gender=gender,
            tgt_fps=tgt_fps,
            device=device,
            min_object_height=projector.min_object_height,
            contact_links=projector.contact_links,
            object_model_path=projector.object_model_path,
            object_scale=cfg.object.scale,
            scale_object_trajectory=cfg.object.scale_trajectory,
            object_traj_smooth_window=cfg.object.traj_smooth_window,
            ground_contact_tol=oc.ground_contact_tol,
            foot_contact_enabled=fc.enabled,
            foot_contact_z_thresh=fc.z_thresh,
            foot_contact_vel_thresh=fc.vel_thresh,
            foot_contact_smplx_joints=fc.smplx_joint_names,
            contact_points_per_frame_mode=oc.contact_points_per_frame,
            contact_points_smooth_window=oc.contact_points_smooth_window,
        )

    # ── Inject the manual shifts ─────────────────────────────────────────
    moved, disabled, retimed = _inject_shifts(md, shiftset)
    _log(
        f"[hoi_retarget.contact] {moved} moved · {retimed} retimed · {disabled} disabled "
        f"of {len(shiftset.shifts)} strips (object_scale={md.get('object_scale', 1.0):.4f})."
    )

    def _save(md_local: dict, stage: str) -> dict:
        meta = build_pkl_meta(
            source_file=source_path,
            robot=robot,
            projection_mode=projector.projection_mode,
            stage=stage,
            cfg=projector.cfg,
        )
        return save_motion_pickle(
            md_local,
            out_dir,
            stage,
            object_model_path=object_model_path,
            meta=meta,
        )

    def _video(md_local: dict, stage: str) -> None:
        if not record_video:
            return
        vpath = os.path.join(out_dir, f"{stage}_{robot}.mp4")
        record_motion_video(
            md_local,
            robot,
            vpath,
            object_model_path=object_model_path,
            headless=headless,
            object_scale=float(md_local.get("object_scale", 1.0)),
        )

    # Stage-1 output for reference, then the contact-stage NLP.
    _save(md, "kinematic_window")
    _log(f"[hoi_retarget.contact] solving contact-stage window NLP (object_scale {md.get('object_scale', 1.0):.3f}) …")
    window_data = projector.project_motion(md)
    window_md = _save(window_data, "contact_window")
    _video(window_md, "contact_window")
    _log(f"[hoi_retarget.contact] done → {out_dir}")
    return out_dir
