# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``hoi-retarget``: run the pipeline on one clip or a folder of clips.

``--mode kinematic`` stops after Sec. III-A (IK with object scaling) and writes ``kinematic_window.pkl``.
``--mode contact`` (default) continues into Sec. III-B (windowed trajectory optimization) and writes
``contact_window.pkl``. ``kinematic_window.pkl`` is written in both modes; it is the contact stage's reference.
"""

import os
import sys

if "--headless" in sys.argv or "--no-headless" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import contextlib
from dataclasses import replace

import torch

from hoi_retarget.config import (
    ProjectionConfig,
)
from hoi_retarget.datasets.cari4d import infer_cari4d_gender
from hoi_retarget.gmr.params import ROBOT_URDF_DICT
from hoi_retarget.io import build_pkl_meta, save_motion_pickle
from hoi_retarget.optimization.window import PinocchioMotionProjectorWindow
from hoi_retarget.paths import require_asset_root
from hoi_retarget.rendering.video import record_motion_video
from hoi_retarget.retargeting.object_model import (
    get_object_motion_defaults,
    resolve_object_model_path,
)
from hoi_retarget.retargeting.source_data import (
    build_projection_motion_data_with_gmr,
)
from hoi_retarget.robots import SUPPORTED_ROBOTS, activate_robot


def process_file(
    projector: PinocchioMotionProjectorWindow,
    in_path: str,
    out_dir: str,
    robot: str,
    gender: str,
    tgt_fps: int,
    device: str,
    record_video: bool,
    object_model_path: str | None = None,
    save_stages: bool = False,
    headless: bool = True,
) -> str:
    video_dir = out_dir.replace("output/", "videos/", 1)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    def _save_pickle(motion_data_local: dict, stage_name: str) -> dict:
        meta = build_pkl_meta(
            source_file=in_path,
            robot=robot,
            projection_mode=projector.projection_mode,
            stage=stage_name,
            cfg=projector.cfg,
        )
        return save_motion_pickle(
            motion_data_local,
            out_dir,
            stage_name,
            object_model_path=object_model_path,
            meta=meta,
        )

    def _record_media(md: dict, stage_name: str, is_final_stage: bool) -> None:
        if not record_video:
            return
        if not (is_final_stage or save_stages):
            return
        vpath = os.path.join(video_dir, f"{stage_name}_{robot}.mp4")
        record_motion_video(
            md,
            robot,
            vpath,
            object_model_path=object_model_path or md.get("object_model_path"),
            headless=headless,
            object_scale=float(md.get("object_scale", 1.0)),
        )

    mode = projector.projection_mode

    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
        motion_data, source_tag = build_projection_motion_data_with_gmr(
            input_path=in_path,
            robot=robot,
            gender=gender,
            tgt_fps=tgt_fps,
            device=device,
            min_object_height=projector.min_object_height,
            contact_links=projector.contact_links,
            object_model_path=projector.object_model_path,
            object_scale=projector.cfg.object.scale,
            scale_object_trajectory=projector.cfg.object.scale_trajectory,
            object_traj_smooth_window=projector.cfg.object.traj_smooth_window,
            ground_contact_tol=projector.cfg.object.ground_contact_tol,
            foot_contact_enabled=projector.cfg.foot_contact.enabled,
            foot_contact_z_thresh=projector.cfg.foot_contact.z_thresh,
            foot_contact_vel_thresh=projector.cfg.foot_contact.vel_thresh,
            foot_contact_smplx_joints=projector.cfg.foot_contact.smplx_joint_names,
            contact_points_per_frame_mode=projector.cfg.object.contact_points_per_frame,
            contact_points_smooth_window=projector.cfg.object.contact_points_smooth_window,
        )
    print(f"[INFO] Loaded motion source '{source_tag}' from '{in_path}'.")
    if gender == "auto":
        is_cari4d = in_path.lower().endswith(".pth") or "cari4d" in in_path.lower()
        resolved_gender = infer_cari4d_gender(in_path, fallback=None) if is_cari4d else "neutral"
        print(f"[INFO] --gender auto resolved to '{resolved_gender}'.")
    fps = float(motion_data.get("fps", tgt_fps))
    if abs(fps - tgt_fps) > 0.5:
        print(f"[WARN] --tgt_fps {tgt_fps} requested; the clip is {fps:g} fps (the source rate is the ceiling).")
    contact_seq = motion_data.get("object_contact_sequence")
    if mode != "kinematic" and contact_seq is not None and not contact_seq.any():
        print(f"[WARN] {in_path}: source has zero labelled contact frames.")

    kinematic_md = _save_pickle(motion_data, "kinematic_window")
    _record_media(kinematic_md, "kinematic_window", is_final_stage=(mode == "kinematic"))

    if mode == "kinematic":
        print("[INFO] --mode kinematic: stopping after Stage 1 (IK retargeting).")
        return out_dir

    stage_tag = f"{mode}_window"
    window_data = projector.project_motion(motion_data)
    window_md = _save_pickle(window_data, stage_tag)
    _record_media(window_md, stage_tag, is_final_stage=True)

    if projector._analysis.enabled and projector._analysis.log:
        projector._analysis.write(out_dir, stage_tag)

    return out_dir


def process_folder(
    projector_cache: dict,
    src_folder: str,
    tgt_folder: str,
    robot: str,
    gender: str,
    tgt_fps: int,
    device: str,
    record_video: bool,
    cfg: ProjectionConfig,
    save_stages: bool = False,
    compute_analysis: bool = False,
    headless: bool = True,
) -> None:
    default_object_model_path = cfg.object_model_path
    failures: list[str] = []
    n_clips = 0
    for root, _, files in os.walk(src_folder, followlinks=True):
        rel_root = os.path.relpath(root, src_folder)
        out_root = os.path.join(tgt_folder, rel_root)
        for fname in sorted(files):
            in_path = os.path.join(root, fname)
            if fname.endswith(".pth"):
                print(f"[SKIP] {in_path}: CARI4D runs one sequence at a time (--cari4d_sequence).")
                continue
            if not fname.endswith(".pt"):
                continue
            n_clips += 1
            stem, _ = os.path.splitext(fname)
            out_dir = os.path.join(out_root, stem)
            print(f"[INFO] Processing: {in_path}")
            try:
                object_model_path = resolve_object_model_path(
                    motion_path=in_path,
                    configured_object_model_path=default_object_model_path,
                    auto_from_motion_path=cfg.auto_object_from_path,
                )
                if object_model_path not in projector_cache:
                    print(f"[INFO] Using object model: {object_model_path}")
                    object_defaults = get_object_motion_defaults(object_model_path)
                    cfg_for_object = replace(cfg, object_model_path=object_model_path)
                    projector_cache[object_model_path] = PinocchioMotionProjectorWindow(
                        cfg=cfg_for_object,
                        min_object_height=object_defaults["min_object_height"],
                        contact_links=object_defaults["contact_links"],
                        compute_analysis=compute_analysis,
                    )
                process_file(
                    projector=projector_cache[object_model_path],
                    in_path=in_path,
                    out_dir=out_dir,
                    robot=robot,
                    gender=gender,
                    tgt_fps=tgt_fps,
                    device=device,
                    record_video=record_video,
                    object_model_path=object_model_path,
                    save_stages=save_stages,
                    headless=headless,
                )
            except Exception as exc:  # one bad clip must not abort the folder
                failures.append(f"{in_path}: {type(exc).__name__}: {exc}")
                print(f"[FAIL] {failures[-1]}")
    print(f"[INFO] Folder done: {n_clips - len(failures)}/{n_clips} clips solved.")
    if failures:
        print("\n".join(f"[FAIL] {f}" for f in failures))
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser, importable so stage-by-stage drivers reuse the same flags and defaults."""
    parser = argparse.ArgumentParser(
        description="Unified-window motion projection: a sequence of overlapping "
        "short-horizon NLPs that optimise the robot configuration "
        "trajectory (q) to track the IK reference and preserve "
        "robot-object contact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("--src_folder", type=str, default=None, help="Source directory of SMPL motion files")
    io_group.add_argument("--tgt_folder", type=str, default=None, help="Target directory for projected outputs")
    io_group.add_argument("--input_file", type=str, help="Single input SMPL motion file")
    io_group.add_argument(
        "--out_dir", type=str, help="Output directory for staged results (kinematic_window.pkl, contact_window.pkl)"
    )
    io_group.add_argument(
        "--cari4d_sequence",
        type=str,
        default="",
        help=(
            "CARI4D sequence stem (e.g. Date03_Sub01_gas_wild002). When set, resolves "
            "--input_file to <cari4d_opt_dir>/<STEM>.pth and --object_model_path to a "
            "staged per-sequence URDF under <cari4d_assets_root>/. Overrides --input_file "
            "and --object_model_path. Pass '' to disable and fall through to --input_file."
        ),
    )
    io_group.add_argument(
        "--cari4d_object_name",
        type=str,
        default="",
        help=("Optional shared object-asset name for a CARI4D sequence (e.g. 'demochair'). "),
    )
    io_group.add_argument(
        "--cari4d_opt_dir",
        type=str,
        default="data/CARI4D/opt/cari4d-release+step031397_demo-hy3d3-optv2",
        help="Directory holding CARI4D opt-stage .pth files.",
    )
    io_group.add_argument(
        "--cari4d_meshes_root",
        type=str,
        default="data/CARI4D/meshes",
        help="Root holding per-sequence Hunyuan3D mesh folders.",
    )
    io_group.add_argument(
        "--cari4d_assets_root",
        type=str,
        default="data/CARI4D/assets",
        help="Where to stage generated URDF + sample_points.npy for CARI4D sequences.",
    )
    io_group.add_argument("--tgt_fps", type=int, default=30, help="Target FPS for retargeting (OMOMO source is 30 fps)")
    io_group.add_argument(
        "--gender",
        type=str,
        default="auto",
        choices=["auto", "male", "female", "neutral"],
        help="Body-model gender. 'auto' (default) reads it from the subject code for "
        "sources that encode one (CARI4D), and falls back to neutral otherwise. "
        "CARI4D's fit is gendered, so the wrong template is a shape error.",
    )
    io_group.add_argument(
        "--robot",
        choices=list(SUPPORTED_ROBOTS),
        default="unitree_g1",
        help="Target robot. Selects its MJCF/URDF, IK config and link roles (see hoi_retarget/robots.py).",
    )
    io_group.add_argument(
        "--record_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record a MuJoCo video for each output. Use --no-record_video to skip "
        "and render later with hoi-retarget-render.",
    )
    io_group.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offscreen (EGL) rendering — no display needed. Use --no-headless for interactive viewer.",
    )
    _d = ProjectionConfig()
    io_group.add_argument(
        "--urdf_path",
        type=str,
        default=None,
        help="Robot URDF path. If omitted, inferred from --robot via ROBOT_URDF_DICT.",
    )
    io_group.add_argument(
        "--object_model_path",
        type=str,
        default=None,
        help="Object model (.urdf/.xml/.obj/.stl/.dae). Required unless it can be "
        "auto-inferred from the motion path (folder/filename object token).",
    )

    obj_group = parser.add_argument_group(
        "Object Scaling",
        "Rescale the object reference so the interaction fits the smaller robot. "
        "These are preprocessing knobs — the object pose is never optimised.",
    )
    obj_group.add_argument(
        "--object_scale",
        type=float,
        default=None,
        help="Object geometry scale. Default follows the robot's released corpus: the fit "
        "scale s_root (0.83) on unitree_g1, 1.0 on unitree_h2. 1.0 keeps the captured size and "
        "leaves the trajectory to --scale_object_trajectory; any other value resizes the object "
        "and always fits the trajectory by s_root.",
    )
    obj_group.add_argument(
        "--scale_object_trajectory",
        action=argparse.BooleanOptionalAction,
        default=_d.object.scale_trajectory,
        help="At --object_scale 1.0 (the unitree_h2 default): scale the object POSITION "
        "trajectory by s_root (XY about the origin, height about its per-frame lowest point so "
        "it never sinks). Any other object scale always fits the trajectory.",
    )
    obj_group.add_argument(
        "--object_traj_smooth_window",
        type=int,
        default=_d.object.traj_smooth_window,
        help="Savitzky-Golay window (frames, odd; <3 disables) that smooths the frozen, "
        "scaled object trajectory. Removes the grounded floor-following noise and the "
        "liftoff/set-down transition steps left by freezing the object (the removed "
        "object-optimisation NLP used to smooth these). Default 11.",
    )

    mode_group = parser.add_argument_group("Pipeline Mode", "Select which stage the projection runs.")
    mode_group.add_argument(
        "--mode",
        choices=["kinematic", "contact"],
        default="contact",
        help="kinematic: stop after Stage 1 (IK retargeting). contact: run the windowed q-only contact NLP (default).",
    )

    win_group = parser.add_argument_group("Window Parameters", "Sizing and stitching for the unified-window NLP.")
    win_group.add_argument("--window_size", type=int, default=_d.window.window_size, help="Frames per window NLP.")
    win_group.add_argument(
        "--overlap", type=int, default=_d.window.overlap, help="Overlap frames between consecutive windows."
    )
    diag_group = parser.add_argument_group("Diagnostics", "Debugging and analysis tools.")
    diag_group.add_argument(
        "--compute_analysis",
        action="store_true",
        default=False,
        help="Print and save NLP statistics per window: variables, constraints, cost terms, solve time.",
    )
    diag_group.add_argument(
        "--save_stages",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also record videos for intermediate (non-final) stages, e.g. the kinematic "
        "stage; the final-stage video and all stage pickles are written regardless.",
    )

    return parser


def main() -> None:
    """Entry point for the ``hoi-retarget`` console script."""
    parser = build_parser()
    args = parser.parse_args()

    # Fail before solving: a wheel install has no assets/ unless HOI_RETARGET_ASSETS is set.
    require_asset_root()
    if args.object_scale is not None and args.object_scale <= 0:
        parser.error("--object_scale must be positive")

    # ── CARI4D convenience resolver ──────────────────────────────────────
    # Derive --input_file / --object_model_path / --out_dir from the sequence stem.
    if args.cari4d_sequence:
        stem = args.cari4d_sequence
        pth = os.path.join(args.cari4d_opt_dir, f"{stem}.pth")
        if not os.path.isfile(pth):
            parser.error(f"--cari4d_sequence {stem!r}: no .pth at {pth}")
        if args.src_folder:
            parser.error(
                "--cari4d_sequence is mutually exclusive with --src_folder. Pass --cari4d_sequence '' to disable."
            )

        # Stage URDF + sample_points.npy (idempotent).
        from hoi_retarget.datasets.cari4d_assets import prepare_cari4d_object_assets

        try:
            urdf_path = prepare_cari4d_object_assets(
                stem=stem,
                assets_root=args.cari4d_assets_root,
                meshes_root=args.cari4d_meshes_root,
                object_name=(args.cari4d_object_name or None),
            )
        except FileNotFoundError as exc:
            parser.error(f"--cari4d_sequence {stem!r}: {exc}")

        args.input_file = pth
        args.object_model_path = urdf_path
        if args.out_dir is None:
            args.out_dir = os.path.join("outputs", "cari4d", stem)
        print(f"[INFO] CARI4D: input={pth}")
        print(f"[INFO] CARI4D: urdf ={urdf_path}")
        print(f"[INFO] CARI4D: out  ={args.out_dir}")

    # Must run before ProjectionConfig.from_args and before any clip is loaded; both read the profile.
    activate_robot(args.robot)

    if args.urdf_path is None:
        urdf_candidate = ROBOT_URDF_DICT.get(args.robot)
        if urdf_candidate is None:
            parser.error(
                f"No URDF mapping for robot '{args.robot}'. "
                "Pass --urdf_path explicitly or add an entry to ROBOT_URDF_DICT."
            )
        args.urdf_path = str(urdf_candidate)

    cfg = ProjectionConfig.from_args(args)

    projector_cache: dict[str, PinocchioMotionProjectorWindow] = {}

    if args.src_folder and args.tgt_folder:
        process_folder(
            projector_cache=projector_cache,
            src_folder=args.src_folder,
            tgt_folder=args.tgt_folder,
            robot=args.robot,
            gender=args.gender,
            tgt_fps=args.tgt_fps,
            device="cuda" if torch.cuda.is_available() else "cpu",
            record_video=args.record_video,
            cfg=cfg,
            save_stages=args.save_stages,
            compute_analysis=args.compute_analysis,
            headless=args.headless,
        )
    elif args.input_file:
        if not os.path.isfile(args.input_file):
            parser.error(f"--input_file not found: {args.input_file}")
        object_model_path = resolve_object_model_path(
            motion_path=args.input_file,
            configured_object_model_path=args.object_model_path,
            auto_from_motion_path=cfg.auto_object_from_path,
        )
        if object_model_path not in projector_cache:
            print(f"[INFO] Using object model: {object_model_path}")
            object_defaults = get_object_motion_defaults(object_model_path)
            cfg_for_object = replace(cfg, object_model_path=object_model_path)
            projector_cache[object_model_path] = PinocchioMotionProjectorWindow(
                cfg=cfg_for_object,
                min_object_height=object_defaults["min_object_height"],
                contact_links=object_defaults["contact_links"],
                compute_analysis=args.compute_analysis,
            )
        projector = projector_cache[object_model_path]

        out_dir = args.out_dir
        if out_dir is None:
            out_dir = os.path.splitext(args.input_file)[0] + "_projected"

        process_file(
            projector=projector,
            in_path=args.input_file,
            out_dir=out_dir,
            robot=args.robot,
            gender=args.gender,
            tgt_fps=args.tgt_fps,
            device="cuda" if torch.cuda.is_available() else "cpu",
            record_video=args.record_video,
            object_model_path=object_model_path,
            save_stages=args.save_stages,
            headless=args.headless,
        )
    elif args.src_folder:
        parser.error("--src_folder needs --tgt_folder")
    else:
        parser.error("no input: pass --input_file, --src_folder/--tgt_folder, or --cari4d_sequence")


if __name__ == "__main__":
    main()
