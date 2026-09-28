# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render MP4s from saved stage pickles (``kinematic_window.pkl``, ``contact_window.pkl``).

Uses the same ``record_motion_video`` as ``hoi-retarget --record_video``.

    hoi-retarget-render --input_pkl path/to/contact_window.pkl
    hoi-retarget-render --input_dir output/ --stages contact_window --size 4096

``--style`` picks a preset from :mod:`hoi_retarget.rendering.render_style` (default
``square``); ``--size`` re-renders it at another resolution.
"""

import os
import sys as _sys

if "--headless" in _sys.argv or "--no-headless" not in _sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pickle
from pathlib import Path

from hoi_retarget.paths import require_asset_root
from hoi_retarget.rendering import render_style
from hoi_retarget.rendering.render_style import DEFAULT_STYLE, STYLE_NAMES
from hoi_retarget.rendering.video import record_motion_video
from hoi_retarget.robots import SUPPORTED_ROBOTS, activate_robot, robot_for_pkl


def _default_video_path(pkl_path: Path, robot: str) -> Path:
    """Replace a ``data`` path segment with ``videos``, else write next to the pickle."""
    parts = pkl_path.parts
    try:
        idx = parts.index("data")
        new_parts = parts[:idx] + ("videos",) + parts[idx + 1 :]
        out_dir = Path(*new_parts).parent
    except ValueError:
        out_dir = pkl_path.parent
    return out_dir / f"{pkl_path.stem}_{robot}.mp4"


def _load_motion_data(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _resolve_object_model_path(motion_data: dict, cli_override: str | None) -> str | None:
    if cli_override:
        return cli_override
    return motion_data.get("object_model_path")


def _iter_pkls(input_dir: Path, stages: set | None) -> list:
    out = []
    for p in sorted(input_dir.rglob("*.pkl")):
        if stages is not None and p.stem not in stages:
            continue
        out.append(p)
    return out


def record_one(
    pkl_path: Path,
    video_path: Path,
    robot: str,
    object_model_path_override: str | None,
    headless: bool,
    style: str = DEFAULT_STYLE,
) -> None:
    motion_data = _load_motion_data(pkl_path)
    activate_robot(robot)
    object_model_path = _resolve_object_model_path(motion_data, object_model_path_override)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    name = style if isinstance(style, str) else style.name
    print(f"[INFO] Recording {pkl_path} -> {video_path}  (style={name})")
    record_motion_video(
        motion_data=motion_data,
        robot=robot,
        video_path=str(video_path),
        object_model_path=object_model_path,
        headless=headless,
        object_scale=float(motion_data.get("object_scale", 1.0)),
        style=style,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render MP4 videos from saved motion-projection pickles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src_group = parser.add_argument_group("Input")
    src = src_group.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_pkl", type=str, help="Path to a single motion pickle.")
    src.add_argument("--input_dir", type=str, help="Directory; every *.pkl inside (recursively) is rendered.")
    src_group.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated pkl stems to record (e.g. 'kinematic_window,contact_window'). Applies to --input_dir.",
    )

    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Output mp4 path (single-pickle mode). Default: sibling of the pkl.",
    )
    out_group.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Output directory (folder mode). Default: mirror 'data/'→'videos/' in each pkl's path.",
    )

    ren_group = parser.add_argument_group("Rendering")
    ren_group.add_argument(
        "--robot",
        choices=list(SUPPORTED_ROBOTS),
        default=None,
        help="Robot to render. Default: the one each pickle records.",
    )
    ren_group.add_argument(
        "--style",
        choices=STYLE_NAMES,
        default=DEFAULT_STYLE,
        help=f"Render style preset (default: {DEFAULT_STYLE}). "
        "'studio' is the same look ~4x faster (no supersampling); "
        "'legacy' is MuJoCo defaults at 640x480.",
    )
    ren_group.add_argument(
        "--size",
        type=int,
        default=None,
        help="Override the output width in px (square presets "
        "become SIZE x SIZE). Use --size 4096 for a hero "
        "figure; supersampling is dropped above 1440 px.",
    )
    ren_group.add_argument(
        "--height", type=int, default=None, help="Override the output height. Default: same as --size."
    )
    ren_group.add_argument("--supersample", type=int, default=None, help="Override the supersample factor.")
    ren_group.add_argument(
        "--object_model_path",
        type=str,
        default=None,
        help="Override the object model. If omitted, uses the pickle's 'object_model_path' key.",
    )
    ren_group.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offscreen (EGL) rendering. Use --no-headless for interactive viewer.",
    )

    args = parser.parse_args()
    require_asset_root()

    style = args.style
    if args.size or args.height or args.supersample:
        base = render_style.get_style(style)
        style = render_style.at_size(base, args.size or base.width, args.height, args.supersample)
        print(f"[INFO] {base.name} at {style.width}x{style.height}, {style.supersample}x supersample")

    stages_filter = None
    if args.stages:
        stages_filter = {s.strip() for s in args.stages.split(",") if s.strip()}

    if args.input_pkl:
        pkl_path = Path(args.input_pkl).resolve()
        if not pkl_path.exists():
            parser.error(f"--input_pkl does not exist: {pkl_path}")
        try:
            robot = robot_for_pkl(_load_motion_data(pkl_path), args.robot)
        except ValueError as exc:
            parser.error(str(exc))
        vpath = Path(args.video_path) if args.video_path else _default_video_path(pkl_path, robot)
        record_one(
            pkl_path=pkl_path,
            video_path=vpath,
            robot=robot,
            object_model_path_override=args.object_model_path,
            headless=args.headless,
            style=style,
        )
        return

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        parser.error(f"--input_dir is not a directory: {input_dir}")
    pkls = _iter_pkls(input_dir, stages_filter)
    if not pkls:
        print(f"[WARN] No pickles matched under {input_dir} (stages filter={stages_filter}).")
        return

    video_dir = Path(args.video_dir).resolve() if args.video_dir else None
    for pkl_path in pkls:
        robot = robot_for_pkl(_load_motion_data(pkl_path), args.robot)
        if video_dir is not None:
            rel = pkl_path.relative_to(input_dir)
            vpath = video_dir / rel.parent / f"{pkl_path.stem}_{robot}.mp4"
        else:
            vpath = _default_video_path(pkl_path, robot)
        record_one(
            pkl_path=pkl_path,
            video_path=vpath,
            robot=robot,
            object_model_path_override=args.object_model_path,
            headless=args.headless,
            style=style,
        )


if __name__ == "__main__":
    main()
