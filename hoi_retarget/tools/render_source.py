# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the SMPL-X source of an OMOMO clip in the robot renderer's style.

    hoi-retarget-render-source --clip sub6_whitechair_036 --pair_pkl <clip>/contact_window.pkl
    hoi-retarget-render-source --clips_file stems.txt --pair_dir output/omomo_all_4421

``--pair_pkl`` takes the clip's ``object_scale`` and shares the robot render's
camera, so side-by-side panels match in size and horizon.
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
from hoi_retarget.rendering.smpl_video import (
    GROUND_MODES,
    TONE_NAMES,
    SmplSourceRenderer,
    object_name_from_stem,
)
from hoi_retarget.rendering.video import solve_camera
from hoi_retarget.robots import SUPPORTED_ROBOTS


def _pair_pkl_for(stem: str, pair_dir: Path) -> Path:
    return pair_dir / object_name_from_stem(stem) / stem / "contact_window.pkl"


def _default_video_path(stem: str, out_dir) -> Path:
    root = Path(out_dir) if out_dir else Path("videos") / "smpl_source"
    return root / object_name_from_stem(stem) / f"{stem}_smplx.mp4"


def render_one(stem, video_path, args, pair_pkl=None):
    subject_scale, camera = args.scale, None
    if pair_pkl is not None and Path(pair_pkl).exists():
        md = pickle.load(open(pair_pkl, "rb"))
        if subject_scale is None:
            subject_scale = float(md.get("object_scale", 1.0))
        if not args.no_share_camera:
            # Frame the human with the robot's camera: identical framing, and
            # therefore an identical horizon, in both panels.
            camera = solve_camera(
                md, args.robot, args.style, md.get("object_model_path"), float(md.get("object_scale", 1.0))
            )
            if camera is None:
                print(f"[INFO] {args.style} uses a fixed camera; both panels already share it.")
    if subject_scale is None:
        subject_scale = 1.0

    name = args.style if isinstance(args.style, str) else args.style.name
    print(
        f"[INFO] {stem} -> {video_path}  (style={name}, tone={args.tone}, "
        f"scale={subject_scale:g}, ground={args.ground}, "
        f"camera={'shared' if camera is not None else 'own'})"
    )
    r = SmplSourceRenderer(style=args.style, tone=args.tone, subject_scale=subject_scale, ground_align=args.ground)
    clip = r.load_clip(stem)
    if args.single_frame is None:
        r.render_video(clip, video_path, camera=camera)
        return
    import imageio.v2 as imageio

    t = min(int(args.single_frame), clip["T"] - 1)
    _t, img = next(iter(r.render_frames(clip, [t], camera=camera)))
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(video_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render OMOMO SMPL-X source clips in the G1 recorder's look.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_argument_group("Input")
    which = src.add_mutually_exclusive_group(required=True)
    which.add_argument("--clip", help="One clip stem, e.g. sub6_whitechair_036.")
    which.add_argument("--clips_file", help="File of clip stems, one per line.")

    out = parser.add_argument_group("Output")
    out.add_argument(
        "--video_path",
        default=None,
        help="Output path (single-clip mode). Default: videos/smpl_source/<object>/<stem>_smplx.mp4",
    )
    out.add_argument("--video_dir", default=None, help="Output root for list mode.")
    out.add_argument("--single_frame", type=int, default=None, help="Write this frame as a PNG instead of an mp4.")

    ren = parser.add_argument_group("Look")
    ren.add_argument(
        "--style",
        choices=STYLE_NAMES,
        default=DEFAULT_STYLE,
        help=f"Render style preset (default: {DEFAULT_STYLE}). Use the same one as the robot panel.",
    )
    ren.add_argument("--tone", choices=TONE_NAMES, default="slate", help="Body colour (default: slate).")
    ren.add_argument(
        "--size",
        type=int,
        default=None,
        help="Override the output width in px (square presets become SIZE x SIZE). --size 4096 for a hero figure.",
    )
    ren.add_argument("--height", type=int, default=None)
    ren.add_argument("--supersample", type=int, default=None)
    ren.add_argument(
        "--ground",
        choices=GROUND_MODES,
        default="clip",
        help="Re-datum the source onto the floor. The OMOMO fits "
        "float or sink by up to ~7 cm per clip while the "
        "retargeted robot does not (default: clip).",
    )

    pair = parser.add_argument_group("Pairing with the robot render")
    pair.add_argument(
        "--pair_pkl",
        default=None,
        help="The clip's contact_window.pkl. Takes object_scale "
        "from it and shares its camera, so the two panels "
        "match in size and horizon.",
    )
    pair.add_argument("--pair_dir", default=None, help="Dataset root to find each stem's pkl in (list mode).")
    pair.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Shrink the human and its object by this factor about "
        "the floor. Default: the pkl's object_scale, else 1.0.",
    )
    pair.add_argument(
        "--no_share_camera",
        action="store_true",
        help="Frame the human on its own content instead of borrowing the robot's camera.",
    )
    pair.add_argument(
        "--robot",
        choices=list(SUPPORTED_ROBOTS),
        default="unitree_g1",
        help="Robot the pkl was retargeted to (for the camera).",
    )
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True, help="Offscreen (EGL) rendering."
    )

    args = parser.parse_args()
    require_asset_root()
    if args.pair_pkl is not None and not Path(args.pair_pkl).exists():
        parser.error(f"--pair_pkl not found: {args.pair_pkl}")
    if args.size or args.height or args.supersample:
        base = render_style.get_style(args.style)
        args.style = render_style.at_size(base, args.size or base.width, args.height, args.supersample)
        print(f"[INFO] {base.name} at {args.style.width}x{args.style.height}, {args.style.supersample}x supersample")
    suffix = ".png" if args.single_frame is not None else ".mp4"

    if args.clip:
        path = (
            Path(args.video_path)
            if args.video_path
            else _default_video_path(args.clip, args.video_dir).with_suffix(suffix)
        )
        pair_pkl = args.pair_pkl
        if pair_pkl is None and args.pair_dir:
            pair_pkl = _pair_pkl_for(args.clip, Path(args.pair_dir))
        render_one(args.clip, path, args, pair_pkl)
        return

    stems = [s.strip() for s in Path(args.clips_file).read_text().splitlines() if s.strip()]
    if not stems:
        parser.error(f"no clip stems in {args.clips_file}")
    failed = False
    for stem in stems:
        pair_pkl = _pair_pkl_for(stem, Path(args.pair_dir)) if args.pair_dir else None
        path = _default_video_path(stem, args.video_dir).with_suffix(suffix)
        try:
            render_one(stem, path, args, pair_pkl)
        except Exception as e:
            print(f"[FAIL] {stem}: {type(e).__name__}: {e}", flush=True)
            failed = True
    if failed:
        _sys.exit(1)


if __name__ == "__main__":
    main()
