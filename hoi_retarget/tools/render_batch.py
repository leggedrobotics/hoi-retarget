# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parallel batch renderer for stage pickles.

Imports the render stack once per worker process and renders each pickle in a
``try/except``; failures go to ``--fail_log`` and the run exits 1 if any failed.
Each mp4 is named ``<stem><--name_suffix>.mp4`` beside its pickle (or mirrored under
``--video_dir``); existing mp4s are skipped unless ``--force``. Always headless.

    hoi-retarget-render-batch --input_dir output/run --stages contact_window --workers 6

The default ``--workers`` is sized per style for an 8 GB GPU.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

from hoi_retarget.paths import REPO_ROOT, require_asset_root, resolve_asset_path
from hoi_retarget.rendering.render_style import (
    DEFAULT_STYLE,
    STYLE_NAMES,
    get_style,
)
from hoi_retarget.robots import SUPPORTED_ROBOTS, activate_robot, robot_for_pkl

# Workers chdir here so other relative paths a pickle carries resolve.
_REPO_ROOT = str(REPO_ROOT)

# Per-worker state, populated by the pool initializer.
_G = {}


def _init_worker(robot: str, object_model_path_override: str | None, quiet: bool, style: str | None = None):
    os.environ["MUJOCO_GL"] = "egl"
    os.chdir(_REPO_ROOT)

    from hoi_retarget.rendering.render_style import DEFAULT_STYLE
    from hoi_retarget.rendering.video import record_motion_video

    if quiet:
        # Failures still come back through the return value.
        sys.stdout = open(os.devnull, "w")

    _G.update(
        record=record_motion_video,
        robot=robot,
        override=object_model_path_override,
        style=style or DEFAULT_STYLE,
    )


def _render_one(task):
    pkl_path, video_path = task
    t0 = time.time()
    try:
        with open(pkl_path, "rb") as f:
            motion_data = pickle.load(f)
        robot_for_pkl(motion_data, _G["robot"])  # a mismatch fails this clip with a clear message
        object_model_path = _G["override"] or motion_data.get("object_model_path")
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        _G["record"](
            motion_data=motion_data,
            robot=_G["robot"],
            video_path=str(video_path),
            object_model_path=object_model_path,
            headless=True,
            object_scale=float(motion_data.get("object_scale", 1.0)),
            style=_G["style"],
        )
        if not os.path.exists(video_path):
            return (str(pkl_path), False, "no mp4 written", time.time() - t0)
        return (str(pkl_path), True, "", time.time() - t0)
    except Exception as e:
        return (str(pkl_path), False, repr(e)[:300], time.time() - t0)


def _video_path_for(pkl_path: Path, input_dir: Path, video_dir: Path | None, suffix: str) -> Path:
    name = f"{pkl_path.stem}{suffix}.mp4"
    if video_dir is not None:
        rel = pkl_path.relative_to(input_dir)
        return video_dir / rel.parent / name
    return pkl_path.parent / name


def default_workers(style) -> int:
    """Workers that fit an 8 GB card at this style's framebuffer size."""
    if not style.styled:
        return 6
    return {1: 4, 2: 2}.get(style.supersample, 1)


def build_tasks(input_dir, video_dir, stages, suffix, force):
    tasks, n_skip = [], 0
    for pkl_path in sorted(input_dir.rglob("*.pkl")):
        if stages is not None and pkl_path.stem not in stages:
            continue
        vpath = _video_path_for(pkl_path, input_dir, video_dir, suffix)
        if vpath.exists() and not force:
            n_skip += 1
            continue
        tasks.append((pkl_path, vpath))
    return tasks, n_skip


def main():
    ap = argparse.ArgumentParser(
        description="Parallel, fault-tolerant batch renderer for motion pickles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input_dir", required=True, help="Directory; every matching *.pkl inside (recursively) is rendered."
    )
    ap.add_argument(
        "--stages",
        default=None,
        help="Comma-separated pkl stems to render (e.g. 'kinematic_window,contact_window'). Default: all *.pkl.",
    )
    ap.add_argument(
        "--video_dir",
        default=None,
        help="Mirror the input tree under this dir. Default: write each mp4 next to its pickle.",
    )
    ap.add_argument(
        "--name_suffix",
        default="",
        help="Suffix before '.mp4' (default: none -> exact <stem>.mp4). "
        "Use e.g. '_unitree_g1' to match hoi-retarget-render.",
    )
    ap.add_argument(
        "--robot",
        default="unitree_g1",
        choices=list(SUPPORTED_ROBOTS),
        help="Robot to render (must match the pickle's retarget target).",
    )
    ap.add_argument(
        "--style",
        choices=STYLE_NAMES,
        default=DEFAULT_STYLE,
        help=f"Render style preset (default: {DEFAULT_STYLE}). "
        "'studio' is the same look ~4x faster and is the right "
        "choice for bulk passes; 'legacy' is MuJoCo defaults at 640x480.",
    )
    ap.add_argument("--object_model_path", default=None, help="Override the object model for all pickles.")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes. Default depends on --style "
        "(2 supersampled / 4 studio / 6 legacy), sized for an 8 GB card.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Cap #renders (smoke test).")
    ap.add_argument("--force", action="store_true", help="Re-render existing mp4s.")
    ap.add_argument("--fail_log", default=None, help="Failure record path. Default: <input_dir>/render_failures.txt")
    ap.add_argument("--verbose", action="store_true", help="Let workers print record_motion_video's per-clip logs.")
    args = ap.parse_args()
    require_asset_root()
    # Activate before rendering so robot-specific tables match --robot.
    activate_robot(args.robot)

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        ap.error(f"--input_dir is not a directory: {input_dir}")
    style = get_style(args.style)
    workers = args.workers if args.workers else default_workers(style)
    video_dir = Path(args.video_dir).resolve() if args.video_dir else None
    stages = {s.strip() for s in args.stages.split(",") if s.strip()} if args.stages else None
    fail_log = Path(args.fail_log) if args.fail_log else input_dir / "render_failures.txt"

    tasks, n_skip = build_tasks(input_dir, video_dir, stages, args.name_suffix, args.force)
    if stages is not None and not tasks and not n_skip:
        ap.error(f"--stages {sorted(stages)} matched no files under {input_dir}")
    if args.limit:
        tasks = tasks[: args.limit]
    n = len(tasks)

    print(f"[batch] input_dir={input_dir}")
    print(
        f"[batch] stages={sorted(stages) if stages else 'ALL'} robot={args.robot} "
        f"workers={workers} suffix='{args.name_suffix}'"
    )
    print(f"[batch] style={style.name} ({style.width}x{style.height}, {style.supersample}x supersample)")
    print(f"[batch] tasks={n}  skipped(existing mp4)={n_skip}  force={args.force}")
    if n == 0:
        print("[batch] nothing to do.")
        return

    t0 = time.time()
    ok = fail = done = 0
    failures = []
    ctx = mp.get_context("spawn")
    object_model_path = str(resolve_asset_path(args.object_model_path)) if args.object_model_path else None
    init_args = (args.robot, object_model_path, not args.verbose, args.style)
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=init_args) as pool:
        for pkl_path, success, msg, _dt in pool.imap_unordered(_render_one, tasks, chunksize=1):
            done += 1
            if success:
                ok += 1
            else:
                fail += 1
                failures.append((pkl_path, msg))
            if done % 25 == 0 or done == n:
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                print(
                    f"[batch] {done}/{n}  ok={ok} fail={fail}  {rate:.2f}/s  "
                    f"elapsed={el / 60:.1f}m  eta={eta / 60:.1f}m",
                    flush=True,
                )

    el = time.time() - t0
    print(f"[batch] DONE  ok={ok} fail={fail}  total={n}  in {el / 60:.1f}m")
    if failures:
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_log, "w") as f:
            f.write(f"# {fail} render failures of {n} tasks\n")
            f.write("# columns: <pkl_path>\\t<error>\n")
            for p, m in failures:
                f.write(f"{p}\t{m}\n")
        print(f"[batch] failures recorded in {fail_log}")
        for p, m in failures[:20]:
            print(f"    FAIL {p}  {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
