# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a stage pickle in the browser with viser: ``hoi-retarget-view <pkl>``.

Playback controls (play, speed, frame) and per-link contact markers: orange (hands)
or cyan (feet) while in contact, with a wireframe sphere at the contact target.
"""

import argparse
import numpy as np
import os
import pickle
import sys
import time
from pathlib import Path

from hoi_retarget.gmr.params import ROBOT_URDF_DICT
from hoi_retarget.paths import require_asset_root
from hoi_retarget.rendering.video import forward_kinematics
from hoi_retarget.rendering.viser_viewer import ViserMotionViewer
from hoi_retarget.retargeting.source_data import compute_object_contact_points_world_by_frame
from hoi_retarget.robots import robot_for_pkl


def _running_over_ssh() -> bool:
    return "SSH_CONNECTION" in os.environ or "SSH_CLIENT" in os.environ


def _print_url_box(url: str, host: str, port: int) -> None:
    bar = "─" * 72
    print(bar)
    print(f"  [inspect_viser] Serving at  {url}")
    if _running_over_ssh() and host != "0.0.0.0":
        print("  [inspect_viser] Over SSH? On the client machine run:")
        print(f"                    ssh -L {port}:localhost:{port} <this-host>")
        print(f"                  then open  http://localhost:{port}/  in the client browser.")
    print("  [inspect_viser] Press Ctrl+C to stop.")
    print(bar)


def _load_frames(motion_data: dict) -> dict:
    return {
        "root_pos": np.asarray(motion_data["root_pos"]),
        "root_rot": np.asarray(motion_data["root_rot"]),
        "dof_pos": np.asarray(motion_data["dof_pos"]),
        "obj_pos": np.asarray(motion_data["object_pos"]) if "object_pos" in motion_data else None,
        "obj_rot": np.asarray(motion_data["object_rot"]) if "object_rot" in motion_data else None,
    }


def _load_contact(motion_data: dict, robot: str, num_frames: int) -> dict:
    """Per-frame contact marker entries for the viewer (empty when the clip has no contact links)."""
    contact_link_names = list(motion_data.get("contact_link_names", []))
    if not contact_link_names:
        return {"names": [], "per_frame": []}

    world_body_pos, _orient, _base_idx, body_names = forward_kinematics(motion_data, robot)
    body_name_to_idx = {name: i for i, name in enumerate(body_names)}
    points_by_frame = compute_object_contact_points_world_by_frame(motion_data, num_frames)
    per_link_flags = motion_data.get("per_link_contact_flags")
    if per_link_flags is not None:
        per_link_flags = np.asarray(per_link_flags)

    per_frame = []
    for t in range(num_frames):
        frame_cp = points_by_frame[t] if t < len(points_by_frame) else {}
        entries = []
        for c, link_name in enumerate(contact_link_names):
            idx = body_name_to_idx.get(link_name)
            if idx is None:
                continue
            if per_link_flags is not None and t < per_link_flags.shape[0] and c < per_link_flags.shape[1]:
                active = bool(per_link_flags[t, c])
            else:
                active = link_name in frame_cp
            entries.append(
                {
                    "link_name": link_name,
                    "link_pos": world_body_pos[t, idx],
                    "contact_point_world": frame_cp.get(link_name),
                    "active": active,
                }
            )
        per_frame.append(entries)
    return {"names": contact_link_names, "per_frame": per_frame}


def _step_frame(viewer: ViserMotionViewer, ctx: dict, t: int) -> None:
    object_data = None
    if ctx["obj_pos"] is not None and ctx["obj_rot"] is not None:
        object_data = (ctx["obj_pos"][t], ctx["obj_rot"][t])
    contact = ctx["contact"]["per_frame"]
    viewer.step(
        root_pos=ctx["root_pos"][t],
        root_rot=ctx["root_rot"][t],
        dof_pos=ctx["dof_pos"][t],
        object_data=object_data,
        per_link_contact_data=contact[t] if t < len(contact) else None,
        rate_limit=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Live-inspect a motion pickle in a browser (robot + object replay).")
    ap.add_argument(
        "motion_pkl",
        type=str,
        help="Path to motion pickle (kinematic_window.pkl / contact_window.pkl).",
    )
    ap.add_argument("--robot", type=str, default=None, help="Robot to draw. Default: the one the pickle records.")
    ap.add_argument(
        "--urdf-path", type=str, default=None, help="Override robot URDF path (defaults to ROBOT_URDF_DICT[--robot])."
    )
    ap.add_argument(
        "--object-model-path",
        type=str,
        default=None,
        help="Override object URDF path (defaults to the path embedded in the pickle under 'object_model_path').",
    )
    ap.add_argument("--host", type=str, default="127.0.0.1", help="Bind address. Use 0.0.0.0 to expose on your LAN.")
    ap.add_argument("--port", type=int, default=0, help="Bind port. Default 0 = auto-pick a free port.")
    ap.add_argument("--fps", type=float, default=None, help="Override playback fps (defaults to pickle 'fps').")
    ap.add_argument(
        "--no-contact",
        action="store_true",
        help="Skip the contact markers. They cost one forward-kinematics pass over the clip at startup.",
    )
    args = ap.parse_args()
    require_asset_root()

    pkl_path = Path(args.motion_pkl).expanduser()
    if not pkl_path.is_file():
        sys.exit(f"[inspect_viser] Not a file: {pkl_path}")
    with open(pkl_path, "rb") as f:
        try:
            motion_data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            sys.exit(f"[inspect_viser] Not a pickle: {pkl_path}")

    try:
        robot = robot_for_pkl(motion_data, args.robot)
    except ValueError as exc:
        sys.exit(f"[inspect_viser] {exc}")
    urdf_path = args.urdf_path or str(ROBOT_URDF_DICT[robot])
    object_model_path = args.object_model_path or motion_data.get("object_model_path")
    if object_model_path is None:
        sys.exit("[inspect_viser] No object_model_path in pickle and --object-model-path not set.")

    ctx = _load_frames(motion_data)
    fps = float(args.fps if args.fps is not None else motion_data["fps"])
    dof_names = list(motion_data.get("dof_names", []))
    num_frames = int(ctx["root_pos"].shape[0])
    ctx["contact"] = (
        {"names": [], "per_frame": []} if args.no_contact else _load_contact(motion_data, robot, num_frames)
    )

    viewer = ViserMotionViewer(
        robot_type=robot,
        urdf_path=urdf_path,
        object_model_path=object_model_path,
        dof_names=dof_names,
        motion_fps=fps,
        object_scale=float(motion_data.get("object_scale") or 1.0),
        contact_link_names=ctx["contact"]["names"],
        host=args.host,
        port=args.port,
    )

    with viewer.server.gui.add_folder("Playback"):
        playing_handle = viewer.server.gui.add_checkbox("Playing", initial_value=True)
        speed_handle = viewer.server.gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
        frame_handle = viewer.server.gui.add_slider("Frame", min=0, max=max(0, num_frames - 1), step=1, initial_value=0)

    contact_readout = None
    if ctx["contact"]["names"]:
        with viewer.server.gui.add_folder("Contact"):
            show_contact = viewer.server.gui.add_checkbox("Show markers", initial_value=True)
            show_contact.on_update(lambda _: viewer.set_contact_visible(show_contact.value))
            contact_readout = viewer.server.gui.add_text("In contact", initial_value="—", disabled=True)
        active_per_frame = [
            ", ".join(e["link_name"].replace("_link", "").replace("_", " ") for e in entries if e["active"]) or "—"
            for entries in ctx["contact"]["per_frame"]
        ]
        frac = sum(1 for a in active_per_frame if a != "—") / max(1, len(active_per_frame))
        print(
            f"  [inspect_viser] Contact on {frac * 100:.0f}% of frames "
            f"({len(ctx['contact']['names'])} links: {', '.join(ctx['contact']['names'])})"
        )

    _print_url_box(viewer.get_url(), args.host, viewer.get_port())

    base_dt = 1.0 / fps
    accumulator = 0.0  # fractional frame cursor
    last_written_frame = -1  # last int(frame) the server wrote back
    last_tick = time.monotonic()

    try:
        while True:
            tick_start = time.monotonic()
            # Cap dt so a stall does not jump the cursor.
            dt = min(tick_start - last_tick, 0.5)
            last_tick = tick_start

            current_slider = int(frame_handle.value)
            if current_slider != last_written_frame:
                accumulator = float(current_slider)

            if playing_handle.value and num_frames > 1:
                accumulator = (accumulator + speed_handle.value * dt * fps) % num_frames

            t = int(accumulator) % max(1, num_frames)
            _step_frame(viewer, ctx, t)
            if t != last_written_frame:
                frame_handle.value = t
                if contact_readout is not None and t < len(active_per_frame):
                    contact_readout.value = active_per_frame[t]
                last_written_frame = t

            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, base_dt - elapsed))
    except KeyboardInterrupt:
        print("\n[inspect_viser] Shutting down.")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
