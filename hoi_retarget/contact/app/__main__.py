# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI entry for the contact-point editor.

    hoi-retarget-edit-contacts <source.pt|.pth> [--object_model_path URDF]

Opens a viser web editor; **Go** runs the contact-mode retarget from the edited
points into ``outputs/contact_editor/<stem>/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime

from hoi_retarget.contact.app.session import (
    build_session,
    detect_source_kind,
)
from hoi_retarget.contact.app.viewer import ContactShifterViewer
from hoi_retarget.contact.shifts import ShiftSet
from hoi_retarget.datasets.cari4d import infer_cari4d_gender
from hoi_retarget.paths import CONTACT_EDITOR_OUTPUT, DATA_ROOT, REPO_ROOT, require_asset_root
from hoi_retarget.retargeting.object_model import resolve_object_model_path
from hoi_retarget.robots import SUPPORTED_ROBOTS

# CARI4D chair clip token -> staged asset key.
_CARI4D_CHAIR = {
    "ChairPush": "demochair_push",
    "ChairPull": "demochair_pull",
    "ChairLiftBack": "demochair_liftback",
    "ChairLiftFront": "demochair_liftfront",
    "ChairLiftLeg": "demochair_liftleg",
    "ChairRotate": "demochair_rotate",
}


def _cari4d_clean_name(stem: str):
    """Staged asset key for a CARI4D clip stem (or None if unmapped)."""
    tok = re.sub(r"^Date\d+_Sub\d+_", "", stem).replace("-wild", "")
    if tok in _CARI4D_CHAIR:
        return _CARI4D_CHAIR[tok]
    if tok == "RubberDuck":
        return "rubberduck"
    if tok.startswith("OrangeTable"):
        return "orangetable" + tok[len("OrangeTable") :]
    if tok.startswith("Wok"):
        return "wok" + tok[len("Wok") :]
    return None


def _cari4d_object_candidates(stem: str):
    """Staged-asset keys to try for a CARI4D stem, most specific first."""
    cands = []
    clean = _cari4d_clean_name(stem)
    if clean:
        cands.append(clean)
    cands.append(stem)
    parts = stem.split("_")
    if len(parts) >= 3:
        token = parts[2].split("-")[0]
        cands += [token, token.lower()]
    cands.append(stem.lower())
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _cari4d_from_clip_map(stem: str):
    """Object URDF for this stem from any ``experiments/*/clip_map.json``, or None."""
    for cm in glob.glob(os.path.join(REPO_ROOT, "experiments", "*", "clip_map.json")):
        try:
            with open(cm) as f:
                data = json.load(f)
        except Exception:
            continue
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            seq = entry.get("seq", "")
            urdf = entry.get("urdf", "")
            if (
                (seq == stem or os.path.basename(entry.get("pth", "")) == stem + ".pth")
                and urdf
                and os.path.isfile(urdf)
            ):
                return urdf
    return None


def resolve_object_model(source_path: str, override) -> str:
    if override:
        return override
    kind = detect_source_kind(source_path)
    if kind == "intermimic":
        return resolve_object_model_path(
            motion_path=source_path, configured_object_model_path=None, auto_from_motion_path=True
        )
    # CARI4D: staged URDF under data/CARI4D/assets/, else a clip_map.json entry.
    stem = os.path.splitext(os.path.basename(source_path))[0]
    assets = os.path.join(str(DATA_ROOT), "CARI4D", "assets")
    for key in _cari4d_object_candidates(stem):
        cand = os.path.join(assets, f"{key}.urdf")
        if os.path.isfile(cand):
            return cand
    via_map = _cari4d_from_clip_map(stem)
    if via_map:
        return via_map
    raise FileNotFoundError(
        f"Could not resolve a CARI4D object URDF for '{stem}'. The clip's object "
        f"may not be staged under {assets} (CARI4D clips are staged by object "
        f"name, not clip token — e.g. Chair* → demochair_*). Stage it with "
        f"python -m hoi_retarget.datasets.cari4d_assets, or pass --object_model_path explicitly. "
        f"Tried keys: {_cari4d_object_candidates(stem)}."
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="hoi-retarget-edit-contacts", description=__doc__)
    ap.add_argument("source", help="Source motion file (.pt OMOMO / .pth CARI4D)")
    ap.add_argument("--object_model_path", default=None, help="Object URDF (auto-resolved if omitted)")
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Use 0.0.0.0 to reach the editor from another machine on your network.",
    )
    ap.add_argument("--port", type=int, default=0, help="Bind port. Default 0 = auto-pick a free one.")
    ap.add_argument(
        "--robot",
        default="unitree_g1",
        choices=SUPPORTED_ROBOTS,
        help="Robot to edit against. Decides the contact links, the end-effector "
        "meshes drawn as markers, and the default object scaling.",
    )
    ap.add_argument("--tgt_fps", type=int, default=30)
    ap.add_argument(
        "--gender",
        default="auto",
        choices=["auto", "male", "female", "neutral"],
        help="auto: from the CARI4D subject code, else neutral (same as hoi-retarget).",
    )
    ap.add_argument(
        "--strip_mean_targets",
        action="store_true",
        help="Solve against one contact point per strip (its mean) instead of the per-frame points hoi-retarget uses.",
    )
    ap.add_argument("--resume", default=None, help="Path to a saved shifts.json to preload edits from")
    ap.add_argument("--no-retarget", action="store_true", help="Edit/save only; do not run the retarget on Go")
    ap.add_argument("--no-video", action="store_true", help="Skip MuJoCo video rendering in the retarget")
    ap.add_argument(
        "--object_scale",
        type=float,
        default=None,
        help="Object geometry scale (as hoi-retarget). Default: 0.83 on the G1, 1.0 on the H2.",
    )
    ap.add_argument(
        "--scale_object_trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reposition the object trajectory by the fit scale (on in both the 0.83 and the 1.0 corpus conventions).",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Write this session's output here instead of "
        "outputs/contact_editor/<stem>/<timestamp>/. A result already in that "
        "dir is never overwritten: the new one gets a _<timestamp> suffix.",
    )
    args = ap.parse_args()
    require_asset_root()
    if args.object_scale is not None and args.object_scale <= 0:
        ap.error("--object_scale must be positive")
    if args.resume and not os.path.isfile(args.resume):
        ap.error(f"--resume not found: {args.resume}")

    source = os.path.abspath(args.source)
    if not os.path.isfile(source):
        ap.error(f"source not found: {source}")
    omp = resolve_object_model(source, args.object_model_path)
    # Resolve once: the preview and the re-solve on Go must use the same body.
    if args.gender == "auto" and detect_source_kind(source) == "cari4d":
        args.gender = infer_cari4d_gender(source, fallback=None) or ap.error(
            f"cannot infer the subject's gender from {os.path.basename(source)!r}; pass --gender male|female"
        )
    elif args.gender == "auto":
        args.gender = "neutral"
    print(f"[hoi_retarget.contact] source : {source}")
    print(f"[hoi_retarget.contact] object : {omp}")
    print("[hoi_retarget.contact] building session (SMPL-X forward + pipeline build)…")

    session = build_session(
        source,
        omp,
        tgt_fps=args.tgt_fps,
        gender=args.gender,
        robot=args.robot,
        object_scale=args.object_scale,
        scale_object_trajectory=args.scale_object_trajectory,
        strip_mean_targets=args.strip_mean_targets,
    )
    print(
        f"[hoi_retarget.contact] {session.n_frames} frames, {len(session.strips)} contact strips, "
        f"object_scale={session.object_scale:.3f}"
    )
    for st in session.strips:
        print(f"    · {st.label}  ({st.kind}/{st.side})")

    # One timestamped dir per session, so re-editing a clip never overwrites.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
        if os.path.exists(os.path.join(out_dir, "contact_window.pkl")):
            out_dir = f"{out_dir}_{stamp}"
    else:
        out_dir = os.path.join(str(CONTACT_EDITOR_OUTPUT), session.stem, stamp)
    print(f"[hoi_retarget.contact] output : {out_dir}")

    viewer = ContactShifterViewer(session, host=args.host, port=args.port, out_dir=out_dir)
    if args.resume:
        viewer.preload(ShiftSet.load(args.resume))
        print(f"[hoi_retarget.contact] preloaded edits from {args.resume}")

    go, shifts = viewer.run()
    if not go:
        print("[hoi_retarget.contact] quit without Go — nothing saved.")
        return
    if args.no_retarget:
        os.makedirs(out_dir, exist_ok=True)
        shifts.save(os.path.join(out_dir, "shifts.json"))
        print(f"[hoi_retarget.contact] --no-retarget: saved shifts to {out_dir}/shifts.json")
        return

    from hoi_retarget.contact.retarget import run_contact_retarget_with_shifts

    out_dir = run_contact_retarget_with_shifts(
        source,
        omp,
        shifts,
        out_dir=out_dir,
        record_video=not args.no_video,
        headless=True,
        verbose=True,
        robot=args.robot,
        gender=args.gender,
        tgt_fps=args.tgt_fps,
        scale_object_trajectory=args.scale_object_trajectory,
        strip_mean_targets=args.strip_mean_targets,
    )
    print(f"[hoi_retarget.contact] retarget complete → {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        print(f"    · {f}")


if __name__ == "__main__":
    main()
