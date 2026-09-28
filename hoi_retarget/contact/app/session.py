# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``ShifterSession``: everything the contact editor needs, built from a source clip.

Two passes from the same converter frames, so frame ``t`` agrees across them:
1. SMPL-X forward (``load_smplx_data`` + ``get_smplx_data_offline_fast``): human mesh,
   real-size object pose (with the InterMimic ``pelvis - transl`` offset), and
   wrist/ankle world orientations.
2. ``build_projection_motion_data_with_gmr`` in contact mode: the contact strips,
   their auto contact points, and the scaled object trajectory and ``object_scale``.

Contact points are stored as real-size object-frame metres.
"""

from __future__ import annotations

import contextlib
import numpy as np
import os
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import torch
import trimesh
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from hoi_retarget.config import ProjectionConfig
from hoi_retarget.contact.shifts import strip_key
from hoi_retarget.datasets.cari4d import convert_cari4d_to_smplx
from hoi_retarget.gmr.utils.smpl import (
    convert_intermimic_to_smplx,
    get_smplx_data_offline_fast,
    load_smplx_data,
    source_body_mesh,
)
from hoi_retarget.paths import ASSET_ROOT
from hoi_retarget.retargeting.object_model import get_object_motion_defaults
from hoi_retarget.retargeting.source_data import (
    _find_contiguous_segments,
    build_projection_motion_data_with_gmr,
)
from hoi_retarget.robots import activate_robot, get_profile

BODY_MODEL_PATH = str(ASSET_ROOT / "body_models")

# Contact link -> SMPL-X joint whose global orientation orients its marker mesh.
LINK_TO_ORIENT_JOINT = {
    "left_palm_pad": "left_wrist",
    "right_palm_pad": "right_wrist",
    "left_ankle_roll_link": "left_ankle",
    "right_ankle_roll_link": "right_ankle",
}

# Per contact link, read from the robot's own files (G1 and H2 differ in all three):
# the visual mesh, the mesh-frame offset to the contact target (a palm pad's `pos` in
# the mocap MJCF; zero for feet), and the source-joint -> mesh rotation from the IK
# config's wrist entry (identity left, Rz(180 deg) right).


def _mocap_xml(robot: str) -> pathlib.Path:
    """The robot's mocap MJCF, which carries the palm-pad frames."""
    from hoi_retarget.gmr.params import ROBOT_XML_DICT

    return pathlib.Path(str(ROBOT_XML_DICT[robot]))


def _mesh_file(mesh_dir: pathlib.Path, stem: str) -> pathlib.Path | None:
    """Find ``<stem>.stl`` regardless of case -- G1 ships .STL, H2 .stl."""
    for cand in (f"{stem}.STL", f"{stem}.stl", f"{stem}.obj"):
        if (mesh_dir / cand).is_file():
            return mesh_dir / cand
    return None


def _link_mesh_spec(robot: str) -> dict[str, tuple[pathlib.Path, np.ndarray]]:
    """(mesh path, mesh-frame offset to the contact target) per contact link."""
    prof = get_profile(robot)
    root = ET.parse(_mocap_xml(robot)).getroot()
    mesh_dir = _mocap_xml(robot).parent / "meshes"
    spec: dict[str, tuple[pathlib.Path, np.ndarray]] = {}

    # ElementTree's XPath has no nested predicate, so map child -> parent once.
    parent_of = {c: b for b in root.iter("body") for c in b.iter("body") if c is not b}
    by_name = {b.get("name"): b for b in root.iter("body")}
    for link in prof.hand_links:
        pad = by_name.get(link)
        parent = parent_of.get(pad)
        if pad is None or parent is None:
            continue
        # A body can hold several meshes (H2's wrist body has wrist and hand); prefer the hand.
        meshes = [g.get("mesh") for g in parent.findall("geom") if g.get("mesh")]
        name = next((m for m in meshes if "hand" in m.lower()), meshes[0] if meshes else None)
        mesh = _mesh_file(mesh_dir, name) if name else None
        if mesh is None:
            continue
        spec[link] = (mesh, np.array([float(x) for x in pad.get("pos", "0 0 0").split()]))

    for link in prof.foot_links:
        mesh = _mesh_file(mesh_dir, link)
        if mesh is not None:
            spec[link] = (mesh, np.zeros(3))
    return spec


def _ik_align_quat(robot: str) -> dict[str, list[float]]:
    """Source-joint -> mesh-frame rotation per contact link, from the IK config."""
    import json

    tag = robot.replace("unitree_", "")
    cfg = pathlib.Path(__file__).resolve().parents[2] / "gmr" / "ik_configs" / f"smplx_to_{tag}.json"
    table = json.loads(cfg.read_text()).get("ik_match_table2", {}) if cfg.is_file() else {}
    prof = get_profile(robot)
    out: dict[str, list[float]] = {}
    for link in prof.contact_links:
        side = "left" if link.startswith("left") else "right"
        entry = table.get(f"{side}_wrist_yaw_link")
        # feet track the world, not a source joint orientation
        out[link] = list(entry[4]) if (entry and link in prof.hand_links) else [1.0, 0.0, 0.0, 0.0]
    return out


def _align_mat(q_wxyz) -> np.ndarray:
    return R.from_quat(np.asarray(q_wxyz, float)[[1, 2, 3, 0]]).as_matrix()


def detect_source_kind(path: str) -> str:
    """Return 'cari4d' or 'intermimic' from the path / extension."""
    low = path.lower()
    if "cari4d" in low or low.endswith(".pth"):
        return "cari4d"
    return "intermimic"


def _quat_wxyz(rot: R) -> np.ndarray:
    return rot.as_quat(scalar_first=True).astype(np.float32)


def _resample_frames(arr: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a (N0, ...) array along axis 0 to length n."""
    n0 = arr.shape[0]
    if n0 == n:
        return arr
    f = interp1d(np.arange(n0), arr, axis=0, kind="linear")
    return f(np.linspace(0, n0 - 1, n)).astype(arr.dtype)


def _human_and_object_world(source_path: str, kind: str, gender: str, tgt_fps: int) -> dict:
    """SMPL-X forward: human verts/faces, object pose and wrist/ankle world quats."""
    if kind == "cari4d":
        smplx_data = convert_cari4d_to_smplx(source_path, gender=gender)
    else:
        smplx_data = convert_intermimic_to_smplx(source_path, gender=gender)

    body_model, smplx_output, _height, obj_data = load_smplx_data(smplx_data, BODY_MODEL_PATH)
    smplx_frames, _fps, object_frames = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=tgt_fps,
        object_data=obj_data,
    )
    if object_frames is None:
        raise ValueError("Source has no object trajectory.")
    n = len(smplx_frames)

    obj_pos = np.asarray([f[0] for f in object_frames], dtype=np.float32)  # (n, 3)
    obj_rot = np.asarray([f[1] for f in object_frames], dtype=np.float32)  # (n, 4) wxyz
    # InterMimic/OMOMO store the object relative to ``transl``; CARI4D is absolute.
    if kind != "cari4d":
        offset = (smplx_frames[-1]["pelvis"][0] - smplx_output.transl[-1].detach().cpu().numpy()).astype(np.float32)
        obj_pos = obj_pos + offset

    # Resample the mesh if FPS alignment changed the frame count.
    verts0 = source_body_mesh(smplx_data, BODY_MODEL_PATH)[0].astype(np.float32)  # (N0, V, 3)
    faces = np.asarray(body_model.faces, dtype=np.int32)
    human_vertices = _resample_frames(verts0, n)

    orient_world: dict[str, np.ndarray] = {}
    for jname in set(LINK_TO_ORIENT_JOINT.values()):
        if jname in smplx_frames[0]:
            orient_world[jname] = np.asarray(
                [smplx_frames[t][jname][1] for t in range(n)], dtype=np.float32
            )  # (n, 4) wxyz

    return {
        "n_frames": n,
        "human_vertices": human_vertices,
        "human_faces": faces,
        "obj_pos_real": obj_pos,
        "obj_rot_real": obj_rot,
        "orient_world": orient_world,
    }


def _load_object_mesh(urdf_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(verts, faces) of a one-link object URDF's visual mesh, real size, URDF ``scale`` applied."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    mesh_el = root.find(".//visual/geometry/mesh")
    if mesh_el is None:
        mesh_el = root.find(".//geometry/mesh")
    if mesh_el is None:
        raise ValueError(f"No <mesh> in {urdf_path}")
    fname = mesh_el.attrib["filename"]
    scale = np.array([float(s) for s in mesh_el.attrib.get("scale", "1 1 1").split()], float)
    mesh_path = fname if os.path.isabs(fname) else os.path.join(os.path.dirname(urdf_path), fname)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32) * scale.astype(np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return verts, faces


def _load_link_meshes(robot: str) -> dict[str, dict]:
    """Per contact-link visual mesh + target offset + source-orientation align."""
    align = _ik_align_quat(robot)
    out: dict[str, dict] = {}
    for link, (mesh_path, t_target) in _link_mesh_spec(robot).items():
        m = trimesh.load(str(mesh_path), force="mesh", process=False)
        out[link] = {
            "vertices": np.asarray(m.vertices, dtype=np.float32),
            "faces": np.asarray(m.faces, dtype=np.int32),
            "t_target": np.asarray(t_target, dtype=np.float32),  # mesh-frame → contact target
            "align": _align_mat(align[link]).astype(np.float32),  # source-joint → mesh frame
        }
    return out


@dataclass
class StripInfo:
    link_name: str  # e.g. left_palm_pad
    link_idx: int  # column in contact_link_names / per_link_contact_flags
    ordinal: int  # k-th strip of this link (0-based)
    start: int  # first frame (inclusive)
    end: int  # last frame (exclusive)
    auto_real: np.ndarray  # (3,) auto contact point, real-size object frame
    side: str  # 'left' | 'right'
    kind: str  # 'hand' | 'foot'

    @property
    def key(self) -> str:
        return strip_key(self.link_name, self.ordinal)

    @property
    def mid(self) -> int:
        return int((self.start + self.end) // 2)

    @property
    def label(self) -> str:
        return f"{self.link_name} #{self.ordinal}  [{self.start}-{self.end}]"


@dataclass
class ShifterSession:
    source_path: str
    object_model_path: str
    stem: str
    kind: str
    fps: float
    n_frames: int
    object_scale: float
    # human (real-size source world)
    human_vertices: np.ndarray  # (N, V, 3)
    human_faces: np.ndarray  # (F, 3)
    # object world poses
    obj_pos_real: np.ndarray  # (N, 3)  real-size (human-aligned)
    obj_rot_real: np.ndarray  # (N, 4) wxyz
    obj_pos_scaled: np.ndarray  # (N, 3)  scaled robot world
    obj_rot_scaled: np.ndarray  # (N, 4) wxyz
    object_vertices: np.ndarray  # (M, 3)  real-size object mesh
    object_faces: np.ndarray  # (K, 3)
    # per-frame source wrist/ankle world orientation, by smplx joint name
    orient_world: dict[str, np.ndarray]  # name -> (N, 4) wxyz
    link_meshes: dict[str, dict]  # contact link -> verts/faces/t_target/align
    strips: list[StripInfo] = field(default_factory=list)
    motion_data: dict | None = None  # the built md (kept for cross-checks)
    #: Go must reuse these to solve what the preview shows.
    robot: str = "unitree_g1"
    scale_object_mesh: bool = True

    def orient_for_strip(self, strip: StripInfo) -> np.ndarray | None:
        jname = LINK_TO_ORIENT_JOINT.get(strip.link_name)
        if jname is None:
            return None
        return self.orient_world.get(jname)


def build_session(
    source_path: str,
    object_model_path: str,
    *,
    tgt_fps: int = 30,
    gender: str = "neutral",
    robot: str = "unitree_g1",
    object_scale: float | None = None,
    scale_object_trajectory: bool = True,
    strip_mean_targets: bool = False,
    verbose: bool = False,
) -> ShifterSession:
    """Build a :class:`ShifterSession` from a source clip and object URDF.

    ``object_scale``: geometry scale, None = the robot's convention (G1 0.83, H2 1.0).
    """
    # Selects the per-robot link names used downstream (e.g. the H2's feet).
    activate_robot(robot)
    source_path = os.path.abspath(source_path)
    kind = detect_source_kind(source_path)
    stem = os.path.splitext(os.path.basename(source_path))[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Pass 1: SMPL-X forward (real-size source world) ──────────────────
    world = _human_and_object_world(source_path, kind, gender, tgt_fps)

    # ── Pass 2: the real pipeline build (strips + scaled world) ──────────
    cfg = ProjectionConfig()  # published defaults, contact mode
    obj_def = get_object_motion_defaults(object_model_path)
    oc = cfg.object
    fc = cfg.foot_contact

    _ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
    if not verbose:
        _stack = contextlib.ExitStack()
        _null = _stack.enter_context(open(os.devnull, "w"))
        _stack.enter_context(contextlib.redirect_stdout(_null))
        _stack.enter_context(contextlib.redirect_stderr(_null))
        _ctx = _stack
    with _ctx:
        md, _src_tag = build_projection_motion_data_with_gmr(
            input_path=source_path,
            robot=robot,
            gender=gender,
            tgt_fps=tgt_fps,
            device=device,
            min_object_height=obj_def["min_object_height"],
            contact_links=obj_def["contact_links"],
            object_model_path=object_model_path,
            object_scale=object_scale,
            scale_object_trajectory=scale_object_trajectory,
            object_traj_smooth_window=cfg.object.traj_smooth_window,
            ground_contact_tol=oc.ground_contact_tol,
            foot_contact_enabled=fc.enabled,
            foot_contact_z_thresh=fc.z_thresh,
            foot_contact_vel_thresh=fc.vel_thresh,
            foot_contact_smplx_joints=fc.smplx_joint_names,
            contact_points_per_frame_mode=not strip_mean_targets,
            contact_points_smooth_window=oc.contact_points_smooth_window,
        )

    object_scale = float(md.get("object_scale", 1.0))
    obj_pos_scaled = np.asarray(md["object_pos"], dtype=np.float32)
    obj_rot_scaled = np.asarray(md["object_rot"], dtype=np.float32)
    per_link_flags = np.asarray(md["per_link_contact_flags"], dtype=bool)  # (N, C)
    contact_link_names = list(md["contact_link_names"])
    cp = np.asarray(md["fixed_contact_points_per_frame_in_object_frame"], dtype=np.float32)

    # ── Align frame counts across the two passes ─────────────────────────
    n = min(world["n_frames"], obj_pos_scaled.shape[0], per_link_flags.shape[0], cp.shape[0])
    human_vertices = world["human_vertices"][:n]
    obj_pos_real = world["obj_pos_real"][:n]
    obj_rot_real = world["obj_rot_real"][:n]
    obj_pos_scaled = obj_pos_scaled[:n]
    obj_rot_scaled = obj_rot_scaled[:n]
    per_link_flags = per_link_flags[:n]
    cp = cp[:n]
    orient_world = {k: v[:n] for k, v in world["orient_world"].items()}

    # ── Object mesh + per-link EE meshes ─────────────────────────────────
    object_vertices, object_faces = _load_object_mesh(object_model_path)
    link_meshes = _load_link_meshes(robot)

    # ── Enumerate strips + real-size auto points ─────────────────────────
    strips: list[StripInfo] = []
    for c, link in enumerate(contact_link_names):
        side = "left" if link.startswith("left") else "right"
        kind_lf = "foot" if "ankle" in link else "hand"
        for k, (s, e) in enumerate(_find_contiguous_segments(per_link_flags[:, c])):
            e = min(e, n)
            if e <= s:
                continue
            with np.errstate(invalid="ignore"):
                scaled_pt = np.nanmean(cp[s:e, c], axis=0)
            if not np.all(np.isfinite(scaled_pt)):
                continue
            real_pt = (scaled_pt / object_scale).astype(np.float32)
            strips.append(
                StripInfo(
                    link_name=link,
                    link_idx=c,
                    ordinal=k,
                    start=int(s),
                    end=int(e),
                    auto_real=real_pt,
                    side=side,
                    kind=kind_lf,
                )
            )

    return ShifterSession(
        source_path=source_path,
        object_model_path=object_model_path,
        stem=stem,
        kind=kind,
        fps=float(md.get("fps", tgt_fps)),
        n_frames=n,
        object_scale=object_scale,
        robot=robot,
        scale_object_mesh=bool(md.get("scale_object_mesh", True)),
        human_vertices=human_vertices,
        human_faces=world["human_faces"],
        obj_pos_real=obj_pos_real,
        obj_rot_real=obj_rot_real,
        obj_pos_scaled=obj_pos_scaled,
        obj_rot_scaled=obj_rot_scaled,
        object_vertices=object_vertices,
        object_faces=object_faces,
        orient_world=orient_world,
        link_meshes=link_meshes,
        strips=strips,
        motion_data=md,
    )
