# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the SMPL-X source motion of an OMOMO clip in the robot recorder's style.

The human is a MuJoCo ``<skin>`` whose vertices are overwritten each frame, so
scene, lighting and camera go through the same :mod:`render_style` code as the robot.

    record_smpl_video("sub6_whitechair_036", "human.mp4", style="square_auto")

``ground_align`` re-plants the feet (the source floats or sinks by up to ~7 cm),
``subject_scale`` shrinks human and object about the floor under the pelvis (pass
the clip's ``object_scale``), and ``camera=solve_camera(...)`` from the robot clip
shares its framing and horizon.
"""

from __future__ import annotations

import numpy as np
import os
import pathlib
import struct
import tempfile
import xml.etree.ElementTree as ET

import mujoco as mj
from scipy.spatial.transform import Rotation as R

from hoi_retarget.gmr.robot_motion_viewer import _inject_urdf_object_into_robot_xml
from hoi_retarget.gmr.utils.smpl import convert_intermimic_to_smplx, source_body_mesh
from hoi_retarget.paths import ASSET_ROOT, DATA_ROOT, smplx_ext
from hoi_retarget.rendering import render_style
from hoi_retarget.rendering.render_style import DEFAULT_STYLE, Style

SOURCE_ROOT = DATA_ROOT / "InterMimic" / "OMOMO_new"
BODY_MODELS = pathlib.Path(ASSET_ROOT) / "body_models"
OBJECT_URDFS = pathlib.Path(ASSET_ROOT) / "objects"

#: Human body colours: ``name -> (rgb, description)``.
HUMAN_TONES = {
    "slate": ((0.60, 0.65, 0.74), "cool slate — separates from the robot"),
    "slate_light": ((0.68, 0.72, 0.80), "slate, one step lighter"),
    "slate_deep": ((0.50, 0.56, 0.67), "slate, one step darker"),
    "slate_warm": ((0.62, 0.64, 0.70), "slate with the blue pulled out"),
    "bone": ((0.87, 0.83, 0.75), "warm off-white — reads human, still neutral"),
    "clay": ((0.79, 0.69, 0.60), "warm clay"),
    "robot_grey": ((0.82, 0.83, 0.85), "the G1's own shell grey"),
}
TONE_NAMES = list(HUMAN_TONES)

GROUND_MODES = ("none", "clip", "frame")

_BONE = "smpl_bone"
# SMPL-X kinematic-tree indices of the two ankles and two feet.
_FOOT_JOINTS = (7, 8, 10, 11)


def object_name_from_stem(stem: str) -> str:
    """``sub6_whitechair_036`` -> ``whitechair``."""
    return "_".join(stem.split("_")[1:-1])


# --------------------------------------------------------------------------- #
# The skin asset
# --------------------------------------------------------------------------- #
def write_skn(path, verts: np.ndarray, faces: np.ndarray, bone: str = _BONE) -> None:
    """Write a MuJoCo binary skin with one bone binding every vertex at weight 1.

    The bind pose is irrelevant: ``scene.skinvert`` is overwritten every frame.
    """
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    nvert = verts.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("<4i", nvert, 0, faces.shape[0], 1))
        f.write(verts.tobytes())
        f.write(faces.tobytes())
        name = bone.encode()[:39]
        f.write(name + b"\0" * (40 - len(name)))
        f.write(np.zeros(3, np.float32).tobytes())  # bindpos
        f.write(np.array([1, 0, 0, 0], np.float32).tobytes())  # bindquat
        f.write(struct.pack("<i", nvert))
        f.write(np.arange(nvert, dtype=np.int32).tobytes())  # vertid
        f.write(np.ones(nvert, np.float32).tobytes())  # vertweight


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (MuJoCo derives skin normals from the bone, not the vertices)."""
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(verts)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def build_scene_xml(skn_path, human_rgba) -> str:
    """Write a minimal scene MJCF (floor, skin bone, skin) and return its path.

    Mirrors ``g1_mocap_29dof.xml``'s statistic extent, ``groundplane`` assets and
    ``floor`` geom so :func:`render_style.patch_model_xml` styles it identically.
    """
    mjcf = ET.Element("mujoco", {"model": "smplx_source"})
    ET.SubElement(mjcf, "statistic", {"center": "0 0 0.9", "extent": "0.8"})
    vis = ET.SubElement(mjcf, "visual")
    ET.SubElement(vis, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.1 0.1 0.1", "specular": "0.9 0.9 0.9"})
    ET.SubElement(vis, "global", {"offwidth": "2080", "offheight": "1170"})

    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(
        asset,
        "texture",
        {"type": "skybox", "builtin": "gradient", "rgb1": "1 1 1", "rgb2": "1 1 1", "width": "800", "height": "800"},
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "1 1 1",
            "rgb2": "1 1 1",
            "markrgb": "0 0 0",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "groundplane", "texture": "groundplane", "texuniform": "true", "texrepeat": "5 5", "reflectance": "0"},
    )
    ET.SubElement(
        asset,
        "skin",
        {
            "name": "smpl_skin",
            "file": str(skn_path),
            "inflate": "0",
            "rgba": " ".join(f"{c:.4f}" for c in human_rgba),
        },
    )

    wb = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(
        wb,
        "geom",
        {
            "name": "floor",
            "size": "0 0 0.01",
            "type": "plane",
            "material": "groundplane",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(wb, "body", {"name": _BONE, "mocap": "true", "pos": "0 0 0"})
    ET.SubElement(wb, "light", {"pos": "-3 -3 5", "dir": "3 3 -5", "diffuse": ".5 .5 .5", "castshadow": "true"})

    fd, out = tempfile.mkstemp(prefix="smpl_scene_", suffix=".xml", dir=str(pathlib.Path(skn_path).parent))
    os.close(fd)
    ET.ElementTree(mjcf).write(out, encoding="utf-8", xml_declaration=True)
    return out


# --------------------------------------------------------------------------- #
# The renderer
# --------------------------------------------------------------------------- #
class SmplSourceRenderer:
    """Renders OMOMO SMPL-X source clips in a :class:`render_style.Style`."""

    def __init__(
        self,
        style: str | Style | None = DEFAULT_STYLE,
        tone: str = "slate",
        gender: str = "neutral",
        subject_scale: float = 1.0,
        ground_align: str = "clip",
        source_root=None,
        workdir=None,
    ):
        """``subject_scale`` shrinks human and object about the floor under the pelvis.

        ``ground_align``: ``none`` keeps the mocap height, ``clip`` applies one
        shift per clip, ``frame`` re-plants the feet every frame.
        """
        import smplx

        if ground_align not in GROUND_MODES:
            raise ValueError(f"ground_align must be one of {GROUND_MODES}")
        if tone not in HUMAN_TONES:
            raise ValueError(f"unknown tone {tone!r}; choose from {', '.join(TONE_NAMES)}")
        self.style = render_style.get_style(style)
        self.human_rgb = HUMAN_TONES[tone][0]
        self.subject_scale = float(subject_scale)
        self.ground_align = ground_align
        self.source_root = pathlib.Path(source_root or SOURCE_ROOT)
        self.gender = gender
        self.body_model = smplx.create(str(BODY_MODELS), "smplx", gender=gender, ext=smplx_ext(), use_pca=False)
        self.faces = np.asarray(self.body_model.faces, dtype=np.int32)
        self.workdir = pathlib.Path(workdir or tempfile.mkdtemp(prefix="smpl_mjcf_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._skn = None

    # ------------------------------------------------------------------ ground
    @property
    def foot_vertices(self) -> np.ndarray:
        """Vertices skinned mostly to an ankle or foot (the whole-mesh min would catch crouching hands)."""
        if not hasattr(self, "_foot_vids"):
            w = self.body_model.lbs_weights.detach().cpu().numpy()
            self._foot_vids = np.where(np.isin(w.argmax(axis=1), _FOOT_JOINTS))[0]
        return self._foot_vids

    def _ground_offset(self, verts) -> np.ndarray | None:
        if self.ground_align == "none":
            return None
        sole = verts[:, self.foot_vertices, 2].min(axis=1)
        if self.ground_align == "frame":
            return (-sole).astype(np.float32)
        # 2nd percentile, not the min: one bad frame should not lift the clip.
        return np.full(verts.shape[0], -np.percentile(sole, 2), np.float32)

    # -------------------------------------------------------------------- data
    def load_clip(self, stem: str) -> dict:
        """Load the SMPL-X mesh and object trajectory, scaled and ground-aligned."""
        pt_path = self.source_root / f"{stem}.pt"
        if not pt_path.is_file():
            raise FileNotFoundError(f"{stem}: render-source supports OMOMO clips only (looked in {pt_path})")
        sd = convert_intermimic_to_smplx(str(pt_path), gender=self.gender)
        T = sd["pose_body"].shape[0]
        verts, joints, _ = source_body_mesh(sd, str(BODY_MODELS))
        verts = verts.astype(np.float32)
        transl = sd["trans"]

        obj_pos = sd["obj_state"][:, :3].astype(np.float32).copy()
        obj_rotvec = sd["obj_state"][:, 3:6].astype(np.float32)
        obj_pos += (joints[-1, 0] - transl[-1]).astype(np.float32)  # pelvis offset
        obj_quat = R.from_rotvec(obj_rotvec).as_quat()[:, [3, 0, 1, 2]]  # -> wxyz

        s = self.subject_scale
        if s != 1.0:
            # Scale about the floor point under the pelvis so only size changes.
            g = np.stack([joints[:, 0, 0], joints[:, 0, 1], np.zeros(T)], axis=1).astype(np.float32)
            verts = g[:, None, :] + s * (verts - g[:, None, :])
            joints = g[:, None, :] + s * (joints - g[:, None, :])
            obj_pos = g + s * (obj_pos - g)

        dz = self._ground_offset(verts)
        if dz is not None:
            up = np.array([0, 0, 1], np.float32)
            verts = verts + up * dz[:, None, None]
            joints = joints + up.astype(np.float64) * dz[:, None, None]
            obj_pos = obj_pos + up * dz[:, None]

        return {
            "stem": stem,
            "obj_name": object_name_from_stem(stem),
            "verts": verts,
            "pelvis": joints[:, 0, :].astype(float),
            "obj_pos": obj_pos,
            "obj_quat": obj_quat.astype(float),
            "T": T,
            "fps": float(np.asarray(sd["mocap_frame_rate"]).item()),
        }

    # ------------------------------------------------------------------- scene
    def _build(self, clip):
        if self._skn is None:
            self._skn = self.workdir / "smplx.skn"
            write_skn(self._skn, clip["verts"][0], self.faces)
        base_xml = build_scene_xml(self._skn, (*self.human_rgb, 1.0))
        runtime_xml, obj_body = _inject_urdf_object_into_robot_xml(
            base_xml, str(OBJECT_URDFS / f"{clip['obj_name']}.urdf"), object_scale=self.subject_scale
        )
        styled = render_style.patch_model_xml(runtime_xml, self.style)
        model = mj.MjModel.from_xml_path(styled)
        data = mj.MjData(model)

        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_body)
        obj_geom = int(model.body_geomadr[bid])
        lights = render_style.apply_to_model(model, self.style, obj_geom)
        # A skin is not a geom, so apply_to_model misses it; give it the robot's light material.
        model.skin_rgba[0] = (*self.human_rgb, 1.0)
        mat = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, render_style.MAT_ROBOT_LIGHT)
        if mat >= 0:
            model.skin_matid[0] = mat

        fb_w = self.style.width * self.style.supersample
        fb_h = self.style.height * self.style.supersample
        model.vis.global_.offwidth = fb_w
        model.vis.global_.offheight = fb_h
        return dict(
            model=model,
            data=data,
            renderer=mj.Renderer(model, height=fb_h, width=fb_w),
            opt=render_style.make_scene_option(self.style),
            lights=lights,
            obj_mocap=int(model.body_mocapid[bid]),
            obj_geom=obj_geom,
            bone_mocap=int(model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _BONE)]),
            xmls=(base_xml, runtime_xml, styled),
        )

    # ------------------------------------------------------------------ camera
    def solve_camera(self, clip, scene=None):
        """``(distance, lookat)`` this style would frame the human with."""
        style = self.style
        scene = scene if scene is not None else self._build(clip)
        base = render_style.smooth_track(clip["pelvis"], style)
        if not style.cam_fit:
            track = base.copy()
            track[:, 2] += style.cam_lookat_dz
            return 3.0 * style.cam_distance_scale, track

        gid = scene["obj_geom"]
        model = scene["model"]
        c, half = model.geom_aabb[gid][:3], model.geom_aabb[gid][3:]
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        local = R.from_quat(np.roll(model.geom_quat[gid], -1)).apply(c + signs * half) + model.geom_pos[gid]
        mats = R.from_quat(np.roll(clip["obj_quat"], -1, axis=-1)).as_matrix()
        corners = np.einsum("tij,kj->tki", mats, local) + clip["obj_pos"][:, None, :]
        verts = clip["verts"][:, ::40, :].astype(float)  # ~260 surface points
        # Surface points, not link origins, so they need less pad.
        fit_style = render_style.Style(**{**style.__dict__, "cam_fit_pad": min(style.cam_fit_pad, 0.04)})
        return render_style.fit_camera(
            np.concatenate([verts, corners], axis=1),
            base,
            fit_style,
            aspect=style.width / style.height,
            fovy_deg=float(model.vis.global_.fovy),
            fps=clip["fps"],
        )

    # ------------------------------------------------------------------ render
    def render_frames(self, clip, frames=None, camera=None):
        """Yield ``(t, rgb)``.  ``camera=(distance, lookat)`` overrides the fit."""
        scene = self._build(clip)
        model, data, renderer = scene["model"], scene["data"], scene["renderer"]
        distance, lookat = camera if camera is not None else self.solve_camera(clip, scene)
        n = clip["T"]
        distance = np.broadcast_to(np.asarray(distance, float).reshape(-1), (n,))
        lookat = np.asarray(lookat, float)
        frames = range(n) if frames is None else frames
        try:
            for t in frames:
                data.mocap_pos[scene["obj_mocap"]] = clip["obj_pos"][t]
                data.mocap_quat[scene["obj_mocap"]] = clip["obj_quat"][t]
                data.mocap_pos[scene["bone_mocap"]] = clip["pelvis"][t]
                render_style.update_lights(model, self.style, scene["lights"], clip["pelvis"][t])
                mj.mj_forward(model, data)

                cam = mj.MjvCamera()
                cam.type = mj.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = lookat[min(t, lookat.shape[0] - 1)]
                cam.distance = float(distance[t])
                cam.elevation = self.style.cam_elevation
                cam.azimuth = self.style.cam_azimuth
                renderer.update_scene(data, camera=cam, scene_option=scene["opt"])
                # Overwrite the bone-driven skin with this frame's mesh (flat, 3*nvert).
                v = clip["verts"][t]
                renderer.scene.skinvert[:] = v.reshape(-1)
                renderer.scene.skinnormal[:] = vertex_normals(v.astype(np.float64), self.faces).reshape(-1)
                render_style.apply_scene_flags(renderer.scene, self.style)
                yield t, render_style.downscale(renderer.render(), self.style)
        finally:
            for p in scene["xmls"]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def render_video(self, clip, out_path, camera=None):
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = render_style.open_video_writer(str(out_path), clip["fps"], self.style)
        try:
            for _t, img in self.render_frames(clip, camera=camera):
                writer.append_data(img)
        finally:
            writer.close()
        return out_path


def record_smpl_video(
    stem: str,
    video_path: str,
    style: str | Style | None = DEFAULT_STYLE,
    tone: str = "slate",
    subject_scale: float = 1.0,
    ground_align: str = "clip",
    camera: tuple | None = None,
    single_frame: int | None = None,
    source_root=None,
) -> None:
    """Render one OMOMO source clip; ``single_frame`` writes that frame as a PNG instead."""
    import imageio.v2 as imageio

    r = SmplSourceRenderer(
        style=style, tone=tone, subject_scale=subject_scale, ground_align=ground_align, source_root=source_root
    )
    clip = r.load_clip(stem)
    if single_frame is None:
        r.render_video(clip, video_path, camera=camera)
        return
    t = min(int(single_frame), clip["T"] - 1)
    _t, img = next(iter(r.render_frames(clip, [t], camera=camera)))
    out = pathlib.Path(video_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out), img)
