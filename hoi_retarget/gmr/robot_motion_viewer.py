# Derived from GMR (General Motion Retargeting).
# Source: https://github.com/YanjieZe/GMR @ 5bac4bd (2025-10-16)
# Copyright 2025 Yanjie Ze
#
# SPDX-License-Identifier: MIT
#
# Modified by ETH Zurich, 2026, under the same MIT terms:
#   - extended substantially for human-object interaction: object meshes injected
#     into the robot scene, contact-point markers, offscreen rendering, and the
#     named render styles used for the paper figures. Little of upstream's viewer
#     remains beyond its structure;
#   - render styling moved to hoi_retarget.rendering.render_style.

import os
import glob
import time
import tempfile

from hoi_retarget import paths
import atexit
import signal
import xml.etree.ElementTree as ET
from dataclasses import replace as _dc_replace

import mujoco as mj
import mujoco.viewer as mjv
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from hoi_retarget.rendering import render_style
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print

_RUNTIME_XML_PATHS = set()
_RUNTIME_TEMP_FILES = set()
_CLEANUP_HOOKS_INSTALLED = False
# Cache of source-OBJ -> cleaned-OBJ (keyed by (path, mtime)) so repeated viewer
# constructions don't re-run the trimesh reindex on the same mesh.
_TEXTURED_OBJ_CACHE = {}

# When True, draws a green sphere on each robot contact link that is currently
# active, and a cyan line connecting it to the expected contact point on the
# object surface.  Inactive contact links get a dim grey sphere.
DEBUG_PER_LINK_CONTACT = False

# When True, recolours the object on body/ground contact and renders spheres at
# the per-frame contact points passed via `object_contact_points_world`.  When
# False, the object keeps its default/textured appearance and no contact spheres
# are drawn.
DEBUG_CONTACT = False

# Custom zoom multiplier to see LONG objects if needed.
VIEWER_CAM_DISTANCE_MULTIPLIER = 1.5

# Object opacity, background, lighting, materials, resolution and encoding all
# come from the render style now — see hoi_retarget/rendering/render_style.py
# and the `style=` argument below.  The default is render_style.DEFAULT_STYLE.


def _cleanup_runtime_xml_paths():
    for p in list(_RUNTIME_XML_PATHS):
        try:
            os.remove(p)
        except OSError:
            pass
        _RUNTIME_XML_PATHS.discard(p)
    for p in list(_RUNTIME_TEMP_FILES):
        try:
            os.remove(p)
        except OSError:
            pass
        _RUNTIME_TEMP_FILES.discard(p)


def _install_runtime_xml_cleanup_hooks():
    global _CLEANUP_HOOKS_INSTALLED
    if _CLEANUP_HOOKS_INSTALLED:
        return
    _CLEANUP_HOOKS_INSTALLED = True
    atexit.register(_cleanup_runtime_xml_paths)

    for sig in (signal.SIGINT, signal.SIGTERM):
        prev = signal.getsignal(sig)

        def _handler(signum, frame, _prev=prev):
            _cleanup_runtime_xml_paths()
            if callable(_prev):
                _prev(signum, frame)
            elif _prev == signal.SIG_DFL:
                raise SystemExit(128 + signum)
            # SIG_IGN: intentionally do nothing after cleanup.

        signal.signal(sig, _handler)


def _parse_float_list(raw, n, default):
    if raw is None:
        vals = list(default)
    else:
        vals = [float(x) for x in raw.strip().split()]
    if len(vals) < n:
        vals.extend([0.0] * (n - len(vals)))
    return vals[:n]


def _resolve_mesh_path(model_path, mesh_ref):
    if mesh_ref is None:
        raise ValueError(f"Missing mesh filename in URDF visual geometry: {model_path}")
    if os.path.isabs(mesh_ref) and os.path.isfile(mesh_ref):
        return mesh_ref

    base = os.path.dirname(os.path.abspath(model_path))
    candidates = [
        os.path.join(base, mesh_ref),
        os.path.join(base, "..", mesh_ref),
        os.path.join(base, "..", "..", mesh_ref),
        os.path.join(os.getcwd(), mesh_ref),
        os.path.join(os.getcwd(), "assets", mesh_ref),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Cannot resolve mesh path '{mesh_ref}' from '{model_path}'")


def _rpy_to_quat_wxyz(rpy_xyz):
    quat_xyzw = R.from_euler("xyz", rpy_xyz).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _find_textured_mesh_for_urdf(urdf_path):
    """Locate a textured OBJ+PNG pair that matches this URDF, if one exists.

    CARI4D ships an un-textured mesh under ``<dataset>/assets/objects/<stem>/``
    (referenced by the URDF) and a textured copy aligned to the same frame
    under ``<dataset>/meshes/<stem>.*/``.  When the latter is present, we
    prefer it so the viewer shows the real texture.

    Returns ``(obj_path, texture_path)`` or ``(None, None)``.
    """
    urdf_abs = os.path.abspath(urdf_path)
    assets_dir = os.path.dirname(urdf_abs)
    stem = os.path.splitext(os.path.basename(urdf_abs))[0]
    meshes_dir = os.path.normpath(os.path.join(assets_dir, "..", "meshes"))
    if not os.path.isdir(meshes_dir):
        return None, None

    # CARI4D mesh dirs follow either <stem>.<cam>.color.mp4_<step>_rgba (older)
    # or <stem>_t<time>_k<int>_rgba (newer) — accept both.
    candidates = sorted(set(
        glob.glob(os.path.join(meshes_dir, f"{stem}.*"))
        + glob.glob(os.path.join(meshes_dir, f"{stem}_*"))
    ))
    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        objs = sorted(glob.glob(os.path.join(cand, "*.obj")))
        pngs = sorted(glob.glob(os.path.join(cand, "*.png")))
        if objs and pngs:
            return objs[0], pngs[0]
    return None, None


def _prepare_textured_obj_for_mujoco(obj_path):
    """Re-export a textured OBJ into a form MuJoCo's strict parser can load.

    CARI4D's textured OBJs have more ``vt`` entries than vertices (shared
    positions with distinct UVs), which MuJoCo refuses.  We load via trimesh
    and write a cleaned OBJ where every face vertex references a unified
    ``v``/``vt`` index.  Returns the path to the cleaned OBJ, or ``None`` if
    trimesh isn't installed or the conversion fails — in which case the
    caller should fall back to the un-textured mesh.
    """
    try:
        key = (os.path.abspath(obj_path), os.path.getmtime(obj_path))
    except OSError:
        return None
    cached = _TEXTURED_OBJ_CACHE.get(key)
    if cached is not None and os.path.isfile(cached):
        return cached

    try:
        import trimesh
    except ImportError:
        return None

    try:
        mesh = trimesh.load(obj_path, force="mesh", process=False)
    except Exception:
        return None
    if getattr(mesh, "visual", None) is None or getattr(mesh.visual, "uv", None) is None:
        return None

    fd, tmp_path = tempfile.mkstemp(prefix="runtime_object_mesh_", suffix=".obj")
    os.close(fd)
    try:
        mesh.export(tmp_path, include_texture=True)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None

    _RUNTIME_TEMP_FILES.add(tmp_path)
    _TEXTURED_OBJ_CACHE[key] = tmp_path
    return tmp_path


def _inject_urdf_object_into_robot_xml(robot_xml_path, object_model_path, object_scale=1.0):
    """
    Merge a URDF visual object into a temporary robot MJCF as a mocap body.
    Returns (temp_xml_path, object_body_name).

    ``object_scale`` (<1 for full_scale) shrinks the rendered geometry about the
    object frame origin — primitive sizes, mesh scale, and the visual origin offset —
    so the mesh matches the shrunk object_pos / contact points in the pkl. Defaults to
    1.0 (no-op) for none/pos_scale and for callers that pre-scale the URDF themselves.
    """
    if object_model_path is None:
        raise ValueError("object_model_path is required and must point to a URDF file.")
    object_scale = float(object_scale)
    object_path = str(paths.resolve_asset_path(object_model_path))
    if not os.path.isfile(object_path):
        raise FileNotFoundError(f"Object URDF not found: {object_path}")
    if not object_path.lower().endswith(".urdf"):
        raise ValueError(f"Object model must be a URDF file, got: {object_path}")

    robot_tree = ET.parse(robot_xml_path)
    robot_root = robot_tree.getroot()

    obj_root = ET.parse(object_path).getroot()
    visual = obj_root.find(".//link/visual")
    if visual is None:
        raise ValueError(f"No <visual> found in URDF: {object_path}")
    geom = visual.find("geometry")
    if geom is None:
        raise ValueError(f"No <geometry> under <visual> in URDF: {object_path}")

    origin = visual.find("origin")
    xyz = _parse_float_list(None if origin is None else origin.attrib.get("xyz"), 3, [0.0, 0.0, 0.0])
    rpy = _parse_float_list(None if origin is None else origin.attrib.get("rpy"), 3, [0.0, 0.0, 0.0])
    quat_wxyz = _rpy_to_quat_wxyz(rpy)
    # Scale the visual origin offset about the object frame origin too (it shifts the
    # mesh relative to object_pos, which is what the contact points are measured from).
    xyz = [c * object_scale for c in xyz]
    pos_str = f"{xyz[0]} {xyz[1]} {xyz[2]}"
    quat_str = f"{quat_wxyz[0]} {quat_wxyz[1]} {quat_wxyz[2]} {quat_wxyz[3]}"

    rgba = [0.8, 0.8, 0.8, 1.0]
    material = visual.find("material")
    if material is not None:
        color = material.find("color")
        if color is not None:
            rgba = _parse_float_list(color.attrib.get("rgba"), 4, rgba)
    rgba_str = f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"

    asset = robot_root.find("asset")
    if asset is None:
        asset = ET.SubElement(robot_root, "asset")
    worldbody = robot_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> in robot XML: {robot_xml_path}")

    object_tag = os.path.splitext(os.path.basename(object_path))[0]
    object_body_name = f"runtime_object_{object_tag}"
    object_mesh_name = f"runtime_object_mesh_{object_tag}"

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": object_body_name,
            "mocap": "true",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
        },
    )

    geom_attrs = {
        "pos": pos_str,
        "quat": quat_str,
        "rgba": rgba_str,
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
    }

    sphere = geom.find("sphere")
    box = geom.find("box")
    cylinder = geom.find("cylinder")
    mesh = geom.find("mesh")
    if sphere is not None:
        radius = float(sphere.attrib.get("radius", "0.12")) * object_scale
        geom_attrs.update({"type": "sphere", "size": f"{radius}"})
    elif box is not None:
        full = _parse_float_list(box.attrib.get("size"), 3, [0.24, 0.24, 0.24])
        half = [0.5 * full[0] * object_scale, 0.5 * full[1] * object_scale, 0.5 * full[2] * object_scale]
        geom_attrs.update({"type": "box", "size": f"{half[0]} {half[1]} {half[2]}"})
    elif cylinder is not None:
        radius = float(cylinder.attrib.get("radius", "0.12")) * object_scale
        length = float(cylinder.attrib.get("length", "0.24")) * object_scale
        geom_attrs.update({"type": "cylinder", "size": f"{radius} {0.5 * length}"})
    elif mesh is not None:
        mesh_ref = mesh.attrib.get("filename") or mesh.attrib.get("file")
        mesh_path = _resolve_mesh_path(object_path, mesh_ref)
        mesh_scale = _parse_float_list(mesh.attrib.get("scale"), 3, [1.0, 1.0, 1.0])
        mesh_scale = [c * object_scale for c in mesh_scale]

        # Prefer a co-registered textured mesh (e.g. CARI4D's meshes/<stem>.*)
        # over the plain OBJ from the URDF when available.  MuJoCo's OBJ parser
        # is strict about v/vt index alignment, so we re-export through trimesh
        # into a unified form; on any failure we silently fall back.
        textured_obj, texture_png = _find_textured_mesh_for_urdf(object_path)
        if textured_obj is not None and texture_png is not None:
            prepared_obj = _prepare_textured_obj_for_mujoco(textured_obj)
            if prepared_obj is not None:
                mesh_path = prepared_obj
                texture_name = f"runtime_object_tex_{object_tag}"
                material_name = f"runtime_object_mat_{object_tag}"
                ET.SubElement(
                    asset,
                    "texture",
                    {
                        "name": texture_name,
                        "type": "2d",
                        "file": texture_png,
                    },
                )
                ET.SubElement(
                    asset,
                    "material",
                    {
                        "name": material_name,
                        "texture": texture_name,
                        "texuniform": "false",
                    },
                )
                geom_attrs["material"] = material_name
                # Drop the flat rgba so the texture shows through.
                geom_attrs.pop("rgba", None)

        ET.SubElement(
            asset,
            "mesh",
            {
                "name": object_mesh_name,
                "file": mesh_path,
                "scale": f"{mesh_scale[0]} {mesh_scale[1]} {mesh_scale[2]}",
            },
        )
        geom_attrs.update({"type": "mesh", "mesh": object_mesh_name})
    else:
        raise ValueError(f"Unsupported URDF visual geometry in: {object_path}")

    ET.SubElement(body, "geom", geom_attrs)

    render_style.absolutize_asset_paths(robot_root, robot_xml_path)
    fd, tmp_xml_path = tempfile.mkstemp(prefix="runtime_robot_with_object_", suffix=".xml")
    os.close(fd)
    robot_tree.write(tmp_xml_path, encoding="utf-8", xml_declaration=True)
    return tmp_xml_path, object_body_name


def draw_frame(
    pos,
    mat,
    scene,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            scene.geoms[scene.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        scene.ngeom += 1


def draw_line(from_pos, to_pos, scene, width=0.008, rgba=(0.0, 0.9, 0.9, 0.8)):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_LINE,
        size=[width, width, width],
        pos=np.asarray(from_pos, dtype=float),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=float),
    )
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_LINE,
        width=width,
        from_=np.asarray(from_pos, dtype=float),
        to=np.asarray(to_pos, dtype=float),
    )
    geom.rgba[:] = np.asarray(rgba, dtype=float)
    scene.ngeom += 1


def draw_capsule(from_pos, to_pos, scene, width=0.006, rgba=(0.0, 0.9, 0.9, 0.9)):
    """Like draw_line but with real thickness — thin mjGEOM_LINE is 1 px and
    effectively invisible at 1080p."""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=[width, width, width],
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=float),
    )
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=np.asarray(from_pos, dtype=float),
        to=np.asarray(to_pos, dtype=float),
    )
    geom.rgba[:] = np.asarray(rgba, dtype=float)
    scene.ngeom += 1


def draw_sphere(pos, scene, radius=0.025, rgba=(1.0, 0.65, 0.0, 1.0)):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, radius, radius],
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=False,
                video_path=None,
                video_width=None,
                video_height=None,
                keyboard_callback=None,
                object_model_path=None,
                # headless mode: skip interactive viewer, only offscreen render
                headless=False,
                # full_scale: shrink the rendered object mesh to match object_scale
                object_scale=1.0,
                # look: a render_style preset name or Style (None -> DEFAULT_STYLE)
                style=None,
                ):

        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.object_mocap_id = None
        self.runtime_model_xml_path = None
        self.styled_model_xml_path = None
        self.runtime_object_body_name = None

        self.style = render_style.get_style(style)
        if video_width is not None:
            self.style = _dc_replace(self.style, width=int(video_width))
        if video_height is not None:
            self.style = _dc_replace(self.style, height=int(video_height))

        if object_model_path is None:
            raise ValueError("object_model_path is required and must be a URDF.")

        runtime_xml, runtime_body_name = _inject_urdf_object_into_robot_xml(
            str(self.xml_path), object_model_path, object_scale=object_scale
        )
        _install_runtime_xml_cleanup_hooks()
        _RUNTIME_XML_PATHS.add(runtime_xml)
        # Rewrite the runtime MJCF with the style's framebuffer / lights / scene.
        # `legacy` returns the same path untouched.
        styled_xml = render_style.patch_model_xml(runtime_xml, self.style)
        if styled_xml != runtime_xml:
            _RUNTIME_XML_PATHS.add(styled_xml)
            self.styled_model_xml_path = styled_xml
        self.model = mj.MjModel.from_xml_path(styled_xml)
        self.runtime_model_xml_path = runtime_xml
        self.runtime_object_body_name = runtime_body_name
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = (
            VIEWER_CAM_DISTANCE_DICT[robot_type]
            * VIEWER_CAM_DISTANCE_MULTIPLIER
            * self.style.cam_distance_scale
        )
        # Height offset of the camera target above the pelvis.  Styles set it
        # statically; ``fit_camera`` (via record_motion_video) overwrites both
        # this and viewer_cam_distance with the clip's fitted framing.
        self.cam_lookat_dz = float(self.style.cam_lookat_dz)
        mj.mj_step(self.model, self.data)

        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.runtime_object_body_name)
        if body_id < 0:
            raise RuntimeError(f"Runtime object body not found in model: {self.runtime_object_body_name}")
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError(f"Runtime object body is not mocap: {self.runtime_object_body_name}")
        self.object_mocap_id = mocap_id
        geom_id = int(self.model.body_geomadr[body_id])
        self.object_geom_id = geom_id if geom_id >= 0 else None

        # Recolour the object and the robot, and collect the style's light rig.
        self.light_ids = render_style.apply_to_model(
            self.model, self.style, self.object_geom_id
        )
        self.object_opacity = self.style.object_alpha
        self.object_rgba_default = (
            self.model.geom_rgba[self.object_geom_id].copy() if self.object_geom_id is not None else None
        )

        # Contact markers: either the module-level DEBUG switch or the style.
        self.show_per_link_contact = DEBUG_PER_LINK_CONTACT or (
            self.style.styled and self.style.contact_overlay
        )
        self._object_trail = []

        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video

        print(f"[viewer] runtime URDF object loaded in MuJoCo model: {self.runtime_object_body_name}")
        print(f"[viewer] render style '{self.style.name}' "
              f"({self.style.width}x{self.style.height}, {self.style.supersample}x supersample)")

        self.headless = headless
        self.viewer = None
        if not headless:
            self.viewer = mjv.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=keyboard_callback
                )
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot

        # Offscreen renderer — always created for video recording and headless
        # render_frame().  Supersampled styles render big and downscale on read.
        fb_w = self.style.width * self.style.supersample
        fb_h = self.style.height * self.style.supersample
        self.model.vis.global_.offwidth = fb_w
        self.model.vis.global_.offheight = fb_h
        self.renderer = mj.Renderer(self.model, height=fb_h, width=fb_w)
        self.scene_option = render_style.make_scene_option(self.style)
        if self.viewer is not None and self.scene_option is not None:
            # Keep the interactive window free of the same z-fighting duplicates.
            self.viewer.opt.geomgroup[:] = self.scene_option.geomgroup

        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)

            if video_dir and not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = render_style.open_video_writer(
                self.video_path, self.motion_fps, self.style
            )
            print(f"Recording video to {self.video_path}")

    def _set_object_color(self, rgb=None):
        if self.object_geom_id is None:
            return

        if rgb is None:
            if self.object_rgba_default is None:
                return
            rgba = self.object_rgba_default.copy()
            rgba[3] = self.object_opacity
        else:
            rgba = np.array([rgb[0], rgb[1], rgb[2], self.object_opacity], dtype=float)
        self.model.geom_rgba[self.object_geom_id] = rgba
        
    def _make_tracking_camera(self, lookat=None) -> "mj.MjvCamera":
        """Build an offscreen camera that tracks the robot base.

        ``lookat`` overrides the raw pelvis position (used to feed a smoothed
        camera track).  Framing comes from the style: ``cam_elevation`` (-10),
        ``cam_azimuth`` (90, the ``MjvCamera()`` default the recorder has always
        inherited), and ``cam_lookat_dz``, which the auto-fit overwrites with the
        offset that centres the robot + object instead of the pelvis.
        """
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        if lookat is None:
            lookat = self.data.xpos[self.model.body(self.robot_base).id]
        cam.lookat[:] = np.asarray(lookat, dtype=float)
        cam.lookat[2] += self.cam_lookat_dz
        cam.distance = self.viewer_cam_distance
        cam.elevation = self.style.cam_elevation
        cam.azimuth = self.style.cam_azimuth
        return cam

    def step(self,
            # robot data
            root_pos, root_rot, dof_pos,
            # human data
            human_motion_data=None,
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization
            human_pos_offset=np.array([0.0, 0.0, 0]),
            object_data=None,
            object_contact_points_world=None,
            per_link_contact_data=None,
            # rate limit
            rate_limit=True,
            follow_camera=True,
            # optional pre-computed (e.g. smoothed) camera lookat for this frame
            camera_lookat=None,
            # render and return the frame even when not recording
            capture=False,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.

        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.

        Returns the rendered RGB frame when recording (or when ``capture``), else None.
        """

        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos

        if object_data is not None:
            self.data.mocap_pos[self.object_mocap_id] = object_data[0]
            self.data.mocap_quat[self.object_mocap_id] = object_data[1]
            if DEBUG_CONTACT and self.object_geom_id is not None:
                body_contact = bool(object_data[2]) if len(object_data) > 2 else False
                ground_contact = bool(object_data[3]) if len(object_data) > 3 else False
                if body_contact and ground_contact:
                    # Both: magenta
                    self._set_object_color((0.95, 0.2, 0.95))
                elif body_contact:
                    # Robot touching object: red (existing behaviour)
                    self._set_object_color((0.95, 0.2, 0.2))
                elif ground_contact:
                    # Ground contact only: blue
                    self._set_object_color((0.2, 0.4, 0.95))
                elif self.object_rgba_default is not None:
                    self._set_object_color(None)
            if self.style.styled and self.style.object_trail:
                self._object_trail.append(np.asarray(object_data[0], dtype=float).copy())
                if len(self._object_trail) > self.style.trail_len:
                    self._object_trail.pop(0)

        # The light rig follows the robot; light_xpos is derived in mj_forward.
        render_style.update_lights(self.model, self.style, self.light_ids, root_pos)

        mj.mj_forward(self.model, self.data)

        if self.viewer is not None:
            if follow_camera:
                lookat = np.array(
                    self.data.xpos[self.model.body(self.robot_base).id]
                    if camera_lookat is None else np.asarray(camera_lookat, dtype=float),
                    dtype=float,
                )
                lookat[2] += self.cam_lookat_dz
                self.viewer.cam.lookat = lookat
                self.viewer.cam.distance = self.viewer_cam_distance
                self.viewer.cam.elevation = self.style.cam_elevation  # 正面视角，轻微向下看
                # self.viewer.cam.azimuth = 180    # 正面朝向机器人

            self.viewer.user_scn.ngeom = 0
            self._draw_custom_geoms(
                self.viewer.user_scn,
                object_data=object_data,
                human_motion_data=human_motion_data,
                human_point_scale=human_point_scale,
                human_pos_offset=human_pos_offset,
                show_human_body_name=show_human_body_name,
                object_contact_points_world=object_contact_points_world,
                per_link_contact_data=per_link_contact_data,
            )

        if self.viewer is not None:
            self.viewer.sync()
        if rate_limit is True and not self.headless:
            self.rate_limiter.sleep()

        if self.record_video or capture:
            cam = (self.viewer.cam if self.viewer is not None
                   else self._make_tracking_camera(camera_lookat))
            if self.scene_option is None:
                self.renderer.update_scene(self.data, camera=cam)
            else:
                self.renderer.update_scene(self.data, camera=cam,
                                           scene_option=self.scene_option)
            render_style.apply_scene_flags(self.renderer.scene, self.style)
            self._draw_custom_geoms(
                self.renderer.scene,
                object_data=object_data,
                human_motion_data=None,  # body frames/labels not shown in recording
                human_point_scale=human_point_scale,
                human_pos_offset=human_pos_offset,
                show_human_body_name=False,
                object_contact_points_world=object_contact_points_world,
                per_link_contact_data=per_link_contact_data,
            )
            img = render_style.downscale(self.renderer.render(), self.style)
            if self.record_video:
                self.mp4_writer.append_data(img)
            return img
        return None

    def _draw_custom_geoms(
        self,
        scene,
        object_data,
        human_motion_data,
        human_point_scale,
        human_pos_offset,
        show_human_body_name,
        object_contact_points_world,
        per_link_contact_data=None,
    ):
        if human_motion_data is not None:
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    scene,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None,
                )
        if DEBUG_CONTACT and object_contact_points_world is not None:
            if isinstance(object_contact_points_world, dict):
                contact_points_iter = object_contact_points_world.values()
            else:
                contact_points_iter = object_contact_points_world
            for contact_pos in contact_points_iter:
                draw_sphere(pos=np.asarray(contact_pos, dtype=float), scene=scene)

        # Fading trail behind the object (contact_overlay style).
        if self.style.styled and self.style.object_trail and len(self._object_trail) >= 2:
            n = len(self._object_trail)
            for i in range(n - 1):
                a = 0.05 + 0.32 * (i / max(n - 2, 1))
                draw_capsule(self._object_trail[i], self._object_trail[i + 1], scene,
                             width=0.0055, rgba=(0.18, 0.48, 0.90, a))

        # Per-link contact debug: green sphere on active links, grey on inactive,
        # cyan line from active link to its contact point on the object.
        if self.show_per_link_contact and per_link_contact_data is not None:
            for entry in per_link_contact_data:
                link_pos = entry["link_pos"]       # robot link world position
                contact_pt = entry.get("contact_point_world")  # point on object
                is_active = entry["active"]        # bool: link in contact this frame
                if is_active:
                    # Active contact link: bright green sphere.
                    #* For debugging, make radius larger if the link hides the spheres. 
                    draw_sphere(pos=link_pos, scene=scene, radius=0.03,
                                rgba=(0.1, 0.95, 0.2, 1.0))
                    # Cyan line from link to contact point on object.
                    if contact_pt is not None:
                        draw_line(link_pos, contact_pt, scene,
                                  width=0.006, rgba=(0.0, 0.9, 0.9, 0.9))
                        draw_sphere(pos=contact_pt, scene=scene, radius=0.02,
                                    rgba=(0.0, 0.9, 0.9, 1.0))
                else:
                    # Inactive: dim grey sphere (smaller).
                    draw_sphere(pos=link_pos, scene=scene, radius=0.015,
                                rgba=(0.5, 0.5, 0.5, 0.4))

    def render_frame(self, root_pos, root_rot, dof_pos, object_data=None):
        """Render a single frame offscreen and return RGB array (H, W, 3).

        Uses identical camera tracking and contact-debug geometry as step().
        Works in headless mode without an interactive viewer.

        Args:
            root_pos: (3,) base position.
            root_rot: (4,) quaternion wxyz.
            dof_pos: (ndof,) joint positions.
            object_data: optional tuple (obj_pos, obj_rot_wxyz, contact, ground_contact,
                         min_height). Only first two required; rest default to 0/False/None.

        Returns:
            np.ndarray of shape (H, W, 3) uint8 RGB.
        """
        # Set robot pose
        self.data.qpos[:3] = root_pos
        w, x, y, z = root_rot
        self.data.qpos[3:7] = [w, x, y, z]
        n_dof = min(len(dof_pos), len(self.data.qpos) - 7)
        self.data.qpos[7:7 + n_dof] = dof_pos[:n_dof]

        # Set object mocap + colour-code contacts (same as step())
        if object_data is not None and self.object_mocap_id is not None:
            self.data.mocap_pos[self.object_mocap_id] = object_data[0]
            w2, x2, y2, z2 = object_data[1]
            self.data.mocap_quat[self.object_mocap_id] = [w2, x2, y2, z2]
            if DEBUG_CONTACT and self.object_geom_id is not None:
                body_contact = bool(object_data[2]) if len(object_data) > 2 else False
                ground_contact = bool(object_data[3]) if len(object_data) > 3 else False
                if body_contact and ground_contact:
                    self._set_object_color((0.95, 0.2, 0.95))
                elif body_contact:
                    self._set_object_color((0.95, 0.2, 0.2))
                elif ground_contact:
                    self._set_object_color((0.2, 0.4, 0.95))
                else:
                    self._set_object_color(None)

        mj.mj_forward(self.model, self.data)

        # Camera: track robot base at same angle as interactive viewer
        cam = self._make_tracking_camera()

        if self.scene_option is None:
            self.renderer.update_scene(self.data, camera=cam)
        else:
            self.renderer.update_scene(self.data, camera=cam,
                                       scene_option=self.scene_option)
        render_style.apply_scene_flags(self.renderer.scene, self.style)

        # Draw contact-debug geoms into renderer scene
        self._draw_custom_geoms(
            self.renderer.scene,
            object_data=object_data,
            human_motion_data=None,
            human_point_scale=0.1,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            object_contact_points_world=None,
            per_link_contact_data=None,
        )

        return render_style.downscale(self.renderer.render(), self.style)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
        for path in (self.runtime_model_xml_path, self.styled_model_xml_path):
            if path is None:
                continue
            try:
                os.remove(path)
            except OSError:
                pass
            _RUNTIME_XML_PATHS.discard(path)
