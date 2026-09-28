# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named render styles for the MuJoCo recorders.

A :class:`Style` holds every visual setting, :data:`PRESETS` maps names to styles
(default ``square``), and ``patch_model_xml`` / ``apply_*`` apply one. Shared by
``RobotMotionViewer`` and ``SmplSourceRenderer``; :func:`at_size` resizes a preset.

Presets: ``square`` (1024 px, fixed distance), ``square_auto`` (per-clip framing via
:func:`fit_camera`), ``studio`` / ``studio_ss2`` (16:9 1080p, 1x / 2x supersample),
``paper_white``, ``dark_studio``, ``contact_overlay``, ``resolution_only`` (MuJoCo's
look at 1080p) and ``legacy`` (MuJoCo defaults at 640x480).
"""

from __future__ import annotations

import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace

import mujoco as mj

DEFAULT_STYLE = "square"

#: Side of the square presets; ``at_size`` overrides it.
SQUARE_PX = 1024


# --------------------------------------------------------------------------- #
# The style
# --------------------------------------------------------------------------- #
@dataclass
class Style:
    """All visual settings of one render variant."""

    name: str = "studio"

    # False: MJCF lighting, scene and materials stay as-is; object colour/alpha still apply.
    styled: bool = True

    # --- framebuffer / sampling ---
    width: int = 1920
    height: int = 1080
    supersample: int = 1  # render at N*W x N*H then Lanczos-downscale
    offsamples: int = 8  # MSAA samples in the offscreen framebuffer
    shadowsize: int = 4096
    shadowscale: float = 0.8  # shadow frustum vs. spot cutoff (smaller = sharper)
    shadowclip: float = 4.0  # shadow box vs. model extent (directional lights)

    # --- lighting ---
    replace_lights: bool = True
    headlight_diffuse: float = 0.05
    headlight_ambient: float = 0.30
    headlight_specular: float = 0.0
    key_diffuse: float = 0.70
    key_specular: float = 0.18
    fill_diffuse: float = 0.28
    rim_diffuse: float = 0.38
    key_offset: tuple = (2.2, -2.6, 3.4)  # relative to the robot pelvis
    fill_offset: tuple = (-3.0, -1.6, 1.7)
    rim_offset: tuple = (-1.4, 3.2, 2.6)
    shadows: bool = True
    lights_follow_robot: bool = True
    key_directional: bool = False  # directional key: even floor lighting, no moving light pool

    # --- background / floor ---
    replace_scene: bool = True
    sky_top: tuple = (0.62, 0.68, 0.78)
    sky_bottom: tuple = (0.97, 0.97, 0.98)
    floor_rgb1: tuple = (0.88, 0.885, 0.90)  # base tone
    floor_rgb2: tuple = (0.845, 0.855, 0.875)  # alternating tile tone
    floor_markrgb: tuple = (0.70, 0.72, 0.76)  # grid line colour
    floor_texrepeat: float = 3.0
    floor_extent: float = 0.0  # half-size (m) forced onto the floor plane; 0 = leave it
    # Clip planes, as multiples of the model extent (0.8 m). At zfar=30 the 24 m cut
    # lands before the horizon, so the sky/floor line moves with camera height.
    znear: float = 0.02
    zfar: float = 30.0
    floor_reflectance: float = 0.06
    floor_specular: float = 0.12
    floor_shininess: float = 0.2
    haze: bool = False
    reflection: bool = True
    skybox: bool = True

    # --- materials ---
    robot_specular: float = 0.22
    robot_shininess: float = 0.30
    robot_light_rgb: tuple | None = (0.82, 0.83, 0.85)
    robot_dark_rgb: tuple | None = (0.17, 0.18, 0.20)
    object_rgb: tuple | None = (0.90, 0.55, 0.16)  # None => keep the URDF's own colour
    object_alpha: float = 0.92
    object_specular: float = 0.18
    object_shininess: float = 0.25
    hide_collision_geoms: bool = True

    # --- overlays ---
    contact_overlay: bool = False
    object_trail: bool = False
    trail_len: int = 45

    # --- camera ---
    smooth_camera: bool = True
    smooth_window: int = 21
    cam_distance_scale: float = 1.0  # times VIEWER_CAM_DISTANCE_DICT[robot] * 1.5
    cam_elevation: float = -10.0  # degrees, negative looks down (more floor)
    cam_azimuth: float = 90.0  # MjvCamera() default
    cam_lookat_dz: float = 0.0  # metres added to the lookat height (pelvis)
    # Per-clip auto-framing; see fit_camera.
    cam_fit: bool = False
    cam_fit_track: str = "pelvis"  # "pelvis" | "content" (aim between robot+object)
    cam_fit_margin: float = 1.10  # >1 leaves slack around the content
    cam_fit_pad: float = 0.12  # sphere radius carried by each link point (m)
    cam_fit_bias: float = 0.0  # >0 pushes the content down (more headroom)
    cam_fit_min: float = 1.4  # clamps on the fitted distance (m)
    cam_fit_max: float = 3.6
    cam_fit_dolly: bool = False  # let the distance drift instead of being fixed
    cam_fit_dolly_sec: float = 2.0  # how slowly it may drift
    cam_fit_pan_sec: float = 1.5  # smoothing window of the "content" lookat

    # --- encoding ---
    crf: int | None = 16  # None => imageio's default encoder settings


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
def _preset_studio() -> Style:
    """Neutral light-grey set, 3-point rig, 1 m grid, sharp key shadow."""
    return Style(
        name="studio",
        shadowsize=8192,
        shadowscale=0.3,
        key_directional=True,
        headlight_ambient=0.34,
        key_diffuse=0.62,
        fill_diffuse=0.28,
        rim_diffuse=0.38,
        sky_top=(0.44, 0.52, 0.64),
        sky_bottom=(0.92, 0.94, 0.96),
        floor_rgb1=(0.66, 0.67, 0.69),
        floor_rgb2=(0.66, 0.67, 0.69),
        floor_markrgb=(0.56, 0.57, 0.60),
        floor_texrepeat=1.0,
        floor_reflectance=0.0,
        floor_specular=0.04,
        floor_shininess=0.1,
        reflection=False,
        # Far clip past the horizon, so the sky/floor line is fixed (see Style.znear).
        znear=0.06,
        zfar=400.0,
    )


def _preset_studio_ss2() -> Style:
    """``studio`` rendered at 2x and Lanczos-downscaled."""
    return replace(_preset_studio(), name="studio_ss2", supersample=2)


def _preset_square() -> Style:
    """1:1 at a fixed distance (3.0 m for G1), camera near level; clips are size-comparable."""
    return replace(
        _preset_studio(), name="square", supersample=2, width=SQUARE_PX, height=SQUARE_PX, cam_elevation=-4.0
    )


def _preset_square_auto() -> Style:
    """1:1, aimed at the robot + object and framed per clip with a ~2 s dolly.

    Never crops, but framing differs per clip, so clips are not size-comparable.
    """
    return replace(
        _preset_square(),
        name="square_auto",
        cam_fit=True,
        cam_fit_track="content",
        cam_fit_margin=1.08,
        cam_fit_dolly=True,
    )


def _preset_paper_white() -> Style:
    """White floor and sky, for print figures."""
    return replace(
        _preset_studio(),
        name="paper_white",
        supersample=2,
        headlight_ambient=0.40,
        key_diffuse=0.60,
        fill_diffuse=0.32,
        rim_diffuse=0.30,
        sky_top=(1.0, 1.0, 1.0),
        sky_bottom=(1.0, 1.0, 1.0),
        floor_rgb1=(0.985, 0.985, 0.99),
        floor_rgb2=(0.985, 0.985, 0.99),
        floor_markrgb=(0.985, 0.985, 0.99),
        floor_specular=0.0,
        floor_shininess=0.0,
        robot_light_rgb=(0.72, 0.735, 0.765),
        robot_dark_rgb=(0.19, 0.20, 0.22),
        robot_specular=0.16,
        robot_shininess=0.25,
        object_rgb=(0.88, 0.47, 0.11),
        object_alpha=1.0,
    )


def _preset_dark_studio() -> Style:
    """Charcoal set with a reflective floor."""
    return replace(
        _preset_studio(),
        name="dark_studio",
        supersample=2,
        headlight_ambient=0.22,
        key_diffuse=0.80,
        key_specular=0.30,
        fill_diffuse=0.22,
        rim_diffuse=0.55,
        sky_top=(0.05, 0.06, 0.08),
        sky_bottom=(0.17, 0.19, 0.23),
        floor_rgb1=(0.105, 0.115, 0.135),
        floor_rgb2=(0.105, 0.115, 0.135),
        floor_markrgb=(0.20, 0.22, 0.27),
        floor_reflectance=0.28,
        floor_specular=0.35,
        floor_shininess=0.5,
        reflection=True,
        robot_light_rgb=(0.80, 0.81, 0.84),
        robot_dark_rgb=(0.13, 0.14, 0.16),
        robot_specular=0.32,
        robot_shininess=0.42,
        object_rgb=(0.95, 0.58, 0.13),
        object_alpha=1.0,
    )


def _preset_contact_overlay() -> Style:
    """Studio + translucent object + contact markers + object trail."""
    return replace(
        _preset_studio(),
        name="contact_overlay",
        supersample=2,
        object_alpha=0.42,
        contact_overlay=True,
        object_trail=True,
    )


def _preset_resolution_only() -> Style:
    """MuJoCo's own lighting and scene at 1080p."""
    return Style(
        name="resolution_only",
        replace_lights=False,
        replace_scene=False,
        offsamples=4,
        robot_light_rgb=None,
        robot_dark_rgb=None,
        robot_specular=-1.0,  # <0 => do not touch robot materials
        object_rgb=None,  # keep the URDF's own colour
        object_alpha=0.3,
        haze=True,
        reflection=False,
        hide_collision_geoms=False,
        smooth_camera=False,
        shadows=True,
    )


def _preset_legacy() -> Style:
    """MuJoCo defaults at 640x480, 30 %-opacity object, imageio's default encoder."""
    return Style(
        name="legacy",
        styled=False,
        width=640,
        height=480,
        supersample=1,
        object_rgb=None,
        object_alpha=0.3,
        smooth_camera=False,
        crf=None,
    )


PRESETS = {
    "studio_ss2": _preset_studio_ss2,
    "studio": _preset_studio,
    "square": _preset_square,
    "square_auto": _preset_square_auto,
    "paper_white": _preset_paper_white,
    "dark_studio": _preset_dark_studio,
    "contact_overlay": _preset_contact_overlay,
    "resolution_only": _preset_resolution_only,
    "legacy": _preset_legacy,
}

STYLE_NAMES = sorted(PRESETS)


def at_size(style: str | Style | None, px: int, height: int | None = None, supersample: int | None = None) -> Style:
    """The same look at width ``px`` (height defaults to ``px``).

    ``supersample`` defaults to the style's own up to 1440 px and 1 above, where the
    supersampled MSAA framebuffer gets too large.
    """
    style = get_style(style)
    if supersample is None:
        supersample = style.supersample if px <= 1440 else 1
    w, h = int(px), int(height or px)
    w, h = w + (w % 2), h + (h % 2)  # libx264/yuv420p rejects odd dimensions
    return replace(style, width=w, height=h, supersample=int(supersample))


def get_style(style: str | Style | None = None) -> Style:
    """Resolve ``None`` / a preset name / a :class:`Style` into a Style."""
    if style is None:
        style = DEFAULT_STYLE
    if isinstance(style, Style):
        return style
    try:
        return PRESETS[style]()
    except KeyError:
        raise ValueError(f"unknown render style {style!r}; choose from {', '.join(STYLE_NAMES)}") from None


# --------------------------------------------------------------------------- #
# Applying a style: MJCF
# --------------------------------------------------------------------------- #
# Names of the assets patch_model_xml injects, looked up again after compilation.
MAT_ROBOT_LIGHT = "pub_robot_light"
MAT_ROBOT_DARK = "pub_robot_dark"
MAT_OBJECT = "pub_object"
LIGHT_KEY = "pub_key"
LIGHT_FILL = "pub_fill"
LIGHT_RIM = "pub_rim"


def _rgb(t) -> str:
    return " ".join(f"{float(v):.4f}" for v in t)


def absolutize_asset_paths(root: ET.Element, xml_path: str) -> None:
    """Make every asset ``file`` absolute, as MuJoCo would resolve it, so the MJCF can move."""
    base = os.path.dirname(os.path.abspath(xml_path))
    comp = root.find("compiler")
    dirs = {k: (comp.attrib.pop(k, None) if comp is not None else None) for k in ("assetdir", "meshdir", "texturedir")}
    mesh_dir = os.path.join(base, dirs["meshdir"] or dirs["assetdir"] or "")
    tex_dir = os.path.join(base, dirs["texturedir"] or dirs["assetdir"] or "")
    for el in root.iter():
        f = el.get("file")
        if f and not os.path.isabs(f):
            d = mesh_dir if el.tag in ("mesh", "skin", "hfield") else tex_dir if el.tag == "texture" else base
            el.set("file", os.path.normpath(os.path.join(d, f)))


def patch_model_xml(xml_path: str, style: Style) -> str:
    """Write a styled copy of the MJCF and return its temp path.

    Returns ``xml_path`` itself for an unstyled style; delete the result only if it differs.
    The G1 MJCF has two ``<asset>`` and two ``<worldbody>`` blocks, so all are scanned.
    """
    if not style.styled:
        return xml_path

    tree = ET.parse(xml_path)
    root = tree.getroot()
    absolutize_asset_paths(root, xml_path)

    assets = root.findall("asset")
    worldbodies = root.findall("worldbody")
    if not assets:
        assets = [ET.SubElement(root, "asset")]
    if not worldbodies:
        raise ValueError("no <worldbody> in model")
    asset = assets[-1]
    worldbody = worldbodies[-1]

    # ---- <visual>: framebuffer size, MSAA, shadow map, haze ---------------- #
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")

    glob = vis.find("global")
    if glob is None:
        glob = ET.SubElement(vis, "global")
    glob.set("offwidth", str(style.width * style.supersample))
    glob.set("offheight", str(style.height * style.supersample))

    quality = vis.find("quality")
    if quality is None:
        quality = ET.SubElement(vis, "quality")
    quality.set("offsamples", str(style.offsamples))
    quality.set("shadowsize", str(style.shadowsize))
    quality.set("numslices", "40")
    quality.set("numstacks", "40")
    quality.set("numquads", "6")

    vmap = vis.find("map")
    if vmap is None:
        vmap = ET.SubElement(vis, "map")
    # The shadow frustum is sized from the 0.8 m model extent; widen it or the robot leaves it.
    vmap.set("shadowclip", str(style.shadowclip))
    vmap.set("shadowscale", str(style.shadowscale))
    vmap.set("znear", str(style.znear))
    vmap.set("zfar", str(style.zfar))
    if not style.haze:
        vmap.set("haze", "0.0")

    head = vis.find("headlight")
    if head is None:
        head = ET.SubElement(vis, "headlight")
    if style.replace_lights:
        head.set("diffuse", _rgb((style.headlight_diffuse,) * 3))
        head.set("ambient", _rgb((style.headlight_ambient,) * 3))
        head.set("specular", _rgb((style.headlight_specular,) * 3))
        head.set("active", "1")

    if style.replace_scene:
        rgba_el = vis.find("rgba")
        if rgba_el is None:
            rgba_el = ET.SubElement(vis, "rgba")
        rgba_el.set("haze", _rgb(style.sky_bottom) + " 1")

    # Floor to group 1 so group 0 (collision geoms) can be hidden.
    floor_geom = None
    for wb in worldbodies:
        for g in wb.findall("geom"):
            if g.get("name") == "floor" or g.get("type") == "plane":
                floor_geom = g
    if floor_geom is not None:
        floor_geom.set("group", "1")
        if style.floor_extent > 0:
            e = style.floor_extent
            floor_geom.set("size", f"{e} {e} 0.01")

    if style.replace_scene:
        # Drop the existing skybox and ground texture/material.
        for a in assets:
            for tex in list(a.findall("texture")):
                if tex.get("type") == "skybox" or tex.get("name") in (
                    "groundplane",
                    "texplane",
                ):
                    a.remove(tex)
            for matel in list(a.findall("material")):
                if matel.get("name") in ("groundplane", "MatPlane"):
                    a.remove(matel)

        if style.skybox:
            ET.SubElement(
                asset,
                "texture",
                {
                    "type": "skybox",
                    "builtin": "gradient",
                    "rgb1": _rgb(style.sky_top),
                    "rgb2": _rgb(style.sky_bottom),
                    "width": "512",
                    "height": "512",
                },
            )
        ET.SubElement(
            asset,
            "texture",
            {
                "type": "2d",
                "name": "pub_ground",
                "builtin": "checker",
                "mark": "edge",
                "markrgb": _rgb(style.floor_markrgb),
                "rgb1": _rgb(style.floor_rgb1),
                "rgb2": _rgb(style.floor_rgb2),
                "width": "1024",
                "height": "1024",
            },
        )
        ET.SubElement(
            asset,
            "material",
            {
                "name": "pub_ground",
                "texture": "pub_ground",
                "texuniform": "true",
                "texrepeat": f"{style.floor_texrepeat} {style.floor_texrepeat}",
                "reflectance": str(style.floor_reflectance),
                "specular": str(style.floor_specular),
                "shininess": str(style.floor_shininess),
            },
        )
        if floor_geom is not None:
            floor_geom.set("material", "pub_ground")

    # ---- robot / object materials ---------------------------------------- #
    if style.robot_specular >= 0.0:
        for nm in (MAT_ROBOT_LIGHT, MAT_ROBOT_DARK):
            ET.SubElement(
                asset,
                "material",
                {
                    "name": nm,
                    "specular": str(style.robot_specular),
                    "shininess": str(style.robot_shininess),
                    "reflectance": "0",
                },
            )
    ET.SubElement(
        asset,
        "material",
        {
            "name": MAT_OBJECT,
            "specular": str(style.object_specular),
            "shininess": str(style.object_shininess),
            "reflectance": "0",
        },
    )

    # ---- lights ----------------------------------------------------------- #
    if style.replace_lights:
        for wb in worldbodies:
            for lt in list(wb.findall("light")):
                wb.remove(lt)

        # Key (shadow caster), fill, rim; update_lights moves them with the robot.
        def _light(name, offset, diffuse, specular, castshadow, directional):
            off = np.asarray(offset, dtype=float)
            attrs = {
                "name": name,
                "pos": _rgb(off),
                "dir": _rgb(-off / np.linalg.norm(off)),
                "diffuse": _rgb((diffuse,) * 3),
                "specular": _rgb((specular,) * 3),
                "ambient": "0 0 0",
                "castshadow": "true" if castshadow else "false",
            }
            if directional:
                attrs["directional"] = "true"
            else:
                attrs["cutoff"] = "120"
                attrs["exponent"] = "1"
            ET.SubElement(worldbody, "light", attrs)

        _light(LIGHT_KEY, style.key_offset, style.key_diffuse, style.key_specular, style.shadows, style.key_directional)
        _light(LIGHT_FILL, style.fill_offset, style.fill_diffuse, 0.0, False, True)
        _light(LIGHT_RIM, style.rim_offset, style.rim_diffuse, 0.0, False, True)

    fd, out_path = tempfile.mkstemp(prefix="styled_render_", suffix=".xml")
    os.close(fd)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


# --------------------------------------------------------------------------- #
# Applying a style: compiled model, scene, image, writer
# --------------------------------------------------------------------------- #
def apply_to_model(model, style: Style, object_geom_id: int | None = None) -> dict:
    """Recolour the object and the robot in a compiled model.

    Returns ``{light_name: light_id}`` of the installed lights for :func:`update_lights`.
    """
    # Object alpha always; colour only if untextured (textured meshes keep their texture).
    if object_geom_id is not None and object_geom_id >= 0:
        rgba = model.geom_rgba[object_geom_id]
        textured = int(model.geom_matid[object_geom_id]) >= 0
        if style.object_rgb is not None and not textured:
            rgba[:3] = style.object_rgb
            mat_obj = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, MAT_OBJECT)
            if mat_obj >= 0:
                model.geom_matid[object_geom_id] = mat_obj
        rgba[3] = style.object_alpha

    if not style.styled:
        return {}

    # Robot: two-tone recolour + a softer material.
    mat_light = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, MAT_ROBOT_LIGHT)
    mat_dark = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, MAT_ROBOT_DARK)
    touch_rgb = style.robot_light_rgb is not None or style.robot_dark_rgb is not None
    if touch_rgb or (mat_light >= 0 and mat_dark >= 0):
        floor_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
        for gid in range(model.ngeom):
            if gid == object_geom_id or gid == floor_gid:
                continue
            is_light = float(model.geom_rgba[gid][:3].mean()) > 0.45
            if is_light and style.robot_light_rgb is not None:
                model.geom_rgba[gid][:3] = style.robot_light_rgb
            elif (not is_light) and style.robot_dark_rgb is not None:
                model.geom_rgba[gid][:3] = style.robot_dark_rgb
            if mat_light >= 0 and mat_dark >= 0:
                model.geom_matid[gid] = mat_light if is_light else mat_dark

    light_ids = {}
    for nm in (LIGHT_KEY, LIGHT_FILL, LIGHT_RIM):
        lid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_LIGHT, nm)
        if lid >= 0:
            light_ids[nm] = lid
    return light_ids


def light_offsets(style: Style) -> dict:
    return {
        LIGHT_KEY: np.asarray(style.key_offset, dtype=float),
        LIGHT_FILL: np.asarray(style.fill_offset, dtype=float),
        LIGHT_RIM: np.asarray(style.rim_offset, dtype=float),
    }


def update_lights(model, style: Style, light_ids: dict, base_pos) -> None:
    """Move the light rig with the robot. Call before ``mj_forward``."""
    if not (style.styled and style.lights_follow_robot) or not light_ids:
        return
    base_pos = np.asarray(base_pos, dtype=float)
    anchor = np.array([base_pos[0], base_pos[1], 0.0])
    offsets = light_offsets(style)
    for nm, lid in light_ids.items():
        off = offsets[nm]
        model.light_pos[lid] = anchor + off
        model.light_dir[lid] = -off / np.linalg.norm(off)


def make_scene_option(style: Style):
    """The ``MjvOption`` for this style, or ``None`` to use MuJoCo's default."""
    if not style.styled:
        return None
    opt = mj.MjvOption()
    mj.mjv_defaultOption(opt)
    if style.hide_collision_geoms:
        # Group 0 duplicates the visual meshes as collision geoms and would z-fight.
        opt.geomgroup[0] = 0
    opt.flags[mj.mjtVisFlag.mjVIS_SKIN] = 1
    return opt


def apply_scene_flags(scene, style: Style) -> None:
    """Set the render flags on a freshly-updated ``MjvScene``."""
    if not style.styled:
        return
    scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 1 if style.shadows else 0
    scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 1 if style.reflection else 0
    scene.flags[mj.mjtRndFlag.mjRND_SKYBOX] = 1 if style.skybox else 0
    scene.flags[mj.mjtRndFlag.mjRND_HAZE] = 1 if style.haze else 0
    scene.flags[mj.mjtRndFlag.mjRND_FOG] = 0


def downscale(img: np.ndarray, style: Style) -> np.ndarray:
    """Lanczos-downscale a supersampled frame to the style's output size."""
    if style.supersample <= 1:
        return img
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((style.width, style.height), Image.LANCZOS))


def open_video_writer(path: str, fps: float, style: Style):
    """imageio writer with this style's encoder settings."""
    import imageio.v2 as imageio

    if style.crf is None:
        return imageio.get_writer(path, fps=fps)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        macro_block_size=1,  # 1080 is not a multiple of 16; do not let imageio pad
        # Here, not in ffmpeg_params: imageio adds its own -pix_fmt and ffmpeg warns on duplicates.
        pixelformat="yuv420p",
        ffmpeg_params=[
            "-crf",
            str(style.crf),
            "-preset",
            "slow",
            "-profile:v",
            "high",
            "-movflags",
            "+faststart",
        ],
    )


def camera_basis(style: Style):
    """``(forward, right, up)`` unit vectors of the free camera; ``eye = lookat - distance * forward``."""
    e = np.deg2rad(style.cam_elevation)
    a = np.deg2rad(style.cam_azimuth)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


def _window(sec: float, fps: float, n: int) -> int:
    win = max(3, round(sec * max(fps, 1.0)) // 2 * 2 + 1)
    return min(win, max(3, n // 2 * 2 + 1)) if n >= 3 else 1


def _movavg(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or a.shape[0] < 3:
        return a
    pad = win // 2
    k = np.ones(win) / win
    xp = np.pad(a, ((pad, pad),) + ((0, 0),) * (a.ndim - 1), mode="edge")
    if a.ndim == 1:
        return np.convolve(xp, k, mode="valid")
    return np.stack([np.convolve(xp[:, i], k, mode="valid") for i in range(a.shape[1])], axis=1)


def _safe_dolly(need: np.ndarray, win: int) -> np.ndarray:
    """Max-filter then moving-average ``need``: smooth, and never below the raw requirement."""
    if win <= 1 or need.shape[0] < 3:
        return need
    pad = win // 2
    xp = np.pad(need, pad, mode="edge")
    dilated = np.array([xp[i : i + win].max() for i in range(need.shape[0])])
    return _movavg(dilated, win)


def _needed_distance(points, lookat, style, aspect, fovy_deg):
    """(T,) smallest distance that keeps every point of each frame in frame."""
    fwd, right, up = camera_basis(style)
    w = points - lookat[:, None, :]
    ty = np.tan(np.deg2rad(fovy_deg) / 2.0)
    tx = ty * aspect
    pad, m = float(style.cam_fit_pad), float(style.cam_fit_margin)
    return (np.maximum((m * np.abs(w @ right) + pad) / tx, (m * np.abs(w @ up) + pad) / ty) - (w @ fwd)).max(axis=1)


def fit_camera(
    points: np.ndarray, lookat: np.ndarray, style: Style, aspect: float, fovy_deg: float = 45.0, fps: float = 30.0
):
    """Solve a camera ``(distance, lookat)`` that keeps ``points`` in frame.

    ``points``: ``(T, N, 3)`` world content; ``lookat``: ``(T, 3)`` pelvis track.
    Distance is a scalar, or ``(T,)`` with ``cam_fit_dolly``. ``cam_fit_track="pelvis"``
    keeps the pelvis track plus one height offset; ``"content"`` aims at the smoothed
    centre of the content. Distance is solved after the aim, so nothing crops.
    """
    pts = np.asarray(points, dtype=float)
    base = np.asarray(lookat, dtype=float)
    fwd, right, up = camera_basis(style)
    pad = float(style.cam_fit_pad)
    n = pts.shape[0]

    if style.cam_fit_track == "content":
        centre = 0.5 * (pts.min(axis=1) + pts.max(axis=1))
        track = _movavg(centre, _window(style.cam_fit_pan_sec, fps, n))
        height = _movavg(((pts @ up).max(axis=1) - (pts @ up).min(axis=1)), _window(style.cam_fit_pan_sec, fps, n))
        track[:, 2] += style.cam_fit_bias * 0.5 * height / up[2]
    else:
        # A point's screen height is distance-independent, so dz has a closed form.
        y = (pts - base[:, None, :]) @ up
        lo, hi = float(y.min() - pad), float(y.max() + pad)
        dz = (0.5 * (lo + hi) + style.cam_fit_bias * 0.5 * (hi - lo)) / up[2]
        track = base.copy()
        track[:, 2] += dz

    need = _needed_distance(pts, track, style, aspect, fovy_deg)
    if style.cam_fit_dolly:
        d = np.clip(_safe_dolly(need, _window(style.cam_fit_dolly_sec, fps, n)), style.cam_fit_min, style.cam_fit_max)
    else:
        d = float(np.clip(need.max(), style.cam_fit_min, style.cam_fit_max))
    return d, track


def smooth_track(track: np.ndarray, style: Style) -> np.ndarray:
    """Moving-average an (N, 3) lookat track to remove the pelvis's gait bob."""
    if not (style.styled and style.smooth_camera):
        return track
    win = style.smooth_window
    if win <= 1 or track.shape[0] < 3:
        return track
    win = min(win, track.shape[0] // 2 * 2 + 1)
    if win < 3:
        return track
    pad = win // 2
    xp = np.pad(track, ((pad, pad), (0, 0)), mode="edge")
    k = np.ones(win) / win
    return np.stack([np.convolve(xp[:, i], k, mode="valid") for i in range(track.shape[1])], axis=1)
