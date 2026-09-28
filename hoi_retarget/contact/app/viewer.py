# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``ContactShifterViewer``: the viser editor for per-strip contact points.

Everything object-attached lives under ``/obj_frame``, re-posed every frame so the
markers, the translate gizmo (selected strip) and the end-effector mesh ride the object.
Edits are held in a ``ShiftSet`` in real-size object-frame metres; the scaled view
shows them at ``object_scale`` x the size slider, which is also the scale Go solves at.
"""

from __future__ import annotations

import numpy as np
import os
import threading
import time

import viser
from scipy.spatial.transform import Rotation as R

from hoi_retarget.contact.app.session import ShifterSession
from hoi_retarget.contact.shifts import Shift, ShiftSet
from hoi_retarget.paths import CONTACT_EDITOR_OUTPUT
from hoi_retarget.rendering.viser_viewer import reserve_port

OUTPUT_ROOT = str(CONTACT_EDITOR_OUTPUT)

# Colors (RGB 0-255).
C_HUMAN = (235, 200, 170)
C_OBJECT = (175, 178, 190)
C_LEFT = (60, 120, 255)
C_RIGHT = (255, 95, 95)
C_SEL = (255, 225, 0)
C_DISABLED = (110, 110, 115)
C_EE = (120, 125, 135)


def _rotmat(wxyz) -> np.ndarray:
    return R.from_quat(np.asarray(wxyz, float)[[1, 2, 3, 0]]).as_matrix()


class ContactShifterViewer:
    def __init__(self, session: ShifterSession, host: str = "127.0.0.1", port: int = 0, out_dir: str | None = None):
        self.session = session
        self.out_dir = out_dir or os.path.join(OUTPUT_ROOT, session.stem)
        self.host = host
        self.port = reserve_port(host, port)
        self.server = viser.ViserServer(host=host, port=self.port)
        self.port = self.server.get_port()  # viser rebinds silently if this port is busy
        self.server.scene.set_up_direction("+z")

        # ── editable state ───────────────────────────────────────────────
        self.world_mode = "real"  # 'real' | 'scaled'
        self.marker_mode = "sphere"  # 'sphere' | 'mesh'
        self.aug = 1.0  # size slider: scales the scaled preview and the Go solve
        self.t = 0
        self.playing = False
        self.sel = 0 if session.strips else -1
        self._go = False
        self._done = threading.Event()
        self._suppress_cb = False
        self.human_visible = True

        # ── shift set (edited == auto, enabled by default) ───────────────
        self.shifts = ShiftSet(
            source_path=session.source_path,
            object_model_path=session.object_model_path,
            stem=session.stem,
            object_scale=session.object_scale,
        )
        for st in session.strips:
            self.shifts.shifts[st.key] = Shift(
                link_name=st.link_name,
                ordinal=st.ordinal,
                start=st.start,
                end=st.end,
                auto_real=[float(x) for x in st.auto_real],
                edited_real=[float(x) for x in st.auto_real],
            )

        self._marker_handles: dict[str, viser.SceneNodeHandle] = {}
        self._build_scene()
        self._build_gui()
        self._select(self.sel)
        self._render_frame(0)
        self._spawn_play_thread()

    # ── geometry helpers ────────────────────────────────────────────────
    @property
    def scale_factor(self) -> float:
        # Scaled world: session object_scale x size slider.
        return self.session.object_scale * self.aug if self.world_mode == "scaled" else 1.0

    def _object_pose(self, t: int):
        s = self.session
        if self.world_mode == "scaled":
            return s.obj_rot_scaled[t], s.obj_pos_scaled[t]
        return s.obj_rot_real[t], s.obj_pos_real[t]

    def _edited_local(self, st_key: str) -> np.ndarray:
        """Edited point expressed in the current world's object frame."""
        return np.asarray(self.shifts.shifts[st_key].edited_real, float) * self.scale_factor

    # ── scene construction ──────────────────────────────────────────────
    def _build_scene(self) -> None:
        sv = self.server.scene
        sv.add_grid(
            "/ground", width=6.0, height=6.0, plane="xy", cell_size=0.25, section_size=1.0, position=(0.0, 0.0, 0.0)
        )
        sv.add_mesh_simple(
            "/human",
            self.session.human_vertices[0],
            self.session.human_faces,
            color=C_HUMAN,
            opacity=0.45,
            side="double",
        )
        self.obj_frame = sv.add_frame("/obj_frame", show_axes=False)
        self._add_object_mesh()
        for st in self.session.strips:
            self._rebuild_plain_marker(st)
        self.gizmo = sv.add_transform_controls("/obj_frame/gizmo", scale=0.16, disable_rotations=True, line_width=3.0)
        self.gizmo.on_update(self._on_gizmo_update)
        self.sel_marker = sv.add_icosphere(
            "/obj_frame/gizmo/marker", radius=0.03, color=C_SEL, opacity=1.0, position=(0, 0, 0)
        )
        # Selected link's EE mesh, posed in _update_marker_mesh.
        first = self.session.strips[0].link_name if self.session.strips else "left_palm_pad"
        lm = self.session.link_meshes[first]
        self.marker_mesh_node = sv.add_mesh_simple(
            "/obj_frame/marker_mesh", lm["vertices"], lm["faces"], color=C_EE, opacity=0.9, side="double"
        )
        self.marker_mesh_node.visible = False

    def _add_object_mesh(self) -> None:
        verts = self.session.object_vertices * self.scale_factor
        self.server.scene.add_mesh_simple(
            "/obj_frame/object",
            verts,
            self.session.object_faces,
            color=C_OBJECT,
            opacity=0.55,
            side="double",
            flat_shading=True,
        )

    def _rebuild_plain_marker(self, st) -> None:
        """(Re)create a strip's sphere: side-coloured, grey if disabled, hidden if selected."""
        sh = self.shifts.shifts[st.key]
        if not sh.enabled:
            col, op = C_DISABLED, 0.4
        else:
            col, op = (C_LEFT if st.side == "left" else C_RIGHT), 0.95
        h = self.server.scene.add_icosphere(
            f"/obj_frame/markers/{st.key}", radius=0.022, color=col, opacity=op, position=self._edited_local(st.key)
        )
        h.visible = self.sel < 0 or self.session.strips[self.sel].key != st.key
        self._marker_handles[st.key] = h

    # ── GUI ─────────────────────────────────────────────────────────────
    def _build_gui(self) -> None:
        g = self.server.gui
        g.add_markdown(
            f"## hoi_retarget.contact\n`{self.session.stem}`  ·  "
            f"{self.session.n_frames} frames  ·  object_scale "
            f"{self.session.object_scale:.3f}"
        )

        with g.add_folder("Playback"):
            self.frame_slider = g.add_slider("frame", 0, max(self.session.n_frames - 1, 0), step=1, initial_value=0)
            self.frame_slider.on_update(lambda _: self._render_frame(int(self.frame_slider.value)))
            self.play_btn = g.add_button("▶ play / ⏸ pause")
            self.play_btn.on_click(lambda _: self._toggle_play())

        rb = getattr(self.session, "robot", "unitree_g1")
        sc = getattr(self.session, "scale_object_mesh", True)
        g.add_markdown(
            f"**{rb.replace('unitree_', '').upper()}** · object mesh "
            f"{'scaled to ' + format(self.session.object_scale, '.2f') if sc else 'at captured size (1.0)'}"
        )

        with g.add_folder("View"):
            self.world_dd = g.add_dropdown("world", ("real-size", "scaled (robot)"), initial_value="real-size")
            self.world_dd.on_update(lambda _: self._set_world("scaled" if "scaled" in self.world_dd.value else "real"))
            self.marker_dd = g.add_dropdown("marker", ("sphere", "robot EE mesh"), initial_value="sphere")
            self.marker_dd.on_update(
                lambda _: self._set_marker("mesh" if self.marker_dd.value != "sphere" else "sphere")
            )
            self.show_human = g.add_checkbox("show human", initial_value=True)
            self.show_human.on_update(lambda _: self._render_frame(self.t))

        with g.add_folder("Object size"):
            self.aug_slider = g.add_slider("object size ×", 0.3, 3.0, step=0.05, initial_value=1.0)
            self.aug_slider.on_update(lambda _: self._set_aug(float(self.aug_slider.value)))
            g.add_markdown(
                "_resizes the object for the robot (preview in the **scaled** world); applied to the retarget._"
            )

        nmax = max(self.session.n_frames - 1, 1)
        with g.add_folder("Strips"):
            labels = [st.label for st in self.session.strips] or ["(none)"]
            self.strip_dd = g.add_dropdown("edit strip", tuple(labels), initial_value=labels[0])
            self.strip_dd.on_update(lambda _: self._select(self._label_index(self.strip_dd.value)))
            self.enabled_cb = g.add_checkbox("enabled (use in solve)", initial_value=True)
            self.enabled_cb.on_update(lambda _: None if self._suppress_cb else self._on_enabled_toggle())
            # Frame range the contact is active in the solve.
            self.start_slider = g.add_slider("contact start frame", 0, nmax, step=1, initial_value=0)
            self.start_slider.on_update(lambda _: None if self._suppress_cb else self._on_range_change())
            self.end_slider = g.add_slider("contact end frame", 0, nmax, step=1, initial_value=nmax)
            self.end_slider.on_update(lambda _: None if self._suppress_cb else self._on_range_change())
            self.info = g.add_markdown("")
            self.reset_btn = g.add_button("reset this strip")
            self.reset_btn.on_click(lambda _: self._reset_selected())
            self.reset_all_btn = g.add_button("reset all")
            self.reset_all_btn.on_click(lambda _: self._reset_all())

        with g.add_folder("Output"):
            self.save_btn = g.add_button("💾 save shifts")
            self.save_btn.on_click(lambda _: self._save_only())
            self.go_btn = g.add_button("🚀 Go — retarget", color="green")
            self.go_btn.on_click(lambda _: self._go_retarget())
            self.status = g.add_markdown("")

    def _label_index(self, label: str) -> int:
        for i, st in enumerate(self.session.strips):
            if st.label == label:
                return i
        return -1

    # ── rendering ───────────────────────────────────────────────────────
    def _render_frame(self, t: int) -> None:
        t = int(np.clip(t, 0, self.session.n_frames - 1))
        self.t = t
        s = self.session
        show_human = (not hasattr(self, "show_human")) or self.show_human.value
        if show_human:
            self.server.scene.add_mesh_simple(
                "/human", s.human_vertices[t], s.human_faces, color=C_HUMAN, opacity=0.45, side="double"
            )
            self.human_visible = True
        elif self.human_visible:
            h = self.server.scene.add_mesh_simple(
                "/human", s.human_vertices[t], s.human_faces, color=C_HUMAN, opacity=0.45
            )
            h.visible = False
            self.human_visible = False
        wxyz, pos = self._object_pose(t)
        self.obj_frame.wxyz = np.asarray(wxyz, float)
        self.obj_frame.position = np.asarray(pos, float)
        if (
            self.marker_mode == "mesh"
            and self.sel >= 0
            and self.shifts.shifts[self.session.strips[self.sel].key].enabled
        ):
            self._update_marker_mesh(t)
        if getattr(self, "info", None) is not None and self.sel >= 0:
            self._refresh_info()  # keeps the ●active/○inactive indicator live

    def _spawn_play_thread(self) -> None:
        def loop():
            while not self._done.is_set():
                if self.playing and self.session.n_frames > 1:
                    nxt = (self.t + 1) % self.session.n_frames
                    self._render_frame(nxt)
                    try:
                        self.frame_slider.value = nxt
                    except Exception:
                        pass
                    time.sleep(1.0 / max(self.session.fps, 1.0))
                else:
                    time.sleep(0.05)

        threading.Thread(target=loop, daemon=True).start()

    def _toggle_play(self) -> None:
        self.playing = not self.playing

    # ── selection / markers ─────────────────────────────────────────────
    def _select(self, idx: int) -> None:
        self.sel = idx
        if idx < 0:
            self.gizmo.visible = False
            self.sel_marker.visible = False
            self.marker_mesh_node.visible = False
            self.info.content = "_no contact strips in this clip_"
            return
        st = self.session.strips[idx]
        sh = self.shifts.shifts[st.key]
        self._suppress_cb = True
        self.enabled_cb.value = sh.enabled
        self.start_slider.value = int(sh.start)
        self.end_slider.value = int(sh.end)
        self._suppress_cb = False
        self.gizmo.position = self._edited_local(st.key)
        self._render_frame(int(sh.start))  # jump to the contact's first frame
        try:
            self.frame_slider.value = int(sh.start)
        except Exception:
            pass
        for s2 in self.session.strips:
            self._rebuild_plain_marker(s2)
        self._apply_selection_visual(st)
        self._refresh_info()

    def _apply_selection_visual(self, st) -> None:
        """Show the gizmo and highlight if enabled, else the grey plain sphere."""
        en = self.shifts.shifts[st.key].enabled
        self.gizmo.visible = en
        self.sel_marker.visible = en and (self.marker_mode == "sphere")
        if en and self.marker_mode == "mesh":
            self._update_marker_mesh(self.t)
            self.marker_mesh_node.visible = True
        else:
            self.marker_mesh_node.visible = False
        self._marker_handles[st.key].visible = not en

    def _refresh_all_markers(self) -> None:
        for st in self.session.strips:
            self._rebuild_plain_marker(st)
        if self.sel >= 0:
            self.gizmo.position = self._edited_local(self.session.strips[self.sel].key)
            self._apply_selection_visual(self.session.strips[self.sel])

    def _update_marker_mesh(self, t: int) -> None:
        """Pose the selected link's EE mesh: contact target at the edited point, source orientation."""
        st = self.session.strips[self.sel]
        lm = self.session.link_meshes[st.link_name]
        local = self._edited_local(st.key)
        wxyz_o, pos_o = self._object_pose(t)
        Ro = _rotmat(wxyz_o)
        p_world = np.asarray(pos_o, float) + Ro @ local
        src = self.session.orient_for_strip(st)
        R_src = _rotmat(src[t]) if src is not None else Ro
        R_mesh = R_src @ lm["align"]
        mesh_pos_world = p_world - R_mesh @ lm["t_target"]
        # express relative to /obj_frame
        R_local = Ro.T @ R_mesh
        p_local = Ro.T @ (mesh_pos_world - np.asarray(pos_o, float))
        q = R.from_matrix(R_local).as_quat(scalar_first=True)
        self.marker_mesh_node = self.server.scene.add_mesh_simple(
            "/obj_frame/marker_mesh",
            lm["vertices"],
            lm["faces"],
            color=C_EE,
            opacity=0.9,
            side="double",
            wxyz=q,
            position=p_local,
        )

    def _on_gizmo_update(self, _handle) -> None:
        if self.sel < 0:
            return
        st = self.session.strips[self.sel]
        real = np.asarray(self.gizmo.position, float) / self.scale_factor
        self.shifts.set_edited(st.link_name, st.ordinal, real)
        if self.marker_mode == "mesh":
            self._update_marker_mesh(self.t)
        self._refresh_info()

    def _on_enabled_toggle(self) -> None:
        if self.sel < 0:
            return
        st = self.session.strips[self.sel]
        self.shifts.set_enabled(st.link_name, st.ordinal, bool(self.enabled_cb.value))
        self._rebuild_plain_marker(st)
        self._apply_selection_visual(st)
        self._refresh_info()

    def _refresh_info(self) -> None:
        if self.sel < 0:
            return
        st = self.session.strips[self.sel]
        sh = self.shifts.shifts[st.key]
        d = sh.delta * 100.0
        state = "ENABLED" if sh.enabled else "DISABLED (dropped from solve)"
        rng = f"`{sh.start}–{sh.end}`"
        if sh.retimed:
            rng += f"  (was `{sh.auto_start}–{sh.auto_end}`)"
        active = "● active" if sh.start <= self.t < sh.end else "○ inactive here"
        self.info.content = (
            f"**{st.link_name}** #{st.ordinal}  ({st.kind}, {st.side}) — {state}\n\n"
            f"contact frames {rng}  · frame {self.t} {active}\n\n"
            f"Δ (obj frame): `{d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}` cm  "
            f"(|Δ|={np.linalg.norm(sh.delta) * 100:.1f} cm)\n\n"
            f"moved: **{self.shifts.moved_count()}**  ·  retimed: "
            f"**{self.shifts.retimed_count()}**  ·  disabled: "
            f"**{self.shifts.disabled_count()}**  ·  total {len(self.shifts.shifts)}"
        )

    # ── view toggles ────────────────────────────────────────────────────
    def _set_world(self, mode: str) -> None:
        self.world_mode = mode
        self._add_object_mesh()
        self._render_frame(self.t)
        self._refresh_all_markers()

    def _set_marker(self, mode: str) -> None:
        self.marker_mode = mode
        if self.sel >= 0:
            self._apply_selection_visual(self.session.strips[self.sel])

    def _set_aug(self, aug: float) -> None:
        self.aug = float(aug)
        self.shifts.object_aug_scale = float(aug)
        self._add_object_mesh()  # rescale the (scaled-world) object mesh
        self._render_frame(self.t)
        self._refresh_all_markers()  # rescale marker / gizmo positions

    def _on_range_change(self) -> None:
        """Adjust the selected strip's contact duration (start/end frames)."""
        if self.sel < 0:
            return
        st = self.session.strips[self.sel]
        s = int(self.start_slider.value)
        e = int(self.end_slider.value)
        e = max(e, s + 1)  # keep at least one active frame
        self._suppress_cb = True
        self.end_slider.value = e
        self._suppress_cb = False
        self.shifts.set_range(st.link_name, st.ordinal, s, e)
        self._render_frame(s)  # jump to the new start for context
        try:
            self.frame_slider.value = s
        except Exception:
            pass
        self._refresh_info()

    # ── edits / output ──────────────────────────────────────────────────
    def _reset_selected(self) -> None:
        if self.sel < 0:
            return
        st = self.session.strips[self.sel]
        self.shifts.reset(st.link_name, st.ordinal)  # point + range back to auto
        sh = self.shifts.shifts[st.key]
        self._suppress_cb = True
        self.start_slider.value = int(sh.start)
        self.end_slider.value = int(sh.end)
        self._suppress_cb = False
        self.gizmo.position = self._edited_local(st.key)
        self._rebuild_plain_marker(st)
        self._apply_selection_visual(st)
        self._refresh_info()

    def _reset_all(self) -> None:
        self.shifts.reset_all()
        self._refresh_all_markers()
        self._refresh_info()

    def _save_only(self) -> None:
        out = os.path.join(self.out_dir, "shifts.json")
        self.shifts.save(out)
        self.status.content = (
            f"💾 saved → `{out}`  ({self.shifts.moved_count()} moved, {self.shifts.disabled_count()} disabled)"
        )

    def _go_retarget(self) -> None:
        self.status.content = "🚀 closing viewer and running contact retarget…"
        self._go = True
        self._done.set()

    def preload(self, other: ShiftSet) -> None:
        """Adopt edited points, enable flags, contact ranges + aug from a saved ShiftSet."""
        for st in self.session.strips:
            sh = other.shifts.get(st.key)
            if sh is not None:
                self.shifts.set_edited(st.link_name, st.ordinal, sh.edited_real)
                self.shifts.set_enabled(st.link_name, st.ordinal, sh.enabled)
                self.shifts.set_range(st.link_name, st.ordinal, sh.start, sh.end)
        self.aug = float(other.object_aug_scale)
        self.shifts.object_aug_scale = self.aug
        try:
            self.aug_slider.value = self.aug
        except Exception:
            pass
        self._add_object_mesh()
        self._refresh_all_markers()
        if self.sel >= 0:
            self._select(self.sel)

    # ── lifecycle ───────────────────────────────────────────────────────
    def run(self):
        """Block until Go or Ctrl-C; returns ``(go, shifts)``. Only save and Go write to disk."""
        print(f"[hoi_retarget.contact] viewer ready → http://localhost:{self.port}/")
        if self.host == "127.0.0.1" and ("SSH_CONNECTION" in os.environ or "SSH_CLIENT" in os.environ):
            print(
                f"[hoi_retarget.contact] over SSH? on the client run:  ssh -L {self.port}:localhost:{self.port} <this-host>"
            )
        print("[hoi_retarget.contact] edit the strips, then press 'Go' (or Ctrl-C to quit without saving).")
        try:
            self._done.wait()
        except KeyboardInterrupt:
            print("\n[hoi_retarget.contact] interrupted — quitting without saving or retargeting.")
        finally:
            try:
                self.server.stop()
            except Exception:
                pass
        return self._go, self.shifts
