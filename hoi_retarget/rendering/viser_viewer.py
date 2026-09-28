# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import io
import numpy as np
import os
import socket
import time
from collections.abc import Sequence
from pathlib import Path

import viser
from viser.extras import ViserUrdf

from hoi_retarget import paths


def reserve_port(host: str, port: int) -> int:
    """Return ``port``, or a free port from the OS when it is 0 (viser does not resolve 0 itself)."""
    if port != 0:
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class ViserMotionViewer:
    """Browser (viser) viewer for robot + object motion with per-link contact markers.

    ``step()`` mirrors ``RobotMotionViewer.step()``. Contact links show a grey marker,
    turning orange (hands) or cyan (feet) while in contact, with a wireframe sphere
    at the contact target on the object.
    """

    def __init__(
        self,
        robot_type: str,
        urdf_path: str,
        object_model_path: str,
        dof_names: Sequence[str],
        motion_fps: float = 30.0,
        object_scale: float = 1.0,
        contact_link_names: Sequence[str] = (),
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        if object_model_path is None:
            raise ValueError("object_model_path is required and must be a URDF.")
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"Robot URDF not found: {urdf_path}")
        # Pickles record this path relative to where they were produced.
        object_model_path = str(paths.resolve_asset_path(object_model_path))
        if not os.path.isfile(object_model_path):
            raise FileNotFoundError(
                f"Object URDF not found: {object_model_path} "
                "(relative corpus paths resolve against HOI_RETARGET_OBJECT_ROOTS)"
            )

        self.robot_type = robot_type
        self.urdf_path = urdf_path
        self.object_model_path = object_model_path
        self.motion_fps = float(motion_fps)
        self.dt = 1.0 / self.motion_fps

        self.object_scale = float(object_scale)

        port = reserve_port(host, port)
        self._host = host

        with contextlib.redirect_stdout(io.StringIO()):
            self.server = viser.ViserServer(host=host, port=port, verbose=False)
        # viser silently rebinds to another port when the requested one is busy.
        self._port = self.server.get_port()
        self._url = f"http://localhost:{self._port}/"
        self.server.scene.set_up_direction("+z")

        # Ground plane at z=0.
        self.server.scene.add_grid(
            "/ground",
            width=20.0,
            height=20.0,
            plane="xy",
            cell_size=0.5,
            section_size=1.0,
            plane_color=(245, 245, 245),
            plane_opacity=0.35,
        )

        self._robot_frame = self.server.scene.add_frame("/robot", show_axes=False)
        self._object_frame = self.server.scene.add_frame("/object", show_axes=False)
        self._robot_urdf = ViserUrdf(
            self.server,
            Path(urdf_path),
            root_node_name="/robot",
        )
        self._object_urdf = ViserUrdf(
            self.server,
            Path(object_model_path),
            scale=self.object_scale,
            root_node_name="/object",
        )

        # Map dof_pos columns to viser's actuated-joint order.
        viser_joint_names = list(self._robot_urdf.get_actuated_joint_names())
        dof_name_to_col = {name: i for i, name in enumerate(dof_names)}
        self._dof_permutation = np.array(
            [dof_name_to_col.get(name, -1) for name in viser_joint_names],
            dtype=int,
        )
        missing = [name for name, col in zip(viser_joint_names, self._dof_permutation) if col < 0]
        if missing:
            print(
                f"[viser] Warning: {len(missing)} URDF joint(s) not in dof_names "
                f"(will be driven to 0): {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        self._contact_link_names = list(contact_link_names)
        self._build_contact_markers()

    # ------------------------------------------------------------------
    # A handle's colour is fixed at creation, so each link gets an idle and a live marker.
    HAND_RGB = (255, 122, 47)
    FOOT_RGB = (53, 196, 232)
    IDLE_RGB = (150, 150, 150)

    def _build_contact_markers(self) -> None:
        self._link_idle: dict[str, object] = {}
        self._link_live: dict[str, object] = {}
        self._target: dict[str, object] = {}
        for name in self._contact_link_names:
            hue = self.FOOT_RGB if ("ankle" in name or "foot" in name or "toe" in name) else self.HAND_RGB
            safe = name.replace("/", "_")
            self._link_idle[name] = self.server.scene.add_icosphere(
                f"/contact/{safe}/idle",
                radius=0.018,
                color=self.IDLE_RGB,
                opacity=0.45,
                visible=False,
            )
            self._link_live[name] = self.server.scene.add_icosphere(
                f"/contact/{safe}/live",
                radius=0.028,
                color=hue,
                visible=False,
            )
            # Contact target, hollow so the link marker shows inside it.
            self._target[name] = self.server.scene.add_icosphere(
                f"/contact/{safe}/target",
                radius=0.038,
                color=hue,
                wireframe=True,
                visible=False,
            )
        self._contact_visible = True

    def set_contact_visible(self, visible: bool) -> None:
        """Master switch for the contact markers (wired to a GUI checkbox)."""
        self._contact_visible = bool(visible)
        if not visible:
            for d in (self._link_idle, self._link_live, self._target):
                for h in d.values():
                    h.visible = False

    def _update_contact(self, per_link_contact_data) -> None:
        if not self._contact_link_names:
            return
        if not self._contact_visible or per_link_contact_data is None:
            if per_link_contact_data is None:
                for d in (self._link_idle, self._link_live, self._target):
                    for h in d.values():
                        h.visible = False
            return
        seen = set()
        for entry, name in zip(per_link_contact_data, self._contact_link_names):
            # Entries may omit links; use "link_name" when present, else positional order.
            name = entry.get("link_name", name)
            if name not in self._link_idle:
                continue
            seen.add(name)
            active = bool(entry.get("active"))
            pos = entry.get("link_pos")
            if pos is not None:
                pos = np.asarray(pos, dtype=float)
                self._link_idle[name].position = pos
                self._link_live[name].position = pos
            self._link_idle[name].visible = pos is not None and not active
            self._link_live[name].visible = pos is not None and active

            tgt = entry.get("contact_point_world")
            if active and tgt is not None:
                self._target[name].position = np.asarray(tgt, dtype=float)
                self._target[name].visible = True
            else:
                self._target[name].visible = False
        for name in self._contact_link_names:
            if name not in seen:
                self._link_idle[name].visible = False
                self._link_live[name].visible = False
                self._target[name].visible = False

    # ------------------------------------------------------------------
    def step(
        self,
        root_pos: np.ndarray,
        root_rot: np.ndarray,
        dof_pos: np.ndarray,
        *,
        object_data: tuple | None = None,
        per_link_contact_data: list[dict] | None = None,
        rate_limit: bool = False,
        **_ignored,
    ) -> None:
        self._robot_frame.position = np.asarray(root_pos, dtype=float)
        self._robot_frame.wxyz = np.asarray(root_rot, dtype=float)

        dof_pos = np.asarray(dof_pos, dtype=float)
        cfg = np.zeros(len(self._dof_permutation), dtype=float)
        for i, col in enumerate(self._dof_permutation):
            if col >= 0:
                cfg[i] = dof_pos[col]
        self._robot_urdf.update_cfg(cfg)

        if object_data is not None:
            self._object_frame.position = np.asarray(object_data[0], dtype=float)
            self._object_frame.wxyz = np.asarray(object_data[1], dtype=float)

        self._update_contact(per_link_contact_data)

        if rate_limit:
            time.sleep(self.dt)

    # ------------------------------------------------------------------
    def get_url(self) -> str:
        return self._url

    def get_port(self) -> int:
        return self._port

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:
            pass
