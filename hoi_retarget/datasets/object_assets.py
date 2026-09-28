# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage an object mesh into the layout the pipeline reads.

Writes ``<assets_root>/<name>.urdf`` and ``<assets_root>/objects/<name>/{<name>.obj, sample_points.npy}``.
Without ``sample_points.npy``, ``compute_min_object_height_per_frame`` returns None and the object
floats or sinks by its half-height, silently. Object poses rotate about the mesh's vertex centroid,
so an off-centre mesh gets a warning; ``--recenter`` moves it.

    hoi-retarget-stage-object mug.obj --name mug --assets-root data/my_dataset/_assets
"""

from __future__ import annotations

import argparse
import numpy as np
import shutil
from pathlib import Path

#: Surface samples per object, as in the released converted datasets (CARI4D stages 500, see ``cari4d_assets``).
N_POINTS = 1024


def urdf_text(name: str) -> str:
    """Object URDF text, as used for the converted datasets' assets (the ``<inertial>`` block is unused)."""
    return f"""<?xml version="1.0" ?>
<robot name="{name}.urdf">
  <dynamics damping="0.5" friction="0.9"/>
  <link name="baseLink">
    <contact>
      <lateral_friction value="0.9"/><rolling_friction value="0.5"/>
      <stiffness value="30000"/><damping value="1000"/>
    </contact>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="objects/{name}/{name}.obj" scale="1.0 1.0 1.0"/></geometry>
      <material name="mat"><color rgba="0.7 0.8 0.9 1"/></material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="objects/{name}/{name}.obj" scale="1.0 1.0 1.0"/></geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
"""


def stage_object(
    mesh_path: str | Path,
    name: str,
    assets_root: str | Path,
    n_points: int = N_POINTS,
    overwrite: bool = False,
    recenter: bool = False,
) -> Path:
    """Write the URDF, mesh copy and point cloud. Idempotent; returns the URDF."""
    import trimesh

    mesh_path, assets_root = Path(mesh_path), Path(assets_root)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    obj_dir = assets_root / "objects" / name
    urdf, dst_mesh = assets_root / f"{name}.urdf", obj_dir / f"{name}.obj"
    pts_path = obj_dir / "sample_points.npy"
    if not overwrite and dst_mesh.exists() and pts_path.exists() and urdf.exists():
        print(f"{name} already staged in {assets_root}; pass --overwrite")
        return urdf
    obj_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_path, force="mesh")
    off = float(np.abs(mesh.vertices.mean(axis=0)).max())
    if recenter:
        mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    elif off > 1e-3:
        print(
            f"warning: {name} centroid is {off * 100:.1f} cm off the origin. Poses rotate "
            f"about the centroid, so this object will sit wrong. Pass --recenter."
        )

    if not dst_mesh.exists() or overwrite:
        if mesh_path.suffix.lower() == ".obj" and not recenter:
            shutil.copy2(mesh_path, dst_mesh)
        else:
            mesh.export(dst_mesh)
    if not pts_path.exists() or overwrite:
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)
        np.save(pts_path, np.asarray(pts, np.float64))
    if not urdf.exists() or overwrite:
        urdf.write_text(urdf_text(name))
    return urdf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mesh", help="object mesh (.obj, or anything trimesh reads)")
    ap.add_argument("--name", help="object name; defaults to the mesh's stem")
    ap.add_argument("--assets-root", required=True, help="directory that --object_model_path will point into")
    ap.add_argument("--n-points", type=int, default=N_POINTS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--recenter", action="store_true", help="move the mesh so its vertex centroid is the origin")
    a = ap.parse_args(argv)
    urdf = stage_object(a.mesh, a.name or Path(a.mesh).stem, a.assets_root, a.n_points, a.overwrite, a.recenter)
    print(urdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
