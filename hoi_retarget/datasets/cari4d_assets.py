# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage per-sequence URDF + sample_points.npy for CARI4D meshes.

Writes ``<assets_root>/<STEM>.urdf`` and, under ``<assets_root>/objects/<STEM>/``, a MuJoCo-loadable
``<STEM>.obj`` plus ``sample_points.npy`` (N, 3) and ``sample_ids.npy`` (N,), the layout
``retargeting.object_model.get_sample_points_path`` expects. Usage:

    python -m hoi_retarget.datasets.cari4d_assets --stem Date03_Sub01_gas_wild002   # or --all
"""

from __future__ import annotations

import argparse
import glob
import numpy as np
import os
import re
import sys
from collections.abc import Sequence

from hoi_retarget.datasets.object_assets import urdf_text

DEFAULT_MESHES_ROOT = "data/CARI4D/meshes"
DEFAULT_ASSETS_ROOT = "data/CARI4D/assets"

# CARI4D mesh folder names:
#   1. <stem>.<cam>.color.mp4_<step>_rgba   (Hunyuan3D-3 export)
#   2. <stem>_t<time>_k<int>_rgba           (export with seed time + k id)
#   3. <stem>_000_align                     (metric-scaled bundle: meshes_metric/)
_MESH_DIR_RES: tuple[re.Pattern, ...] = (
    re.compile(r"^(?P<stem>.+?)\.\d+\.color\.mp4_\d+_rgba$"),
    re.compile(r"^(?P<stem>.+?)_t[\d.]+_k\d+_rgba$"),
    re.compile(r"^(?P<stem>.+?)_\d+_align$"),
)


def _farthest_point_sample_vertices(vertices: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(points, indices)`` of ``n`` vertices chosen by farthest-point sampling."""
    V = np.asarray(vertices, dtype=np.float64)
    total = V.shape[0]
    n = int(min(n, total))
    if n <= 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    idx0 = int(rng.integers(total))
    selected = [idx0]
    dist = np.linalg.norm(V - V[idx0], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(V - V[nxt], axis=1))
    ids = np.asarray(selected, dtype=np.int64)
    return V[ids], ids


def _resolve_mesh_for_stem(stem: str, meshes_root: str) -> str:
    """First ``*_align.obj`` for ``stem`` across the known mesh-folder layouts."""
    patterns = (
        os.path.join(meshes_root, f"{stem}.*.color.mp4_*_rgba", f"{stem}.*.color.mp4_*_align.obj"),
        os.path.join(meshes_root, f"{stem}_t*_k*_rgba", "*_align.obj"),
        os.path.join(meshes_root, f"{stem}_*_align", "*_align.obj"),
        os.path.join(meshes_root, f"{stem}*_rgba", "*_align.obj"),
    )
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No CARI4D mesh found for stem '{stem}' under {meshes_root} (tried: {list(patterns)})")


def _voxel_cluster_decimate(verts: np.ndarray, faces: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Decimate by voxel vertex clustering on a ``grid``-cell lattice; falsy ``grid`` returns the input."""
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if not grid or V.shape[0] == 0:
        return V, F
    lo = V.min(0)
    vox = float((V.max(0) - lo).max()) / float(grid)
    if vox <= 0:
        return V, F
    keys = np.floor((V - lo) / vox).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    nV = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(nV, inv, V)
    nV /= counts[:, None]
    nF = inv[F]
    good = (nF[:, 0] != nF[:, 1]) & (nF[:, 1] != nF[:, 2]) & (nF[:, 0] != nF[:, 2])
    nF = nF[good]
    if nF.shape[0] == 0:
        return V, F  # decimation collapsed everything — keep original
    used = np.unique(nF)
    remap = np.full(nV.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    return nV[used], remap[nF]


def _export_mujoco_clean(verts: np.ndarray, faces: np.ndarray, dst_obj: str) -> None:
    """Write a plain ``v``/``vn``/``f`` OBJ (no ``mtllib``/``vt``), which MuJoCo's strict parser accepts."""
    import trimesh
    from trimesh.exchange.obj import export_obj

    clean = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    data = export_obj(
        clean,
        include_normals=True,
        include_color=False,
        include_texture=False,
        return_texture=False,
        write_texture=False,
        header=None,
    )
    with open(dst_obj, "w") as f:
        f.write(data)


def _write_mujoco_clean_obj(src_obj: str, dst_obj: str) -> None:
    """Rewrite ``src_obj`` as a plain OBJ; MuJoCo cannot parse Hunyuan3D's textured ``mtllib``/``v/vt/vn`` OBJs."""
    import trimesh

    src_mesh = trimesh.load(src_obj, force="mesh", process=False)
    _export_mujoco_clean(src_mesh.vertices, src_mesh.faces, dst_obj)


def prepare_cari4d_object_assets(
    stem: str,
    mesh_path: str | None = None,
    assets_root: str = DEFAULT_ASSETS_ROOT,
    meshes_root: str = DEFAULT_MESHES_ROOT,
    n_sample_points: int = 500,
    seed: int = 0,
    force: bool = False,
    verbose: bool = True,
    object_name: str | None = None,
    center_mesh: bool = False,
    crop_center: Sequence[float] | None = None,
    decimate_grid: int | None = None,
) -> str:
    """Stage the URDF, MuJoCo-clean mesh and sample points for one CARI4D object; return the URDF path.

    Idempotent; ``force`` regenerates.

    Args:
        stem: sequence stem, e.g. ``Date03_Sub01_gas_wild002``.
        mesh_path: explicit mesh; else resolved under ``meshes_root`` by ``object_name`` or ``stem``.
        object_name: stage under this shared name instead of ``stem``, so several sequences can use one
            asset (e.g. ``demochair``).
        center_mesh: subtract ``crop_center`` (default: the vertex centroid) from the mesh. CARI4D's metric
            meshes are posed as ``R @ (V - crop_center) + t``; pass ``scale.json['crop_center']`` to be explicit.
        decimate_grid: voxel-cluster the visual mesh at this resolution (e.g. 96). Sample points always come
            from the full vertex set.
    """
    asset_key = object_name or stem
    assets_root_abs = os.path.abspath(assets_root)
    obj_dir = os.path.join(assets_root_abs, "objects", asset_key)
    os.makedirs(obj_dir, exist_ok=True)

    urdf_path = os.path.join(assets_root_abs, f"{asset_key}.urdf")
    obj_out = os.path.join(obj_dir, f"{asset_key}.obj")
    sp_path = os.path.join(obj_dir, "sample_points.npy")
    ids_path = os.path.join(obj_dir, "sample_ids.npy")

    need_obj = force or not os.path.isfile(obj_out) or os.path.islink(obj_out)
    need_pts = force or not (os.path.isfile(sp_path) and os.path.isfile(ids_path))

    # Load the (large) source mesh only when something needs rebuilding.
    if need_obj or need_pts:
        if mesh_path is None:
            mesh_path = _resolve_mesh_for_stem(asset_key, meshes_root)
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"CARI4D mesh not found: {mesh_path}")
        import trimesh

        # process=False keeps CARI4D's vertex set, so the centroid equals crop_center.
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if verts.size == 0:
            raise ValueError(f"Mesh has zero vertices: {mesh_path}")
        if center_mesh:
            cc = np.asarray(crop_center, dtype=np.float64) if crop_center is not None else verts.mean(0)
            verts = verts - cc
            if verbose:
                print(f"[cari4d_asset_prep] centred mesh by crop_center={np.round(cc, 4)}")

    # 1. URDF
    if force or not os.path.isfile(urdf_path):
        with open(urdf_path, "w") as f:
            f.write(urdf_text(asset_key))
        if verbose:
            print(f"[cari4d_asset_prep] wrote URDF: {urdf_path}")

    # 2. MuJoCo-clean mesh (optionally decimated). Not a symlink: MuJoCo rejects the source's mtllib.
    if os.path.islink(obj_out):
        os.unlink(obj_out)  # stale symlink
    if need_obj:
        v_vis, f_vis = _voxel_cluster_decimate(verts, faces, decimate_grid) if decimate_grid else (verts, faces)
        _export_mujoco_clean(v_vis, f_vis, obj_out)
        if verbose:
            mb = os.path.getsize(obj_out) / 1e6
            print(
                f"[cari4d_asset_prep] wrote MuJoCo-clean mesh: {obj_out} "
                f"({len(v_vis)} verts, {mb:.1f} MB; from {mesh_path})"
            )

    # 3. sample_points.npy + sample_ids.npy (from FULL centred vertices)
    if need_pts:
        pts, ids = _farthest_point_sample_vertices(verts, n_sample_points, seed=seed)
        np.save(sp_path, pts)
        np.save(ids_path, ids)
        if verbose:
            print(f"[cari4d_asset_prep] sampled {pts.shape[0]} points from {verts.shape[0]} verts → {sp_path}")

    return urdf_path


def _discover_all_stems(meshes_root: str) -> list[str]:
    stems: list[str] = []
    if not os.path.isdir(meshes_root):
        return stems
    for name in sorted(os.listdir(meshes_root)):
        for regex in _MESH_DIR_RES:
            m = regex.match(name)
            if m:
                stems.append(m.group("stem"))
                break
    return stems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--stem", type=str, help="Stage assets for one sequence.")
    grp.add_argument(
        "--all",
        action="store_true",
        help=f"Walk {DEFAULT_MESHES_ROOT}/ and stage every mesh dir found.",
    )
    parser.add_argument("--meshes_root", type=str, default=DEFAULT_MESHES_ROOT)
    parser.add_argument("--assets_root", type=str, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--n_sample_points", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    targets = [args.stem] if args.stem else _discover_all_stems(args.meshes_root)
    if not targets:
        print(f"No sequences found under {args.meshes_root}.", file=sys.stderr)
        return 1

    for stem in targets:
        urdf = prepare_cari4d_object_assets(
            stem=stem,
            assets_root=args.assets_root,
            meshes_root=args.meshes_root,
            n_sample_points=args.n_sample_points,
            force=args.force,
        )
        print(f"[cari4d_asset_prep] ready: {urdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
