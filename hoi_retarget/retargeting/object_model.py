# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import re
from collections.abc import Iterable

from scipy.spatial.transform import Rotation as R

from hoi_retarget import paths

# Candidate contact links; the InterMimic .pt flags filter out the inactive ones.
DEFAULT_CONTACT_LINKS = [
    "left_palm_pad",
    "right_palm_pad",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

# Maps every G1 contact link to its SMPL-X position anchor joint.
G1_LINK_TO_SMPLX_ANCHOR_STATIC = {
    "left_palm_pad": "left_palm",
    "right_palm_pad": "right_palm",
    "left_ankle_roll_link": "left_ankle",
    "right_ankle_roll_link": "right_ankle",
}

# min_object_height is a fallback; the per-frame height comes from sample_points.npy.
OBJECT_MOTION_DEFAULTS = {
    "basketball": {
        "min_object_height": 0.16,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "clothesstand": {
        "min_object_height": 0.34,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "floorlamp": {
        "min_object_height": 0.34,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "largebox": {
        "min_object_height": 0.30,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "largetable": {
        "min_object_height": 0.43,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "monitor": {
        "min_object_height": 0.17,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "mop": {
        "min_object_height": 0.51,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "plasticbox": {
        "min_object_height": 0.27,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "smallbox": {
        "min_object_height": 0.17,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "smalltable": {
        "min_object_height": 0.222,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "suitcase": {
        "min_object_height": 0.32,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "trashcan": {
        "min_object_height": 0.205,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "tripod": {
        "min_object_height": 0.29,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "vacuum": {
        "min_object_height": 0.37,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "whitechair": {
        "min_object_height": 0.60,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
    "woodchair": {
        "min_object_height": 0.50,
        "contact_links": DEFAULT_CONTACT_LINKS,
    },
}


def _name_candidates(folder_name: str) -> Iterable[str]:
    base = folder_name.strip()
    if not base:
        return

    lowered = base.lower()
    variants = {
        base,
        lowered,
        lowered.replace("-", "_"),
        lowered.replace(" ", "_"),
    }

    # Remove trailing numeric suffixes such as "_001" or "-2".
    stripped = re.sub(r"([_-]\d+)$", "", lowered)
    if stripped:
        variants.add(stripped)

    for v in variants:
        if v:
            yield v


def infer_object_model_path_from_motion_path(
    motion_path: str,
    objects_root: str | None = None,
) -> str | None:
    """Infer the object URDF from ``motion_path``'s folders, then an OMOMO stem; else None.

    ``.../largebox/foo.pt`` or ``.../sub10_largebox_002.pt`` -> ``<objects_root>/largebox.urdf``.
    ``objects_root`` defaults to ``paths.OBJECT_ROOT``.
    """
    objects_root = str(paths.OBJECT_ROOT) if objects_root is None else objects_root
    norm_path = os.path.abspath(motion_path)
    curr = os.path.dirname(norm_path)
    root = os.path.abspath(os.path.sep)

    while curr and curr != root:
        folder = os.path.basename(curr)
        for name in _name_candidates(folder):
            candidate = os.path.join(objects_root, f"{name}.urdf")
            if os.path.isfile(candidate):
                return candidate
        curr = os.path.dirname(curr)

    # Flat datasets: sub<id>_<object>_<id>.pt
    file_stem = os.path.splitext(os.path.basename(norm_path))[0].lower()

    m = re.match(r"^sub\d+_([a-z0-9_\-]+)_\d+$", file_stem)
    if m:
        for name in _name_candidates(m.group(1)):
            candidate = os.path.join(objects_root, f"{name}.urdf")
            if os.path.isfile(candidate):
                return candidate

    # No looser match: other datasets reuse OMOMO object names for different meshes.
    return None


def resolve_object_model_path(
    motion_path: str | None,
    configured_object_model_path: str | None,
    auto_from_motion_path: bool = True,
) -> str:
    """Configured path if given (must exist), else inferred from ``motion_path``; raises otherwise.

    The configured path wins because inference can match another dataset's object to an OMOMO mesh.
    """
    if configured_object_model_path:
        if not paths.resolve_asset_path(configured_object_model_path).is_file():
            raise FileNotFoundError(
                f"Object model not found: {configured_object_model_path} "
                "(searched the working directory, the repository, the asset root and "
                "HOI_RETARGET_OBJECT_ROOTS)."
            )
        return configured_object_model_path

    if auto_from_motion_path and motion_path:
        inferred = infer_object_model_path_from_motion_path(motion_path=motion_path)
        if inferred:
            return inferred

    raise FileNotFoundError(
        f"Could not resolve an object model: auto-inference from {motion_path!r} found "
        "nothing and no object_model_path was provided. Pass --object_model_path explicitly."
    )


def get_object_motion_defaults(object_model_path: str) -> dict:
    model_name = os.path.splitext(os.path.basename(object_model_path))[0].lower()
    cfg = OBJECT_MOTION_DEFAULTS.get(model_name)
    if cfg is None:
        # Per-sequence assets (e.g. CARI4D); the per-frame floor clearance is recomputed downstream.
        cfg = {
            "min_object_height": 0.0,
            "contact_links": DEFAULT_CONTACT_LINKS,
        }
    return {
        "min_object_height": float(cfg["min_object_height"]),
        "contact_links": list(cfg["contact_links"]),
    }


def get_sample_points_path(object_model_path: str) -> str | None:
    """``<urdf dir>/objects/<name>/sample_points.npy`` for ``<urdf dir>/<name>.urdf``, or None."""
    name = os.path.splitext(os.path.basename(object_model_path))[0]
    base_dir = os.path.dirname(paths.resolve_asset_path(object_model_path))
    npy_path = os.path.join(base_dir, "objects", name, "sample_points.npy")
    return npy_path if os.path.isfile(npy_path) else None


def compute_min_object_height_per_frame(
    object_model_path: str,
    object_rot_wxyz: np.ndarray,
) -> np.ndarray | None:
    """Per-frame height of the object origin above its lowest sample point, shape (T,).

    ``object_rot_wxyz``: (T, 4). Returns None without ``sample_points.npy``.
    """
    npy_path = get_sample_points_path(object_model_path)
    if npy_path is None:
        return None

    points = np.load(npy_path)  # (N_pts, 3)
    T = object_rot_wxyz.shape[0]

    # scipy expects xyzw
    rotations = R.from_quat(object_rot_wxyz[:, [1, 2, 3, 0]])

    min_heights = np.zeros(T, dtype=np.float32)
    for t in range(T):
        rotated = rotations[t].apply(points)  # (N_pts, 3)
        min_heights[t] = -rotated[:, 2].min()
    return min_heights
