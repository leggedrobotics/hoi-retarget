# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-pickle saving with a provenance ``meta`` block and schema-version stamp."""

from __future__ import annotations

import os
import pickle
from typing import Any

# Bump when the stage-pickle schema changes (keys, shapes, or semantics).
PKL_SCHEMA_VERSION = 2


def build_pkl_meta(*, source_file, robot, projection_mode, stage, cfg) -> dict[str, Any]:
    """Provenance block stored under the pickle's ``meta`` key (resolved object_scale and scale flags are top-level)."""
    return {
        "schema_version": PKL_SCHEMA_VERSION,
        "source_file": source_file,
        "robot": robot,
        "projection_mode": projection_mode,
        "stage": stage,
        "object_scale_arg": cfg.object.scale,  # None = robot default
        "scale_object_trajectory": bool(cfg.object.scale_trajectory),
    }


def save_motion_pickle(
    motion_data: dict,
    out_dir: str,
    stage_name: str,
    *,
    object_model_path: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict:
    """Write ``<out_dir>/<stage_name>.pkl`` and return the saved dict."""
    pkl_path = os.path.join(out_dir, f"{stage_name}.pkl")
    md = dict(motion_data)
    if object_model_path is not None:
        md["object_model_path"] = object_model_path
    if meta is not None:
        # Schema-2 readers expect the resolved mesh flag in meta too.
        md["meta"] = {**meta, "scale_object_mesh": bool(md.get("scale_object_mesh", True))}
    with open(pkl_path, "wb") as f:
        pickle.dump(md, f)
    print(f"[INFO] Saved {stage_name}: {pkl_path}")
    return md
