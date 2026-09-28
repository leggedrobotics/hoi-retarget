# Derived from GMR (General Motion Retargeting).
# Source: https://github.com/YanjieZe/GMR @ 5bac4bd (2025-10-16)
# Copyright 2025 Yanjie Ze
#
# SPDX-License-Identifier: MIT
#
# Modified by ETH Zurich, 2026, under the same MIT terms:
#   - dropped `from rich import print`, which shadowed the builtin;
#   - exported ROBOT_URDF_DICT and SUPPORTED_ROBOTS.
"""Vendored fork of GMR: the SMPL-X -> humanoid IK retargeting of Sec. III-A.

Everything in this subpackage is MIT-licensed (see ``LICENSE``); everything
outside it is BSD-3-Clause. ``README.md`` records what was changed, file by
file, against the upstream baseline.
"""

from .data_loader import load_robot_motion
from .kinematics_model import KinematicsModel
from .motion_retarget import GeneralMotionRetargeting
from .params import (
    ASSET_ROOT,
    IK_CONFIG_DICT,
    IK_CONFIG_ROOT,
    ROBOT_BASE_DICT,
    ROBOT_URDF_DICT,
    ROBOT_XML_DICT,
    SUPPORTED_ROBOTS,
    VIEWER_CAM_DISTANCE_DICT,
)
from .robot_motion_viewer import RobotMotionViewer

__all__ = [
    "ASSET_ROOT",
    "IK_CONFIG_DICT",
    "IK_CONFIG_ROOT",
    "ROBOT_BASE_DICT",
    "ROBOT_URDF_DICT",
    "ROBOT_XML_DICT",
    "SUPPORTED_ROBOTS",
    "VIEWER_CAM_DISTANCE_DICT",
    "GeneralMotionRetargeting",
    "KinematicsModel",
    "RobotMotionViewer",
    "load_robot_motion",
]
