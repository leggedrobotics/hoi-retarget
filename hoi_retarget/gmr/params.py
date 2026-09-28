# Derived from GMR (General Motion Retargeting).
# Source: https://github.com/YanjieZe/GMR @ 5bac4bd (2025-10-16)
# Copyright 2025 Yanjie Ze
#
# SPDX-License-Identifier: MIT
#
# Modified by ETH Zurich, 2026, under the same MIT terms:
#   - asset paths resolve through hoi_retarget.paths instead of a package-relative
#     constant, so an installed copy can point at a shared asset tree;
#   - added ROBOT_URDF_DICT: the trajectory-optimization stage builds a
#     Pinocchio model, which needs a URDF rather than the MJCF used for IK;
#   - added the Unitree H2 entries;
#   - removed the BVH, NOKOV and FBX input paths together with their IK configs
#     (see THIRD_PARTY_NOTICES.md, "Removed for licence reasons").
"""Robot and IK-config registries."""

from __future__ import annotations

from hoi_retarget.paths import ASSET_ROOT, PACKAGE_ROOT, ROBOT_ROOT

IK_CONFIG_ROOT = PACKAGE_ROOT / "gmr" / "ik_configs"

#: MJCF used by the IK stage and by every MuJoCo renderer.
ROBOT_XML_DICT = {
    "unitree_g1": ROBOT_ROOT / "g1" / "g1_mocap_29dof.xml",
    "unitree_h2": ROBOT_ROOT / "h2" / "h2_hoi_mocap.xml",
}

#: URDF used by the Pinocchio model in the trajectory-optimization stage.
ROBOT_URDF_DICT = {
    "unitree_g1": ROBOT_ROOT / "g1" / "g1_29dof_w_hands_capsule.urdf",
    # H2 ships the SERIAL urdf, and the MJCF above is that same urdf re-imported
    # through MuJoCo. Unitree's own H2 MJCF models the parallel ankle and knee
    # linkages explicitly (nq 68 against the urdf's 38, plus 6 equality
    # constraints), which breaks the MuJoCo/Pinocchio joint-order agreement that
    # stage 1 -> stage 2 depends on. See tools/check_robot_assets.py.
    "unitree_h2": ROBOT_ROOT / "h2" / "h2_hoi.urdf",
}

IK_CONFIG_DICT = {
    "smplx": {
        "unitree_g1": IK_CONFIG_ROOT / "smplx_to_g1.json",
        "unitree_h2": IK_CONFIG_ROOT / "smplx_to_h2.json",
    },
}

ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
    "unitree_h2": "pelvis",
}

VIEWER_CAM_DISTANCE_DICT = {
    "unitree_g1": 2.0,
    "unitree_h2": 2.7,  # 1.80 m robot: the G1's 2.0 crops the head and feet
}

#: Robots this package ships assets and an IK config for.
SUPPORTED_ROBOTS = tuple(ROBOT_URDF_DICT)
