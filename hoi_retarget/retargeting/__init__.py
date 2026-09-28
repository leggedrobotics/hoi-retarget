# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sec. III-A: IK retargeting with object scaling, and the contact targets it derives."""

from hoi_retarget.retargeting.object_scaling import apply_object_scaling
from hoi_retarget.retargeting.source_data import build_projection_motion_data_with_gmr

__all__ = ["apply_object_scaling", "build_projection_motion_data_with_gmr"]
