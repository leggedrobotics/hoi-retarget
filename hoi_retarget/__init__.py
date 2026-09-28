# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HOI-Retarget: contact-centric retargeting of human-object interaction onto humanoids.

Turns a human HOI clip (SMPL-X motion, object pose trajectory, per-body contact
labels) into a robot trajectory. Modules follow the paper's Sec. III:

- :mod:`hoi_retarget.retargeting` (III-A): IK with a rescaled object; contacts become object-frame targets.
- :mod:`hoi_retarget.optimization` (III-B): windowed trajectory optimization that recovers those targets.
- :mod:`hoi_retarget.contact` (III-C): Viser contact editor that re-runs III-B from edited targets.

Command line: ``hoi-retarget`` (see :mod:`hoi_retarget.cli`).
"""

__version__ = "1.0.0"
