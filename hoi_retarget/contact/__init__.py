# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manual contact-point editing for contact-mode retargeting.

Open a source clip in a 3D editor, move, retime or disable each contact strip's object-frame
contact point, then re-run the contact-mode solve from the edited points.

- ``app``      : viser editor (``hoi-retarget-edit-contacts <src>``)
- ``retarget`` : injects the edits and runs the solve
- ``shifts``   : the resumable ``shifts.json`` schema
"""

import os as _os

# Headless MuJoCo rendering; must be set before mujoco is first imported.
_os.environ.setdefault("MUJOCO_GL", "egl")
_os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
