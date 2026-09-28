# `hoi_retarget/gmr/` — vendored fork of GMR

Fork of [GMR (General Motion Retargeting)](https://github.com/YanjieZe/GMR) by Yanjie Ze,
baseline **`5bac4bd`, 2025-10-16**. It provides the SMPL-X → humanoid IK that Sec. III-A builds on.

**Licence: files under `hoi_retarget/gmr/` are MIT (`LICENSE` here); files outside it are
BSD-3-Clause.** Exceptions: `torch_utils.py` keeps NVIDIA's BSD-3 header (GMR's MIT does not
supersede it), and the robot and object assets carry their own licences (see
`THIRD_PARTY_NOTICES.md`).

| File | Status | Change |
|---|---|---|
| `data_loader.py` | Verbatim | — |
| `kinematics_model.py` | Verbatim | — |
| `torch_utils.py` | Verbatim | — (NVIDIA BSD-3 header) |
| `utils/__init__.py` | Verbatim | — |
| `LICENSE` | Verbatim | GMR's MIT text |
| `ik_configs/DOC.md` | Verbatim | IK-config format notes (upstream keeps it at the repo root) |
| `motion_retarget.py` | Modified | Ground clearance from IK-config key `ground_offset` (default 0.02 m) instead of 0.1 m |
| `params.py` | Modified | Paths via `hoi_retarget.paths`; `ROBOT_URDF_DICT`; Unitree H2; BVH/NOKOV/FBX paths removed |
| `rot_utils.py` | Modified | CasADi-compatible quaternion helpers |
| `utils/smpl.py` | Modified | Sources read through `hoi_retarget.datasets`; LAFAN1 call removed |
| `robot_motion_viewer.py` | Modified | Rewritten for HOI (161 → 888 lines): objects, contact markers, offscreen rendering, render styles |
| `__init__.py` | Modified | Dropped `from rich import print`; exports added symbols |
| `ik_configs/smplx_to_g1.json` | Modified | Added `"ground_offset": 0.02` |
| `ik_configs/smplx_to_h2.json` | Modified | H2 retuning of the G1 config (386 of 477 lines identical) |

Verbatim files carry no added header; `tests/test_gmr_provenance.py` pins their sha256
against upstream. The directory is excluded from `ruff format` and the licence-header stamp.

**Removed:** upstream's BVH/LAFAN1 input path (`utils/lafan_vendor/`, `utils/lafan1.py`
and the four G1 BVH/FBX IK configs), because its helpers are CC BY-NC-ND 4.0. Its quaternion
product was replaced by `rot_utils.quat_mul_wxyz` (used by the optimization; checked
against SciPy by `tests/test_gmr_provenance.py`).
