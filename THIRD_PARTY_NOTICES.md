# Third-Party Notices

HOI-Retarget is BSD-3-Clause, Copyright (c) 2026 ETH Zurich. This file covers the
third-party material redistributed **in this repository**, as those licences require.

---

## GMR — General Motion Retargeting

| | |
|---|---|
| Location | `hoi_retarget/gmr/` |
| Upstream | <https://github.com/YanjieZe/GMR> @ `5bac4bd` (2025-10-16) |
| Licence | MIT |
| Copyright | Copyright 2025 Yanjie Ze |
| Modified | Partly — `hoi_retarget/gmr/README.md` lists each file's status |

`gmr/torch_utils.py` is NVIDIA BSD-3-Clause, which GMR's MIT does not supersede;
its header carries the NVIDIA notice. `tests/test_gmr_provenance.py` pins the
unmodified files by hash and requires a provenance header on the rest.

## Unitree G1 and H2 robot descriptions

| | |
|---|---|
| Location | `assets/robots/g1/`, `assets/robots/h2/` |
| Upstream | <https://github.com/unitreerobotics/unitree_ros>, <https://github.com/unitreerobotics/unitree_mujoco>; the G1 MJCFs reached us via GMR |
| Licence | BSD 3-Clause |
| Copyright | Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") |
| Modified | Partly — see below |

`LICENSE` sits beside the meshes in each directory, as BSD-3 requires;
`test_robot_licences_travel_with_the_meshes` checks it is still there.

| File | Status |
|---|---|
| `g1/g1_mocap_29dof.xml` | Modified — 2 palm-pad `<body>` elements added |
| `g1/g1_29dof_w_hands_capsule.urdf` | Ours — collision primitives |
| `g1/LICENSE` | Unitree's, unmodified |
| `g1/README.md` | Ours |
| `h2/*` | Ours — the serial URDF re-imported through MuJoCo as `h2_hoi_mocap.xml`, plus palm-pad frames |

## Object assets

| | |
|---|---|
| Location | `assets/objects/` — 13 categories: URDF, mesh, surface samples |
| Upstream | <https://github.com/Sirui-Xu/InterMimic> |
| Licence | MIT |
| Copyright | Copyright (c) 2025 Sirui Xu |
| Modified | No — all 39 files are byte-identical to InterMimic's copies |

The geometry originates with OMOMO (MIT, Copyright (c) 2023 Jiaman Li), which
InterMimic reprocessed into this form. `assets/objects/LICENSE-OBJECTS` carries
both notices.

---

*If you believe material here is misattributed, open an issue and it will be
corrected or removed.*
