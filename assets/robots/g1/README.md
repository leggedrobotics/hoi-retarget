# Unitree G1 — robot description

Derived from Unitree's G1 description package
(<https://github.com/unitreerobotics/unitree_ros>), BSD-3-Clause. The `LICENSE`
beside these files is Unitree's and must travel with them.

Unitree publishes nine G1 variants; only what this pipeline loads is kept here.

| File | Provenance |
|---|---|
| `g1_mocap_29dof.xml` | Unitree's MJCF plus two palm-pad `<body>` elements, which carry the contact frames |
| `g1_29dof_w_hands_capsule.urdf` | Ours. Collision primitives (cylinders and boxes) and the hand links the contact targets attach to |
| `meshes/` | Unitree's, unmodified |

`THIRD_PARTY_NOTICES.md` at the repository root is the authoritative statement.
