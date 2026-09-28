# Installation

Linux, Python 3.10+. Pinocchio and CasADi are awkward from pip and reliable from
conda-forge, so that is the recommended route. GNU `parallel` drives
`batch_retarget.sh`.

```bash
conda create -n hoi-retarget python=3.10 -y
conda activate hoi-retarget
conda install -c conda-forge pinocchio casadi parallel -y
pip install -e .
```

Verify (the test runner comes with the `dev` extra):

```bash
pip install -e ".[dev]"
hoi-retarget --help
pytest tests/ -q
```

The tests need no data, so they stay green against a broken solver. Run one real
clip before trusting an install. Two data tests run when pointed at data:
`HOI_RETARGET_TEST_CONTACT_PKL=<a contact_window.pkl>` and
`HOI_RETARGET_TEST_CARI4D=<a CARI4D .pth>`.

## Browser tools

`hoi-retarget-view` and `hoi-retarget-edit-contacts` are viser servers. They bind
`127.0.0.1` on an auto-picked port and print the URL. `--host 0.0.0.0` opens them
to your network; over SSH, forward the port instead:

```bash
ssh -L 8791:127.0.0.1:8791 <host>
```

## Rendering

MuJoCo renders offscreen through EGL, and the entry points set `MUJOCO_GL=egl`
themselves. A libstdc++ symbol error usually means conda's libstdc++ is older
than the system driver:

```bash
conda install -c conda-forge libstdcxx-ng -y
```

Headless machines need a GPU with EGL. Without one, use `--no-record_video`;
only the mp4s are skipped, the trajectories are unaffected.

## Data

The body models and source motions are separate, licensed downloads — see
[`DATA.md`](DATA.md). Nothing runs without SMPL-X; CARI4D also needs SMPL-H.

## Where things are looked up

| What | Default | Override |
|---|---|---|
| Robots, objects, body models | `assets/` | `HOI_RETARGET_ASSETS` |
| Extra object-mesh roots (released rows) | none | `HOI_RETARGET_OBJECT_ROOTS` |

The overrides are read at import time, so set them before starting Python.

`assets/` sits beside the package rather than inside it, so a built wheel does
not carry it. Install with `pip install -e .` from a checkout, or point
`HOI_RETARGET_ASSETS` at a copy of `assets/`.
