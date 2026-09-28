# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset and data paths.

Assets come from ``$HOI_RETARGET_ASSETS`` if set, else ``assets/`` beside the package.
SMPL-X body models are not shipped; the user places them in ``<assets>/body_models/smplx/``.
"""

from __future__ import annotations

import os
import pathlib

#: The installed package directory.
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

#: The repository root, when running from a source checkout.
REPO_ROOT = PACKAGE_ROOT.parent


def asset_root() -> pathlib.Path:
    """Directory holding ``robots/``, ``objects/`` and ``body_models/``."""
    override = os.environ.get("HOI_RETARGET_ASSETS")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    return REPO_ROOT / "assets"


#: Evaluated at import time: set ``HOI_RETARGET_ASSETS`` before importing the package.
ASSET_ROOT = asset_root()


def object_roots() -> list[pathlib.Path]:
    """Extra object-mesh directories from ``$HOI_RETARGET_OBJECT_ROOTS`` (``os.pathsep``-separated).

    Highest priority first; read at call time. Unlike ``HOI_RETARGET_ASSETS`` it does not hide ``robots/``.
    """
    raw = os.environ.get("HOI_RETARGET_OBJECT_ROOTS", "")
    return [pathlib.Path(r).expanduser() for r in raw.split(os.pathsep) if r]


def resolve_asset_path(path: str | os.PathLike) -> pathlib.Path:
    """Resolve an asset path recorded in a pickle (``assets/objects/<x>.urdf`` or ``<dataset>/_assets/<x>.urdf``).

    Search order: cwd, repo, asset root, then each ``HOI_RETARGET_OBJECT_ROOTS`` entry
    (as recorded and without a leading ``assets/``). Missing files return the input made absolute.
    """
    p = pathlib.Path(path)
    if p.is_absolute():
        if p.is_file():
            return p
        # Absolute path from the producing machine: try its tail under the object roots.
        for root in object_roots():
            for n in (3, 2, 1):
                if len(p.parts) > n:
                    candidate = root.joinpath(*p.parts[-n:])
                    if candidate.is_file():
                        return candidate
        return p
    tail = pathlib.Path(*p.parts[1:]) if p.parts[:1] == ("assets",) else p
    candidates = [pathlib.Path.cwd() / p, REPO_ROOT / p, ASSET_ROOT / tail]
    for root in object_roots():
        candidates += [root / p, root / tail]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return p.absolute()


ROBOT_ROOT = ASSET_ROOT / "robots"
OBJECT_ROOT = ASSET_ROOT / "objects"
BODY_MODEL_ROOT = ASSET_ROOT / "body_models"


def require_asset_root() -> pathlib.Path:
    """Return ``ASSET_ROOT``, or raise FileNotFoundError (``assets/`` is not in a built wheel)."""
    if ASSET_ROOT.is_dir():
        return ASSET_ROOT
    how = (
        f"HOI_RETARGET_ASSETS is set to {ASSET_ROOT}, which does not exist."
        if os.environ.get("HOI_RETARGET_ASSETS")
        else f"Looked beside the package, at {ASSET_ROOT}."
    )
    raise FileNotFoundError(
        f"Robot and object assets not found. {how}\n"
        "assets/ is not carried by a built wheel. Either run from a source "
        "checkout of the repository, or set HOI_RETARGET_ASSETS to a copy of "
        "its assets/ directory."
    )


def body_model_dir() -> pathlib.Path:
    """SMPL-X model directory; raises FileNotFoundError with download instructions if empty."""
    path = BODY_MODEL_ROOT / "smplx"
    if not any(path.glob("SMPLX_*.npz")) and not any(path.glob("SMPLX_*.pkl")):
        raise FileNotFoundError(
            f"SMPL-X body models not found at {path}.\n"
            "They are not redistributable and are not part of this repository.\n"
            "Register at https://smpl-x.is.tue.mpg.de/, download the model files, "
            f"and place SMPLX_{{NEUTRAL,MALE,FEMALE}}.npz (or .pkl) in {path}. See docs/DATA.md."
        )
    return path


def smplx_ext() -> str:
    """``npz`` if SMPL-X ``.npz`` files are installed, else ``pkl`` (``smplx.create`` defaults to npz)."""
    return "npz" if any(body_model_dir().glob("SMPLX_*.npz")) else "pkl"


#: Where the contact editor writes its re-solved clips.
CONTACT_EDITOR_OUTPUT = REPO_ROOT / "outputs" / "contact_editor"

#: Where source motion datasets are expected (not shipped; see ``docs/DATA.md``).
DATA_ROOT = REPO_ROOT / "data"
