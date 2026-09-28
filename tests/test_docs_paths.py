# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every repository path named in a docstring or Markdown file must exist.

A rename leaves documentation pointing at nothing while the rest of the suite stays green.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Backticked or double-backticked paths with a file extension.
PATH_RE = re.compile(r"`{1,2}([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|sh|md|json|urdf|xml|toml|yaml|yml|cff|txt))`{1,2}")

#: Only paths rooted in one of these are ours to check; anything else is an
#: upstream project's file or a bare filename.
OWNED_PREFIXES = ("hoi_retarget/", "docs/", "tests/", "scripts/", "assets/")

#: Deliberate placeholders in worked examples, not real files.
PLACEHOLDERS = {"assets/objects/foo.urdf", "assets/objects/x.urdf"}

SKIP_DIRS = {".git", "__pycache__", "assets", "data", "outputs", ".ruff_cache", ".pytest_cache"}
SKIP_FILES = {"RELEASE_TODO.md", "BUILD_STATE.md"}
SUFFIXES = {".py", ".md", ".sh", ".toml", ".cff"}


def _sources():
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(REPO).parts):
            continue
        if path.name in SKIP_FILES:
            continue
        yield path


def test_documented_repo_paths_exist():
    dangling = {}
    for source in _sources():
        try:
            text = source.read_text()
        except UnicodeDecodeError:
            continue
        for named in set(PATH_RE.findall(text)):
            if not named.startswith(OWNED_PREFIXES) or named in PLACEHOLDERS:
                continue
            if not (REPO / named).exists():
                dangling.setdefault(named, set()).add(str(source.relative_to(REPO)))
    assert not dangling, "documentation points at paths that do not exist:\n" + "\n".join(
        f"  {name}  <- {', '.join(sorted(srcs))}" for name, srcs in sorted(dangling.items())
    )
