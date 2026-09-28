# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-disk schema (``shifts.json``) for edited contact-point shifts.

One ``Shift`` per contact strip, keyed ``(link_name, ordinal)``: the k-th contiguous
contact run of that link. Strip enumeration is deterministic for a given source and
config, so the key is stable. Points are metres in the real-size object frame
(before ``object_scale``); the connector multiplies by ``object_scale``.
"""

from __future__ import annotations

import json
import numpy as np
import os
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = 1


def strip_key(link_name: str, ordinal: int) -> str:
    """Canonical dict key for a strip, e.g. ``left_palm_pad#0``."""
    return f"{link_name}#{ordinal}"


@dataclass
class Shift:
    link_name: str
    ordinal: int
    start: int  # edited active range, inclusive
    end: int  # edited active range, exclusive
    auto_real: list[float]  # auto-derived point, real-size object frame (3,)
    edited_real: list[float]  # user-edited point, real-size object frame (3,)
    enabled: bool = True  # False drops this strip's contact target
    auto_start: int = -1  # detected range; -1 = copy start/end
    auto_end: int = -1

    def __post_init__(self) -> None:
        if self.auto_start < 0:
            self.auto_start = int(self.start)
        if self.auto_end < 0:
            self.auto_end = int(self.end)

    @property
    def key(self) -> str:
        return strip_key(self.link_name, self.ordinal)

    @property
    def delta(self) -> np.ndarray:
        return np.asarray(self.edited_real, float) - np.asarray(self.auto_real, float)

    @property
    def moved(self) -> bool:
        return bool(np.linalg.norm(self.delta) > 1e-6)

    @property
    def retimed(self) -> bool:
        return (int(self.start), int(self.end)) != (int(self.auto_start), int(self.auto_end))


@dataclass
class ShiftSet:
    """All per-strip edits for one source clip."""

    source_path: str
    object_model_path: str
    stem: str
    object_scale: float  # geometry scale the session was built at
    shifts: dict[str, Shift] = field(default_factory=dict)
    object_aug_scale: float = 1.0  # editor size slider; multiplies object_scale
    schema_version: int = SCHEMA_VERSION

    # -- access -------------------------------------------------------------
    def get(self, link_name: str, ordinal: int) -> Shift | None:
        return self.shifts.get(strip_key(link_name, ordinal))

    def set_edited(self, link_name: str, ordinal: int, edited_real) -> None:
        s = self.shifts[strip_key(link_name, ordinal)]
        s.edited_real = [float(x) for x in edited_real]

    def reset(self, link_name: str, ordinal: int) -> None:
        s = self.shifts[strip_key(link_name, ordinal)]
        s.edited_real = list(s.auto_real)
        s.start, s.end = s.auto_start, s.auto_end

    def reset_all(self) -> None:
        for s in self.shifts.values():
            s.edited_real = list(s.auto_real)
            s.start, s.end = s.auto_start, s.auto_end

    def set_enabled(self, link_name: str, ordinal: int, enabled: bool) -> None:
        self.shifts[strip_key(link_name, ordinal)].enabled = bool(enabled)

    def set_range(self, link_name: str, ordinal: int, start: int, end: int) -> None:
        s = self.shifts[strip_key(link_name, ordinal)]
        s.start, s.end = int(start), int(end)

    def moved_count(self) -> int:
        return sum(1 for s in self.shifts.values() if s.moved and s.enabled)

    def disabled_count(self) -> int:
        return sum(1 for s in self.shifts.values() if not s.enabled)

    def retimed_count(self) -> int:
        return sum(1 for s in self.shifts.values() if s.retimed and s.enabled)

    # -- IO -----------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "object_model_path": self.object_model_path,
            "stem": self.stem,
            "object_scale": float(self.object_scale),
            "object_aug_scale": float(self.object_aug_scale),
            "shifts": {k: asdict(v) for k, v in self.shifts.items()},
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> ShiftSet:
        shifts = {k: Shift(**v) for k, v in d.get("shifts", {}).items()}
        return cls(
            source_path=d["source_path"],
            object_model_path=d.get("object_model_path", ""),
            stem=d.get("stem", ""),
            object_scale=float(d.get("object_scale", 1.0)),
            shifts=shifts,
            object_aug_scale=float(d.get("object_aug_scale", 1.0)),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )

    @classmethod
    def load(cls, path: str) -> ShiftSet:
        with open(path) as f:
            return cls.from_dict(json.load(f))
