# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-window NLP statistics (variables, constraints, cost terms, solve time).

Enabled with ``--compute_analysis``; otherwise every mutator is a no-op.
"""

from __future__ import annotations

import os
from typing import Any

from hoi_retarget.config import ProjectionConfig


def _empty_pass_stats(label: str, n_frames: int) -> dict[str, Any]:
    return {
        "label": label,
        "n_frames": n_frames,
        "variables": {},
        "constraints": {},
        "costs": {},
        "solve_time": 0.0,
    }


class NLPAnalysisRecorder:
    """Collect NLP stats per solver pass: ``start_pass``, ``add_*``, ``end_pass``."""

    def __init__(self, enabled: bool, config: ProjectionConfig):
        self.enabled: bool = bool(enabled)
        self.config: ProjectionConfig = config
        self.log: list[dict[str, Any]] = []
        self.pipeline_total_time: float | None = None
        self._current: dict[str, Any] | None = None

    # ---- lifecycle ----
    def reset(self) -> None:
        self.log = []
        self.pipeline_total_time = None
        self._current = None

    def start_pass(self, label: str, n_frames: int) -> None:
        if not self.enabled:
            return
        self._current = _empty_pass_stats(label, n_frames)

    def end_pass(self, solve_time: float) -> None:
        if not self.enabled or self._current is None:
            return
        self._current["solve_time"] = solve_time
        self.log.append(self._current)
        self._current = None

    # ---- builders ----
    def add_variable(self, name: str, shape: tuple[int, ...]) -> None:
        if not self.enabled or self._current is None:
            return
        self._current["variables"][name] = tuple(shape)

    def add_constraint(self, kind: str, n: int = 1) -> None:
        if not self.enabled or self._current is None:
            return
        self._current["constraints"][kind] = self._current["constraints"].get(kind, 0) + n

    def add_cost(self, kind: str, n: int = 1) -> None:
        if not self.enabled or self._current is None:
            return
        self._current["costs"][kind] = self._current["costs"].get(kind, 0) + n

    # ---- state I/O ----
    def set_pipeline_time(self, t: float | None) -> None:
        self.pipeline_total_time = t

    # ---- output ----
    def format(self) -> str:
        """Multi-pass comparison report (header, per-pass summary, side-by-side)."""
        if not self.log:
            return "[INFO] No analysis data recorded."

        cfg = self.config
        mode = cfg.projection_mode
        W = 80
        lines: list[str] = []

        lines.append("=" * W)
        lines.append(f"  COMPUTE ANALYSIS  —  mode={mode}")
        lines.append("=" * W)

        # Per-pass summary
        lines.append("")
        for i, entry in enumerate(self.log):
            label = entry["label"]
            total_vars_n = sum((s[0] if len(s) == 1 else s[0] * s[1]) for s in entry["variables"].values())
            total_con = sum(entry["constraints"].values())
            total_cost = sum(entry["costs"].values())
            lines.append(f"  Pass {i + 1} — {label}")
            lines.append("  " + "─" * (W - 2))
            lines.append(f"  Frames: {entry['n_frames']}  |  Solve time: {entry['solve_time']:.2f}s")
            lines.append(f"  Decision vars: {total_vars_n}  |  Constraints: {total_con}  |  Cost terms: {total_cost}")
            lines.append("")

        # Side-by-side comparison table
        lines.append("=" * W)
        lines.append("  SIDE-BY-SIDE COMPARISON")
        lines.append("=" * W)

        labels = [entry["label"] for entry in self.log]
        col_width = max(20, max(len(l) for l in labels) + 2)
        header = f"{'Metric':<30s}" + "".join(f"{l:>{col_width}s}" for l in labels)
        lines.append(header)
        lines.append("-" * len(header))

        lines.append(f"{'Frames':<30s}" + "".join(f"{entry['n_frames']:>{col_width}d}" for entry in self.log))
        lines.append(
            f"{'Solve time (s)':<30s}" + "".join(f"{entry['solve_time']:>{col_width}.2f}" for entry in self.log)
        )

        # Variables
        lines.append("")
        lines.append("-- Decision Variables --")
        all_var_keys: set[str] = set()
        for entry in self.log:
            all_var_keys.update(entry["variables"].keys())
        for vk in sorted(all_var_keys):
            row = f"  {vk:<28s}"
            for entry in self.log:
                shape = entry["variables"].get(vk)
                if shape is not None:
                    total = 1
                    for s in shape:
                        total *= s
                    row += f"{total:>{col_width}d}"
                else:
                    row += f"{'--':>{col_width}s}"
            lines.append(row)
        row = f"  {'TOTAL':<28s}"
        for entry in self.log:
            total = 0
            for shape in entry["variables"].values():
                n = 1
                for s in shape:
                    n *= s
                total += n
            row += f"{total:>{col_width}d}"
        lines.append(row)

        # Constraints
        lines.append("")
        lines.append("-- Constraints --")
        all_con_keys: set[str] = set()
        for entry in self.log:
            all_con_keys.update(entry["constraints"].keys())
        for ck in sorted(all_con_keys):
            row = f"  {ck:<28s}"
            for entry in self.log:
                val = entry["constraints"].get(ck, 0)
                row += f"{val:>{col_width}d}"
            lines.append(row)
        row = f"  {'TOTAL':<28s}"
        for entry in self.log:
            total = sum(entry["constraints"].values())
            row += f"{total:>{col_width}d}"
        lines.append(row)

        # Costs
        lines.append("")
        lines.append("-- Cost Terms --")
        all_cost_keys: set[str] = set()
        for entry in self.log:
            all_cost_keys.update(entry["costs"].keys())
        for ck in sorted(all_cost_keys):
            row = f"  {ck:<28s}"
            for entry in self.log:
                val = entry["costs"].get(ck, 0)
                row += f"{val:>{col_width}d}"
            lines.append(row)
        row = f"  {'TOTAL':<28s}"
        for entry in self.log:
            total = sum(entry["costs"].values())
            row += f"{total:>{col_width}d}"
        lines.append(row)

        # Total pipeline time
        total_solve_time = sum(entry["solve_time"] for entry in self.log)
        lines.append("")
        lines.append(f"  Total solve time: {total_solve_time:.2f}s")
        if self.pipeline_total_time is not None:
            lines.append(f"  Total pipeline time (incl. preprocessing + FK): {self.pipeline_total_time:.2f}s")
        lines.append("=" * 80)
        return "\n".join(lines)

    def write(self, out_dir: str, mode_tag: str) -> str:
        """Write ``compute_analysis_{mode_tag}.txt`` under ``out_dir``. Returns the path."""
        path = os.path.join(out_dir, f"compute_analysis_{mode_tag}.txt")
        with open(path, "w") as f:
            f.write(self.format())
        print(f"[INFO] Saved compute analysis: {path}")
        return path
