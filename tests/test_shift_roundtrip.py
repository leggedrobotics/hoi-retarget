# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Editor shifts round-trip, in both contact-point modes (per-strip mean and per-frame).

An untouched ShiftSet reproduces flags and points exactly; a moved strip is a flat offset that
keeps the per-frame variation. Needs ``HOI_RETARGET_TEST_CONTACT_PKL`` (any ``contact_window.pkl``).
"""

from __future__ import annotations

import copy
import numpy as np
import os
import pickle
from pathlib import Path

import pytest

from hoi_retarget.contact.retarget import _inject_shifts
from hoi_retarget.contact.shifts import Shift, ShiftSet, strip_key

CLIP = Path(os.environ.get("HOI_RETARGET_TEST_CONTACT_PKL", ""))
CP_KEY = "fixed_contact_points_per_frame_in_object_frame"


def _segments(flags):
    out, start = [], None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def _untouched_shiftset(md):
    """Exactly what the app builds before the user touches anything (auto = strip centroid)."""
    cp = np.asarray(md[CP_KEY], np.float32)
    flags = np.asarray(md["per_link_contact_flags"])
    scale = float(md.get("object_scale", 1.0))
    ss = ShiftSet(source_path="_", object_model_path="_", stem="_", object_scale=scale)
    for c, link in enumerate(md["contact_link_names"]):
        for k, (s, e) in enumerate(_segments(flags[:, c])):
            with np.errstate(invalid="ignore"):
                auto = np.nanmean(cp[s:e, c], axis=0) / scale
            if not np.all(np.isfinite(auto)):
                continue
            ss.shifts[strip_key(link, k)] = Shift(
                link_name=link,
                ordinal=k,
                start=int(s),
                end=int(e),
                auto_real=[float(x) for x in auto],
                edited_real=[float(x) for x in auto],
            )
    return ss


@pytest.fixture(scope="module")
def md_segmean():
    if not CLIP.is_file():
        pytest.skip("set HOI_RETARGET_TEST_CONTACT_PKL to a contact_window.pkl written by hoi-retarget")
    return pickle.load(open(CLIP, "rb"))


@pytest.fixture(scope="module")
def md_perframe(md_segmean):
    """Same clip with a synthetic per-frame ramp, so points VARY inside each strip."""
    md = copy.deepcopy(md_segmean)
    cp = np.asarray(md[CP_KEY], np.float32).copy()
    flags = np.asarray(md["per_link_contact_flags"])
    for c in range(cp.shape[1]):
        for s, e in _segments(flags[:, c]):
            if not np.isfinite(cp[s:e, c]).all():
                continue
            ramp = np.linspace(-0.02, 0.02, e - s)[:, None] * np.array([1.0, 0.5, -0.3])
            cp[s:e, c] += ramp.astype(np.float32)
    md[CP_KEY] = cp
    return md


@pytest.mark.parametrize("mode", ["segmean", "perframe"])
def test_untouched_shiftset_is_identity(mode, md_segmean, md_perframe):
    md0 = md_segmean if mode == "segmean" else md_perframe
    cp0 = np.asarray(md0[CP_KEY], np.float32).copy()
    flags0 = np.asarray(md0["per_link_contact_flags"]).copy()

    md = copy.deepcopy(md0)
    _inject_shifts(md, _untouched_shiftset(md0))
    cp1 = np.asarray(md[CP_KEY], np.float32)

    assert np.array_equal(flags0, np.asarray(md["per_link_contact_flags"])), "flags changed"
    assert np.array_equal(np.isfinite(cp0), np.isfinite(cp1)), "NaN pattern changed"
    both_nan = ~np.isfinite(cp0) & ~np.isfinite(cp1)
    diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(cp0) - np.nan_to_num(cp1)))
    assert diff.max() < 1e-6, f"{mode}: untouched round-trip altered points by {diff.max():.3e}"


@pytest.mark.parametrize("mode", ["segmean", "perframe"])
def test_moved_strip_is_a_flat_offset(mode, md_segmean, md_perframe):
    """The gizmo must translate the strip, not flatten it."""
    md0 = md_segmean if mode == "segmean" else md_perframe
    cp0 = np.asarray(md0[CP_KEY], np.float32).copy()
    scale = float(md0.get("object_scale", 1.0))
    links = list(md0["contact_link_names"])

    ss = _untouched_shiftset(md0)
    sh = ss.shifts[next(iter(ss.shifts))]
    sh.edited_real = [sh.auto_real[0] + 0.05, sh.auto_real[1], sh.auto_real[2]]

    md = copy.deepcopy(md0)
    _inject_shifts(md, ss)
    cp1 = np.asarray(md[CP_KEY], np.float32)

    c, s, e = links.index(sh.link_name), sh.start, sh.end
    dx = cp1[s:e, c, 0] - cp0[s:e, c, 0]
    assert np.allclose(dx, 0.05 * scale, atol=1e-6), "x offset is not flat across the strip"
    assert np.allclose(cp1[s:e, c, 1:], cp0[s:e, c, 1:], atol=1e-6, equal_nan=True), (
        "untouched axes moved -- the strip was overwritten, not offset"
    )
