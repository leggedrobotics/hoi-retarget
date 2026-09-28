#!/usr/bin/env python3
# Copyright (c) 2026, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve a loose motion name to one contact-editor source file (.pt or .pth).

    python -m hoi_retarget.contact.resolve_clip sub10_whitechair_45     # -> one path
    python -m hoi_retarget.contact.resolve_clip --list whitechair       # -> all matches

Exit codes: 0 resolved (path on stdout), 2 no match, 3 ambiguous (candidates on stderr).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from hoi_retarget.paths import DATA_ROOT, REPO_ROOT

REPO_ROOT = str(REPO_ROOT)

# (dataset label, root, extension); overlapping roots are de-duplicated by path.
SEARCH_ROOTS = [
    ("OMOMO", os.path.join(str(DATA_ROOT), "InterMimic", "OMOMO_new"), ".pt"),
    ("CARI4D", os.path.join(str(DATA_ROOT), "CARI4D"), ".pth"),
    ("test", os.path.join(str(DATA_ROOT), "InterMimic"), ".pt"),
]


def _index_all():
    """[(dataset, stem, abspath)] for every source file under the search roots."""
    out, seen = [], set()
    for dataset, root, ext in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(ext):
                    continue
                path = os.path.join(dirpath, fn)
                if path in seen:
                    continue
                seen.add(path)
                out.append((dataset, os.path.splitext(fn)[0], path))
    return out


def _normalise(q: str) -> str:
    """Loose user text -> a stem-shaped token: spaces/dashes -> underscores."""
    q = q.strip()
    q = re.sub(r"\.(pt|pth)$", "", q)
    q = re.sub(r"[\s\-]+", "_", q)
    return q


def _pad_variants(q: str):
    """OMOMO stems zero-pad the index to 3: 'sub10_whitechair_45' -> '..._045'."""
    m = re.match(r"^(.*_)(\d{1,3})$", q)
    if not m:
        return []
    head, idx = m.group(1), m.group(2)
    return [f"{head}{int(idx):03d}"] if len(idx) < 3 else []


def resolve(query: str, entries=None):
    """(path, [candidates]). path is None when 0 or >1 entries match."""
    entries = entries if entries is not None else _index_all()

    # 0. A real path the user pasted verbatim.
    for cand in (query, os.path.join(REPO_ROOT, query)):
        if os.path.isfile(cand):
            return os.path.abspath(cand), []

    q = _normalise(query)
    forms = [q] + _pad_variants(q)

    # 1./2. Exact stem, then case-insensitive exact stem; >1 hit is ambiguous.
    for form in forms:
        for matcher in (lambda s, f: s == f, lambda s, f: s.lower() == f.lower()):
            hits = [e for e in entries if matcher(e[1], form)]
            if len(hits) == 1:
                return hits[0][2], []
            if len(hits) > 1:
                return None, hits

    # 3. Substring, case-insensitive.
    hits = [e for e in entries if q.lower() in e[1].lower()]
    if len(hits) == 1:
        return hits[0][2], []
    return None, hits


def _fmt(hits, limit=40):
    lines = [
        f"  {d:7s} {s:40s} {os.path.relpath(p, REPO_ROOT)}" for d, s, p in sorted(hits, key=lambda e: e[1])[:limit]
    ]
    if len(hits) > limit:
        lines.append(f"  … and {len(hits) - limit} more")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", help="Motion name, partial name, or path")
    ap.add_argument("--list", action="store_true", help="List every match instead of resolving to one")
    args = ap.parse_args()

    entries = _index_all()
    if args.list:
        q = _normalise(args.query).lower()
        hits = [e for e in entries if q in e[1].lower()]
        if not hits:
            print(f"no source file matches '{args.query}'", file=sys.stderr)
            return 2
        print(f"{len(hits)} match(es) for '{args.query}':", file=sys.stderr)
        print(_fmt(hits), file=sys.stderr)
        return 0

    path, cands = resolve(args.query, entries)
    if path:
        print(path)
        return 0
    if not cands:
        print(f"no source file matches '{args.query}' under data/InterMimic or data/CARI4D", file=sys.stderr)
        return 2
    print(f"'{args.query}' is ambiguous — {len(cands)} matches:", file=sys.stderr)
    print(_fmt(cands), file=sys.stderr)
    print("Re-run with a fuller name (e.g. sub10_whitechair_045).", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
