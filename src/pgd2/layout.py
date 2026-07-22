"""Abstract pseudotime-only layout of cells from a long-form trajectory table.

A from-scratch layout (contrast :func:`pgd2.diffuse_features`, which perturbs an
*existing* embedding): :func:`dendrogram_from_table` builds a rooted URD-style
forking tree from cells' overlapping-lineage membership, with pseudotime as the y
axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .graph import _column
from .pseudotime import aggregate_pseudotime_from_table

__all__ = [
    "Dendrogram",
    "dendrogram_from_table",
]


@dataclass
class Dendrogram:
    """Result of :func:`dendrogram_from_table` -- a rooted URD-style tree layout.

    ``coords`` (one row per cell, aligned to ``cell_ids``) holds each cell's
    ``(x, y)`` where y is pseudotime and x is its segment's branch position plus
    jitter. ``segment_x`` maps each segment (a sorted tuple of the lineage labels a
    cell shares) to its x. ``lines`` is an ``(m, 4)`` array of ``(x1, y1, x2, y2)``
    right-angled skeleton segments for drawing the tree spine.
    """

    coords: np.ndarray
    cell_ids: tuple[str, ...]
    cell_segments: tuple[tuple[str, ...], ...]
    segment_x: dict[tuple[str, ...], float]
    lines: np.ndarray


def dendrogram_from_table(
    table: Any,
    *,
    adata: Any = None,
    pseudotime: dict[str, float] | None = None,
    branch_col: str = "branch",
    cell_col: str = "cell_id",
    jitter: float = 0.15,
    jitter_push: float = 0.05,
    seed: int | None = None,
    **pt_kwargs: Any,
) -> Dendrogram:
    """Build a rooted dendrogram (URD tree-plot style) from an overlapping-lineage table.

    Each cell's *segment* is the set of lineages it belongs to; segments partition
    into a trunk that splits into branches. The tree is derived from those segments
    and a per-cell pseudotime: y = pseudotime, x = a tidy-dendrogram branch position
    (leaves spread evenly, each parent at the mean x of its children), cells jittered
    beside their segment's spine.

    ``pseudotime`` is a dict ``cell_id -> value``; if omitted it is computed with
    :func:`~pgd2.aggregate_pseudotime_from_table` (which needs ``adata`` and accepts
    ``root_cell`` / ``backbone_selector`` via ``pt_kwargs``).

    ponytail: nearest-lower-pseudotime parent assignment -- fine for a trunk with one
    split per node; revisit if a single segment forks into 3+ children at one point.
    """

    branch_vals = [str(b) for b in _column(table, branch_col).tolist()]
    row_cells = [str(c) for c in _column(table, cell_col).tolist()]

    membership: dict[str, set[str]] = {}
    for b, c in zip(branch_vals, row_cells, strict=True):
        membership.setdefault(c, set()).add(b)
    cells = list(membership.keys())

    pt = _resolve_pseudotime(pseudotime, table, adata, cells, branch_col, cell_col, pt_kwargs)

    cell_seg = {c: tuple(sorted(membership[c])) for c in cells}
    segments = sorted(set(cell_seg.values()))
    seg_pt = {s: np.array([pt[c] for c in cells if cell_seg[c] == s]) for s in segments}
    start = {s: float(v.min()) for s, v in seg_pt.items()}
    end = {s: float(v.max()) for s, v in seg_pt.items()}

    parent, children_of = _segment_tree(segments, start, end)
    seg_x = _dendrogram_x(segments, children_of, parent)

    rng = np.random.default_rng(seed)
    coords = np.empty((len(cells), 2), dtype=float)
    for i, c in enumerate(cells):
        s = cell_seg[c]
        direction = 1.0 if rng.random() < 0.5 else -1.0
        coords[i, 0] = seg_x[s] + direction * rng.uniform(jitter_push, jitter + jitter_push)
        coords[i, 1] = pt[c]

    lines = _skeleton(segments, parent, children_of, seg_x, start, end)

    return Dendrogram(
        coords=coords,
        cell_ids=tuple(cells),
        cell_segments=tuple(cell_seg[c] for c in cells),
        segment_x=seg_x,
        lines=lines,
    )


def _resolve_pseudotime(pseudotime, table, adata, cells, branch_col, cell_col, pt_kwargs):
    if pseudotime is None:
        pt_arr = aggregate_pseudotime_from_table(
            table, adata=adata, branch_col=branch_col, cell_col=cell_col, **pt_kwargs
        )
        obs = [str(c) for c in adata.obs_names]
        return {c: float(v) for c, v in zip(obs, pt_arr, strict=True)}
    if isinstance(pseudotime, dict):
        return {str(k): float(v) for k, v in pseudotime.items()}
    raise TypeError("pseudotime must be None (computed via aggregate) or a dict cell_id -> value")


def _segment_tree(segments, start, end):
    """Parent = nearest lower-start segment sharing >=1 lineage. Returns (parent, children_of)."""

    parent: dict[tuple, tuple | None] = {}
    children_of: dict[tuple, list[tuple]] = {}
    for s in segments:
        cands = [
            p for p in segments
            if p != s and start[p] < start[s] and set(p) & set(s)
        ]
        parent[s] = max(cands, key=lambda p: start[p]) if cands else None
        if parent[s] is not None:
            children_of.setdefault(parent[s], []).append(s)
    return parent, children_of


def _dendrogram_x(segments, children_of, parent):
    """Tidy layout: leaves spread 1,2,3,...; each parent at the mean x of its children."""

    roots = [s for s in segments if parent[s] is None]
    x: dict[tuple, float] = {}
    counter = [0]

    def assign(seg):
        kids = children_of.get(seg, [])
        if not kids:
            counter[0] += 1
            x[seg] = float(counter[0])
        else:
            x[seg] = float(np.mean([assign(k) for k in kids]))
        return x[seg]

    for r in roots:
        assign(r)
    return x


def _skeleton(segments, parent, children_of, seg_x, start, end):
    """Right-angled spine: each segment vertical + a horizontal connector to each child."""

    lines = []
    for s in segments:
        lines.append((seg_x[s], start[s], seg_x[s], end[s]))  # vertical spine
        for child in children_of.get(s, []):
            lines.append((seg_x[s], end[s], seg_x[child], start[child]))  # squared corner
    return np.array(lines, dtype=float) if lines else np.empty((0, 4), dtype=float)


def _demo() -> None:
    """Self-check: a trunk that splits into two branches."""
    # two lineages A,B share a trunk (cells in both), then each has its own branch
    table = {
        "branch": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "cell_id": ["c0", "c1", "c2", "c3", "c0", "c1", "c4", "c5"],
    }
    pt = {"c0": 0.0, "c1": 0.3, "c2": 0.7, "c3": 1.0, "c4": 0.7, "c5": 0.9}
    d = dendrogram_from_table(table, pseudotime=pt, jitter=0.0, jitter_push=0.0, seed=0)
    assert set(d.segment_x) == {("A", "B"), ("A",), ("B",)}
    assert d.segment_x[("A", "B")] == 1.5  # trunk at mean of its two leaves
    i0 = d.cell_ids.index("c0")
    assert d.coords[i0, 1] == 0.0 and d.coords[i0, 0] == 1.5
    # branch cell c3 is on leaf A, above the branchpoint
    i3 = d.cell_ids.index("c3")
    assert d.cell_segments[i3] == ("A",) and d.coords[i3, 1] == 1.0
    print("ok")


if __name__ == "__main__":
    _demo()
