"""Aggregate a single pseudotime per cell from a multi-branch table.

Auxiliary utility: not part of the PGD paper method and not required by
:func:`pgd2.diffuse_features`. Useful when a trajectory tool emits multiple
pseudotime values per cell (one per branch) and you want a single canonical
value for visualization or downstream analysis.

The function builds a directed pseudotime graph from the table, picks a root
cell (explicit, backbone-derived, or globally earliest), runs unweighted
Dijkstra from the root, and min-max scales distances to [0, 1].
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .graph import _column, _pairs_to_csr, _table_to_forward_pairs


def aggregate_pseudotime_from_table(
    table: Any,
    *,
    adata: Any,
    branch_col: str = "branch",
    pseudotime_col: str = "pseudotime",
    cell_col: str = "cell_id",
    root_cell: str | None = None,
    backbone_mask: np.ndarray | None = None,
    backbone_selector: Callable[[Any], bool] | None = None,
    k: int = 1,
    delta: float | None = None,
) -> np.ndarray:
    """Compute one canonical pseudotime value per cell from a multi-branch table.

    Pseudotime is returned in ``adata.obs_names`` order. Root selection priority:
    explicit ``root_cell`` → backbone row with minimum pseudotime
    (via ``backbone_mask`` or ``backbone_selector``) → globally minimum-pseudotime row.
    """

    if adata is None:
        raise TypeError("adata is required so pseudotime aligns to adata.obs_names")

    node_ids_t, n, pairs = _table_to_forward_pairs(
        table,
        branch_col=branch_col,
        pseudotime_col=pseudotime_col,
        cell_col=cell_col,
        adata=adata,
        k=k,
        delta=delta,
    )
    A_dir = _pairs_to_csr(n, pairs, symmetric=False)

    pt_vals = _column(table, pseudotime_col)
    cell_vals = _column(table, cell_col)

    if root_cell is None:
        mask = _backbone_mask(table, branch_col, backbone_mask, backbone_selector, len(pt_vals))
        if mask is not None and mask.any():
            cand = np.where(mask)[0]
            root_row = cand[int(np.nanargmin(pt_vals[cand]))]
        else:
            root_row = int(np.nanargmin(pt_vals))
        root_cell = str(cell_vals[root_row])

    idx_map = {cid: i for i, cid in enumerate(node_ids_t)}
    root_idx = idx_map.get(str(root_cell))
    if root_idx is None:
        raise KeyError(
            f"root_cell '{root_cell}' is not in graph node_ids; "
            "check adata.obs_names / table cell IDs"
        )

    dist = np.asarray(
        dijkstra(A_dir, directed=True, indices=root_idx, unweighted=True)
    ).ravel()

    finite = np.isfinite(dist)
    n_unreachable = int((~finite).sum())
    if n_unreachable:
        warnings.warn(
            f"{n_unreachable}/{n} cells unreachable from root '{root_cell}'; "
            "their pseudotime is clamped to 1.0. The directed graph likely has "
            "multiple sources (e.g. tip-to-tip branches) — consider reorienting "
            "branches so they all flow outward from a single root.",
            stacklevel=2,
        )
    if not finite.any():
        return np.zeros(n, dtype=float)
    dmin = float(dist[finite].min())
    dmax = float(dist[finite].max())
    if dmax == dmin:
        pt = np.zeros_like(dist, dtype=float)
    else:
        pt = (dist - dmin) / (dmax - dmin)
    pt[~finite] = 1.0
    return pt


def _backbone_mask(
    table: Any,
    branch_col: str,
    backbone_mask: np.ndarray | None,
    backbone_selector: Callable[[Any], bool] | None,
    n_rows: int,
) -> np.ndarray | None:
    if backbone_mask is not None:
        mask = np.asarray(backbone_mask, dtype=bool).ravel()
        if mask.shape[0] != n_rows:
            raise ValueError(
                "backbone_mask must be the same length as the number of table rows"
            )
        return mask
    if backbone_selector is not None:
        branch_vals = _column(table, branch_col)
        return np.fromiter(
            (bool(backbone_selector(b)) for b in branch_vals),
            dtype=bool,
            count=branch_vals.shape[0],
        )
    return None
