from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from .graph import construct_pseudotime_graph_from_table


def compute_pseudotime_from_table(
    table: Any,
    *,
    adata,
    branch_col: str = "branch",
    pseudotime_col: str = "pseudotime",
    cell_col: str = "cell_id",
    backbone_mask: np.ndarray | None = None,
    backbone_selector: Callable[[Any], bool] | None = None,
    root_cell: str | None = None,
    k: int = 1,
    delta: float | None = None,
) -> np.ndarray:
    """Compute a single pseudotime value per cell from a branch table.

    Many trajectory tools can assign *multiple* pseudotime values per cell (e.g., one per
    branch/path). For visualization and downstream tasks, it can be useful to derive a
    single, consistent pseudotime per cell.

    This function:
    1) builds a directed pseudotime graph from the table (increasing pseudotime direction)
    2) chooses a root cell (explicit `root_cell`, else earliest backbone row, else earliest row)
    3) computes unweighted directed shortest-path distances from the root
    4) min-max scales distances to [0, 1]

    Parameters
    ----------
    table
        DataFrame-like object with at least (branch, pseudotime, cell_id) columns.
    adata
        AnnData (or AnnData-like) object. Pseudotime is returned in the same order as
        `adata.obs_names`.
    branch_col, pseudotime_col, cell_col
        Column names.
    backbone_mask
        Optional boolean mask over table rows indicating which rows belong to the backbone.
        If provided, used to pick the root cell as the backbone row with minimum pseudotime.
    backbone_selector
        Optional predicate `fn(branch_value) -> bool` to identify backbone rows from the
        branch column. Ignored if `backbone_mask` is provided.
    root_cell
        Optional explicit root cell ID. If provided, overrides backbone-based selection.
    k, delta
        Graph construction parameters forwarded to `construct_pseudotime_graph_from_table`.
        By default uses `k=1` directed edges.

    Returns
    -------
    pseudotime
        1D numpy array of length `adata.n_obs`, scaled to [0, 1].
    """

    if adata is None:
        raise TypeError("adata is required so pseudotime aligns to adata.obs_names")

    # Directed graph that respects increasing pseudotime ordering.
    g_dir = construct_pseudotime_graph_from_table(
        table,
        branch_col=branch_col,
        pseudotime_col=pseudotime_col,
        cell_col=cell_col,
        adata=adata,
        k=k,
        delta=delta,
        directed=True,
    )

    # Pull raw columns for selecting the root.
    try:
        col_branch = table[branch_col]
        col_pt = table[pseudotime_col]
        col_cell = table[cell_col]
    except Exception as e:  # pragma: no cover
        raise TypeError(
            "table must support table[col] access for required columns"
        ) from e

    if hasattr(col_pt, "to_numpy"):
        pt_vals = col_pt.to_numpy()
    else:
        pt_vals = np.asarray(col_pt)
    pt_vals = np.asarray(pt_vals).ravel()

    if hasattr(col_cell, "to_numpy"):
        cell_vals = col_cell.to_numpy()
    else:
        cell_vals = np.asarray(col_cell)
    cell_vals = np.asarray(cell_vals).ravel()

    if root_cell is None:
        if backbone_mask is not None:
            mask = np.asarray(backbone_mask, dtype=bool).ravel()
            if mask.shape[0] != pt_vals.shape[0]:
                raise ValueError(
                    "backbone_mask must be the same length as the number of rows in table"
                )
        elif backbone_selector is not None:
            if hasattr(col_branch, "to_numpy"):
                branch_vals = col_branch.to_numpy()
            else:
                branch_vals = np.asarray(col_branch)
            branch_vals = np.asarray(branch_vals).ravel()
            mask = np.fromiter(
                (bool(backbone_selector(b)) for b in branch_vals),
                dtype=bool,
                count=branch_vals.shape[0],
            )
        else:
            mask = None

        if mask is not None and mask.any():
            backbone_idx = np.where(mask)[0]
            root_row = backbone_idx[int(np.nanargmin(pt_vals[backbone_idx]))]
            root_cell = str(cell_vals[root_row])
        else:
            root_row = int(np.nanargmin(pt_vals))
            root_cell = str(cell_vals[root_row])

    root_idx = g_dir.index().get(str(root_cell))
    if root_idx is None:
        raise KeyError(
            f"root_cell '{root_cell}' is not present in graph node_ids; check adata.obs_names/table cell IDs"
        )

    dist = sp.csgraph.dijkstra(
        g_dir.adjacency,
        directed=True,
        indices=root_idx,
        unweighted=True,
    )
    dist = np.asarray(dist).ravel()

    finite = np.isfinite(dist)
    if not finite.any():
        return np.zeros(g_dir.n_nodes, dtype=float)

    dmin = float(dist[finite].min())
    dmax = float(dist[finite].max())
    if dmax == dmin:
        pt = np.zeros_like(dist, dtype=float)
    else:
        pt = (dist - dmin) / (dmax - dmin)

    pt[~finite] = 1.0
    return pt
