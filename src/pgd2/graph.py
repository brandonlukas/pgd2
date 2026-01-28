from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp


def _stable_unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_obs_names(adata) -> list[str]:
    if adata is None:
        raise TypeError("adata is None")
    obs_names = getattr(adata, "obs_names", None)
    if obs_names is None:
        raise TypeError("adata must have .obs_names (AnnData-like)")
    return list(obs_names)


def _get_1d_column(table: Any, column: str) -> np.ndarray:
    """Extract a 1D column from a dataframe-like object without requiring pandas.

    Supports common patterns:
    - pandas.DataFrame: table[column] -> Series
    - dict of lists: table[column]
    - any object implementing __getitem__
    """

    try:
        col = table[column]
    except Exception as e:  # pragma: no cover
        raise TypeError(
            f"table must support column access via table[{column!r}]"
        ) from e

    if hasattr(col, "to_numpy"):
        arr = col.to_numpy()
    else:
        arr = np.asarray(col)

    if arr.ndim != 1:
        arr = np.asarray(arr).ravel()
    return arr


@dataclass(frozen=True)
class PseudotimeGraph:
    """A pseudotime graph over cells.

    Node order is critical: feature matrices X must be aligned to `node_ids`.

    Attributes
    ----------
    node_ids
        Cell IDs in node order.
    adjacency
        Sparse adjacency matrix A of shape (n_nodes, n_nodes). May be weighted and/or directed.
    """

    node_ids: tuple[str, ...]
    adjacency: sp.csr_matrix

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def index(self) -> dict[str, int]:
        return {cid: i for i, cid in enumerate(self.node_ids)}

    def degree(self, kind: str = "out") -> np.ndarray:
        """Return node degrees.

        kind="out" uses row-sums (recommended for directed graphs).
        kind="in" uses col-sums.
        """

        if kind not in {"out", "in"}:
            raise ValueError("kind must be 'out' or 'in'")
        A = self.adjacency
        if kind == "out":
            deg = np.asarray(A.sum(axis=1)).ravel()
        else:
            deg = np.asarray(A.sum(axis=0)).ravel()
        return deg

    def transition_matrix(self) -> sp.csr_matrix:
        """Row-stochastic random-walk matrix P = D^{-1} A.

        For isolated nodes (zero out-degree), this sets P[i,i] = 1 so that diffusion leaves
        those nodes unchanged.
        """

        A = self.adjacency.tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()

        # Invert degrees safely
        with np.errstate(divide="ignore"):
            inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)

        Dinv = sp.diags(inv_deg, offsets=0, shape=A.shape, format="csr")
        P = Dinv @ A

        # Fix isolated nodes: set diagonal to 1
        isolated = np.where(deg == 0)[0]
        if isolated.size:
            P = P.tolil(copy=False)
            P[isolated, isolated] = 1.0
            P = P.tocsr()

        return P

    def to_undirected(self, mode: str = "max") -> "PseudotimeGraph":
        """Return an undirected version of this graph.

        mode="max" sets A_ij = max(A_ij, A_ji).
        mode="sum" sets A_ij = A_ij + A_ji.
        """

        if mode not in {"max", "sum"}:
            raise ValueError("mode must be 'max' or 'sum'")
        A = self.adjacency.tocsr()
        if mode == "sum":
            Au = A + A.T
        else:
            Au = A.maximum(A.T)
        Au.eliminate_zeros()
        return PseudotimeGraph(node_ids=self.node_ids, adjacency=Au.tocsr())


def construct_pseudotime_graph(
    branches: Mapping[str, Sequence[str]],
    *,
    adata=None,
    k: int = 50,
    weighted: bool = False,
    directed: bool = False,
) -> PseudotimeGraph:
    """Construct a pseudotime graph from branch-specific pseudotime orderings.

    This implements Eq. (1) from the PGD paper:
    within each branch ordering C = (c1, ..., cn), connect cells within window radius k.

    Parameters
    ----------
    branches
        Dict mapping branch name -> ordered list of cell IDs (in increasing pseudotime).
    adata
        Optional AnnData (or AnnData-like) object. If provided, graph node order is
        exactly `list(adata.obs_names)` and the adjacency will have shape (n_cells, n_cells).
    k
        Window radius (k >= 1). Each cell connects to up to 2k neighbors along pseudotime.
    weighted
        If True, uses edge weights 1 / (|i-j|) within the window. If False, edges are 1.
    directed
        If True, edges point forward in pseudotime (i -> j for j>i). If False, makes edges
        bidirectional (undirected adjacency).

    Returns
    -------
    PseudotimeGraph
        Graph with adjacency matrix A.
    """

    if k < 1:
        raise ValueError("k must be >= 1")

    if adata is not None:
        node_ids = _extract_obs_names(adata)
    else:
        all_ids: list[str] = []
        for _, cells in branches.items():
            all_ids.extend(list(cells))
        node_ids = _stable_unique(all_ids)

    node_ids_t = tuple(node_ids)
    idx = {cid: i for i, cid in enumerate(node_ids_t)}
    n = len(node_ids_t)

    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    for branch_name, ordered_cells in branches.items():
        if len(ordered_cells) == 0:
            continue

        try:
            branch_idx = np.fromiter((idx[c] for c in ordered_cells), dtype=np.int64)
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(
                f"Cell ID '{missing}' from branch '{branch_name}' is not present in node_ids. "
                "If using AnnData, ensure adata.obs_names includes all branch cell IDs."
            ) from None

        m = branch_idx.size
        if m <= 1:
            continue

        max_offset = min(k, m - 1)
        for offset in range(1, max_offset + 1):
            left = branch_idx[:-offset]
            right = branch_idx[offset:]

            rows_parts.append(left)
            cols_parts.append(right)
            if weighted:
                data_parts.append(
                    np.full(left.shape[0], 1.0 / float(offset), dtype=float)
                )
            else:
                data_parts.append(np.ones(left.shape[0], dtype=float))

            if not directed:
                rows_parts.append(right)
                cols_parts.append(left)
                if weighted:
                    data_parts.append(
                        np.full(left.shape[0], 1.0 / float(offset), dtype=float)
                    )
                else:
                    data_parts.append(np.ones(left.shape[0], dtype=float))

    if rows_parts:
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        data = np.concatenate(data_parts)
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        if not weighted:
            # collapse duplicates to 1
            A.data[:] = 1.0
        A.eliminate_zeros()
    else:
        A = sp.csr_matrix((n, n), dtype=float)

    return PseudotimeGraph(node_ids=node_ids_t, adjacency=A)


def construct_pseudotime_graph_from_table(
    table: Any,
    *,
    branch_col: str = "branch",
    pseudotime_col: str = "pseudotime",
    cell_col: str = "cell_id",
    adata=None,
    k: int | None = 50,
    delta: float | None = None,
    weighted: bool = False,
    directed: bool = False,
    connect_within_pseudotime: bool = False,
) -> PseudotimeGraph:
    """Construct a pseudotime graph from a long-form table of (branch, pseudotime, cell_id).

    This is a tie-friendly variant of Eq. (1) from the PGD paper.

    Within each branch:
    - group cells by identical pseudotime values (ties)
    - sort unique pseudotime values
        - connect all cells between pseudotime levels within a window radius of `k` levels
            and/or within a pseudotime-value distance threshold `delta`

    Parameters
    ----------
    table
        DataFrame-like object with at least three columns.
    branch_col, pseudotime_col, cell_col
        Column names for branch label, pseudotime value, and cell ID.
    adata
        Optional AnnData (or AnnData-like) object. If provided, graph node order is
        exactly `list(adata.obs_names)`.
    k
        Window radius in *pseudotime-level index space* (k >= 1). Each pseudotime level
        connects to levels within ±k in the sorted unique pseudotime list. Set to None to
        disable level-based connections.
    delta
        Optional pseudotime-value distance threshold. When provided, also connect pseudotime
        levels whose *values* differ by at most `delta` within a branch. Set to None to
        disable value-based connections.
    weighted
        If True, uses edge weights 1 / (level_offset). If False, edges are 1.
    directed
        If True, edges point forward in pseudotime (lower level -> higher level).
        If False, makes edges bidirectional.
    connect_within_pseudotime
        If True, connect cells that share the same pseudotime value (clique excluding
        self-loops). This can be expensive when many cells share a pseudotime value.

    Returns
    -------
    PseudotimeGraph
        Graph with adjacency matrix A.
    """

    if k is None and delta is None:
        raise ValueError("At least one of k or delta must be provided")
    if k is not None and k < 1:
        raise ValueError("k must be >= 1")
    if delta is not None and delta < 0:
        raise ValueError("delta must be >= 0")

    branch_vals = _get_1d_column(table, branch_col)
    pt_vals = _get_1d_column(table, pseudotime_col)
    cell_vals = _get_1d_column(table, cell_col)
    if not (len(branch_vals) == len(pt_vals) == len(cell_vals)):
        raise ValueError(
            "branch_col, pseudotime_col, and cell_col must have the same length"
        )

    if adata is not None:
        node_ids = _extract_obs_names(adata)
    else:
        node_ids = _stable_unique([str(x) for x in cell_vals.tolist()])

    node_ids_t = tuple(node_ids)
    idx = {cid: i for i, cid in enumerate(node_ids_t)}
    n = len(node_ids_t)

    # branch -> pseudotime_value -> list[cell_id]
    branch_map: dict[Any, dict[Any, list[str]]] = {}
    for b, pt, cid in zip(
        branch_vals.tolist(), pt_vals.tolist(), cell_vals.tolist(), strict=True
    ):
        cid_s = str(cid)
        branch_map.setdefault(b, {}).setdefault(pt, []).append(cid_s)

    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    for branch_name, pt_to_cells in branch_map.items():
        if not pt_to_cells:
            continue

        try:
            levels = sorted(pt_to_cells.keys())
        except TypeError as e:
            raise TypeError(
                f"Pseudotime values for branch {branch_name!r} are not sortable. "
                "Ensure pseudotime_col contains comparable values (all numeric, or all strings, etc.)."
            ) from e

        level_indices: list[np.ndarray] = []
        for pt in levels:
            cell_ids = pt_to_cells[pt]
            try:
                inds = np.fromiter((idx[c] for c in cell_ids), dtype=np.int64)
            except KeyError as e:
                missing = e.args[0]
                raise KeyError(
                    f"Cell ID '{missing}' from branch {branch_name!r} is not present in node_ids. "
                    "If using AnnData, ensure adata.obs_names includes all branch cell IDs."
                ) from None
            level_indices.append(inds)

            if connect_within_pseudotime and inds.size > 1:
                # Clique within the level (excluding self)
                rr = np.repeat(inds, inds.size)
                cc = np.tile(inds, inds.size)
                mask = rr != cc
                rr = rr[mask]
                cc = cc[mask]
                rows_parts.append(rr)
                cols_parts.append(cc)
                data_parts.append(np.ones(rr.shape[0], dtype=float))

        L = len(level_indices)
        if L <= 1:
            continue

        level_vals_float: np.ndarray | None = None
        if delta is not None:
            try:
                level_vals_float = np.asarray(levels, dtype=float)
            except Exception as e:
                raise TypeError(
                    f"delta was provided but pseudotime values for branch {branch_name!r} "
                    "are not numeric."
                ) from e

        # Connect forward pairs (i -> j, j>i) and mirror if not directed.
        for i in range(L - 1):
            end = i
            if k is not None:
                end = max(end, min(L - 1, i + k))
            if level_vals_float is not None:
                # levels are sorted, so value-based neighbors are contiguous
                hi = int(
                    np.searchsorted(
                        level_vals_float,
                        level_vals_float[i] + float(delta),
                        side="right",
                    )
                    - 1
                )
                if hi > end:
                    end = hi

            if end <= i:
                continue

            left = level_indices[i]
            if left.size == 0:
                continue

            for j in range(i + 1, end + 1):
                right = level_indices[j]
                if right.size == 0:
                    continue

                offset = j - i
                rr = np.repeat(left, right.size)
                cc = np.tile(right, left.size)
                rows_parts.append(rr)
                cols_parts.append(cc)
                if weighted:
                    data_parts.append(
                        np.full(rr.shape[0], 1.0 / float(offset), dtype=float)
                    )
                else:
                    data_parts.append(np.ones(rr.shape[0], dtype=float))

                if not directed:
                    rows_parts.append(cc)
                    cols_parts.append(rr)
                    if weighted:
                        data_parts.append(
                            np.full(rr.shape[0], 1.0 / float(offset), dtype=float)
                        )
                    else:
                        data_parts.append(np.ones(rr.shape[0], dtype=float))

    if rows_parts:
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        data = np.concatenate(data_parts)
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        if not weighted:
            A.data[:] = 1.0
        A.eliminate_zeros()
    else:
        A = sp.csr_matrix((n, n), dtype=float)

    return PseudotimeGraph(node_ids=node_ids_t, adjacency=A)
