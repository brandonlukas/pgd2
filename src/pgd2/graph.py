"""Pseudotime graph construction and the row-stochastic transition operator.

A pseudotime graph connects cells that are close along an inferred trajectory.
It is built from a long-form (branch, pseudotime, cell_id) table, connecting
cells within a sliding pseudotime window per branch (PGD paper Eq. 1) while
handling ties on pseudotime values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass
class PseudotimeGraph:
    """A pseudotime graph over cells.

    Feature matrices must be row-aligned to ``node_ids``. Treat instances as
    immutable after construction — mutating ``adjacency`` does not invalidate
    the cached ``transition_matrix``.
    """

    node_ids: tuple[str, ...]
    adjacency: sp.csr_matrix

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @cached_property
    def transition_matrix(self) -> sp.csr_matrix:
        """Row-stochastic random-walk matrix P = D^{-1} A.

        Isolated nodes get a self-loop so diffusion leaves them unchanged.
        """

        A = self.adjacency
        deg = np.asarray(A.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            inv = np.where(deg > 0, 1.0 / deg, 0.0)
        P = sp.diags(inv, shape=A.shape, format="csr") @ A
        isolated = deg == 0
        if isolated.any():
            P = P + sp.diags(isolated.astype(float), shape=A.shape, format="csr")
        return P


def construct_pseudotime_graph_from_table(
    table: Any,
    *,
    branch_col: str = "branch",
    pseudotime_col: str = "pseudotime",
    cell_col: str = "cell_id",
    adata: Any = None,
    k: int = 50,
) -> PseudotimeGraph:
    """Build a pseudotime graph from a long-form (branch, pseudotime, cell_id) table.

    Within each branch, cells are grouped by identical pseudotime values (ties)
    and pseudotime levels are connected when within ``k`` levels. Ties never
    produce edges between cells at the same pseudotime.
    """

    node_ids_t, n, pairs = _table_to_forward_pairs(
        table,
        branch_col=branch_col,
        pseudotime_col=pseudotime_col,
        cell_col=cell_col,
        adata=adata,
        k=k,
    )
    return PseudotimeGraph(node_ids=node_ids_t, adjacency=_pairs_to_csr(n, pairs, symmetric=True))


def _resolve_node_ids(adata: Any, fallback: Sequence[str]) -> list[str]:
    if adata is None:
        return list(dict.fromkeys(fallback))
    obs_names = getattr(adata, "obs_names", None)
    if obs_names is None:
        raise TypeError("adata must have .obs_names (AnnData-like)")
    return list(obs_names)


def _pairs_to_csr(
    n: int,
    forward_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    symmetric: bool,
) -> sp.csr_matrix:
    """Materialize an unweighted CSR adjacency from forward (i -> j) edges.

    If ``symmetric``, edges are mirrored via ``A.maximum(A.T)`` to produce an
    undirected adjacency. Duplicate forward edges (e.g. the same i->j from two
    branches) are collapsed to 1.
    """

    if not forward_pairs:
        return sp.csr_matrix((n, n), dtype=float)

    rows = np.concatenate([i for i, _ in forward_pairs])
    cols = np.concatenate([j for _, j in forward_pairs])
    A = sp.coo_matrix((np.ones(rows.size, dtype=float), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    if symmetric:
        A = A.maximum(A.T)
    A.eliminate_zeros()
    return A


def _column(table: Any, col: str) -> np.ndarray:
    """Extract a 1-D ndarray for ``col`` from a DataFrame-like or dict-of-arrays."""

    v = table[col]
    arr = v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v)
    return arr.ravel()


def _table_to_forward_pairs(
    table: Any,
    *,
    branch_col: str,
    pseudotime_col: str,
    cell_col: str,
    adata: Any,
    k: int,
) -> tuple[tuple[str, ...], int, list[tuple[np.ndarray, np.ndarray]]]:
    """Group a (branch, pseudotime, cell_id) table into forward (i -> j) edge pairs.

    Within each branch, cells are bucketed by pseudotime value, levels are sorted,
    and each level is linked to levels within +k positions. Returned pairs are
    directed from earlier to later pseudotime (cells sharing a pseudotime value
    never produce an edge to each other).
    """

    if k < 1:
        raise ValueError("k must be >= 1")

    branch_vals = _column(table, branch_col)
    pt_vals = _column(table, pseudotime_col)
    cell_vals = _column(table, cell_col)
    if not len(branch_vals) == len(pt_vals) == len(cell_vals):
        raise ValueError("table columns must have the same length")

    cell_strs = [str(c) for c in cell_vals.tolist()]
    node_ids = _resolve_node_ids(adata, cell_strs)
    node_ids_t = tuple(node_ids)
    idx = {cid: i for i, cid in enumerate(node_ids_t)}
    n = len(node_ids_t)

    by_branch: dict[Any, dict[Any, list[int]]] = {}
    for b, pt, cid_s in zip(branch_vals.tolist(), pt_vals.tolist(), cell_strs, strict=True):
        try:
            i = idx[cid_s]
        except KeyError:
            raise KeyError(
                f"Cell '{cid_s}' from branch '{b}' is not in node_ids "
                "(check adata.obs_names if provided)"
            ) from None
        by_branch.setdefault(b, {}).setdefault(pt, []).append(i)

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for pt_to_inds in by_branch.values():
        levels = sorted(pt_to_inds.keys())
        level_inds = [np.asarray(pt_to_inds[pt], dtype=np.int64) for pt in levels]
        L = len(levels)
        if L <= 1:
            continue

        for i in range(L - 1):
            end = min(L - 1, i + k)
            left = level_inds[i]
            if end <= i or left.size == 0:
                continue
            for j in range(i + 1, end + 1):
                right = level_inds[j]
                if right.size == 0:
                    continue
                pairs.append((np.repeat(left, right.size), np.tile(right, left.size)))

    return node_ids_t, n, pairs
