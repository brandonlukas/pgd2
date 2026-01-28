from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
import scipy.sparse as sp

from .graph import PseudotimeGraph

ArrayLike = np.ndarray | sp.spmatrix
TransitionLike = PseudotimeGraph | np.ndarray | sp.spmatrix | Any


def _check_X_shape(X: ArrayLike, n: int) -> None:
    if X.shape[0] != n:
        raise ValueError(
            f"X has {X.shape[0]} rows, but transition operator has {n} states. "
            "Ensure X is aligned to the transition matrix row/column order."
        )


def _row_normalize_transition_matrix(
    P: np.ndarray | sp.spmatrix,
) -> np.ndarray | sp.spmatrix:
    """Row-normalize a transition-like matrix and fix isolated rows.

    For rows with sum 0, sets P[i, i] = 1 so those states remain unchanged by diffusion.
    """

    if sp.issparse(P):
        P = P.tocsr(copy=True)
        row_sums = np.asarray(P.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
        Dinv = sp.diags(inv, offsets=0, shape=P.shape, format="csr")
        out = Dinv @ P
        isolated = np.where(row_sums == 0)[0]
        if isolated.size:
            out = out.tolil(copy=False)
            out[isolated, isolated] = 1.0
            out = out.tocsr()
        out.eliminate_zeros()
        return out

    Pn = np.asarray(P, dtype=float)
    row_sums = Pn.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(Pn, row_sums, out=np.zeros_like(Pn), where=row_sums != 0)
    isolated = np.where(row_sums.ravel() == 0)[0]
    if isolated.size:
        out[isolated, isolated] = 1.0
    return out


def _coerce_transition_matrix(
    transition: TransitionLike, *, normalize: bool
) -> np.ndarray | sp.spmatrix:
    """Coerce a transition operator into a concrete matrix P.

    - If `transition` provides `.transition_matrix()`, that is used.
    - Otherwise `transition` is treated as the transition matrix itself.
    """

    if hasattr(transition, "transition_matrix") and callable(
        getattr(transition, "transition_matrix")
    ):
        P = transition.transition_matrix()  # type: ignore[no-any-return]
    else:
        P = transition

    if not (isinstance(P, np.ndarray) or sp.issparse(P)):
        raise TypeError(
            "transition must be a PseudotimeGraph, a numpy array, a scipy sparse matrix, "
            "or an object exposing transition_matrix()."
        )

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"transition matrix must be square; got shape {P.shape!r}")

    return _row_normalize_transition_matrix(P) if normalize else P


@overload
def diffuse_features(
    X: np.ndarray,
    transition: TransitionLike,
    *,
    alpha: float = 0.5,
    t: int = 1,
    backend: Literal["scipy", "torch"] = "scipy",
    normalize: bool = False,
) -> np.ndarray: ...


@overload
def diffuse_features(
    X: sp.spmatrix,
    transition: TransitionLike,
    *,
    alpha: float = 0.5,
    t: int = 1,
    backend: Literal["scipy", "torch"] = "scipy",
    normalize: bool = False,
) -> sp.spmatrix: ...


def diffuse_features(
    X: ArrayLike,
    transition: TransitionLike,
    *,
    alpha: float = 0.5,
    t: int = 1,
    backend: Literal["scipy", "torch"] = "scipy",
    normalize: bool = False,
):
    """Diffuse (smooth) a feature matrix along a cell-cell transition operator.

    Implements Eq. (2)-(5) from the PGD paper using iterative updates:

        X^{(s+1)} = (1-α) X^{(s)} + α P X^{(s)},  where P = D^{-1} A.

    Parameters
    ----------
    X
        Feature matrix of shape (n_cells, n_features). Can be a dense numpy array (e.g.
        adata.obsm['X_pca']) or a scipy sparse matrix.
    transition
        Cell-cell transition operator $P$ (shape (n_cells, n_cells)) or any object exposing
        `.transition_matrix()` (e.g., `PseudotimeGraph`, some CellRank kernels). If you pass
        a PGD graph, its random-walk transition matrix $P = D^{-1}A$ is used.
    alpha
        Diffusion strength in [0,1].
    t
        Number of diffusion steps (t >= 0).
    backend
        "scipy" (default) uses scipy sparse matrix multiplication.
        "torch" converts P and X to torch tensors and uses torch sparse-dense matmul.
    normalize
        If True, row-normalize the provided transition matrix before diffusion (and set
        isolated rows to self-loops). Leave False if your transition matrix is already
        row-stochastic (common for CellRank kernels).

    Notes
    -----
    Torch is not universally "best" here:
    - If your graph is sparse (common) and you're on CPU, scipy is typically fast and simple.
    - Torch can be beneficial when running on GPU and doing many repeated operations, but
      converting scipy sparse -> torch sparse has overhead.

    Returns
    -------
    X_smooth
        Same type as X (numpy array or scipy sparse matrix).
    """

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    if t < 0:
        raise ValueError("t must be >= 0")

    P = _coerce_transition_matrix(transition, normalize=normalize)
    n = int(P.shape[0])
    _check_X_shape(X, n)

    if t == 0 or alpha == 0.0:
        return X.copy() if hasattr(X, "copy") else X

    if backend == "scipy":
        if sp.issparse(X) and not sp.issparse(P):
            raise TypeError(
                "When X is scipy sparse, transition must also be scipy sparse (or use backend='torch')."
            )
        Xs = X.copy() if hasattr(X, "copy") else X
        for _ in range(t):
            Xs = (1.0 - alpha) * Xs + alpha * (P @ Xs)
        return Xs

    if backend == "torch":
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise ImportError("backend='torch' requires torch to be installed") from e

        if sp.issparse(P):
            # Convert P to torch sparse COO
            Pcoo = P.tocoo()
            indices = torch.tensor(np.vstack([Pcoo.row, Pcoo.col]), dtype=torch.int64)
            values = torch.tensor(Pcoo.data, dtype=torch.float32)
            Pt = torch.sparse_coo_tensor(indices, values, size=Pcoo.shape).coalesce()
            P_is_sparse = True
        else:
            Pt = torch.tensor(np.asarray(P), dtype=torch.float32)
            P_is_sparse = False

        if sp.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)

        Xt = torch.tensor(X_dense, dtype=torch.float32)
        for _ in range(t):
            if P_is_sparse:
                Xt = (1.0 - alpha) * Xt + alpha * torch.sparse.mm(Pt, Xt)
            else:
                Xt = (1.0 - alpha) * Xt + alpha * (Pt @ Xt)

        out = Xt.cpu().numpy()
        if sp.issparse(X):
            return sp.csr_matrix(out)
        return out

    raise ValueError("backend must be 'scipy' or 'torch'")
