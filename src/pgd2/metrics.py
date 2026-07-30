"""Evaluation metrics for embeddings.

For the other half of embedding evaluation -- whether original-space neighbors
survive into the embedding -- use ``sklearn.manifold.trustworthiness``; pgd2 does
not reimplement it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree


def pseudotime_smoothness(
    embedding: npt.ArrayLike, pseudotime: npt.ArrayLike, *, k: int = 15
) -> float:
    """Smoothness of the pseudotime gradient over an embedding's kNN graph.

    ``1 - C``, where ``C`` is Geary's C of ``pseudotime`` on the ``k``-nearest-
    neighbor graph of ``embedding``:

        C = mean over edges (pt_i - pt_j)^2 / (2 * var(pt))

    Returns 1.0 for a perfectly smooth gradient (neighbors share a pseudotime
    value), ~0.0 when pseudotime is scattered at random over the embedding, and
    below 0 when neighbors are more dissimilar than random pairs. Higher is
    smoother; compare values across embeddings of the same cells at the same ``k``.

    ``embedding`` is (n_cells, n_dims) with any number of dims; ``pseudotime`` is a
    length-n_cells array, row-aligned to it (e.g. from
    :func:`pgd2.aggregate_pseudotime_from_table` with the same ``cell_ids``).
    """

    X = np.asarray(embedding, dtype=float)
    pt = np.asarray(pseudotime, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"embedding must be 2-D (n_cells, n_dims); got shape {X.shape!r}")
    if pt.size != X.shape[0]:
        raise ValueError(f"pseudotime has {pt.size} values but embedding has {X.shape[0]} rows")
    if not np.isfinite(pt).all():  # cKDTree rejects non-finite embeddings itself
        raise ValueError("pseudotime must be finite (no NaN/inf)")
    n = X.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, n_cells - 1]; got k={k} with {n} cells")

    var = pt.var(ddof=1)
    if var == 0:
        raise ValueError("pseudotime is constant; smoothness is undefined")

    # ponytail: exact kNN via cKDTree -- fine through a few dozen dims and ~1e5
    # cells; swap in an approximate-NN index if it ever drags.
    _, nbrs = cKDTree(X).query(X, k=k + 1, workers=-1)
    j = nbrs[:, 1:]  # column 0 is the point itself
    d2 = (pt[:, None] - pt[j]) ** 2
    return float(1.0 - d2.mean() / (2.0 * var))
