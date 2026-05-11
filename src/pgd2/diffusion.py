from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
import scipy.sparse as sp

from .graph import PseudotimeGraph

FeatureMatrix = TypeVar("FeatureMatrix", np.ndarray, sp.spmatrix)


class _HasTransitionMatrix(Protocol):
    @property
    def transition_matrix(self) -> np.ndarray | sp.spmatrix: ...


Transition = PseudotimeGraph | _HasTransitionMatrix | np.ndarray | sp.spmatrix


def diffuse_features(
    X: FeatureMatrix,
    transition: Transition,
    *,
    alpha: float = 0.5,
    t: int = 1,
) -> FeatureMatrix:
    """Lazy random-walk diffusion of a feature matrix.

        X <- (1 - alpha) * X + alpha * (P @ X),  repeated t times.

    ``transition`` is one of:
      - a ``PseudotimeGraph`` or CellRank-style kernel exposing a
        ``transition_matrix`` property,
      - a row-stochastic matrix P aligned to ``X``'s rows (dense or sparse).

    If you pass a raw matrix that isn't row-stochastic, row-normalize it first.
    Returns the same type as ``X`` (numpy array or scipy sparse matrix).
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if t < 0:
        raise ValueError("t must be >= 0")

    P = getattr(transition, "transition_matrix", transition)
    if not (isinstance(P, np.ndarray) or sp.issparse(P)):
        raise TypeError("transition must be a PseudotimeGraph, numpy array, or scipy sparse matrix")
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"transition matrix must be square; got shape {P.shape!r}")
    if X.shape[0] != P.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but P has {P.shape[0]}; align X to graph.node_ids"
        )

    if t == 0 or alpha == 0.0:
        return X.copy()

    Xs = X.copy()
    for _ in range(t):
        Xs = (1.0 - alpha) * Xs + alpha * (P @ Xs)
    return Xs
