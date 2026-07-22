"""pgd2: Pseudotime Graph Diffusion.

Build an unweighted pseudotime graph by connecting cells within a sliding
window along pseudotime, then smooth features with a lazy random walk:

    X <- (1 - alpha) * X + alpha * P X,   P = D^{-1} A.
"""

from .diffusion import diffuse_features
from .graph import PseudotimeGraph, construct_pseudotime_graph_from_table
from .layout import Dendrogram, dendrogram_from_table
from .pseudotime import aggregate_pseudotime_from_table

__all__ = [
    "Dendrogram",
    "PseudotimeGraph",
    "aggregate_pseudotime_from_table",
    "construct_pseudotime_graph_from_table",
    "dendrogram_from_table",
    "diffuse_features",
]
