"""pgd2: Pseudotime Graph Diffusion utilities.

Core API:
- construct_pseudotime_graph
- diffuse_features

The implementation follows the method section of `pgd_ieee.pdf`:
- Build an (unweighted) pseudotime graph by connecting cells within a sliding window of radius k.
- Perform lazy random-walk diffusion: X_{t+1} = (1-α) X_t + α P X_t, with P = D^{-1} A.
"""

from .diffusion import diffuse_features
from .graph import (
    PseudotimeGraph,
    construct_pseudotime_graph,
    construct_pseudotime_graph_from_table,
)
from .pseudotime import compute_pseudotime_from_table

__all__ = [
    "PseudotimeGraph",
    "construct_pseudotime_graph",
    "construct_pseudotime_graph_from_table",
    "compute_pseudotime_from_table",
    "diffuse_features",
]
