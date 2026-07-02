# Quickstart

The two core steps are **construct a graph** and **diffuse features** over it.

## 1. Construct a pseudotime graph

Given cells already ordered along one or more branches, connect each cell to the
`2k` neighbours within a sliding window of radius `k` (PGD paper Eq. 1):

```python
import numpy as np
import pgd2

branches = {
    "branch1": ["cell1", "cell2", "cell3"],
    "branch2": ["cell2", "cell4"],
}

graph = pgd2.construct_pseudotime_graph(branches, k=1)
graph.n_nodes          # number of unique cells
graph.node_ids         # tuple of cell IDs; feature rows must align to this order
graph.transition_matrix  # row-stochastic P = D^{-1} A (cached)
```

## 2. Diffuse a feature matrix

`diffuse_features` applies the lazy random walk `X <- (1 - alpha) X + alpha P X`,
`t` times. Rows of `X` must align to `graph.node_ids`.

```python
X = np.random.randn(graph.n_nodes, 5)
X_smooth = pgd2.diffuse_features(X, graph, alpha=0.5, t=1)
```

- `alpha` in `[0, 1]` controls smoothing strength (`0` = no change).
- `t >= 0` is the number of diffusion steps.
- The return type matches the input (NumPy array or SciPy sparse matrix).

### Passing a transition matrix directly

Instead of a `PseudotimeGraph`, you can pass any row-stochastic matrix `P`, or an
object exposing a `transition_matrix` property (e.g. a CellRank kernel):

```python
P = graph.transition_matrix
X_smooth = pgd2.diffuse_features(X, P, alpha=0.5, t=1)
```

If your matrix is not row-stochastic, row-normalize it first.

## Working with AnnData

Pass an `AnnData` so node IDs align to `adata.obs_names`. You can then diffuse
any matrix from `adata.obsm` directly:

```python
graph = pgd2.construct_pseudotime_graph(branches, adata=adata, k=50)
X_smooth = pgd2.diffuse_features(adata.obsm["X_pca"], graph, alpha=0.5, t=1)
```
