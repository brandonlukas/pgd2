# pgd2

A lightweight Python implementation of **Pseudotime Graph Diffusion (PGD)** as described in `pgd_ieee.pdf`.

Documentation:
- API reference: `docs/api.md`
- Figure reproductions: `docs/figures.md`

## Install

```bash
pip install -e .
```

Optional integrations:

```bash
pip install -e ".[scverse]"   # AnnData helpers
pip install -e ".[torch]"     # torch backend helpers
pip install -e ".[viz]"       # plotting/UMAP for figure recreation
```

## Quickstart

```python
import numpy as np
import pgd2

branches = {
    "branch1": ["cell1", "cell2", "cell3"],
    "branch2": ["cell2", "cell4"],
}

graph = pgd2.construct_pseudotime_graph(branches, adata=None, k=1)

X = np.random.randn(graph.n_nodes, 5)
X_smooth = pgd2.diffuse_features(X, graph, alpha=0.5, t=1)

# Or pass a transition matrix directly (e.g., from CellRank)
P = graph.transition_matrix()
X_smooth2 = pgd2.diffuse_features(X, P, alpha=0.5, t=1)
```

With AnnData (scverse):

```python
# graph aligns to adata.obs_names so you can pass adata.obsm["X_pca"] directly
graph = pgd2.construct_pseudotime_graph(branches, adata=adata, k=50)
X_smooth = pgd2.diffuse_features(adata.obsm["X_pca"], graph, alpha=0.5, t=1)
```

For fuller API details (including table-based graph construction, `delta`, and pseudotime helpers), see `docs/api.md`.
