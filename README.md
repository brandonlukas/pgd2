# pgd2

A lightweight Python implementation of **Pseudotime Graph Diffusion (PGD)** as described in `pgd_ieee.pdf`.

## Install

```bash
pip install -e .
```

`pgd2` depends only on NumPy and SciPy. It interoperates with AnnData and pandas
by duck typing, so if you want those (or plotting libraries), install them
yourself — `pgd2` will not pull them in.

## Quickstart

```python
import numpy as np
import pgd2

branches = {
    "branch1": ["cell1", "cell2", "cell3"],
    "branch2": ["cell2", "cell4"],
}

graph = pgd2.construct_pseudotime_graph(branches, k=1)

X = np.random.randn(graph.n_nodes, 5)
X_smooth = pgd2.diffuse_features(X, graph, alpha=0.5, t=1)

# Or pass a transition matrix directly (e.g., from CellRank).
P = graph.transition_matrix
X_smooth2 = pgd2.diffuse_features(X, P, alpha=0.5, t=1)
```

With AnnData (scverse):

```python
# graph aligns to adata.obs_names so you can pass adata.obsm["X_pca"] directly
graph = pgd2.construct_pseudotime_graph(branches, adata=adata, k=50)
X_smooth = pgd2.diffuse_features(adata.obsm["X_pca"], graph, alpha=0.5, t=1)
```

## Table-based input

For trajectory tools that emit a long-form `(branch, pseudotime, cell_id)` table (e.g. Lamian), build the graph directly. Cells sharing a pseudotime value are grouped as ties — no spurious edges between them.

```python
import pandas as pd

df = pd.read_csv("data/lamian_nc_5.tsv", sep="\t")
graph = pgd2.construct_pseudotime_graph_from_table(df, adata=adata, k=50)
```

If each cell appears on multiple branches and you want a single canonical pseudotime per cell (auxiliary; not paper-method):

```python
pt = pgd2.aggregate_pseudotime_from_table(
    df,
    adata=adata,
    backbone_selector=lambda b: str(b).startswith("backbone"),
)
```

This builds a directed graph, runs unweighted Dijkstra from a backbone-rooted cell, and min-max scales to `[0, 1]`.

## Documentation

Full documentation is built with Sphinx and hosted on Read the Docs. To build it
locally:

```bash
pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
# open docs/_build/html/index.html
```

## API surface

Public exports (see docstrings via `help(pgd2.<name>)`):

- `pgd2.PseudotimeGraph`
- `pgd2.construct_pseudotime_graph`
- `pgd2.construct_pseudotime_graph_from_table`
- `pgd2.aggregate_pseudotime_from_table`
- `pgd2.diffuse_features`
