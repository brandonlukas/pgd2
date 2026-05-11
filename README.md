# pgd2

A lightweight Python implementation of **Pseudotime Graph Diffusion (PGD)** as described in `pgd_ieee.pdf`.

## Install

```bash
pip install -e .
```

Optional integrations:

```bash
pip install -e ".[scverse]"   # AnnData helpers
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
pt = pgd2.compute_pseudotime_from_table(
    df,
    adata=adata,
    backbone_selector=lambda b: str(b).startswith("backbone"),
)
```

This builds a directed graph, runs unweighted Dijkstra from a backbone-rooted cell, and min-max scales to `[0, 1]`.

## Reproducing Figure 4

```bash
python scripts/recreate_figure4_alpha_sweep.py \
  --h5ad data/GSE203244_processed_noX.h5ad \
  --lamian data/lamian_nc_5.tsv \
  --show-edges \
  --out figures/figure4_alpha_sweep.png
```

The faint lines with `--show-edges` are sparse branch path traces (anchors along each branch connected in order), shown for visual guidance.

## API surface

Public exports (see docstrings via `help(pgd2.<name>)`):

- `pgd2.PseudotimeGraph`
- `pgd2.construct_pseudotime_graph`
- `pgd2.construct_pseudotime_graph_from_table`
- `pgd2.compute_pseudotime_from_table`
- `pgd2.diffuse_features`
