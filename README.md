# pgd2

A lightweight Python implementation of **Pseudotime Graph Diffusion (PGD)** (IEEE).

## Install

```bash
pip install -e .
```

`pgd2` depends only on NumPy and SciPy. It interoperates with AnnData and pandas
by duck typing, so if you want those (or plotting libraries), install them
yourself — `pgd2` will not pull them in.

## Quickstart

Trajectory tools emit a long-form `(branch, pseudotime, cell_id)` table (e.g. Lamian).
Build the pseudotime graph from it, then smooth an embedding along the trajectory.
Cells sharing a pseudotime value are grouped as ties — no spurious edges between them.

```python
import pandas as pd
import pgd2

df = pd.read_csv("data/lamian_nc_5.tsv", sep="\t")

# graph aligns to adata.obs_names so you can pass adata.obsm[...] directly
graph = pgd2.construct_pseudotime_graph_from_table(df, adata=adata, k=50)
X_smooth = pgd2.diffuse_features(adata.obsm["X_pca"], graph, alpha=0.5, t=1)

# Or pass a transition matrix directly (e.g., from CellRank).
P = graph.transition_matrix
X_smooth2 = pgd2.diffuse_features(adata.obsm["X_pca"], P, alpha=0.5, t=1)
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

## Abstract tree layout

`dendrogram_from_table` lays cells out as a rooted URD-style tree (y = pseudotime,
x = branch position) from their overlapping-lineage membership — no embedding needed.

```python
d = pgd2.dendrogram_from_table(df, adata=adata,
                               backbone_selector=lambda b: str(b).startswith("backbone"))
# d.coords (x, y per cell), d.lines (tree spine segments) — plot with your own library
```

## API surface

Public exports (see docstrings via `help(pgd2.<name>)`):

- `pgd2.PseudotimeGraph`
- `pgd2.construct_pseudotime_graph_from_table`
- `pgd2.aggregate_pseudotime_from_table`
- `pgd2.diffuse_features`
- `pgd2.dendrogram_from_table`
