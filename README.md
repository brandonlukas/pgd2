# pgd2

A lightweight Python implementation of **Pseudotime Graph Diffusion (PGD)** (IEEE).

## Install

```bash
pip install -e .
```

`pgd2` depends only on NumPy and SciPy. It reads pandas/dict tables by duck typing
and returns plain NumPy/SciPy arrays, so bring your own AnnData and plotting
libraries — `pgd2` will not pull them in.

## Quickstart

Trajectory tools emit a long-form `(branch, pseudotime, cell_id)` table (e.g. Lamian).
Build the pseudotime graph from it, then smooth an embedding along the trajectory.
Cells sharing a pseudotime value are grouped as ties — no spurious edges between them.

The repo ships a small tutorial dataset so this runs as-is: `data/macrophages.tsv`
(the trajectory table) and `data/macrophages_embedding.npz` (matching `X_pca` /
`X_umap` for 2848 wound-healing macrophages from GSE203244). In your own work the
embedding comes from `adata.obsm[...]`; here it's a small `.npz` so the example
needs only NumPy, SciPy, and pandas.

```python
import numpy as np
import pandas as pd
import pgd2

df = pd.read_csv("data/macrophages.tsv", sep="\t")
emb = np.load("data/macrophages_embedding.npz", allow_pickle=True)

# cell_ids fixes the node order so the embedding rows line up with graph.node_ids.
# In a scanpy workflow pass cell_ids=adata.obs_names and diffuse adata.obsm[...].
graph = pgd2.construct_pseudotime_graph_from_table(df, cell_ids=emb["cell_ids"], k=50)
X_smooth = pgd2.diffuse_features(emb["X_pca"], graph, alpha=0.5, t=1)

# Or pass a transition matrix directly (e.g., from CellRank).
P = graph.transition_matrix
X_smooth2 = pgd2.diffuse_features(emb["X_pca"], P, alpha=0.5, t=1)
```

If each cell appears on multiple branches and you want a single canonical pseudotime per cell (auxiliary; not paper-method):

```python
# Output is returned in cell_ids order (pass adata.obs_names in a scanpy workflow).
pt = pgd2.aggregate_pseudotime_from_table(
    df,
    cell_ids=emb["cell_ids"],
    backbone_selector=lambda b: str(b).startswith("backbone"),
)
```

This builds a directed graph, runs unweighted Dijkstra from a backbone-rooted cell, and min-max scales to `[0, 1]`.

## Abstract tree layout

`dendrogram_from_table` lays cells out as a rooted URD-style tree (y = pseudotime,
x = branch position) from their overlapping-lineage membership — no embedding needed.

```python
d = pgd2.dendrogram_from_table(df, cell_ids=emb["cell_ids"],
                               backbone_selector=lambda b: str(b).startswith("backbone"))
# d.coords (x, y per cell), d.lines (tree spine segments) — plot with your own library
```

## Tip: comparing an embedding before and after smoothing

When you plot the original embedding beside the PGD-smoothed one (or animate a
morph between them), the smoothed cloud is often rotated or reflected relative to
the original, so the two don't line up. Align them with an orthogonal Procrustes
fit — `scipy` (already a dependency) ships it, no `pgd2` helper needed:

```python
from scipy.linalg import orthogonal_procrustes

A = original - original.mean(0)     # e.g. adata.obsm["X_umap"]
B = smoothed - smoothed.mean(0)     # UMAP of the diffused features, same cell order
R, _ = orthogonal_procrustes(B, A)  # best rotation/reflection of B onto A
B_aligned = B @ R
```

Now cells slide into their branches instead of the whole cloud spinning. Both
point sets must be in the same cell order.

## API surface

Public exports (see docstrings via `help(pgd2.<name>)`):

- `pgd2.PseudotimeGraph`
- `pgd2.construct_pseudotime_graph_from_table`
- `pgd2.aggregate_pseudotime_from_table`
- `pgd2.diffuse_features`
- `pgd2.dendrogram_from_table`
