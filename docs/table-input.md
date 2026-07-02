# Table-based input

Trajectory tools such as Lamian emit a long-form table with one row per
`(branch, pseudotime, cell_id)`. `pgd2` can build a graph directly from such a
table, correctly handling **ties** — cells that share a pseudotime value are
grouped together and never connected to each other.

## Building a graph from a table

```python
import pandas as pd
import pgd2

df = pd.read_csv("data/lamian_nc_5.tsv", sep="\t")
graph = pgd2.construct_pseudotime_graph_from_table(df, adata=adata, k=50)
```

Within each branch, cells are bucketed by pseudotime value and consecutive
pseudotime *levels* are connected when they fall within `k` levels of each other,
within numeric distance `delta`, or both. Supply at least one of `k` or `delta`.

Column names default to `branch`, `pseudotime`, and `cell_id`; override with
`branch_col`, `pseudotime_col`, and `cell_col` if your table differs. The table
may be a pandas `DataFrame` or a dict of arrays.

## One canonical pseudotime per cell

When a cell appears on several branches and you want a single pseudotime value
per cell (for visualization or downstream analysis), use
`compute_pseudotime_from_table`. This is an **auxiliary** utility — it is not part
of the PGD paper method and is not required by `diffuse_features`.

```python
pt = pgd2.compute_pseudotime_from_table(
    df,
    adata=adata,
    backbone_selector=lambda b: str(b).startswith("backbone"),
)
```

It builds a **directed** pseudotime graph, chooses a root cell, runs unweighted
Dijkstra from the root, and min-max scales distances to `[0, 1]`. Values are
returned in `adata.obs_names` order.

### Root selection

The root is chosen by the first rule that applies:

1. An explicit `root_cell`.
2. The backbone row with minimum pseudotime, where the backbone is given by
   `backbone_mask` (a boolean array over table rows) or `backbone_selector`
   (a predicate on branch labels).
3. The globally minimum-pseudotime row.

### Unreachable cells

If the directed graph has multiple sources (for example tip-to-tip branches),
some cells may be unreachable from the root. Those cells are clamped to
pseudotime `1.0` and a warning is emitted. Reorient branches so they all flow
outward from a single root to avoid this.
