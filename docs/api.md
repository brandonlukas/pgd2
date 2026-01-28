# API reference

This project intentionally keeps the public API small and composable.

## Graph construction

### `pgd2.construct_pseudotime_graph`

Construct a pseudotime graph from already-ordered cells per branch (PGD paper Eq. 1).

- Inputs: `branches: dict[str, list[str]]`, optional `adata` (aligns node order to `adata.obs_names`)
- Key params: `k` (window radius), `directed`, `weighted`

Typical use:

```python
graph = pgd2.construct_pseudotime_graph(branches, adata=adata, k=50)
```

### `pgd2.construct_pseudotime_graph_from_table`

Construct a pseudotime graph from a long-form table with columns:
- branch label
- pseudotime value
- cell id

This supports:
- cells appearing on multiple branches
- pseudotime ties (multiple cells can share the same pseudotime)

Key params:
- `k`: connect pseudotime *levels* within ±k (set `k=None` to disable)
- `delta`: connect pseudotime levels within a numeric value distance (set `delta=None` to disable)
- `directed`: if True, edges point forward in pseudotime
- `connect_within_pseudotime`: optionally connect cells tied at the same pseudotime value

Matches your Lamian TSV layout:

```python
import pandas as pd

df = pd.read_csv("data/lamian_nc_5.tsv", sep="\t")
graph = pgd2.construct_pseudotime_graph_from_table(
    df,
    branch_col="branch",
    pseudotime_col="pseudotime",
    cell_col="cell_id",
    adata=adata,
    k=50,
    # delta=0.5,
)
```

## Diffusion

### `pgd2.diffuse_features`

Diffuse a feature matrix along a **transition operator**:

- You can pass a `PseudotimeGraph` (it uses `graph.transition_matrix()` internally)
- Or pass a transition matrix `P` directly (dense numpy or scipy sparse)

Key params:
- `alpha`: diffusion strength in [0, 1]
- `t`: number of diffusion steps
- `normalize`: if True, row-normalize the provided transition matrix before diffusion

Examples:

```python
Xs = pgd2.diffuse_features(adata.obsm["X_pca_harmony"], graph, alpha=0.5, t=1)

P = graph.transition_matrix()
Xs = pgd2.diffuse_features(adata.obsm["X_pca_harmony"], P, alpha=0.5, t=1)
```

## Pseudotime utilities

### `pgd2.compute_pseudotime_from_table`

Compute a single pseudotime value per cell from a table where each cell may appear on multiple branches.

It builds a directed pseudotime graph from the table and uses unweighted shortest-path distance from a chosen root,
then rescales to [0, 1].

Root selection options:
- `root_cell="<cell-id>"` (explicit override)
- `backbone_mask=<bool array over table rows>`
- `backbone_selector=lambda branch_value: ...`

Example:

```python
pt = pgd2.compute_pseudotime_from_table(
    df,
    adata=adata,
    backbone_selector=lambda b: str(b).startswith("backbone"),
)
```
