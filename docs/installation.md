# Installation

`pgd2` requires Python 3.10 or newer. The core install pulls in only NumPy and
SciPy.

```bash
pip install -e .
```

## Optional integrations

```bash
pip install -e ".[scverse]"   # AnnData helpers
pip install -e ".[viz]"       # matplotlib / pandas / UMAP for figure recreation
pip install -e ".[dev]"       # pytest
```

| Extra     | Adds                          | Enables                                            |
|-----------|-------------------------------|----------------------------------------------------|
| `scverse` | `anndata`                     | Passing an `AnnData` so the graph aligns to `obs_names` |
| `viz`     | `matplotlib`, `pandas`, `umap-learn` | Reading tables and recreating Figure 4      |
| `dev`     | `pytest`                      | Running the test suite                             |

## Verifying the install

```python
import pgd2
print(pgd2.__all__)
# ['PseudotimeGraph', 'compute_pseudotime_from_table',
#  'construct_pseudotime_graph', 'construct_pseudotime_graph_from_table',
#  'diffuse_features']
```
