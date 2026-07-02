# Installation

`pgd2` requires Python 3.10 or newer and depends only on NumPy and SciPy.

```bash
pip install -e .
```

## Optional interoperability

`pgd2` works with [AnnData](https://anndata.readthedocs.io/) and pandas by duck
typing — it never imports them itself. If you want to pass an `AnnData` (so the
graph aligns to `obs_names`) or read tables with pandas, install those packages
yourself; `pgd2` deliberately does not pull them in.

To run the test suite:

```bash
pip install -e ".[dev]"   # pytest
```

## Verifying the install

```python
import pgd2
print(pgd2.__all__)
# ['PseudotimeGraph', 'aggregate_pseudotime_from_table',
#  'construct_pseudotime_graph', 'construct_pseudotime_graph_from_table',
#  'diffuse_features']
```
