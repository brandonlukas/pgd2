# pgd2

A lightweight Python implementation of **Pseudotime Graph Diffusion (PGD)**.

`pgd2` builds an unweighted graph over single cells by connecting cells that are
close along an inferred trajectory (pseudotime), then smooths a feature matrix
over that graph with a lazy random walk:

```{math}
X \leftarrow (1 - \alpha)\, X + \alpha\, P X, \qquad P = D^{-1} A
```

where `A` is the pseudotime adjacency and `P` its row-stochastic transition
operator. The package depends only on NumPy and SciPy; AnnData and plotting are
optional integrations.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
table-input
api
```

## At a glance

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
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
