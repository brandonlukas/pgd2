# API reference

The public API consists of five symbols, all importable from the top-level
`pgd2` namespace.

```{eval-rst}
.. currentmodule:: pgd2

.. autosummary::

   PseudotimeGraph
   construct_pseudotime_graph
   construct_pseudotime_graph_from_table
   aggregate_pseudotime_from_table
   diffuse_features
```

## Graph construction

```{eval-rst}
.. autoclass:: pgd2.PseudotimeGraph
   :members:

.. autofunction:: pgd2.construct_pseudotime_graph

.. autofunction:: pgd2.construct_pseudotime_graph_from_table
```

## Diffusion

```{eval-rst}
.. autofunction:: pgd2.diffuse_features
```

## Pseudotime aggregation

```{eval-rst}
.. autofunction:: pgd2.aggregate_pseudotime_from_table
```
