import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import pgd2


def graph_from_branch(cells, *, k=1, cell_ids=None):
    """Build a graph from one branch of ordered cells (pseudotime = position)."""
    table = {
        "branch": ["b1"] * len(cells),
        "pseudotime": list(range(len(cells))),
        "cell_id": list(cells),
    }
    return pgd2.pseudotime_graph_from_table(table, k=k, cell_ids=cell_ids)


def node_index(g):
    return {cid: i for i, cid in enumerate(g.node_ids)}


@pytest.fixture
def tiny_graph():
    return graph_from_branch(["c1", "c2"], k=1)


def test_pseudotime_graph_matches_cell_ids_order():
    names = ["a", "b", "c", "d"]
    g = graph_from_branch(["b", "c", "d"], k=1, cell_ids=names)
    assert g.node_ids == tuple(names)
    assert g.adjacency.shape == (4, 4)


def test_diffusion_runs_dense_and_sparse():
    g = graph_from_branch(["c1", "c2", "c3", "c4"], k=1)
    P = g.transition_matrix

    X = np.arange(g.n_nodes * 2, dtype=float).reshape(g.n_nodes, 2)
    Xs = pgd2.diffuse_features(X, g, alpha=0.5, t=1)
    assert Xs.shape == X.shape

    Xs2 = pgd2.diffuse_features(X, P, alpha=0.5, t=1)
    assert Xs2.shape == X.shape

    Xsp = sp.csr_matrix(X)
    Xsp_s = pgd2.diffuse_features(Xsp, g, alpha=0.5, t=2)
    assert Xsp_s.shape == Xsp.shape

    Xsp_s2 = pgd2.diffuse_features(Xsp, P, alpha=0.5, t=2)
    assert Xsp_s2.shape == Xsp.shape


def test_pseudotime_graph_from_table_preserves_ties():
    # Two cells share pseudotime=0; we should not have to arbitrarily order them.
    table = {
        "branch": ["b1", "b1", "b1"],
        "pseudotime": [0, 0, 1],
        "cell_id": ["c1", "c2", "c3"],
    }

    g = pgd2.pseudotime_graph_from_table(table, k=1)
    A = g.adjacency.tocsr()
    i = node_index(g)

    # c1,c2 should connect to c3 (both directions by default)
    assert A[i["c1"], i["c3"]] != 0
    assert A[i["c2"], i["c3"]] != 0
    assert A[i["c3"], i["c1"]] != 0
    assert A[i["c3"], i["c2"]] != 0

    # no forced tie-breaking edge between c1 and c2 by default
    assert A[i["c1"], i["c2"]] == 0
    assert A[i["c2"], i["c1"]] == 0


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_diffuse_features_rejects_alpha_out_of_range(tiny_graph, alpha):
    X = np.zeros((tiny_graph.n_nodes, 1))
    with pytest.raises(ValueError, match="alpha"):
        pgd2.diffuse_features(X, tiny_graph, alpha=alpha, t=1)


def test_diffuse_features_rejects_negative_t(tiny_graph):
    X = np.zeros((tiny_graph.n_nodes, 1))
    with pytest.raises(ValueError, match="t must be"):
        pgd2.diffuse_features(X, tiny_graph, alpha=0.5, t=-1)


def test_diffuse_features_rejects_shape_mismatch(tiny_graph):
    X = np.zeros((tiny_graph.n_nodes + 1, 1))
    with pytest.raises(ValueError, match="rows"):
        pgd2.diffuse_features(X, tiny_graph, alpha=0.5, t=1)


def test_pseudotime_graph_rejects_invalid_k():
    table = {"branch": ["b1", "b1"], "pseudotime": [0, 1], "cell_id": ["c1", "c2"]}
    with pytest.raises(ValueError, match="k must be"):
        pgd2.pseudotime_graph_from_table(table, k=0)


def test_pseudotime_graph_reports_unknown_cell():
    table = {"branch": ["b1", "b1"], "pseudotime": [0, 1], "cell_id": ["a", "missing"]}
    with pytest.raises(KeyError, match="missing"):
        pgd2.pseudotime_graph_from_table(table, cell_ids=["a", "b"])


def test_transition_matrix_is_row_stochastic():
    g = graph_from_branch(["c1", "c2", "c3"], k=1)
    P = g.transition_matrix
    row_sums = np.asarray(P.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.ones(g.n_nodes))


def test_aggregate_pseudotime_from_table_respects_backbone_selector():
    # Two branches assign different pseudotime values per cell.
    # The selector should constrain root selection to the backbone rows.
    table = {
        "branch": ["backbone", "backbone", "b1", "b1"],
        "pseudotime": [0.0, 1.0, 100.0, 101.0],
        "cell_id": ["c1", "c2", "c1", "c2"],
    }
    pt = pgd2.aggregate_pseudotime_from_table(
        table, cell_ids=["c1", "c2"], backbone_selector=lambda b: b == "backbone"
    )
    assert pt.shape == (2,)
    assert float(pt[0]) <= float(pt[1])


def test_aggregate_pseudotime_from_table_warns_on_unreachable():
    table = {
        "branch": ["b1", "b1", "b2", "b2"],
        "pseudotime": [0.0, 1.0, 0.0, 1.0],
        "cell_id": ["c1", "c2", "c3", "c4"],
    }
    with pytest.warns(UserWarning, match="unreachable from root"):
        pt = pgd2.aggregate_pseudotime_from_table(
            table, cell_ids=["c1", "c2", "c3", "c4"], root_cell="c1"
        )
    assert float(pt[2]) == 1.0
    assert float(pt[3]) == 1.0


def test_aggregate_pseudotime_from_table_no_warning_when_all_reachable():
    table = {
        "branch": ["b1", "b1", "b1"],
        "pseudotime": [0.0, 0.5, 1.0],
        "cell_id": ["c1", "c2", "c3"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pgd2.aggregate_pseudotime_from_table(table, cell_ids=["c1", "c2", "c3"], root_cell="c1")


def test_diffuse_features_accepts_property_style_kernel():
    """CellRank-style kernels expose `transition_matrix` as a property, not a method."""

    g = graph_from_branch(["c1", "c2", "c3"], k=1)

    class PropertyKernel:
        def __init__(self, P):
            self._P = P

        @property
        def transition_matrix(self):
            return self._P

    kernel = PropertyKernel(g.transition_matrix)
    X = np.arange(g.n_nodes * 2, dtype=float).reshape(g.n_nodes, 2)
    Xs_kernel = pgd2.diffuse_features(X, kernel, alpha=0.5, t=2)
    Xs_graph = pgd2.diffuse_features(X, g, alpha=0.5, t=2)
    np.testing.assert_allclose(Xs_kernel, Xs_graph)


def test_dendrogram_from_table_derives_trunk_and_split():
    # two lineages A,B share a trunk (cells in both), then each has its own branch
    table = {
        "branch": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "cell_id": ["c0", "c1", "c2", "c3", "c0", "c1", "c4", "c5"],
    }
    pt = {"c0": 0.0, "c1": 0.3, "c2": 0.7, "c3": 1.0, "c4": 0.7, "c5": 0.9}
    d = pgd2.dendrogram_from_table(table, pseudotime=pt, jitter=0.0, jitter_push=0.0, seed=0)
    trunk = ("A", "B")
    assert set(d.segment_x) == {trunk, ("A",), ("B",)}
    # leaves spread to 1 and 2, trunk sits at their mean
    assert {d.segment_x[("A",)], d.segment_x[("B",)]} == {1.0, 2.0}
    assert d.segment_x[trunk] == 1.5
    # y is pseudotime; trunk cell c0 at its trunk x
    i0 = d.cell_ids.index("c0")
    assert d.coords[i0, 1] == 0.0 and d.coords[i0, 0] == 1.5
    assert d.cell_segments[i0] == trunk
    # skeleton starts at the trunk root (pt 0) and squares off at the branchpoint
    assert (d.lines[:, 1].min() == 0.0) and (0.3 in d.lines[:, 1])
