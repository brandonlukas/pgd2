import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import pgd2


class DummyAdata:
    def __init__(self, obs_names):
        self.obs_names = obs_names


@pytest.fixture
def tiny_graph():
    return pgd2.construct_pseudotime_graph({"b1": ["c1", "c2"]}, k=1)


def test_construct_graph_matches_adata_order():
    adata = DummyAdata(["a", "b", "c", "d"])
    branches = {"b1": ["b", "c", "d"]}

    g = pgd2.construct_pseudotime_graph(branches, adata=adata, k=1)
    assert g.node_ids == tuple(adata.obs_names)
    assert g.adjacency.shape == (4, 4)


def test_diffusion_runs_dense_and_sparse():
    branches = {"b1": ["c1", "c2", "c3", "c4"]}
    g = pgd2.construct_pseudotime_graph(branches, k=1)
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


def test_construct_graph_from_table_preserves_ties():
    class DummyTable(dict):
        pass

    # Two cells share pseudotime=0; we should not have to arbitrarily order them.
    table = DummyTable(
        {
            "branch": ["b1", "b1", "b1"],
            "pseudotime": [0, 0, 1],
            "cell_id": ["c1", "c2", "c3"],
        }
    )

    g = pgd2.construct_pseudotime_graph_from_table(table, k=1)
    A = g.adjacency.tocsr()

    # c1,c2 should connect to c3 (both directions by default)
    i = g.index()
    assert A[i["c1"], i["c3"]] != 0
    assert A[i["c2"], i["c3"]] != 0
    assert A[i["c3"], i["c1"]] != 0
    assert A[i["c3"], i["c2"]] != 0

    # no forced tie-breaking edge between c1 and c2 by default
    assert A[i["c1"], i["c2"]] == 0
    assert A[i["c2"], i["c1"]] == 0


def test_construct_graph_from_table_delta_only():
    table = {
        "branch": ["b1", "b1", "b1"],
        "pseudotime": [0.0, 0.2, 1.0],
        "cell_id": ["c1", "c2", "c3"],
    }

    # delta connects c1<->c2 but not to c3
    g = pgd2.construct_pseudotime_graph_from_table(table, k=None, delta=0.25)
    A = g.adjacency.tocsr()
    i = g.index()

    assert A[i["c1"], i["c2"]] != 0
    assert A[i["c2"], i["c1"]] != 0
    assert A[i["c1"], i["c3"]] == 0
    assert A[i["c3"], i["c1"]] == 0


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


def test_construct_pseudotime_graph_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be"):
        pgd2.construct_pseudotime_graph({"b1": ["c1", "c2"]}, k=0)


def test_construct_pseudotime_graph_reports_unknown_cell():
    adata = DummyAdata(["a", "b"])
    with pytest.raises(KeyError, match="missing"):
        pgd2.construct_pseudotime_graph({"b1": ["a", "missing"]}, adata=adata, k=1)


def test_construct_pseudotime_graph_from_table_requires_k_or_delta():
    table = {"branch": ["b1"], "pseudotime": [0.0], "cell_id": ["c1"]}
    with pytest.raises(ValueError, match="k or delta"):
        pgd2.construct_pseudotime_graph_from_table(table, k=None, delta=None)


def test_transition_matrix_is_row_stochastic():
    g = pgd2.construct_pseudotime_graph({"b1": ["c1", "c2", "c3"]}, k=1)
    P = g.transition_matrix
    row_sums = np.asarray(P.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.ones(g.n_nodes))


def test_compute_pseudotime_from_table_respects_backbone_mask():
    # Two branches assign different pseudotime values per cell.
    # The mask should constrain root selection to the backbone rows.
    table = {
        "branch": ["backbone", "backbone", "b1", "b1"],
        "pseudotime": [0.0, 1.0, 100.0, 101.0],
        "cell_id": ["c1", "c2", "c1", "c2"],
    }
    adata = DummyAdata(["c1", "c2"])
    backbone_mask = np.array([True, True, False, False])
    pt = pgd2.compute_pseudotime_from_table(table, adata=adata, backbone_mask=backbone_mask)
    assert pt.shape == (2,)
    assert float(pt[0]) <= float(pt[1])


def test_compute_pseudotime_from_table_warns_on_unreachable():
    table = {
        "branch": ["b1", "b1", "b2", "b2"],
        "pseudotime": [0.0, 1.0, 0.0, 1.0],
        "cell_id": ["c1", "c2", "c3", "c4"],
    }
    adata = DummyAdata(["c1", "c2", "c3", "c4"])
    with pytest.warns(UserWarning, match="unreachable from root"):
        pt = pgd2.compute_pseudotime_from_table(table, adata=adata, root_cell="c1")
    assert float(pt[2]) == 1.0
    assert float(pt[3]) == 1.0


def test_compute_pseudotime_from_table_no_warning_when_all_reachable():
    table = {
        "branch": ["b1", "b1", "b1"],
        "pseudotime": [0.0, 0.5, 1.0],
        "cell_id": ["c1", "c2", "c3"],
    }
    adata = DummyAdata(["c1", "c2", "c3"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pgd2.compute_pseudotime_from_table(table, adata=adata, root_cell="c1")


def test_diffuse_features_accepts_property_style_kernel():
    """CellRank-style kernels expose `transition_matrix` as a property, not a method."""

    g = pgd2.construct_pseudotime_graph({"b1": ["c1", "c2", "c3"]}, k=1)

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
