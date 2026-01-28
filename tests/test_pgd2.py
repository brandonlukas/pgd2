import numpy as np
import scipy.sparse as sp

import pgd2


def test_construct_graph_matches_adata_order():
    class DummyAdata:
        def __init__(self, obs_names):
            self.obs_names = obs_names

    adata = DummyAdata(["a", "b", "c", "d"])
    branches = {"b1": ["b", "c", "d"]}

    g = pgd2.construct_pseudotime_graph(branches, adata=adata, k=1)
    assert g.node_ids == tuple(adata.obs_names)
    assert g.adjacency.shape == (4, 4)


def test_diffusion_runs_dense_and_sparse():
    branches = {"b1": ["c1", "c2", "c3", "c4"]}
    g = pgd2.construct_pseudotime_graph(branches, k=1)
    P = g.transition_matrix()

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


def test_compute_pseudotime_from_table_respects_backbone_mask():
    class DummyAdata:
        def __init__(self, obs_names):
            self.obs_names = obs_names

    # Two branches assign different pseudotime values per cell.
    table = {
        "branch": ["backbone", "backbone", "b1", "b1"],
        "pseudotime": [0.0, 1.0, 100.0, 101.0],
        "cell_id": ["c1", "c2", "c1", "c2"],
    }
    adata = DummyAdata(["c1", "c2"])

    # Mask only the backbone rows; root should be c1 (pt=0) not influenced by b1.
    backbone_mask = np.array([True, True, False, False])
    pt = pgd2.compute_pseudotime_from_table(
        table, adata=adata, backbone_mask=backbone_mask
    )
    assert pt.shape == (2,)
    assert float(pt[0]) <= float(pt[1])
