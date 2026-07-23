"""README hero: the original UMAP blob morphing into the PGD-smoothed trajectory.

A looping GIF (~800px). Two-endpoint Procrustes morph (original <-> PGD-smoothed),
cells colored by pseudotime -- the clean "blob becomes a trajectory" story. Both
endpoints are UMAP'd with identical settings so the morph reflects only the
diffusion, and the smoothed embedding is Procrustes-aligned to the original so
cells slide into their branches instead of the whole cloud spinning.

Dogfoods pgd2 end to end (graph -> diffuse -> pseudotime). Dev-only deps, not part
of the package. Excluded from the wheel/sdist; only the resulting GIF is tracked.

    uv run --with anndata --with umap-learn --with matplotlib python make_figures/hero.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import scipy.sparse.csgraph as csg
import umap
from scipy.linalg import orthogonal_procrustes

import pgd2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H5AD = ROOT / "data/GSE203244_processed_noX.h5ad"
TSV = ROOT / "data/lamian_nc_9.tsv"
OUT = ROOT / "figures/hero.gif"
SEED, ALPHA, T = 0, 0.7, 1


def main() -> None:
    adata = ad.read_h5ad(H5AD)
    cells = [str(c) for c in adata.obs_names]
    X = np.asarray(adata.obsm["X_pca_harmony"])

    rows = list(csv.DictReader(open(TSV), delimiter="\t"))
    table = {k: [r[k] for r in rows] for k in rows[0]}
    table["pseudotime"] = [float(x) for x in table["pseudotime"]]

    # pgd2: build the pseudotime graph and smooth the embedding along it
    graph = pgd2.construct_pseudotime_graph_from_table(table, adata=adata, k=50)
    Xs = pgd2.diffuse_features(X, graph, alpha=ALPHA, t=T)

    pt = _undirected_pseudotime(table, adata, cells)

    # UMAP both endpoints with identical settings, then align smoothed -> original
    A = _umap(X)
    B = _umap(Xs)
    A -= A.mean(0)
    B -= B.mean(0)
    R, _ = orthogonal_procrustes(B, A)
    B = B @ R

    # shared scale + fixed limits so panel geometry is identical every frame
    s = np.abs(np.vstack([A, B])).max()
    A, B = A / s, B / s
    pad = 0.05
    both = np.vstack([A, B])
    xlim = (both[:, 0].min() - pad, both[:, 0].max() + pad)
    ylim = (both[:, 1].min() - pad, both[:, 1].max() + pad)

    _render(A, B, pt, xlim, ylim)


def _umap(M: np.ndarray) -> np.ndarray:
    return umap.UMAP(n_neighbors=15, min_dist=0.5, random_state=SEED).fit_transform(M)


def _undirected_pseudotime(table, adata, cells) -> np.ndarray:
    """Distance-from-root on the undirected graph (nc_9 branches are tip-oriented,
    so the directed aggregate_pseudotime would clamp unreachable cells)."""
    root = next(
        c for b, p, c in zip(table["branch"], table["pseudotime"], table["cell_id"])
        if str(b).startswith("backbone") and p == 0.0
    )
    g = pgd2.construct_pseudotime_graph_from_table(table, adata=adata, k=5)
    dist = csg.dijkstra(g.adjacency, directed=False, indices=cells.index(root), unweighted=True)
    dist[~np.isfinite(dist)] = np.nanmax(dist[np.isfinite(dist)])
    return (dist - dist.min()) / (dist.max() - dist.min())


def _render(A, B, pt, xlim, ylim) -> None:
    fps, morph, hold = 25, 26, 5
    ease = lambda t: (1 - np.cos(np.pi * t)) / 2  # noqa: E731
    ts = [0.0] * hold + [ease(i / morph) for i in range(morph + 1)] + [1.0] * hold
    ts = ts + ts[::-1]  # boomerang: blob -> trajectory -> blob

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    scat = ax.scatter(A[:, 0], A[:, 1], c=pt, cmap="viridis", s=6, alpha=0.85, linewidths=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    cax = inset_axes(ax, width="2.5%", height="26%", loc="upper left", borderpad=1.2)
    cbar = fig.colorbar(scat, cax=cax, ticks=[pt.min(), pt.max()])
    cbar.ax.set_yticklabels(["early", "late"], fontsize=10)
    cbar.set_label("pseudotime", fontsize=11)
    fig.tight_layout()

    def update(t):
        scat.set_offsets(A * (1 - t) + B * t)
        return (scat,)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=ts, blit=True).save(OUT, writer=PillowWriter(fps=fps))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(ts)} frames)")


if __name__ == "__main__":
    main()
