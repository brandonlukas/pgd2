from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pgd2


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recreate PGD Figure 4: alpha sweep on embeddings with UMAP layouts."
    )
    ap.add_argument(
        "--h5ad", default="data/GSE203244_processed_noX.h5ad", help="Path to .h5ad"
    )
    ap.add_argument(
        "--lamian",
        default="data/lamian_nc_5.tsv",
        help="Path to Lamian branch/pseudotime table TSV",
    )
    ap.add_argument(
        "--X-key",
        default="X_pca_harmony",
        help="AnnData .obsm key to smooth (paper used Harmony-corrected PCs)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=50,
        help="Pseudotime window radius (levels) used to build the pseudotime graph",
    )
    ap.add_argument("--t", type=int, default=1, help="Diffusion steps")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument(
        "--out",
        default="figures/figure4_alpha_sweep.png",
        help="Output image path",
    )
    ap.add_argument("--umap-n-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.5)
    ap.add_argument("--umap-metric", default="euclidean")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument(
        "--show-edges", action="store_true", help="Overlay sparse branch path lines"
    )

    args = ap.parse_args()

    try:
        import anndata as ad
    except Exception as e:  # pragma: no cover
        raise SystemExit("anndata is required to load the .h5ad file") from e

    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise SystemExit("pandas is required to load the Lamian TSV") from e

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required for plotting") from e

    try:
        import umap
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "umap-learn is required. Install with: pip install umap-learn"
        ) from e

    np.random.seed(args.seed)

    h5ad_path = Path(args.h5ad)
    lamian_path = Path(args.lamian)

    adata = ad.read_h5ad(h5ad_path)
    df = pd.read_csv(lamian_path, sep="\t")

    if args.X_key not in adata.obsm:
        raise SystemExit(
            f"{args.X_key!r} not found in adata.obsm. Available: {list(adata.obsm.keys())}"
        )

    X0 = np.asarray(adata.obsm[args.X_key])

    graph = pgd2.construct_pseudotime_graph_from_table(df, adata=adata, k=args.k)

    # Pseudotime for coloring
    # Use a selector instead of substring matching hidden inside the implementation.
    pt = pgd2.compute_pseudotime_from_table(
        df,
        adata=adata,
        backbone_selector=lambda b: str(b).startswith("backbone"),
    )

    alphas = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    # Precompute branch anchor cell IDs if requested
    branch_anchors: dict[str, list[str]] = {}
    if args.show_edges:
        for branch_name, sub in df.groupby("branch"):
            sub = sub.sort_values("pseudotime", ascending=True)
            cell_ids = sub["cell_id"].astype(str).to_numpy()
            if cell_ids.size < 2:
                continue
            # pick ~25 anchors (or fewer)
            num = min(25, cell_ids.size)
            anchor_idx = np.unique(
                np.linspace(0, cell_ids.size - 1, num=num, dtype=int)
            )
            branch_anchors[str(branch_name)] = cell_ids[anchor_idx].tolist()

    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    axes = axes.ravel()

    for ax, alpha in zip(axes, alphas, strict=True):
        Xs = pgd2.diffuse_features(X0, graph, alpha=alpha, t=args.t)

        reducer = umap.UMAP(
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.seed,
        )
        emb = reducer.fit_transform(Xs)

        if args.show_edges and branch_anchors:
            id_to_row = {str(cid): i for i, cid in enumerate(adata.obs_names)}
            for _, anchors in branch_anchors.items():
                rows = [id_to_row.get(a) for a in anchors]
                rows = [r for r in rows if r is not None]
                if len(rows) < 2:
                    continue
                xy = emb[np.asarray(rows)]
                ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=0.5, alpha=0.25)

        sc = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=pt,
            s=args.point_size,
            cmap="viridis",
            linewidths=0,
            alpha=0.9,
        )
        ax.set_title(f"α = {alpha:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    # Shared colorbar
    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Pseudotime (early → late)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
