# Figure reproductions

## Figure 4 (alpha sweep)

This repo includes a script that recreates the manuscript’s Figure 4-style panel grid: UMAPs computed from PGD-smoothed Harmony PCs for increasing α.

Inputs:
- `data/GSE203244_processed.h5ad`
- `data/lamian_nc_5.tsv`

Run:

```bash
/Users/brandonlukas/code/brandon/pgd2/.venv/bin/python scripts/recreate_figure4_alpha_sweep.py \
  --show-edges \
  --out figures/figure4_alpha_sweep.png
```

Notes:
- The faint lines shown with `--show-edges` are sparse “branch path traces” (anchors along each branch connected in order) for visual guidance.
- For a closer match to the paper’s caption (“edges indicate the Lamian-inferred trajectory graph”), we’d need the cluster-level trajectory graph or cluster labels/edges emitted by Lamian.
