"""
src/oncoprint.py
================
Cluster-partitioned OncoPrint for a chosen gene set (e.g. the Focal Adhesion
pathway). Patients are ordered by cluster and, within cluster, by mutation load;
genes are ordered by overall frequency. A top track shows per-patient burden, a
left track shows per-gene frequency, and the title carries the global
cluster-vs-mutation association test (Fisher's exact for 2 clusters,
chi-square otherwise).

Input is the binary patient x gene matrix (values in {0,1}); this matches the
cohort data. If a mutation-type MAF is available it can be layered later, but a
binary OncoPrint is the correct default for presence/absence calls.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, fisher_exact

from . import utils

_CLUSTER_BAND_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _global_association(merged: pd.DataFrame, cluster_col: str, genes: List[str]):
    """Association of 'any pathway gene mutated' with cluster."""
    any_mut = (merged[genes].sum(axis=1) > 0).astype(int)
    contingency = pd.crosstab(merged[cluster_col], any_mut)
    for s in (0, 1):
        if s not in contingency.columns:
            contingency[s] = 0
    contingency = contingency[[0, 1]]
    if contingency.shape == (2, 2):
        _, p = fisher_exact(contingency.values)
        return "Fisher's exact", p
    chi2, p, _, _ = chi2_contingency(contingency.values)
    return "chi-square", p


def oncoprint(
    merged: pd.DataFrame,
    cluster_col: str,
    genes: List[str],
    out_path_no_ext: str,
    figure_cfg: dict,
    pathway_name: str = "",
    display_offset: int = 1,
    max_genes: int = 25,
) -> str:
    genes = [g for g in genes if g in merged.columns]
    # order genes by overall frequency
    order = merged[genes].mean().sort_values(ascending=False)
    genes = order.head(max_genes).index.tolist()

    # order patients: by cluster, then by pathway mutation load (desc)
    load = merged[genes].sum(axis=1)
    ordering = merged.assign(_load=load).sort_values(
        [cluster_col, "_load"], ascending=[True, False]
    )
    mat = ordering[genes].T.values  # genes x patients, values {0,1}
    clusters = ordering[cluster_col].values
    n_genes, n_pat = mat.shape

    fig = plt.figure(figsize=(min(22, 0.03 * n_pat + 8), 0.45 * n_genes + 3))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.35, 8], width_ratios=[6, 1],
                          hspace=0.05, wspace=0.03)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_band = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_main = fig.add_subplot(gs[2, 0], sharex=ax_top)
    ax_freq = fig.add_subplot(gs[2, 1], sharey=ax_main)

    # top: per-patient burden
    ax_top.bar(np.arange(n_pat), mat.sum(axis=0), color="#8B0000", width=1.0)
    ax_top.set_ylabel("muts", fontsize=8)
    ax_top.set_xlim(-0.5, n_pat - 0.5)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(False)

    # cluster colour band
    uniq = sorted(pd.unique(clusters))
    cmap_band = {c: _CLUSTER_BAND_COLORS[i % len(_CLUSTER_BAND_COLORS)] for i, c in enumerate(uniq)}
    band = np.array([[list(int(cmap_band[c].lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4)) for c in clusters]])
    ax_band.imshow(band, aspect="auto")
    ax_band.set_yticks([])
    ax_band.tick_params(labelbottom=False)
    ax_band.grid(False)

    # main oncoprint
    cmap = ListedColormap(["#EDEDED", "#2C3E50"])
    ax_main.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="none")
    ax_main.set_yticks(range(n_genes))
    ylabels = ax_main.set_yticklabels(genes, fontsize=8)
    for lbl in ylabels:
        if utils.is_length_artifact(lbl.get_text()):
            lbl.set_style("italic"); lbl.set_color("0.4")
    ax_main.set_xticks([])
    ax_main.set_xlabel("Patients (ordered by cluster, then burden)")
    ax_main.grid(False)

    # left/right: per-gene frequency
    freqs = [100 * merged[g].mean() for g in genes]
    ax_freq.barh(range(n_genes), freqs, color="#555555")
    ax_freq.invert_yaxis()
    ax_freq.set_xlabel("freq %", fontsize=8)
    ax_freq.tick_params(labelleft=False)
    ax_freq.grid(False)

    test_name, p = _global_association(merged, cluster_col, genes)
    ptxt = "P < 0.0001" if p < 1e-4 else f"P = {p:.3g}"
    counts = ", ".join(
        f"{utils.cluster_display_name(c, display_offset)}: {int(np.sum(clusters == c))}"
        for c in uniq
    )
    fig.suptitle(f"{pathway_name} OncoPrint  ({test_name}, {ptxt})\n{counts}",
                 fontsize=11, y=0.98)

    legend = [Patch(facecolor="#2C3E50", label="mutated"),
              Patch(facecolor="#EDEDED", label="wild-type")]
    legend += [Patch(facecolor=cmap_band[c],
                     label=utils.cluster_display_name(c, display_offset)) for c in uniq]
    ax_top.legend(handles=legend, loc="upper right", ncol=2, fontsize=7)

    return utils.save_publication_figure(fig, out_path_no_ext, figure_cfg, tight=False)
