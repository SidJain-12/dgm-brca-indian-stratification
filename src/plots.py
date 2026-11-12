"""
src/plots.py
============
All publication figures for the pipeline, consolidated from the original
monolith's duplicated plotting blocks into a handful of parameterised
functions. Every figure is exported natively as TIFF at the configured DPI
(>=600) via utils.save_publication_figure, with an optional vector PDF sibling.

Figures provided:
  * top_mutated_genes_bar   (Fig 2a / 2b): top-N gene frequencies, one cohort
  * tmb_boxplot             (Fig 5):       per-cluster TMB distribution + KW test
  * cluster_gene_heatmap    (Fig 4c/4d):   gene x cluster frequency heatmap
  * pathway_alteration_bars (Fig 3):       5% / 10% pathway alteration rates
  * grouped_gene_barplot    (Fig 7b):      per-gene mutation frequency by cluster
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

from . import utils


# ----------------------------------------------------------------------------
# Fig 2: top mutated genes for one cohort
# ----------------------------------------------------------------------------
def top_mutated_genes_bar(
    matrix: pd.DataFrame,
    cohort_label: str,
    out_path_no_ext: str,
    figure_cfg: dict,
    top_n: int = 15,
    palette: str = "viridis",
) -> str:
    freqs = (matrix.mean(axis=0) * 100).sort_values(ascending=False).head(top_n)
    flags = [utils.is_length_artifact(g) for g in freqs.index]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette(palette, len(freqs))
    bars = ax.bar(range(len(freqs)), freqs.values, color=colors)
    # mark length-artefact genes with a hatch + asterisk so the caveat is visible
    for bar, flag in zip(bars, flags):
        if flag:
            bar.set_hatch("///")
            bar.set_edgecolor("0.25")
    for i, (v, flag) in enumerate(zip(freqs.values, flags)):
        ax.text(i, v + 1, f"{v:.1f}%" + ("*" if flag else ""), ha="center", fontsize=10)

    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels(freqs.index, rotation=45, ha="right")
    ax.set_xlabel("Gene")
    ax.set_ylabel("Patients mutated (%)")
    ax.set_title(f"Top {top_n} mutated genes — {cohort_label} (n = {len(matrix)})")
    ax.set_ylim(0, freqs.max() * 1.15)
    if any(flags):
        ax.text(
            0.99, 0.97,
            "* hatched: long/repetitive gene — possible length/mapping artefact",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, style="italic",
        )
    return utils.save_publication_figure(fig, out_path_no_ext, figure_cfg)


# ----------------------------------------------------------------------------
# Fig 5: TMB boxplot per cluster (+ whole population) with Kruskal-Wallis
# ----------------------------------------------------------------------------
def tmb_boxplot(
    merged: pd.DataFrame,
    cluster_col: str,
    out_path_no_ext: str,
    figure_cfg: dict,
    display_offset: int = 1,
    title: str = "",
) -> str:
    if "total_mutations" not in merged.columns:
        merged = utils.add_tmb(merged, cluster_col)
    clusters = sorted(merged[cluster_col].unique())
    records, groups = [], []
    for c in clusters:
        vals = merged.loc[merged[cluster_col] == c, "total_mutations"]
        groups.append(vals.values)
        for v in vals:
            records.append({"Group": utils.cluster_display_name(c, display_offset), "TMB": v})
    for v in merged["total_mutations"]:
        records.append({"Group": "Whole cohort", "TMB": v})
    box_df = pd.DataFrame(records)

    p_txt = ""
    if len(groups) >= 2:
        h, p = kruskal(*groups)
        p_txt = "Kruskal-Wallis: p < 0.0001" if p < 1e-4 else f"Kruskal-Wallis: p = {p:.3g}"

    order = [utils.cluster_display_name(c, display_offset) for c in clusters] + ["Whole cohort"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=box_df, x="Group", y="TMB", order=order, ax=ax, hue="Group",
                legend=False, palette="Set2", showfliers=True, width=0.6)
    means = box_df.groupby("Group")["TMB"].mean().reindex(order)
    ax.scatter(range(len(order)), means.values, marker="D", color="red", zorder=5, label="mean")
    ax.set_ylabel("Total mutated genes (TMB)")
    ax.set_xlabel("")
    ax.set_title((title + "  " + p_txt).strip())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    return utils.save_publication_figure(fig, out_path_no_ext, figure_cfg)


# ----------------------------------------------------------------------------
# Fig 4c/4d: gene x cluster frequency heatmap
# ----------------------------------------------------------------------------
def cluster_gene_heatmap(
    merged: pd.DataFrame,
    cluster_col: str,
    out_path_no_ext: str,
    figure_cfg: dict,
    top_n: int = 20,
    display_offset: int = 1,
    title: str = "",
) -> str:
    genes_all = utils.gene_columns(merged, cluster_col)
    top_genes = (merged[genes_all].mean().sort_values(ascending=False).head(top_n).index.tolist())
    clusters = sorted(merged[cluster_col].unique())
    data = {}
    for c in clusters:
        sub = merged[merged[cluster_col] == c]
        col = utils.cluster_display_name(c, display_offset) + f"\n(n={len(sub)})"
        data[col] = [100 * sub[g].mean() for g in top_genes]
    heat = pd.DataFrame(data, index=top_genes)

    fig, ax = plt.subplots(figsize=(1.7 * len(clusters) + 3, 0.45 * top_n + 2))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=100,
                cbar_kws={"label": "Mutation frequency (%)"}, ax=ax, linewidths=0.5,
                linecolor="white")
    # italicise artefact-flagged gene labels
    for lbl in ax.get_yticklabels():
        if utils.is_length_artifact(lbl.get_text()):
            lbl.set_style("italic")
            lbl.set_color("0.35")
    ax.set_title(title or "Top mutated genes by cluster")
    ax.set_xlabel("")
    ax.set_ylabel("Gene")
    return utils.save_publication_figure(fig, out_path_no_ext, figure_cfg)


# ----------------------------------------------------------------------------
# Fig 3: pathway alteration rates (5% / 10% thresholds), side by side
# ----------------------------------------------------------------------------
def pathway_alteration_bars(
    pathway_results: Dict[str, Dict[str, float]],
    cohort_label: str,
    out_path_no_ext: str,
    figure_cfg: dict,
) -> str:
    names = [p.replace("_", " ").title() for p in pathway_results]
    r5 = [v["pathway_alteration_rate_5pct"] for v in pathway_results.values()]
    r10 = [v["pathway_alteration_rate_10pct"] for v in pathway_results.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    for ax, rates, thr, color in ((ax1, r5, "5", "orange"), (ax2, r10, "10", "coral")):
        bars = ax.bar(names, rates, color=color, alpha=0.85)
        for b, val in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.5, f"{val:.1f}%",
                    ha="center", fontsize=8)
        ax.set_title(f"Pathway alteration ({cohort_label})\n(>= {thr}% of genes altered)")
        ax.set_ylabel("Patients with pathway altered (%)")
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    return utils.save_publication_figure(fig, out_path_no_ext, figure_cfg)


# ----------------------------------------------------------------------------
# Fig 7b: per-gene mutation frequency grouped by cluster
# ----------------------------------------------------------------------------
def grouped_gene_barplot(
    merged: pd.DataFrame,
    cluster_col: str,
    genes: Sequence[str],
    out_path_no_ext: str,
    figure_cfg: dict,
    display_offset: int = 1,
    title: str = "",
) -> str:
    clusters = sorted(merged[cluster_col].unique())
    records = []
    for g in genes:
        for c in clusters:
            sub = merged[merged[cluster_col] == c]
            records.append(
                {
                    "Gene": g,
                    "Cluster": utils.cluster_display_name(c, display_offset),
                    "Frequency": 100 * sub[g].mean(),
                }
            )
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(1.1 * len(genes) + 4, 7))
    sns.barplot(data=df, x="Gene", y="Frequency", hue="Cluster", ax=ax, palette="Set1")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.0f%%", fontsize=8)
    ax.set_ylabel("Mutation frequency (%)")
    ax.set_xlabel("Gene")
    ax.set_title(title or "Per-gene mutation frequency by cluster")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    return utils.save_publication_figure(fig, out_path_no_ext, figure_cfg)
