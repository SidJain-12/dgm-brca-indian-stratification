"""
src/cooccurrence.py
===================
Pairwise gene co-occurrence and mutual-exclusivity analysis, computed
separately for each cluster (and for the whole cohort).

For every gene pair (i, j) within a cluster we build the 2 x 2 table of
co-mutation and run Fisher's exact test:

        odds ratio > 1  -> tendency to CO-OCCUR
        odds ratio < 1  -> tendency to be MUTUALLY EXCLUSIVE

We report a signed association score

        score = -log10(p) * sign(log2(OR))

so a single heatmap encodes both direction (co-occurrence vs exclusivity) and
significance. p-values are BH-FDR corrected across all tested pairs within a
cluster. Output is a publication TIFF per cluster plus a tidy CSV of every pair.
"""
from __future__ import annotations

import os
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from . import utils


def _pair_table(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2x2 contingency of two binary vectors: [[both, a_only],[b_only, neither]]."""
    both = int(np.sum((a == 1) & (b == 1)))
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    neither = int(np.sum((a == 0) & (b == 0)))
    return np.array([[both, a_only], [b_only, neither]])


def pairwise_associations(
    sub: pd.DataFrame, genes: List[str], min_mutated: int = 5
) -> pd.DataFrame:
    """Return a tidy frame of Fisher associations for all gene pairs."""
    # keep only genes mutated in at least `min_mutated` patients in this subset
    usable = [g for g in genes if sub[g].sum() >= min_mutated]
    records = []
    for g1, g2 in combinations(usable, 2):
        a = sub[g1].values
        b = sub[g2].values
        table = _pair_table(a, b)
        # Haldane-Anscombe correction so OR is finite
        odds, p = fisher_exact(table)
        both = table[0, 0]
        or_corr = ((table[0, 0] + 0.5) * (table[1, 1] + 0.5)) / (
            (table[0, 1] + 0.5) * (table[1, 0] + 0.5)
        )
        direction = "co_occurrence" if or_corr > 1 else "mutual_exclusivity"
        records.append(
            {
                "gene1": g1,
                "gene2": g2,
                "n_both": int(both),
                "odds_ratio": float(or_corr),
                "log2_OR": float(np.log2(or_corr)),
                "p_value": float(p),
                "direction": direction,
            }
        )
    df = pd.DataFrame(records)
    if not df.empty:
        _, q, _, _ = multipletests(df["p_value"], method="fdr_bh")
        df["q_value"] = q
        df["signed_score"] = -np.log10(df["p_value"].clip(lower=1e-300)) * np.sign(df["log2_OR"])
    return df


def _score_matrix(pairs: pd.DataFrame, genes: List[str]) -> pd.DataFrame:
    """Symmetric gene x gene matrix of signed association scores."""
    mat = pd.DataFrame(0.0, index=genes, columns=genes)
    for _, r in pairs.iterrows():
        mat.loc[r["gene1"], r["gene2"]] = r["signed_score"]
        mat.loc[r["gene2"], r["gene1"]] = r["signed_score"]
    return mat


def plot_cooccurrence_heatmap(
    matrix: pd.DataFrame,
    title: str,
    out_path_no_ext: str,
    figure_cfg: dict,
) -> str:
    """Diverging heatmap: blue = mutual exclusivity, red = co-occurrence."""
    n = matrix.shape[0]
    size = max(6, 0.32 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    vmax = np.nanmax(np.abs(matrix.values)) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(matrix.values, cmap="RdBu_r", norm=norm, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title(title)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("signed -log10(p)\n(red: co-occurrence, blue: exclusivity)", fontsize=8)
    return utils.save_publication_figure(fig, out_path_no_ext, figure_cfg)


def run_for_solution(
    merged: pd.DataFrame,
    cluster_col: str,
    cfg: dict,
    solution_name: str,
) -> Tuple[str, List[str]]:
    """Compute + plot co-occurrence for every cluster of one solution.

    Returns (tidy_csv_path, list_of_tiff_paths).
    """
    offset = cfg["clustering"].get("display_offset", 1)
    top_n = cfg["analysis"]["cooccurrence"]["top_n_genes"]
    min_mut = cfg["analysis"]["cooccurrence"]["min_mutated"]
    out_dir = utils.ensure_dir(
        os.path.join(cfg["paths"]["results_dir"], "cooccurrence", solution_name)
    )

    all_genes = utils.gene_columns(merged, cluster_col)
    tiffs: List[str] = []
    tidy_frames: List[pd.DataFrame] = []

    clusters = sorted(merged[cluster_col].unique())
    for c in clusters:
        sub = merged[merged[cluster_col] == c]
        # rank genes by frequency *within this cluster* for interpretability
        freq = sub[all_genes].mean().sort_values(ascending=False)
        genes = freq.head(top_n).index.tolist()

        pairs = pairwise_associations(sub, genes, min_mutated=min_mut)
        cname = utils.cluster_display_name(c, offset)
        if not pairs.empty:
            pairs.insert(0, "cluster", cname)
            tidy_frames.append(pairs)
            mat = _score_matrix(pairs, genes)
            tiffs.append(
                plot_cooccurrence_heatmap(
                    mat,
                    f"{solution_name} — {cname}: co-occurrence / mutual exclusivity",
                    os.path.join(out_dir, f"{solution_name}_{cname.replace(' ', '')}_cooccurrence"),
                    cfg["figure"],
                )
            )

    csv_path = os.path.join(out_dir, f"{solution_name}_cooccurrence_pairs.csv")
    if tidy_frames:
        pd.concat(tidy_frames, ignore_index=True).to_csv(csv_path, index=False)
    return csv_path, tiffs


def run(cfg: dict, cohort: str = "indian") -> Dict[str, str]:
    frames = utils.build_solution_frames(cfg, cohort=cohort)
    out = {}
    for sol, (cluster_col, merged) in frames.items():
        csv_path, _ = run_for_solution(merged, cluster_col, cfg, sol)
        out[sol] = csv_path
    return out
