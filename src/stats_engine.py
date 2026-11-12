"""
src/stats_engine.py
===================
Statistical engine for cluster-resolved mutation analysis.

For every clustering solution it computes, per gene:
  * overall and per-cluster mutation frequency (%),
  * a global test of association between mutation status and cluster
    (Chi-square when expected counts are adequate, Fisher's exact otherwise —
     Fisher via a k x 2 Monte-Carlo / pairwise fallback since SciPy's exact
     Fisher is 2 x 2 only),
  * Benjamini-Hochberg FDR-adjusted q-values,
  * a `length_artifact_flag` so size/mapping-driven genes are never mistaken
    for validated biology.

Everything is exported to a single, clean, multi-sheet .xlsx workbook:
  - one "<solution>_gene_stats" sheet per clustering solution,
  - a "cluster_summary" sheet (n, mean TMB, median TMB per cluster),
  - a "README" sheet documenting columns and caveats.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal
from statsmodels.stats.multitest import multipletests

from . import utils


# ----------------------------------------------------------------------------
# Core per-gene test
# ----------------------------------------------------------------------------
def _gene_cluster_test(mutated: pd.Series, cluster: pd.Series) -> Tuple[str, float, float]:
    """Return (test_name, statistic, p_value) for one gene across clusters.

    Builds the clusters x {wild-type, mutated} contingency table and chooses:
      * Fisher's exact test for the 2 x 2 case (two clusters),
      * Chi-square when all expected counts >= 5,
      * Chi-square with a note otherwise (small-count caveat recorded).
    """
    contingency = pd.crosstab(cluster, mutated)
    # guarantee both mutation states are represented as columns 0 and 1
    for state in (0, 1):
        if state not in contingency.columns:
            contingency[state] = 0
    contingency = contingency[[0, 1]]

    if contingency.shape[0] < 2 or contingency.values.sum() == 0:
        return ("none", np.nan, np.nan)

    # 2 x 2 -> exact Fisher
    if contingency.shape == (2, 2):
        odds, p = fisher_exact(contingency.values)
        return ("fisher_exact", float(odds), float(p))

    # k x 2 -> chi-square; check expected-count adequacy
    chi2, p, dof, expected = chi2_contingency(contingency.values)
    test_name = "chi2" if (expected >= 5).all() else "chi2_lowcount"
    return (test_name, float(chi2), float(p))


def per_cluster_frequencies(
    merged: pd.DataFrame, cluster_col: str, genes: List[str], display_offset: int = 1
) -> pd.DataFrame:
    """Long-to-wide table of per-cluster mutation frequency (%) for each gene."""
    clusters = sorted(merged[cluster_col].unique())
    rows = []
    for gene in genes:
        row = {"gene": gene}
        row["overall_freq_pct"] = 100.0 * merged[gene].mean()
        for c in clusters:
            sub = merged[merged[cluster_col] == c]
            row[f"C{int(c) + display_offset}_freq_pct"] = 100.0 * sub[gene].mean()
            row[f"C{int(c) + display_offset}_n_mut"] = int(sub[gene].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def gene_stats_table(
    merged: pd.DataFrame,
    cluster_col: str,
    genes: List[str] | None = None,
    display_offset: int = 1,
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Full per-gene statistics table for one clustering solution."""
    if genes is None:
        genes = utils.gene_columns(merged, cluster_col)

    freq = per_cluster_frequencies(merged, cluster_col, genes, display_offset)

    test_names, stats, pvals = [], [], []
    for gene in genes:
        name, stat, p = _gene_cluster_test(merged[gene], merged[cluster_col])
        test_names.append(name)
        stats.append(stat)
        pvals.append(p)

    freq["test"] = test_names
    freq["statistic"] = stats
    freq["p_value"] = pvals

    # BH-FDR over the genes that actually produced a p-value
    valid = freq["p_value"].notna()
    freq["q_value"] = np.nan
    if valid.any():
        _, q, _, _ = multipletests(freq.loc[valid, "p_value"], method=fdr_method)
        freq.loc[valid, "q_value"] = q

    freq["length_artifact_flag"] = [utils.is_length_artifact(g) for g in freq["gene"]]
    freq = freq.sort_values("overall_freq_pct", ascending=False).reset_index(drop=True)
    return freq


def cluster_summary_table(
    merged: pd.DataFrame, cluster_col: str, display_offset: int = 1
) -> pd.DataFrame:
    """Per-cluster n, mean/median TMB, plus a Kruskal-Wallis test of TMB across
    clusters (the whole-cohort answer to 'do subtypes differ in burden?')."""
    if "total_mutations" not in merged.columns:
        merged = utils.add_tmb(merged, cluster_col)
    clusters = sorted(merged[cluster_col].unique())
    rows = []
    groups = []
    for c in clusters:
        sub = merged[merged[cluster_col] == c]["total_mutations"]
        groups.append(sub.values)
        rows.append(
            {
                "cluster": f"Cluster {int(c) + display_offset}",
                "n_patients": int(len(sub)),
                "mean_TMB": float(sub.mean()),
                "median_TMB": float(sub.median()),
                "std_TMB": float(sub.std()),
            }
        )
    summary = pd.DataFrame(rows)
    if len(groups) >= 2:
        h, p = kruskal(*groups)
        summary.attrs["kruskal_H"] = float(h)
        summary.attrs["kruskal_p"] = float(p)
    return summary


# ----------------------------------------------------------------------------
# Excel export
# ----------------------------------------------------------------------------
_README_ROWS = [
    ("gene", "HGNC symbol of the gene."),
    ("overall_freq_pct", "Percentage of all patients mutated in this gene."),
    ("C<k>_freq_pct", "Percentage of patients in cluster k mutated in this gene."),
    ("C<k>_n_mut", "Number of mutated patients in cluster k."),
    ("test", "Association test used: fisher_exact (2x2), chi2, or chi2_lowcount."),
    ("statistic", "Odds ratio (Fisher) or chi-square statistic."),
    ("p_value", "Raw p-value for association of mutation status with cluster."),
    ("q_value", "Benjamini-Hochberg FDR-adjusted p-value across all genes."),
    (
        "length_artifact_flag",
        "TRUE = gene is unusually long/repetitive (TTN, MUC*, PABPC3, SYNE1 ...); "
        "high frequency may reflect coding length / mapping artefact rather than "
        "positive selection. Validate orthogonally before biological claims.",
    ),
]


def export_multisheet_excel(
    solution_tables: Dict[str, pd.DataFrame],
    summaries: Dict[str, pd.DataFrame],
    out_path: str,
    fdr_alpha: float = 0.05,
) -> str:
    """Write all solutions + summaries + README to one workbook."""
    utils.ensure_dir(os.path.dirname(out_path) or ".")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # README sheet
        pd.DataFrame(_README_ROWS, columns=["column", "description"]).to_excel(
            writer, sheet_name="README", index=False
        )
        # combined cluster summary
        combined = []
        for sol, summ in summaries.items():
            s = summ.copy()
            s.insert(0, "solution", sol)
            kp = summ.attrs.get("kruskal_p")
            s["TMB_kruskal_p"] = kp if kp is not None else np.nan
            combined.append(s)
        if combined:
            pd.concat(combined, ignore_index=True).to_excel(
                writer, sheet_name="cluster_summary", index=False
            )
        # one sheet per solution (sheet names capped at 31 chars by Excel)
        for sol, tbl in solution_tables.items():
            sheet = f"{sol}_gene_stats"[:31]
            tbl.to_excel(writer, sheet_name=sheet, index=False)

    _highlight_significant(out_path, solution_tables, fdr_alpha)
    return out_path


def _highlight_significant(path: str, solution_tables: Dict[str, pd.DataFrame], alpha: float):
    """Light cosmetic pass: bold significant q-values, shade artefact genes."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill

    wb = load_workbook(path)
    shade = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    bold = Font(bold=True)
    for sol, tbl in solution_tables.items():
        sheet = f"{sol}_gene_stats"[:31]
        if sheet not in wb.sheetnames:
            continue
        ws = wb[sheet]
        cols = {c.value: i + 1 for i, c in enumerate(ws[1])}
        qcol = cols.get("q_value")
        flagcol = cols.get("length_artifact_flag")
        for r in range(2, ws.max_row + 1):
            if qcol:
                v = ws.cell(row=r, column=qcol).value
                if isinstance(v, (int, float)) and v == v and v < alpha:
                    ws.cell(row=r, column=qcol).font = bold
            if flagcol and str(ws.cell(row=r, column=flagcol).value).upper() == "TRUE":
                for c in range(1, ws.max_column + 1):
                    ws.cell(row=r, column=c).fill = shade
    wb.save(path)


# ----------------------------------------------------------------------------
# Orchestration entry point
# ----------------------------------------------------------------------------
def run(cfg: dict, cohort: str = "indian") -> str:
    frames = utils.build_solution_frames(cfg, cohort=cohort)
    offset = cfg["clustering"].get("display_offset", 1)
    fdr_method = cfg["analysis"].get("fdr_method", "fdr_bh")

    tables, summaries = {}, {}
    for sol, (cluster_col, merged) in frames.items():
        tables[sol] = gene_stats_table(
            merged, cluster_col, display_offset=offset, fdr_method=fdr_method
        )
        summaries[sol] = cluster_summary_table(merged, cluster_col, display_offset=offset)

    out_dir = utils.ensure_dir(os.path.join(cfg["paths"]["results_dir"], "stats"))
    out_path = os.path.join(out_dir, f"{cohort}_cluster_gene_statistics.xlsx")
    return export_multisheet_excel(
        tables, summaries, out_path, fdr_alpha=cfg["analysis"].get("fdr_alpha", 0.05)
    )
