"""
src/utils.py
============
Shared utilities for the DGM-BRCA stratification pipeline:

* config loading
* patient-ID harmonisation between cohorts and cluster files
* loading binary mutation matrices and cluster assignments
* publication-grade matplotlib styling + native TIFF (>=600 DPI) export
* a small registry of genes whose extreme mutation frequencies are most
  plausibly driven by coding length / mapping artefacts rather than biology.

The last point is a deliberate scientific safeguard: in this cohort several
"top" genes (TTN, the MUC family, PABPC3, SYNE1, OBSCN ...) are among the
longest / most repetitive in the genome and are classic passenger/false-positive
flags in tumour-only mutation calling. We never silently drop them, but every
frequency table produced downstream carries an explicit `length_artifact_flag`
column so reviewers and readers can see which signals need orthogonal validation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Length / mapping artefact registry
# ----------------------------------------------------------------------------
# Genes flagged here are NOT excluded automatically. They are annotated so that
# a human can decide. References: Lawrence et al. Nature 2013 (MutSigCV, gene
# length covariate); TTN/MUC/OBSCN/SYNE1 routinely appear as significantly
# mutated purely because of their size and are commonly filtered in BRCA studies.
LENGTH_ARTIFACT_GENES = {
    "TTN", "MUC16", "MUC17", "MUC6", "MUC4", "MUC5B", "MUC12", "MUC2",
    "MUCL3", "PABPC3", "SYNE1", "SYNE2", "OBSCN", "NEB", "RYR2", "RYR3",
    "FLG", "DNAH5", "DNAH11", "DST", "HMCN1", "FAT1", "FAT3", "CSMD3",
    "USH2A", "LRP1B", "PCLO", "AHNAK2", "HLA-DRB5",
}


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
def load_config(config_path: str | os.PathLike) -> dict:
    """Load the YAML config and resolve relative paths against the repo root."""
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    repo_root = config_path.parent.parent  # config/ lives one level below root
    cfg["_repo_root"] = str(repo_root)

    def _resolve(p: str) -> str:
        p = Path(p)
        return str(p if p.is_absolute() else (repo_root / p))

    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = _resolve(val)
    return cfg


def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


# ----------------------------------------------------------------------------
# Data loading + ID harmonisation
# ----------------------------------------------------------------------------
def harmonise_ids(index: pd.Index, id_len: int = 12) -> pd.Index:
    """Truncate barcodes (e.g. TCGA-AN-A0FL-01A -> TCGA-AN-A0FL) so that
    matrices, cluster files and clinical tables join on a common key."""
    return index.astype(str).str.slice(0, id_len)


def load_mutation_matrix(
    path: str | os.PathLike,
    id_len: int = 12,
    drop_duplicate_patients: bool = True,
    binarise: bool = True,
) -> pd.DataFrame:
    """Load a binary patient x gene matrix with a harmonised patient index."""
    mat = pd.read_csv(path, index_col=0)
    mat.index = harmonise_ids(mat.index, id_len)
    mat.index.name = "patient_id"
    if drop_duplicate_patients:
        mat = mat[~mat.index.duplicated(keep="first")]
    if binarise:
        mat = (mat.fillna(0) > 0).astype(int)
    return mat


def load_clusters(
    cluster_dir: str | os.PathLike,
    cluster_col: str,
    filename: Optional[str] = None,
    id_len: int = 12,
    relabel: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """Load one cluster-assignment CSV -> DataFrame[patient_id, cluster_col]."""
    filename = filename or f"{cluster_col}.csv"
    df = pd.read_csv(Path(cluster_dir) / filename)
    id_col = "patient_id" if "patient_id" in df.columns else df.columns[0]
    df = df.rename(columns={id_col: "patient_id"})
    df["patient_id"] = harmonise_ids(pd.Index(df["patient_id"]), id_len)
    if cluster_col not in df.columns:
        # fall back to the first non-id integer column
        cand = [c for c in df.columns if c != "patient_id"][0]
        df = df.rename(columns={cand: cluster_col})
    if relabel:
        df[cluster_col] = df[cluster_col].map(lambda x: relabel.get(x, x))
    df[cluster_col] = df[cluster_col].astype(int)
    return df[["patient_id", cluster_col]]


def merge_matrix_clusters(
    matrix: pd.DataFrame, clusters: pd.DataFrame, cluster_col: str
) -> pd.DataFrame:
    """Inner-join a mutation matrix with a cluster assignment on patient_id."""
    merged = matrix.reset_index().merge(clusters, on="patient_id", how="inner")
    return merged


def gene_columns(df: pd.DataFrame, cluster_col: str) -> list[str]:
    """Return the gene feature columns of a merged frame."""
    drop = {"patient_id", cluster_col, "total_mutations"}
    return [c for c in df.columns if c not in drop]


def add_tmb(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Add a per-patient total mutation burden column (count of mutated genes)."""
    genes = gene_columns(df, cluster_col)
    df = df.copy()
    df["total_mutations"] = df[genes].sum(axis=1)
    return df


def is_length_artifact(gene: str) -> bool:
    return gene.upper() in LENGTH_ARTIFACT_GENES


def cluster_display_name(raw_label: int, offset: int = 1) -> str:
    return f"Cluster {int(raw_label) + offset}"


# ----------------------------------------------------------------------------
# Publication styling + TIFF export
# ----------------------------------------------------------------------------
def set_pub_style(figure_cfg: dict) -> None:
    """Apply a clean, consistent publication style globally."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,  # on-screen; save dpi handled separately
            "savefig.dpi": figure_cfg.get("dpi", 600),
            "font.family": figure_cfg.get("font_family", "DejaVu Sans"),
            "font.size": figure_cfg.get("base_fontsize", 12),
            "axes.titlesize": figure_cfg.get("base_fontsize", 12) + 2,
            "axes.labelsize": figure_cfg.get("base_fontsize", 12),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "figure.autolayout": False,
            "pdf.fonttype": 42,   # editable text in vector output
            "ps.fonttype": 42,
        }
    )


def save_publication_figure(
    fig: "plt.Figure",
    out_path_no_ext: str | os.PathLike,
    figure_cfg: dict,
    tight: bool = True,
) -> str:
    """Save a matplotlib figure natively as TIFF at the configured DPI, with an
    optional vector PDF sibling for LaTeX. Returns the TIFF path."""
    out_path_no_ext = str(out_path_no_ext)
    ensure_dir(os.path.dirname(out_path_no_ext) or ".")
    dpi = figure_cfg.get("dpi", 600)
    fmt = figure_cfg.get("format", "tiff")
    ext = "tif" if fmt in ("tif", "tiff") else fmt
    tiff_path = f"{out_path_no_ext}.{ext}"

    save_kwargs = dict(dpi=dpi, bbox_inches="tight" if tight else None)
    # Pillow-backed lossless compression for TIFF
    if fmt in ("tif", "tiff"):
        save_kwargs["pil_kwargs"] = {"compression": figure_cfg.get("compression", "tiff_lzw")}

    fig.savefig(tiff_path, format="tiff" if fmt in ("tif", "tiff") else fmt, **save_kwargs)
    if figure_cfg.get("also_save_pdf", False):
        fig.savefig(f"{out_path_no_ext}.pdf", bbox_inches="tight" if tight else None)
    plt.close(fig)
    return tiff_path


# ----------------------------------------------------------------------------
# Convenience: build the {solution_name: merged_df} dict used everywhere
# ----------------------------------------------------------------------------
def build_solution_frames(cfg: dict, cohort: str = "indian") -> Dict[str, Tuple[str, pd.DataFrame]]:
    """Return {solution_name: (cluster_col, merged_df_with_tmb)} for every
    clustering solution declared in the config."""
    matrix_key = f"{cohort}_matrix"
    matrix = load_mutation_matrix(
        cfg["paths"][matrix_key], id_len=cfg["cohorts"]["patient_id_len"]
    )
    out: Dict[str, Tuple[str, pd.DataFrame]] = {}
    for sol_name, cluster_col in cfg["clustering"]["solutions"].items():
        relabel = cfg["clustering"].get("relabel", {}).get(sol_name)
        clusters = load_clusters(
            cfg["paths"]["cluster_dir"],
            cluster_col,
            id_len=cfg["cohorts"]["patient_id_len"],
            relabel=relabel,
        )
        merged = add_tmb(merge_matrix_clusters(matrix, clusters, cluster_col), cluster_col)
        out[sol_name] = (cluster_col, merged)
    return out
