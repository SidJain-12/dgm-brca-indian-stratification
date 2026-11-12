#!/usr/bin/env python
"""
main.py
=======
Command-line orchestrator for the DGM-BRCA downstream analysis pipeline.

Model training (the DeepGraphMut graph autoencoder) lives in the upstream
`ta_single_omics.py` / DeepGraphMut codebase and is intentionally out of scope
here: this repository handles everything *after* embeddings + cluster
assignments exist (statistics, figures, co-occurrence, survival, oncoprints).

Usage
-----
    python main.py --config config/config.yaml --cohort indian --stage all
    python main.py --stage stats,cooccurrence
    python main.py --stage survival --cohort indian

Stages: figures | stats | cooccurrence | survival | oncoprint | all
"""
from __future__ import annotations

import argparse
import os

from src import utils, plots, stats_engine, cooccurrence, survival, oncoprint


# Default Focal-Adhesion gene set for the demonstration OncoPrint (Fig 7).
FOCAL_ADHESION_GENES = [
    "PIK3CA", "EGFR", "LAMA1", "LAMA3", "TNXB", "ERBB2", "PTEN", "RELN",
    "VTN", "VWF", "LAMA5", "TLN2", "FLNC", "MAP2K1", "LAMA2",
]


def run_figures(cfg: dict, cohort: str) -> None:
    fig_dir = utils.ensure_dir(os.path.join(cfg["paths"]["results_dir"], "figures"))
    figcfg = cfg["figure"]
    offset = cfg["clustering"].get("display_offset", 1)
    top_n = cfg["analysis"]["top_n_genes"]

    # Fig 2a/2b: top mutated genes for whichever cohort matrices exist
    for name, key in (("indian", "indian_matrix"), ("tcga", "tcga_matrix")):
        if os.path.exists(cfg["paths"][key]):
            m = utils.load_mutation_matrix(cfg["paths"][key], cfg["cohorts"]["patient_id_len"])
            plots.top_mutated_genes_bar(
                m, cfg["cohorts"][f"{name}_label"],
                os.path.join(fig_dir, f"fig2_top_genes_{name}"), figcfg, top_n=15,
            )

    frames = utils.build_solution_frames(cfg, cohort=cohort)
    for sol, (cluster_col, merged) in frames.items():
        plots.tmb_boxplot(merged, cluster_col,
                          os.path.join(fig_dir, f"fig5_tmb_{sol}"), figcfg,
                          offset, title=f"{sol}")
        plots.cluster_gene_heatmap(merged, cluster_col,
                                   os.path.join(fig_dir, f"fig4_heatmap_{sol}"), figcfg,
                                   top_n=top_n, display_offset=offset, title=sol)
        present = [g for g in FOCAL_ADHESION_GENES if g in merged.columns]
        if present:
            plots.grouped_gene_barplot(merged, cluster_col, present,
                                       os.path.join(fig_dir, f"fig7b_focal_{sol}"), figcfg,
                                       offset, title=f"Focal Adhesion — {sol}")


def run_oncoprints(cfg: dict, cohort: str) -> None:
    onc_dir = utils.ensure_dir(os.path.join(cfg["paths"]["results_dir"], "oncoprints"))
    frames = utils.build_solution_frames(cfg, cohort=cohort)
    offset = cfg["clustering"].get("display_offset", 1)
    for sol, (cluster_col, merged) in frames.items():
        present = [g for g in FOCAL_ADHESION_GENES if g in merged.columns]
        if present:
            oncoprint.oncoprint(merged, cluster_col, present,
                                os.path.join(onc_dir, f"fig7a_focal_oncoprint_{sol}"),
                                cfg["figure"], pathway_name="Focal Adhesion",
                                display_offset=offset)


STAGES = {
    "figures": run_figures,
    "stats": lambda cfg, cohort: print("  -> wrote", stats_engine.run(cfg, cohort)),
    "cooccurrence": lambda cfg, cohort: print("  -> wrote", cooccurrence.run(cfg, cohort)),
    "survival": lambda cfg, cohort: print("  -> wrote", survival.run(cfg, cohort)),
    "oncoprint": run_oncoprints,
}


def main():
    ap = argparse.ArgumentParser(description="DGM-BRCA downstream analysis pipeline")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--cohort", default="indian", choices=["indian", "tcga"])
    ap.add_argument("--stage", default="all",
                    help="comma-separated: figures,stats,cooccurrence,survival,oncoprint,all")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    utils.set_pub_style(cfg["figure"])

    stages = list(STAGES) if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    for stage in stages:
        if stage not in STAGES:
            raise SystemExit(f"Unknown stage '{stage}'. Valid: {', '.join(STAGES)} or 'all'.")
        print(f"[stage] {stage}")
        STAGES[stage](cfg, args.cohort)
    print("Done.")


if __name__ == "__main__":
    main()
