# DGM-BRCA: Downstream Stratification & Analysis Pipeline

Modular, publication-grade analysis code for *Stratifying Indian Breast Cancer
Cohorts Using Graph Autoencoders and Somatic Mutation Networks*.

This repository covers everything **downstream of the DeepGraphMut (DGM) graph
autoencoder** — statistics, figures, co-occurrence/mutual-exclusivity, survival,
and OncoPrints. Model training (the graph autoencoder itself) is the upstream
in-house DeepGraphMut framework and is kept separate (`ta_single_omics.py`).

## Structure

```
dgm_brca/
├── config/config.yaml        # all paths + analysis parameters live here
├── data/
│   ├── raw/                   # patient×gene matrices, clinical table
│   └── clusters/              # cluster_PAM_k2.csv, cluster_PAM_k3.csv
├── src/
│   ├── utils.py               # IO, ID harmonisation, TIFF export, artefact registry
│   ├── plots.py               # Fig 2/3/4/5/7b — native TIFF @600 DPI
│   ├── stats_engine.py        # per-cluster freq + Fisher/χ² + BH-FDR → multi-sheet XLSX
│   ├── cooccurrence.py        # per-cluster co-occurrence / mutual exclusivity
│   ├── survival.py            # KM + log-rank + Cox PH (cluster / TMB / driver)
│   └── oncoprint.py           # cluster-partitioned OncoPrint (Fig 7a)
├── main.py                    # CLI orchestrator
├── smoke_test.py              # synthetic end-to-end check (no real data needed)
└── requirements.txt
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data contracts

| File | Format |
|------|--------|
| `data/raw/indian_patient_gene_matrix.csv` | rows = patients (index), cols = genes, values ∈ {0,1} |
| `data/raw/tcga_patient_gene_matrix.csv`   | same schema (for cross-cohort comparison) |
| `data/clusters/cluster_PAM_k{2,3}.csv`    | columns `patient_id`, `cluster_PAM_k{2,3}` |
| `data/raw/BRCA_surv.csv`                  | index = patient_id, cols `OS_time`, `OS_event` |

Patient IDs are harmonised by truncation to 12 chars (`cohorts.patient_id_len`)
so matrices, clusters and clinical data join on a common key.

## Run

```bash
python main.py --stage all --cohort indian          # everything
python main.py --stage stats,cooccurrence           # selected stages
python main.py --stage survival                      # needs BRCA_surv.csv
python smoke_test.py                                  # verify install on synthetic data
```

Outputs land in `results/` (`figures/`, `stats/`, `cooccurrence/`, `survival/`,
`oncoprints/`). Every figure is a lossless TIFF at 600 DPI (+ an optional vector
PDF sibling for LaTeX).

## Scientific safeguard: length/mapping artefacts

Several top-ranked genes in this cohort (TTN, the MUC family, PABPC3, SYNE1,
OBSCN, …) are among the longest/most repetitive in the genome and are classic
false-positive flags in tumour-only mutation calling (cf. MutSigCV's gene-length
covariate, Lawrence et al. 2013). The pipeline **never silently drops them**;
instead every frequency table carries a `length_artifact_flag`, these genes are
hatched/italicised in figures, and the README sheet of the Excel export explains
the caveat. Treat their high frequencies as hypotheses requiring orthogonal
validation (VAF distributions, MutSigCV/dNdS, replication in an independent
cohort), not as established biology.
