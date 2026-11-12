"""End-to-end smoke test on synthetic data (no real patient data required)."""
import os, numpy as np, pandas as pd
from PIL import Image

from src import utils, plots, stats_engine, cooccurrence, survival, oncoprint

rng = np.random.default_rng(0)
ROOT = os.path.dirname(__file__)

# ---- synthetic cohort: 300 patients, 60 genes, 3 latent clusters ----
n, genes = 300, [f"GENE{i}" for i in range(55)] + ["TP53", "PIK3CA", "TTN", "MUC6", "EGFR"]
pids = [f"TCGA-XX-{i:04d}-01A" for i in range(n)]
clabels = rng.integers(0, 3, n)
base = rng.uniform(0.03, 0.25, len(genes))
M = np.zeros((n, len(genes)), int)
for i in range(n):
    lift = np.zeros(len(genes)); c = clabels[i]
    lift[c*5:(c*5)+5] += 0.4                      # cluster-specific genes
    if c == 2: lift[:] += 0.25                     # cluster 3 hypermutated
    M[i] = (rng.uniform(0, 1, len(genes)) < np.clip(base + lift, 0, 0.95)).astype(int)
mat = pd.DataFrame(M, index=pids, columns=genes)

os.makedirs(f"{ROOT}/data/raw", exist_ok=True)
os.makedirs(f"{ROOT}/data/clusters", exist_ok=True)
mat.to_csv(f"{ROOT}/data/raw/indian_patient_gene_matrix.csv")
mat.sample(frac=0.8, random_state=1).to_csv(f"{ROOT}/data/raw/tcga_patient_gene_matrix.csv")

trunc = [p[:12] for p in pids]
pd.DataFrame({"patient_id": trunc, "cluster_PAM_k2": (clabels > 0).astype(int)}).to_csv(
    f"{ROOT}/data/clusters/cluster_PAM_k2.csv", index=False)
pd.DataFrame({"patient_id": trunc, "cluster_PAM_k3": clabels}).to_csv(
    f"{ROOT}/data/clusters/cluster_PAM_k3.csv", index=False)

# clinical: worse survival for hypermutated cluster + TP53-mut
t = rng.exponential(60, n) * (0.5 + 0.5*(clabels < 2)) * (1 - 0.3*mat["TP53"].values)
pd.DataFrame({"OS_time": np.clip(t, 1, 200), "OS_event": rng.integers(0, 2, n)},
             index=trunc).to_csv(f"{ROOT}/data/raw/BRCA_surv.csv")

# ---- run pipeline ----
cfg = utils.load_config(f"{ROOT}/config/config.yaml")
utils.set_pub_style(cfg["figure"])

import main as pipeline
pipeline.run_figures(cfg, "indian")
xlsx = stats_engine.run(cfg, "indian")
cooccurrence.run(cfg, "indian")
survival.run(cfg, "indian")
pipeline.run_oncoprints(cfg, "indian")

# ---- verify outputs ----
tif = f"{ROOT}/results/figures/fig5_tmb_PAM_k3.tif"
im = Image.open(tif)
print("TIFF dpi:", im.info.get("dpi"), "size:", im.size)

import openpyxl
wb = openpyxl.load_workbook(xlsx)
print("Excel sheets:", wb.sheetnames)

tifs = []
for dp, _, fs in os.walk(f"{ROOT}/results"):
    tifs += [f for f in fs if f.endswith(".tif")]
print("total TIFFs:", len(tifs))
print("survival result exists:", os.path.exists(f"{ROOT}/results/survival/survival_results.csv"))
print("SMOKE TEST OK")
