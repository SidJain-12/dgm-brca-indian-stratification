"""
src/survival.py
===============
Overall-survival (OS) analysis linking DGM-defined subtypes and mutation
features to outcome. Implements the contrasts requested for the manuscript:

  1. Subtype (cluster) stratification            -> KM + multi-group log-rank
  2. TMB-high vs TMB-low (config quantile split) -> KM + log-rank + Cox HR
  3. Driver mutated vs wild-type (e.g. TP53)     -> KM + log-rank + Cox HR
  4. Multivariable Cox PH: cluster + TMB + driver status (+ optional age)

This module *computes* results from the user's clinical table; it never
fabricates hazard ratios or p-values. If the clinical file is absent the
pipeline logs a clear message and skips this stage. A tidy results table
(`survival_results.csv`) records HR, 95% CI, and p for each contrast so the
manuscript's Results text can be populated directly from real output.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from . import utils


# ----------------------------------------------------------------------------
# Assembly of the survival analysis frame
# ----------------------------------------------------------------------------
def build_survival_frame(
    merged: pd.DataFrame,
    cluster_col: str,
    clinical: pd.DataFrame,
    time_col: str,
    event_col: str,
    tmb_high_quantile: float = 0.75,
    driver_genes: Optional[List[str]] = None,
    display_offset: int = 1,
) -> pd.DataFrame:
    """Join mutation/cluster data to clinical outcomes and derive covariates."""
    if "total_mutations" not in merged.columns:
        merged = utils.add_tmb(merged, cluster_col)

    driver_genes = driver_genes or []
    keep = ["patient_id", cluster_col, "total_mutations"] + [
        g for g in driver_genes if g in merged.columns
    ]
    df = merged[keep].copy()

    clin = clinical.copy()
    clin.index = utils.harmonise_ids(clin.index)
    clin = clin[[time_col, event_col]].dropna()
    df = df.merge(clin, left_on="patient_id", right_index=True, how="inner")

    df["cluster"] = df[cluster_col].map(lambda c: utils.cluster_display_name(c, display_offset))
    thr = df["total_mutations"].quantile(tmb_high_quantile)
    df["TMB_group"] = np.where(df["total_mutations"] >= thr, "TMB-high", "TMB-low")
    for g in driver_genes:
        if g in df.columns:
            df[f"{g}_status"] = np.where(df[g] == 1, f"{g}-mut", f"{g}-wt")
    df = df.rename(columns={time_col: "time", event_col: "event"})
    df["event"] = df["event"].astype(int)
    return df


# ----------------------------------------------------------------------------
# Kaplan-Meier + log-rank for a categorical grouping
# ----------------------------------------------------------------------------
def km_by_group(
    df: pd.DataFrame,
    group_col: str,
    out_path_no_ext: str,
    figure_cfg: dict,
    title: str = "",
) -> Dict[str, float]:
    fig, ax = plt.subplots(figsize=(7, 6))
    kmf = KaplanMeierFitter()
    groups = sorted(df[group_col].dropna().unique())
    for g in groups:
        mask = df[group_col] == g
        kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=f"{g} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=True)

    if len(groups) == 2:
        a, b = groups
        res = logrank_test(
            df.loc[df[group_col] == a, "time"], df.loc[df[group_col] == b, "time"],
            df.loc[df[group_col] == a, "event"], df.loc[df[group_col] == b, "event"],
        )
        p = res.p_value
    else:
        res = multivariate_logrank_test(df["time"], df[group_col], df["event"])
        p = res.p_value

    ax.set_xlabel("Overall survival time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ptxt = "log-rank p < 0.0001" if p < 1e-4 else f"log-rank p = {p:.3g}"
    ax.set_title((title + "\n" + ptxt).strip())
    utils.save_publication_figure(fig, out_path_no_ext, figure_cfg)
    return {"grouping": group_col, "logrank_p": float(p), "n": int(len(df))}


# ----------------------------------------------------------------------------
# Cox proportional hazards (univariable and multivariable)
# ----------------------------------------------------------------------------
def cox_model(
    df: pd.DataFrame,
    covariates: List[str],
    label: str,
) -> pd.DataFrame:
    """Fit a Cox PH model and return a tidy summary (HR, CI, p) per covariate."""
    model_df = df[["time", "event"] + covariates].dropna().copy()
    # one-hot encode categoricals, dropping first level as reference
    cat = [c for c in covariates if model_df[c].dtype == object]
    if cat:
        model_df = pd.get_dummies(model_df, columns=cat, drop_first=True)
    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="time", event_col="event")
    s = cph.summary.reset_index().rename(columns={"index": "covariate"})
    out = pd.DataFrame(
        {
            "model": label,
            "covariate": s["covariate"],
            "HR": np.exp(s["coef"]),
            "CI_lower": np.exp(s["coef lower 95%"]),
            "CI_upper": np.exp(s["coef upper 95%"]),
            "p_value": s["p"],
        }
    )
    return out


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_for_solution(
    merged: pd.DataFrame,
    cluster_col: str,
    clinical: pd.DataFrame,
    cfg: dict,
    solution_name: str,
) -> pd.DataFrame:
    scfg = cfg["survival"]
    offset = cfg["clustering"].get("display_offset", 1)
    out_dir = utils.ensure_dir(
        os.path.join(cfg["paths"]["results_dir"], "survival", solution_name)
    )
    df = build_survival_frame(
        merged, cluster_col, clinical,
        time_col=scfg["time_col"], event_col=scfg["event_col"],
        tmb_high_quantile=scfg["tmb_high_quantile"],
        driver_genes=scfg["driver_genes"], display_offset=offset,
    )
    if df.empty:
        print(f"[survival] no overlapping patients for {solution_name}; skipping.")
        return pd.DataFrame()

    results: List[dict] = []

    # 1. subtype
    results.append(
        km_by_group(df, "cluster",
                    os.path.join(out_dir, f"{solution_name}_KM_cluster"),
                    cfg["figure"], title=f"{solution_name}: OS by subtype")
    )
    # 2. TMB
    results.append(
        km_by_group(df, "TMB_group",
                    os.path.join(out_dir, f"{solution_name}_KM_TMB"),
                    cfg["figure"], title=f"{solution_name}: OS by TMB group")
    )
    # 3. driver mut vs wt
    for g in scfg["driver_genes"]:
        col = f"{g}_status"
        if col in df.columns and df[col].nunique() == 2:
            results.append(
                km_by_group(df, col,
                            os.path.join(out_dir, f"{solution_name}_KM_{g}"),
                            cfg["figure"], title=f"{solution_name}: OS by {g} status")
            )

    logrank_df = pd.DataFrame(results)
    logrank_df.insert(0, "solution", solution_name)

    # 4. multivariable Cox
    cox_covs = ["cluster", "TMB_group"] + [
        f"{g}_status" for g in scfg["driver_genes"] if f"{g}_status" in df.columns
    ]
    try:
        cox_df = cox_model(df, cox_covs, label=f"{solution_name}_multivariable")
    except Exception as exc:  # convergence / separation issues on small cohorts
        print(f"[survival] Cox model skipped for {solution_name}: {exc}")
        cox_df = pd.DataFrame()

    # persist
    logrank_df.to_csv(os.path.join(out_dir, f"{solution_name}_logrank.csv"), index=False)
    if not cox_df.empty:
        cox_df.to_csv(os.path.join(out_dir, f"{solution_name}_cox.csv"), index=False)
    return cox_df


def run(cfg: dict, cohort: str = "indian") -> Optional[str]:
    clin_path = cfg["paths"]["clinical"]
    if not os.path.exists(clin_path):
        print(f"[survival] clinical file not found ({clin_path}); stage skipped.")
        return None
    clinical = pd.read_csv(clin_path, index_col=0)
    frames = utils.build_solution_frames(cfg, cohort=cohort)

    all_cox = []
    for sol, (cluster_col, merged) in frames.items():
        cox_df = run_for_solution(merged, cluster_col, clinical, cfg, sol)
        if not cox_df.empty:
            all_cox.append(cox_df)

    if all_cox:
        out = os.path.join(cfg["paths"]["results_dir"], "survival", "survival_results.csv")
        pd.concat(all_cox, ignore_index=True).to_csv(out, index=False)
        return out
    return None
