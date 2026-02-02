# src/tools_drift.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class ToolResult:
    name: str
    summary: str
    payload: Dict[str, Any]


def _psi_from_bins(exp_pct: np.ndarray, act_pct: np.ndarray, eps: float = 1e-6) -> float:
    exp = np.clip(exp_pct, eps, 1.0)
    act = np.clip(act_pct, eps, 1.0)
    return float(np.sum((act - exp) * np.log(act / exp)))


def psi_numeric(
    baseline: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    b = baseline.dropna().to_numpy()
    c = current.dropna().to_numpy()

    if len(b) < 20 or len(c) < 20:
        return float("nan")

    # Use quantile bins from baseline (stable reference)
    quantiles = np.linspace(0, 1, n_bins + 1)
    cuts = np.unique(np.quantile(b, quantiles))
    if len(cuts) < 3:
        return float("nan")

    b_counts, _ = np.histogram(b, bins=cuts)
    c_counts, _ = np.histogram(c, bins=cuts)

    b_pct = b_counts / max(b_counts.sum(), 1)
    c_pct = c_counts / max(c_counts.sum(), 1)
    return _psi_from_bins(b_pct, c_pct)


def psi_categorical(baseline: pd.Series, current: pd.Series) -> float:
    b = baseline.fillna("NA").astype(str)
    c = current.fillna("NA").astype(str)

    # Align categories
    cats = sorted(set(b.unique()).union(set(c.unique())))
    b_freq = b.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c_freq = c.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi_from_bins(b_freq, c_freq)


def drift_psi(
    df_baseline: pd.DataFrame,
    df_current: pd.DataFrame,
    top_k: int = 10,
    n_bins: int = 10,
) -> ToolResult:
    num_cols = df_baseline.select_dtypes(include="number").columns.intersection(
        df_current.select_dtypes(include="number").columns
    )
    cat_cols = df_baseline.select_dtypes(exclude="number").columns.intersection(
        df_current.select_dtypes(exclude="number").columns
    )

    psi_scores = []

    for col in num_cols:
        score = psi_numeric(df_baseline[col], df_current[col], n_bins=n_bins)
        psi_scores.append((col, score, "numeric"))

    for col in cat_cols:
        score = psi_categorical(df_baseline[col], df_current[col])
        psi_scores.append((col, score, "categorical"))

    out = pd.DataFrame(psi_scores, columns=["feature", "psi", "type"]).sort_values("psi", ascending=False)
    top = out.head(top_k)

    # Common rule of thumb interpretation
    # <0.1: no shift, 0.1-0.25: moderate shift, >0.25: major shift
    summary = f"Computed PSI for {len(out)} features. Top drift: {top.iloc[0]['feature']} (PSI={top.iloc[0]['psi']:.3f})" if len(top) else "No overlapping features for PSI."
    return ToolResult("drift_psi", summary, {"psi_table": out, "top": top})


def drift_ks(
    df_baseline: pd.DataFrame,
    df_current: pd.DataFrame,
    alpha: float = 0.05,
    top_k: int = 10,
) -> ToolResult:
    num_cols = df_baseline.select_dtypes(include="number").columns.intersection(
        df_current.select_dtypes(include="number").columns
    )

    rows = []
    for col in num_cols:
        b = df_baseline[col].dropna()
        c = df_current[col].dropna()
        if len(b) < 20 or len(c) < 20:
            continue
        ks_stat, p = ks_2samp(b, c)
        rows.append((col, float(ks_stat), float(p), bool(p < alpha)))

    out = pd.DataFrame(rows, columns=["feature", "ks_stat", "p_value", "significant"]).sort_values(
        ["significant", "ks_stat"], ascending=[False, False]
    )
    top = out.head(top_k)

    summary = f"Computed KS drift for {len(out)} numeric features. Significant shifts: {int(out['significant'].sum())}."
    return ToolResult("drift_ks", summary, {"ks_table": out, "top": top, "alpha": alpha})
