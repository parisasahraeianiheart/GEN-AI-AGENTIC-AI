from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


@dataclass
class ToolResult:
    name: str
    summary: str
    payload: Dict[str, Any]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def profile_schema(df: pd.DataFrame) -> ToolResult:
    dtypes = df.dtypes.astype(str).to_dict()
    n = len(df)
    m = df.shape[1]
    mem = df.memory_usage(deep=True).sum()

    summary = f"Rows={n:,}, Cols={m}, Memory={mem/1024/1024:.2f} MB"
    payload = {
        "shape": (n, m),
        "dtypes": dtypes,
        "columns": list(df.columns),
    }
    return ToolResult("profile_schema", summary, payload)


def missingness(df: pd.DataFrame, top_n: int = 20) -> ToolResult:
    miss_rate = df.isna().mean().sort_values(ascending=False)
    miss_cnt = df.isna().sum().sort_values(ascending=False)
    out = pd.DataFrame({"missing_rate": miss_rate, "missing_count": miss_cnt}).head(top_n)

    summary = f"Top missing columns: {', '.join(out.index.tolist()[:5]) if len(out)>0 else 'None'}"
    return ToolResult("missingness", summary, {"missing_table_top": out})


def duplicates(df: pd.DataFrame) -> ToolResult:
    dup_rows = int(df.duplicated().sum())
    summary = f"Duplicate rows: {dup_rows:,}"
    return ToolResult("duplicates", summary, {"duplicate_rows": dup_rows})


def numeric_summary(df: pd.DataFrame, num_cols: List[str]) -> ToolResult:
    if not num_cols:
        return ToolResult("numeric_summary", "No numeric columns.", {})
    desc = df[num_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    skew = df[num_cols].skew(numeric_only=True)
    kurt = df[num_cols].kurtosis(numeric_only=True)
    summary = f"Computed describe/skew/kurtosis for {len(num_cols)} numeric columns."
    return ToolResult(
        "numeric_summary",
        summary,
        {"describe": desc, "skew": skew.sort_values(ascending=False), "kurtosis": kurt.sort_values(ascending=False)},
    )


def categorical_summary(df: pd.DataFrame, cat_cols: List[str], top_n: int = 10) -> ToolResult:
    if not cat_cols:
        return ToolResult("categorical_summary", "No categorical columns.", {})
    nunique = df[cat_cols].nunique(dropna=False).sort_values(ascending=False)
    top_cols = nunique.head(min(top_n, len(nunique))).to_dict()
    summary = f"Computed nunique for {len(cat_cols)} categorical columns. Highest-cardinality: {next(iter(top_cols)) if top_cols else 'None'}"
    return ToolResult("categorical_summary", summary, {"nunique": nunique})


def correlation_matrix(df: pd.DataFrame, num_cols: List[str]) -> ToolResult:
    if len(num_cols) < 2:
        return ToolResult("correlation_matrix", "Not enough numeric columns for correlation.", {})
    corr = df[num_cols].corr(numeric_only=True)
    summary = "Computed Pearson correlation matrix for numeric columns."
    return ToolResult("correlation_matrix", summary, {"corr": corr})


def outlier_quantiles(df: pd.DataFrame, num_cols: List[str], qs=(0.01, 0.99)) -> ToolResult:
    if not num_cols:
        return ToolResult("outlier_quantiles", "No numeric columns.", {})
    q = df[num_cols].quantile(list(qs))
    summary = f"Computed quantiles {qs} for numeric columns (boundary view, not all outliers)."
    return ToolResult("outlier_quantiles", summary, {"quantiles": q})


def target_relationships(
    df: pd.DataFrame,
    target: str,
    num_cols: List[str],
    cat_cols: List[str],
    task: str = "auto",
    top_k: int = 10,
) -> ToolResult:
    """
    Returns quick feature-target relationships:
    - For numeric features: correlation or t-test style summaries (classification)
    - For categorical features: chi-square association (classification)
    - Mutual Information ranking (classification/regression)
    """
    if target not in df.columns:
        return ToolResult("target_relationships", f"Target '{target}' not in columns.", {})

    y = df[target]
    task_used = task

    # Determine task if auto
    if task == "auto":
        if y.dropna().nunique() <= 2:
            task_used = "classification"
        else:
            task_used = "regression"

    payload: Dict[str, Any] = {"task": task_used}

    # Numeric feature relationships
    if num_cols:
        xnum = df[num_cols].copy()

        # simple correlations
        if task_used == "regression":
            corr = xnum.apply(lambda s: s.corr(y), axis=0).sort_values(key=lambda s: s.abs(), ascending=False)
            payload["num_target_corr"] = corr.head(top_k)
        else:
            # classification: difference in means (Welch t-test) for numeric features if y is binary
            if y.dropna().nunique() == 2:
                cls_vals = sorted(y.dropna().unique())
                a = df[df[target] == cls_vals[0]]
                b = df[df[target] == cls_vals[1]]
                tests = []
                for col in num_cols:
                    aa = a[col].dropna()
                    bb = b[col].dropna()
                    if len(aa) > 2 and len(bb) > 2:
                        stat, p = stats.ttest_ind(aa, bb, equal_var=False)
                        tests.append((col, stat, p, float(aa.mean()), float(bb.mean())))
                tdf = pd.DataFrame(tests, columns=["feature", "t_stat", "p_value", "mean_class0", "mean_class1"])
                if not tdf.empty:
                    tdf = tdf.sort_values("p_value").head(top_k)
                payload["welch_ttest_top"] = tdf

    # Categorical feature relationships (chi-square for classification)
    if task_used == "classification" and cat_cols:
        chi = []
        for col in cat_cols:
            # build contingency table
            tab = pd.crosstab(df[col].fillna("NA"), y)
            if tab.shape[0] > 1 and tab.shape[1] > 1:
                chi2, p, dof, exp = stats.chi2_contingency(tab)
                chi.append((col, chi2, p, dof, int(tab.values.sum())))
        cdf = pd.DataFrame(chi, columns=["feature", "chi2", "p_value", "dof", "n"])
        if not cdf.empty:
            cdf = cdf.sort_values("p_value").head(top_k)
        payload["chi_square_top"] = cdf

    # Mutual Information ranking (simple, practical)
    # We'll one-hot encode categoricals and fill NaNs in numerics
    X = df.drop(columns=[target]).copy()
    X_num = X.select_dtypes(include="number").fillna(X.select_dtypes(include="number").median(numeric_only=True))
    X_cat = X.select_dtypes(exclude="number").fillna("NA")

    X_enc = pd.get_dummies(pd.concat([X_num, X_cat], axis=1), drop_first=False)
    y_clean = y.fillna(y.mode().iloc[0]) if task_used == "classification" else y.fillna(y.median())

    if task_used == "classification":
        mi = mutual_info_classif(X_enc, y_clean, random_state=42)
    else:
        mi = mutual_info_regression(X_enc, y_clean, random_state=42)

    mi_s = pd.Series(mi, index=X_enc.columns).sort_values(ascending=False).head(top_k)
    payload["mutual_info_top"] = mi_s

    summary = f"Computed target relationships for task={task_used} with MI ranking + basic stats."
    return ToolResult("target_relationships", summary, payload)


# -----------------------------
# Plotting tools (save plots to disk)
# -----------------------------
def plot_distributions(df: pd.DataFrame, num_cols: List[str], out_dir: str, max_plots: int = 6) -> ToolResult:
    _ensure_dir(out_dir)
    cols = num_cols[:max_plots]
    saved = []

    for col in cols:
        s = df[col].dropna()
        if s.empty:
            continue
        plt.figure()
        plt.hist(s, bins=30)
        plt.title(f"Distribution: {col}")
        path = os.path.join(out_dir, f"dist_{col}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        saved.append(path)

    summary = f"Saved {len(saved)} distribution plots to {out_dir}."
    return ToolResult("plot_distributions", summary, {"paths": saved})


def plot_boxplots(df: pd.DataFrame, num_cols: List[str], out_dir: str, max_plots: int = 6) -> ToolResult:
    _ensure_dir(out_dir)
    cols = num_cols[:max_plots]
    saved = []

    for col in cols:
        s = df[col].dropna()
        if s.empty:
            continue
        plt.figure()
        plt.boxplot(s, vert=True)
        plt.title(f"Boxplot: {col}")
        path = os.path.join(out_dir, f"box_{col}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        saved.append(path)

    summary = f"Saved {len(saved)} boxplots to {out_dir}."
    return ToolResult("plot_boxplots", summary, {"paths": saved})


def plot_target_relationships(
    df: pd.DataFrame,
    target: str,
    num_cols: List[str],
    out_dir: str,
    max_plots: int = 6,
) -> ToolResult:
    """
    For binary target: boxplots by class
    For regression: scatter plots
    """
    _ensure_dir(out_dir)
    saved = []
    if target not in df.columns:
        return ToolResult("plot_target_relationships", f"Target '{target}' not found.", {})

    y = df[target]
    task = "classification" if y.dropna().nunique() <= 2 else "regression"

    cols = num_cols[:max_plots]
    for col in cols:
        tmp = df[[col, target]].dropna()
        if tmp.empty:
            continue

        plt.figure()
        if task == "classification":
            classes = sorted(tmp[target].unique())
            data = [tmp[tmp[target] == c][col].values for c in classes]
            plt.boxplot(data, labels=[str(c) for c in classes])
            plt.title(f"{col} by {target}")
        else:
            plt.scatter(tmp[col].values, tmp[target].values, s=10)
            plt.title(f"{col} vs {target}")
            plt.xlabel(col)
            plt.ylabel(target)

        path = os.path.join(out_dir, f"target_{col}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        saved.append(path)

    summary = f"Saved {len(saved)} target relationship plots (task={task}) to {out_dir}."
    return ToolResult("plot_target_relationships", summary, {"paths": saved, "task": task})
