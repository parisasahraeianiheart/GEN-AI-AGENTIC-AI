from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import time
import pandas as pd

from src.tools import (
    ToolResult,
    profile_schema,
    missingness,
    duplicates,
    numeric_summary,
    categorical_summary,
    correlation_matrix,
    outlier_quantiles,
    target_relationships,
    plot_distributions,
    plot_boxplots,
    plot_target_relationships,
)
from src.report import EDAReport


@dataclass
class AgentConfig:
    target: Optional[str] = None
    task: str = "auto"  # "auto" | "classification" | "regression"
    max_steps: int = 10
    max_plots: int = 12
    plots_dir: str = "artifacts/plots"
    top_k: int = 10


class AutonomousEDAAgent:
    """
    A simple, deterministic "agentic" EDA:
    - Plans a sequence of tool calls
    - Executes them with a budget (steps + plots)
    - Produces a structured report
    """

    def __init__(self, config: AgentConfig):
        self.cfg = config
        self.state: Dict[str, Any] = {
            "steps_used": 0,
            "plots_used": 0,
            "tool_log": [],
        }

    def _step(self, tool_name: str, result: ToolResult):
        self.state["steps_used"] += 1
        self.state["tool_log"].append({"tool": tool_name, "summary": result.summary})

    def run(self, df: pd.DataFrame, dataset_name: str = "data") -> EDAReport:
        rep = EDAReport(
            title="Autonomous EDA Agent Report",
            dataset=dataset_name,
        )

        # Identify columns
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        # If target is provided, remove from feature lists
        if self.cfg.target and self.cfg.target in num_cols:
            num_cols = [c for c in num_cols if c != self.cfg.target]
        if self.cfg.target and self.cfg.target in cat_cols:
            cat_cols = [c for c in cat_cols if c != self.cfg.target]

        # -------------------------
        # PLAN (fixed but agent-like)
        # -------------------------
        plan = ["profile_schema", "missingness", "duplicates"]

        if num_cols:
            plan += ["numeric_summary", "outlier_quantiles"]
            if len(num_cols) >= 2:
                plan += ["correlation_matrix"]

        if cat_cols:
            plan += ["categorical_summary"]

        if self.cfg.target:
            plan += ["target_relationships"]

        # Add plot tasks at the end (budgeted)
        plan += ["plot_distributions", "plot_boxplots"]
        if self.cfg.target:
            plan += ["plot_target_relationships"]

        # -------------------------
        # EXECUTE with budget
        # -------------------------
        for task_name in plan:
            if self.state["steps_used"] >= self.cfg.max_steps:
                rep.add_anomaly(f"Stopped early: reached max_steps={self.cfg.max_steps}.")
                break

            if task_name == "profile_schema":
                r = profile_schema(df); self._step(task_name, r)
                rep.add_finding(r.summary)
                rep.artifacts["schema"] = r.payload

            elif task_name == "missingness":
                r = missingness(df); self._step(task_name, r)
                rep.add_finding(r.summary)
                rep.artifacts["missingness_top"] = r.payload.get("missing_table_top")

                # Flag high missing columns
                miss_tbl = r.payload.get("missing_table_top")
                if miss_tbl is not None and len(miss_tbl) > 0:
                    high = miss_tbl[miss_tbl["missing_rate"] >= 0.30]
                    if len(high) > 0:
                        rep.add_anomaly(f"{len(high)} columns have >=30% missingness: {', '.join(high.index.tolist()[:6])}")

            elif task_name == "duplicates":
                r = duplicates(df); self._step(task_name, r)
                rep.add_finding(r.summary)
                if r.payload.get("duplicate_rows", 0) > 0:
                    rep.add_anomaly("Dataset contains duplicate rows; consider deduplication rules.")

            elif task_name == "numeric_summary":
                r = numeric_summary(df, num_cols); self._step(task_name, r)
                rep.add_finding(r.summary)
                rep.artifacts["numeric_describe"] = r.payload.get("describe")
                rep.artifacts["skew_top"] = r.payload.get("skew")
                rep.artifacts["kurtosis_top"] = r.payload.get("kurtosis")

                # Flag heavy skew / heavy tails
                skew = r.payload.get("skew")
                kurt = r.payload.get("kurtosis")
                if skew is not None and len(skew) > 0:
                    bad = skew[skew.abs() > 2].head(5)
                    if len(bad) > 0:
                        rep.add_anomaly(f"Heavy skew detected (|skew|>2) in: {', '.join(bad.index.tolist())}")
                        rep.add_next_step("Consider log/Box-Cox/Yeo-Johnson transforms for heavily skewed variables.")
                if kurt is not None and len(kurt) > 0:
                    bad = kurt[kurt > 10].head(5)
                    if len(bad) > 0:
                        rep.add_anomaly(f"Heavy tails / outliers suggested (kurtosis>10) in: {', '.join(bad.index.tolist())}")
                        rep.add_next_step("Consider RobustScaler, winsorization, or outlier handling for heavy-tailed variables.")

            elif task_name == "outlier_quantiles":
                r = outlier_quantiles(df, num_cols); self._step(task_name, r)
                rep.add_finding(r.summary)
                rep.artifacts["outlier_quantiles"] = r.payload.get("quantiles")

            elif task_name == "correlation_matrix":
                r = correlation_matrix(df, num_cols); self._step(task_name, r)
                rep.add_finding(r.summary)
                corr = r.payload.get("corr")
                rep.artifacts["corr"] = corr

                # Flag highly correlated pairs
                if corr is not None:
                    # quick scan for large |corr| off-diagonal
                    pairs = []
                    cols = corr.columns.tolist()
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            val = corr.iloc[i, j]
                            if abs(val) >= 0.85:
                                pairs.append((cols[i], cols[j], float(val)))
                    if pairs:
                        rep.add_anomaly(f"High correlation (|r|>=0.85) pairs found: {', '.join([f'{a}-{b}({v:.2f})' for a,b,v in pairs[:5]])}")
                        rep.add_next_step("Consider dropping one of correlated features or applying dimension reduction.")

            elif task_name == "categorical_summary":
                r = categorical_summary(df, cat_cols); self._step(task_name, r)
                rep.add_finding(r.summary)
                nunique = r.payload.get("nunique")
                rep.artifacts["categorical_nunique"] = nunique

                # Flag high-cardinality
                if nunique is not None and len(nunique) > 0:
                    high = nunique[nunique > 50].head(5)
                    if len(high) > 0:
                        rep.add_anomaly(f"High-cardinality categorical features (>50 unique): {', '.join(high.index.tolist())}")
                        rep.add_next_step("Consider frequency encoding or target encoding for high-cardinality categoricals.")

            elif task_name == "target_relationships":
                r = target_relationships(
                    df=df,
                    target=self.cfg.target,
                    num_cols=[c for c in df.select_dtypes(include="number").columns if c != self.cfg.target],
                    cat_cols=[c for c in df.select_dtypes(exclude="number").columns if c != self.cfg.target],
                    task=self.cfg.task,
                    top_k=self.cfg.top_k,
                )
                self._step(task_name, r)
                rep.add_finding(r.summary)
                rep.artifacts["target_relationships"] = r.payload

                task_used = r.payload.get("task", "auto")
                rep.add_hypothesis(f"Task likely: {task_used} based on target distribution.")

                mi = r.payload.get("mutual_info_top")
                if mi is not None and len(mi) > 0:
                    rep.add_hypothesis(f"Top MI features (potentially predictive): {', '.join(list(mi.index[:5]))}")

                rep.add_next_step("Use cross-validation and evaluate baseline models; start with simple models before complex ones.")

            elif task_name == "plot_distributions":
                if self.state["plots_used"] >= self.cfg.max_plots:
                    continue
                r = plot_distributions(df, df.select_dtypes(include="number").columns.tolist(), self.cfg.plots_dir, max_plots=6)
                self._step(task_name, r)
                paths = r.payload.get("paths", [])
                self.state["plots_used"] += len(paths)
                rep.artifacts.setdefault("plots", []).extend(paths)

            elif task_name == "plot_boxplots":
                if self.state["plots_used"] >= self.cfg.max_plots:
                    continue
                r = plot_boxplots(df, df.select_dtypes(include="number").columns.tolist(), self.cfg.plots_dir, max_plots=6)
                self._step(task_name, r)
                paths = r.payload.get("paths", [])
                self.state["plots_used"] += len(paths)
                rep.artifacts.setdefault("plots", []).extend(paths)

            elif task_name == "plot_target_relationships":
                if not self.cfg.target or self.state["plots_used"] >= self.cfg.max_plots:
                    continue
                num_all = df.select_dtypes(include="number").columns.tolist()
                num_all = [c for c in num_all if c != self.cfg.target]
                r = plot_target_relationships(df, self.cfg.target, num_all, self.cfg.plots_dir, max_plots=6)
                self._step(task_name, r)
                paths = r.payload.get("paths", [])
                self.state["plots_used"] += len(paths)
                rep.artifacts.setdefault("plots", []).extend(paths)

        # Final summary
        rep.artifacts["agent_state"] = {
            "steps_used": self.state["steps_used"],
            "plots_used": self.state["plots_used"],
            "tool_log": self.state["tool_log"],
        }

        # Add final “next steps” if empty
        if not rep.next_steps:
            rep.add_next_step("Define target variable and objective (classification vs regression).")
            rep.add_next_step("Decide missing value strategy and encode categoricals appropriately.")
            rep.add_next_step("Train a baseline model and evaluate with appropriate metrics.")

        return rep
