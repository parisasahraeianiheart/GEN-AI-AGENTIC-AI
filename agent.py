# src/agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

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
from src.tools_drift import drift_psi, drift_ks
from src.report import EDAReport
from src.llm_planner import llm_plan


@dataclass
class AgentConfig:
    target: Optional[str] = None
    task: str = "auto"         # "auto" | "classification" | "regression"
    max_steps: int = 12
    max_plots: int = 12
    plots_dir: str = "artifacts/plots"
    top_k: int = 10            # top-k features in MI / tests
    enable_llm_planner: bool = True
    enable_drift: bool = True
    drift_alpha: float = 0.05
    drift_bins: int = 10


class AutonomousEDAAgent:
    """
    Agentic EDA:
      - Uses an LLM planner to propose a tool plan (JSON)
      - Executes tool calls under budgets (max_steps, max_plots)
      - Optionally runs drift/stability checks (PSI + KS) if baseline/current provided
      - Produces a structured report with artifacts (tables, stats, plot paths)
    """

    def __init__(self, config: AgentConfig):
        self.cfg = config
        self.state: Dict[str, Any] = {
            "steps_used": 0,
            "plots_used": 0,
            "tool_log": [],
            "plan_used": [],
        }

        # Optional drift references (set via set_drift_data or by assigning attributes)
        self.baseline_df: Optional[pd.DataFrame] = None
        self.current_df: Optional[pd.DataFrame] = None

    def set_drift_data(self, baseline_df: Optional[pd.DataFrame], current_df: Optional[pd.DataFrame]) -> None:
        self.baseline_df = baseline_df
        self.current_df = current_df

    def _log_step(self, tool_name: str, summary: str) -> None:
        self.state["steps_used"] += 1
        self.state["tool_log"].append({"tool": tool_name, "summary": summary})

    def _can_plot(self) -> bool:
        return self.state["plots_used"] < self.cfg.max_plots

    def _add_plots_used(self, n: int) -> None:
        self.state["plots_used"] = min(self.cfg.max_plots, self.state["plots_used"] + n)

    def run(self, df: pd.DataFrame, dataset_name: str = "data") -> EDAReport:
        rep = EDAReport(
            title="Autonomous EDA Agent Report",
            dataset=dataset_name,
        )

        # -------------------------
        # Build small "brief" for planner (no raw data)
        # -------------------------
        schema_res = profile_schema(df)
        schema_payload = schema_res.payload

        top_missing = df.isna().mean().sort_values(ascending=False).head(8).index.tolist()
        cat_cols_all = df.select_dtypes(exclude="number").columns.tolist()
        top_card = (
            df[cat_cols_all].nunique(dropna=False).sort_values(ascending=False).head(5).index.tolist()
            if cat_cols_all else []
        )

        quick_stats = {
            "top_missing": top_missing,
            "top_cardinality": top_card,
            "target": self.cfg.target,
            "drift_available": bool(self.baseline_df is not None and self.current_df is not None),
        }

        # -------------------------
        # Create plan: LLM planner (preferred) or fallback deterministic plan
        # -------------------------
        if self.cfg.enable_llm_planner:
            try:
                plan = llm_plan(
                    schema=schema_payload,
                    quick_stats=quick_stats,
                    target=self.cfg.target,
                    task=self.cfg.task,
                    max_steps=self.cfg.max_steps,
                )
            except Exception as e:
                # Fallback plan if planner fails
                rep.add_anomaly(f"LLM planner failed; falling back to deterministic plan. Error: {e}")
                plan = []
        else:
            plan = []

        if not plan:
            # Deterministic fallback plan (safe & strong)
            plan = [
                {"tool": "profile_schema", "args": {}},
                {"tool": "missingness", "args": {"top_n": 20}},
                {"tool": "duplicates", "args": {}},
                {"tool": "numeric_summary", "args": {}},
                {"tool": "outlier_quantiles", "args": {}},
                {"tool": "correlation_matrix", "args": {}},
                {"tool": "categorical_summary", "args": {}},
            ]
            if self.cfg.target:
                plan.append({"tool": "target_relationships", "args": {}})
                plan.append({"tool": "plot_target_relationships", "args": {}})
            plan += [
                {"tool": "plot_distributions", "args": {}},
                {"tool": "plot_boxplots", "args": {}},
            ]
            if self.cfg.enable_drift and self.baseline_df is not None and self.current_df is not None:
                plan += [
                    {"tool": "drift_psi", "args": {"top_k": 10, "n_bins": self.cfg.drift_bins}},
                    {"tool": "drift_ks", "args": {"alpha": self.cfg.drift_alpha, "top_k": 10}},
                ]

        # Keep within max steps
        plan = plan[: self.cfg.max_steps]
        self.state["plan_used"] = plan

        # -------------------------
        # Tool router (dispatcher)
        # -------------------------
        def _num_cols_no_target() -> List[str]:
            cols = df.select_dtypes(include="number").columns.tolist()
            if self.cfg.target and self.cfg.target in cols:
                cols = [c for c in cols if c != self.cfg.target]
            return cols

        def _cat_cols_no_target() -> List[str]:
            cols = df.select_dtypes(exclude="number").columns.tolist()
            if self.cfg.target and self.cfg.target in cols:
                cols = [c for c in cols if c != self.cfg.target]
            return cols

        TOOL_ROUTER = {
            "profile_schema": lambda **kwargs: profile_schema(df),
            "missingness": lambda top_n=20, **kwargs: missingness(df, top_n=top_n),
            "duplicates": lambda **kwargs: duplicates(df),
            "numeric_summary": lambda **kwargs: numeric_summary(df, _num_cols_no_target()),
            "categorical_summary": lambda top_n=10, **kwargs: categorical_summary(df, _cat_cols_no_target(), top_n=top_n),
            "correlation_matrix": lambda **kwargs: correlation_matrix(df, _num_cols_no_target()),
            "outlier_quantiles": lambda qs=(0.01, 0.99), **kwargs: outlier_quantiles(df, _num_cols_no_target(), qs=qs),
            "target_relationships": lambda **kwargs: target_relationships(
                df=df,
                target=self.cfg.target,
                num_cols=_num_cols_no_target(),
                cat_cols=_cat_cols_no_target(),
                task=self.cfg.task,
                top_k=self.cfg.top_k,
            ),
            "plot_distributions": lambda max_plots=6, **kwargs: plot_distributions(
                df, df.select_dtypes(include="number").columns.tolist(), self.cfg.plots_dir, max_plots=max_plots
            ),
            "plot_boxplots": lambda max_plots=6, **kwargs: plot_boxplots(
                df, df.select_dtypes(include="number").columns.tolist(), self.cfg.plots_dir, max_plots=max_plots
            ),
            "plot_target_relationships": lambda max_plots=6, **kwargs: plot_target_relationships(
                df, self.cfg.target, _num_cols_no_target(), self.cfg.plots_dir, max_plots=max_plots
            ),
            # Drift tools (require baseline/current)
            "drift_psi": lambda top_k=10, n_bins=10, **kwargs: drift_psi(
                df_baseline=self.baseline_df,
                df_current=self.current_df,
                top_k=top_k,
                n_bins=n_bins,
            ),
            "drift_ks": lambda alpha=0.05, top_k=10, **kwargs: drift_ks(
                df_baseline=self.baseline_df,
                df_current=self.current_df,
                alpha=alpha,
                top_k=top_k,
            ),
        }

        # -------------------------
        # Execute plan with budgets
        # -------------------------
        for step in plan:
            if self.state["steps_used"] >= self.cfg.max_steps:
                rep.add_anomaly(f"Stopped early: reached max_steps={self.cfg.max_steps}.")
                break

            tool = step.get("tool")
            args = step.get("args", {}) or {}

            if tool not in TOOL_ROUTER:
                rep.add_anomaly(f"Planner requested unknown tool '{tool}'. Skipped.")
                continue

            # Plot budget checks
            if tool in ("plot_distributions", "plot_boxplots", "plot_target_relationships") and not self._can_plot():
                continue
            if tool == "plot_target_relationships" and not self.cfg.target:
                continue

            # Drift availability checks
            if tool in ("drift_psi", "drift_ks"):
                if not self.cfg.enable_drift:
                    continue
                if self.baseline_df is None or self.current_df is None:
                    rep.add_anomaly("Drift requested but baseline/current data not provided. Skipped.")
                    continue

            # Execute tool
            try:
                result: ToolResult = TOOL_ROUTER[tool](**args)
            except TypeError:
                # Planner may pass args that a tool doesn't accept; retry without args
                result = TOOL_ROUTER[tool]()
            except Exception as e:
                rep.add_anomaly(f"Tool '{tool}' failed: {e}")
                continue

            self._log_step(tool, result.summary)
            rep.add_finding(f"[{tool}] {result.summary}")

            # Store artifacts
            rep.artifacts.setdefault("tool_outputs", {})
            rep.artifacts["tool_outputs"][tool] = result.payload

            # Track plot usage
            if tool in ("plot_distributions", "plot_boxplots", "plot_target_relationships"):
                paths = (result.payload or {}).get("paths", [])
                rep.artifacts.setdefault("plots", [])
                rep.artifacts["plots"].extend(paths)
                self._add_plots_used(len(paths))

            # Add “smart” interpretations for a few tools (helps interview polish)
            if tool == "missingness":
                miss_tbl = (result.payload or {}).get("missing_table_top")
                if miss_tbl is not None and len(miss_tbl) > 0:
                    high = miss_tbl[miss_tbl["missing_rate"] >= 0.30]
                    if len(high) > 0:
                        rep.add_anomaly(
                            f"{len(high)} columns have >=30% missingness: {', '.join(high.index.tolist()[:6])}"
                        )
                        rep.add_next_step("Decide missing-value strategy: drop, impute, or model missingness explicitly.")

            if tool == "numeric_summary":
                skew = (result.payload or {}).get("skew")
                kurt = (result.payload or {}).get("kurtosis")
                if skew is not None and len(skew) > 0:
                    bad = skew[skew.abs() > 2].head(5)
                    if len(bad) > 0:
                        rep.add_anomaly(f"Heavy skew detected (|skew|>2) in: {', '.join(bad.index.tolist())}")
                        rep.add_next_step("Consider log/PowerTransformer for heavily skewed variables.")
                if kurt is not None and len(kurt) > 0:
                    bad = kurt[kurt > 10].head(5)
                    if len(bad) > 0:
                        rep.add_anomaly(f"Heavy tails suggested (kurtosis>10) in: {', '.join(bad.index.tolist())}")
                        rep.add_next_step("Consider RobustScaler or outlier handling for heavy-tailed variables.")

            if tool == "correlation_matrix":
                corr = (result.payload or {}).get("corr")
                if corr is not None and hasattr(corr, "columns"):
                    pairs = []
                    cols = corr.columns.tolist()
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            val = corr.iloc[i, j]
                            if abs(val) >= 0.85:
                                pairs.append((cols[i], cols[j], float(val)))
                    if pairs:
                        rep.add_anomaly(
                            "High correlation (|r|>=0.85) pairs: "
                            + ", ".join([f"{a}-{b}({v:.2f})" for a, b, v in pairs[:5]])
                        )
                        rep.add_next_step("Consider dropping correlated features or using dimension reduction.")

            if tool == "categorical_summary":
                nunique = (result.payload or {}).get("nunique")
                if nunique is not None and len(nunique) > 0:
                    high = nunique[nunique > 50].head(5)
                    if len(high) > 0:
                        rep.add_anomaly(f"High-cardinality categoricals (>50 unique): {', '.join(high.index.tolist())}")
                        rep.add_next_step("Use frequency encoding or target encoding for high-cardinality categoricals.")

            if tool == "target_relationships":
                payload = result.payload or {}
                task_used = payload.get("task", "auto")
                rep.add_hypothesis(f"Task inferred: {task_used} based on target distribution.")
                mi = payload.get("mutual_info_top")
                if mi is not None and hasattr(mi, "index") and len(mi) > 0:
                    rep.add_hypothesis(f"Top MI features (potential signal): {', '.join(list(mi.index[:5]))}")
                rep.add_next_step("Start with a simple baseline model; then compare with stronger non-linear models if needed.")

            if tool == "drift_psi":
                top = (result.payload or {}).get("top")
                if top is not None and len(top) > 0:
                    # PSI rule-of-thumb
                    max_feat = top.iloc[0]["feature"]
                    max_psi = float(top.iloc[0]["psi"])
                    if max_psi > 0.25:
                        rep.add_anomaly(f"Major drift detected by PSI: {max_feat} PSI={max_psi:.3f} (>0.25).")
                        rep.add_next_step("Investigate drift source; consider retraining or feature review.")
                    elif max_psi > 0.10:
                        rep.add_anomaly(f"Moderate drift detected by PSI: {max_feat} PSI={max_psi:.3f} (0.10–0.25).")

            if tool == "drift_ks":
                ks_tbl = (result.payload or {}).get("ks_table")
                alpha = (result.payload or {}).get("alpha", self.cfg.drift_alpha)
                if ks_tbl is not None and "significant" in ks_tbl.columns:
                    sig = int(ks_tbl["significant"].sum())
                    if sig > 0:
                        rep.add_anomaly(f"KS drift: {sig} numeric features shifted significantly (p<{alpha}).")
                        rep.add_next_step("If shifts impact model performance, recalibrate thresholds or retrain.")

        # -------------------------
        # Wrap up: attach agent state & defaults
        # -------------------------
        rep.artifacts["agent_state"] = {
            "steps_used": self.state["steps_used"],
            "plots_used": self.state["plots_used"],
            "tool_log": self.state["tool_log"],
            "plan_used": self.state["plan_used"],
        }

        if not rep.next_steps:
            rep.add_next_step("Confirm objective (classification vs regression) and define target precisely.")
            rep.add_next_step("Choose missing-value strategy and encoding/scaling approach.")
            rep.add_next_step("Train baseline model and evaluate with appropriate metrics.")

        return rep
