# run_agent.py
import argparse
import os
import pandas as pd

from src.agent import AutonomousEDAAgent, AgentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Autonomous EDA Agent (LLM planner + optional drift checks).")
    parser.add_argument("--csv", required=True, help="Path to the main/current CSV file (the dataset to analyze).")
    parser.add_argument("--target", default=None, help="Target column name (optional).")
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Task type (auto infers from target cardinality).",
    )

    # Outputs
    parser.add_argument("--out", default="artifacts/report.json", help="Output report JSON path.")
    parser.add_argument("--plots_dir", default="artifacts/plots", help="Directory to save plots.")

    # Budgets
    parser.add_argument("--max_steps", type=int, default=12, help="Maximum number of tool steps to run.")
    parser.add_argument("--max_plots", type=int, default=12, help="Maximum number of plots to save.")

    # LLM planner toggle
    parser.add_argument(
        "--no_llm_planner",
        action="store_true",
        help="Disable the LLM planner and use a deterministic plan instead.",
    )

    # Drift / stability checks
    parser.add_argument("--baseline", default=None, help="Path to baseline CSV for drift checks (optional).")
    parser.add_argument("--current", default=None, help="Path to current CSV for drift checks (optional).")
    parser.add_argument("--drift_alpha", type=float, default=0.05, help="Significance level for KS drift test.")
    parser.add_argument("--drift_bins", type=int, default=10, help="Number of bins for PSI (numeric features).")
    parser.add_argument(
        "--no_drift",
        action="store_true",
        help="Disable drift checks even if baseline/current are provided.",
    )

    args = parser.parse_args()

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(args.csv)

    baseline_df = pd.read_csv(args.baseline) if args.baseline else None
    # If args.current is provided, we use that for drift comparison; otherwise compare baseline vs df
    current_df = pd.read_csv(args.current) if args.current else (df if baseline_df is not None else None)

    # -------------------------
    # Configure agent
    # -------------------------
    cfg = AgentConfig(
        target=args.target,
        task=args.task,
        max_steps=args.max_steps,
        max_plots=args.max_plots,
        plots_dir=args.plots_dir,
        top_k=10,
        enable_llm_planner=(not args.no_llm_planner),
        enable_drift=(not args.no_drift),
        drift_alpha=args.drift_alpha,
        drift_bins=args.drift_bins,
    )

    agent = AutonomousEDAAgent(cfg)
    agent.set_drift_data(baseline_df=baseline_df, current_df=current_df)

    # -------------------------
    # Run
    # -------------------------
    report = agent.run(df, dataset_name=args.csv)

    # -------------------------
    # Save outputs
    # -------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    report.to_json(args.out)

    # -------------------------
    # Print a friendly summary
    # -------------------------
    print("\n✅ Autonomous EDA Agent finished")
    print("📄 Report saved to:", args.out)
    print("🖼️  Plots directory:", args.plots_dir)

    state = report.artifacts.get("agent_state", {})
    print(f"🧠 Steps used: {state.get('steps_used', 'NA')} / {args.max_steps}")
    print(f"🖼️  Plots used: {state.get('plots_used', 'NA')} / {args.max_plots}")

    print("\nTop Findings:")
    for f in report.findings[:8]:
        print("-", f)

    if report.anomalies:
        print("\nAnomalies / Flags:")
        for a in report.anomalies[:8]:
            print("-", a)

    if report.hypotheses:
        print("\nHypotheses:")
        for h in report.hypotheses[:6]:
            print("-", h)

    if report.next_steps:
        print("\nNext Steps:")
        for ns in report.next_steps[:8]:
            print("-", ns)

    # Quick note about drift inputs
    if args.no_drift:
        print("\nℹ️ Drift checks disabled.")
    elif args.baseline and (args.current or True):
        print("\nℹ️ Drift checks: baseline vs current comparison enabled.")
        if args.current is None:
            print("   (Using --csv as current dataset.)")
    else:
        print("\nℹ️ Drift checks: not run (no --baseline provided).")


if __name__ == "__main__":
    main()
