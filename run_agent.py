import argparse
import pandas as pd

from src.agent import AutonomousEDAAgent, AgentConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--target", default=None, help="Target column name (optional)")
    parser.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    parser.add_argument("--out", default="artifacts/report.json", help="Output report JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    cfg = AgentConfig(
        target=args.target,
        task=args.task,
        max_steps=12,
        max_plots=12,
        plots_dir="artifacts/plots",
        top_k=10,
    )

    agent = AutonomousEDAAgent(cfg)
    report = agent.run(df, dataset_name=args.csv)

    # Save report
    import os
    os.makedirs("artifacts", exist_ok=True)
    report.to_json(args.out)

    print("✅ Report saved to:", args.out)
    print("✅ Plots saved to:", cfg.plots_dir)
    print("\nTop Findings:")
    for f in report.findings[:5]:
        print("-", f)

    if report.anomalies:
        print("\nAnomalies:")
        for a in report.anomalies[:5]:
            print("-", a)

    if report.hypotheses:
        print("\nHypotheses:")
        for h in report.hypotheses[:5]:
            print("-", h)


if __name__ == "__main__":
    main()
