from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

client = OpenAI()


ALLOWED_TOOLS = [
    "profile_schema",
    "missingness",
    "duplicates",
    "numeric_summary",
    "categorical_summary",
    "correlation_matrix",
    "outlier_quantiles",
    "target_relationships",
    "plot_distributions",
    "plot_boxplots",
    "plot_target_relationships",
    "drift_psi",
    "drift_ks",
]


def build_dataset_brief(schema: Dict[str, Any], quick_stats: Dict[str, Any]) -> str:
    """
    Keep this short. We do NOT want to paste full data into the prompt.
    """
    cols = schema.get("columns", [])
    dtypes = schema.get("dtypes", {})
    shape = schema.get("shape", None)

    top_missing = quick_stats.get("top_missing", [])
    top_card = quick_stats.get("top_cardinality", [])
    maybe_target = quick_stats.get("target", None)

    return (
        f"DATASET\n"
        f"- shape: {shape}\n"
        f"- columns: {cols}\n"
        f"- dtypes: {dtypes}\n"
        f"- target (if any): {maybe_target}\n"
        f"- top_missing_columns: {top_missing}\n"
        f"- top_cardinality_categoricals: {top_card}\n"
    )


def llm_plan(
    schema: Dict[str, Any],
    quick_stats: Dict[str, Any],
    target: Optional[str],
    task: str,
    max_steps: int,
) -> List[Dict[str, Any]]:
    """
    Returns a list of steps:
      [{"tool": "...", "args": {...}}, ...]
    """

    brief = build_dataset_brief(schema, quick_stats)

    instructions = (
        "You are an EDA planning assistant.\n"
        "Your job: produce a concise tool-execution plan for an autonomous EDA agent.\n"
        "Rules:\n"
        f"1) Only use tools from this allowlist: {ALLOWED_TOOLS}\n"
        "2) Keep the plan within max_steps.\n"
        "3) Prefer high-signal steps first (schema, missingness, numeric/cat summaries).\n"
        "4) If target exists, include target_relationships and target plots.\n"
        "5) If drift inputs are available (baseline/current), include drift_psi and drift_ks.\n"
        "6) Output MUST be valid JSON array only.\n"
        "Each item: {\"tool\": \"tool_name\", \"args\": { ... }}\n"
    )

    input_text = (
        f"{brief}\n\n"
        f"Constraints:\n"
        f"- max_steps: {max_steps}\n"
        f"- target: {target}\n"
        f"- task: {task}\n\n"
        "Return JSON plan now."
    )

    # Using Responses API (recommended for new projects)
    resp = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=input_text,
    )

    text = resp.output_text.strip()

    # Parse JSON safely
    plan = json.loads(text)

    # Basic validation
    if not isinstance(plan, list):
        raise ValueError("Planner did not return a JSON list.")

    cleaned = []
    for step in plan[:max_steps]:
        tool = step.get("tool")
        args = step.get("args", {})
        if tool not in ALLOWED_TOOLS:
            continue
        cleaned.append({"tool": tool, "args": args})

    return cleaned
