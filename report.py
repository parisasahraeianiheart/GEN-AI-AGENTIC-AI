from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List
import json


@dataclass
class EDAReport:
    title: str
    dataset: str
    findings: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def add_finding(self, s: str) -> None:
        self.findings.append(s)

    def add_anomaly(self, s: str) -> None:
        self.anomalies.append(s)

    def add_hypothesis(self, s: str) -> None:
        self.hypotheses.append(s)

    def add_next_step(self, s: str) -> None:
        self.next_steps.append(s)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "dataset": self.dataset,
            "findings": self.findings,
            "anomalies": self.anomalies,
            "hypotheses": self.hypotheses,
            "next_steps": self.next_steps,
            "artifacts": self.artifacts,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
