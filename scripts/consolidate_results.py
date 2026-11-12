"""Consolida os resultados das execuções em artifacts/ (tabela, Wilson e custo)."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RUNS = [
    ("Desenvolvimento, versão inicial (híbrido)", "run-dev-hybrid-v1", "hybrid"),
    ("Desenvolvimento, baseline vetorial", "run-dev-vector", "vector"),
    ("Retido, híbrido", "run-heldout-hybrid", "hybrid"),
    ("Retido, baseline vetorial", "run-heldout-vector", "vector"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    lines = [
        "| Execução | Positivas | IC 95% | Afirmações contraditas | Negativas | Auto | Custo |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    total = 0.0
    for label, run, mode in RUNS:
        directory = ART / run
        judge = load(directory / f"{mode}-judge.json")
        neg = load(directory / f"{mode}-judge-negatives.json")
        report = load(directory / f"{mode}-report.json")
        usage = load(directory / "usage.json")
        if not judge or not judge["summary"].get("total"):
            continue
        s = judge["summary"]
        low, high = wilson(s["correct"], s["total"])
        cost = (usage or {}).get("usd", 0.0)
        total += cost
        negative = (
            f"{neg['summary']['correct']}/{neg['summary']['total']}" if neg else "-"
        )
        auto = f"{report['passed']}/{report['total']}" if report else "-"
        lines.append(
            f"| {label} | {s['correct']}/{s['total']} | {low:.0%} a {high:.0%} | "
            f"{s['with_false_claims']} | {negative} | {auto} | US$ {cost:.2f} |"
        )
    lines.append(
        f"\nCusto total das execuções (geração e juiz, sem a rejulgagem): US$ {total:.2f}"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
