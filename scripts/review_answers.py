"""Exporta respostas, fontes citadas e caminhos para revisão factual documentada."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    lines = [
        "# Caderno de revisão das respostas",
        "",
        "Extraído das respostas reais. As métricas automáticas não substituem "
        "a revisão de adequação à pergunta.",
        "",
    ]
    for mode in ("hybrid", "vector"):
        path = directory / f"{mode}-answers.jsonl"
        if not path.exists():
            continue
        lines += [f"## {mode}", ""]
        for record in path.read_text().splitlines():
            answer = json.loads(record)["result"]
            retrieval = answer["retrieval"]
            lines += [
                f"### {retrieval['question']}",
                "",
                answer["answer"],
                "",
                f"Abstenção declarada: {answer.get('abstained', False)}",
                "",
            ]
            sources = {f"F{i}": hit for i, hit in enumerate(retrieval["hits"], 1)}
            names = {node["id"]: node["name"] for node in retrieval["nodes"]}
            lines += ["Fontes citadas:", ""]
            for ref in answer["citations"]:
                hit = sources[ref]
                lines += [f"- [{ref}] {hit['filename']}: {hit['text']}"]
            lines += ["", "Relações declaradas como apoio:", ""]
            for i, edge in enumerate(retrieval["edges"], 1):
                if f"G{i}" in answer.get("supporting_edges", []):
                    lines.append(
                        f"- G{i}: {names.get(edge['source'], edge['source'])} "
                        f"— {edge['relationship_type']} → "
                        f"{names.get(edge['target'], edge['target'])}; "
                        f"trecho {edge['chunk_id']}"
                    )
            lines += [
                "",
                "Ressalvas:",
                "",
                *["- " + item for item in answer["caveats"]],
                "",
            ]
    (directory / "review-notebook.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
