"""Mede se a atividade descrita em fala coloquial é identificada no grafo."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from licenciagraph.config import Settings
from licenciagraph.graph import GraphConnection
from licenciagraph.ingestion.llm import OpenAIClient
from licenciagraph.retrieval.retriever import build_retriever

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/evaluation/colloquial_questions.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/resolution.json"
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    llm = OpenAIClient.from_settings(settings)
    cases = json.loads(args.dataset.read_text())
    with GraphConnection(settings) as connection:
        retriever = build_retriever(connection.graph, llm, settings)
        repository, resolver = retriever.repository, retriever.resolver
        rows = []
        for case in cases:
            gold = re.compile(case["gold"], re.IGNORECASE)
            question = case["question"]
            embedding = llm.embed(
                model=settings.embedding_model,
                texts=[question],
                dimensions=settings.embedding_dimension,
            )[0]
            before = [
                n.name
                for n in [
                    *(
                        n
                        for n in repository.linked_entities(question)
                        if n.label == "Atividade"
                    ),
                    *repository.similar_activities(embedding),
                ]
            ]
            resolution = resolver.resolve(question)
            after = [n.name for n in resolution.activities]
            rows.append(
                {
                    "id": case["id"],
                    "question": question,
                    "gold": case["gold"],
                    "origem": case.get("origem", "autor"),
                    "baseline": before,
                    "baseline_hit": any(gold.search(n) for n in before),
                    "resolver": after,
                    "match": resolution.match,
                    "resolver_hit": any(gold.search(n) for n in after),
                    "terms": list(resolution.terms),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    total = len(rows)
    print(
        f"baseline {sum(r['baseline_hit'] for r in rows)}/{total}; "
        f"resolvedor {sum(r['resolver_hit'] for r in rows)}/{total}"
    )
    for row in rows:
        flag = ("B" if row["baseline_hit"] else "-") + (
            "R" if row["resolver_hit"] else "-"
        )
        print(flag, row["id"], row["match"], row["resolver"][:2])
    print("tokens:", llm.usage)


if __name__ == "__main__":
    main()
