"""Compara relações extraídas com o gabarito independente do catálogo autoral."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_catalog import COURSES

from licenciagraph.config import Settings
from licenciagraph.graph import GraphConnection
from licenciagraph.ingestion.canonicalization import normalize_text


def fact(filename, source, relation, target):
    return (filename, normalize_text(source), relation, normalize_text(target))


def audit(graph):
    expected = set()
    for code, name, area, skill, level, profile, _, prerequisites, trail in COURSES:
        filename = f"{code}.txt"
        expected.update(
            [
                fact(filename, name, "ABORDA", area),
                fact(filename, name, "DESENVOLVE", skill),
                fact(filename, name, "CLASSIFICADO_EM", level),
                fact(filename, name, "INDICADO_PARA", profile),
                fact(filename, trail, "INCLUI", name),
            ]
        )
        expected.update(fact(filename, name, "PRE_REQUISITO", p) for p in prerequisites)
    rows = graph.ro_query("""
        MATCH (d:Documento)-[:CONTEM]->(t:Trecho)
        MATCH (a)-[r]->(b)
        WHERE d.ingestao_completa = true AND r.trecho_id = t.id
        RETURN d.arquivo, a.nome, type(r), b.nome
    """).result_set
    actual = {fact(*row) for row in rows}
    matched = actual & expected
    return {
        "expected_facts": len(expected),
        "actual_facts": len(actual),
        "matched_facts": len(matched),
        "precision": len(matched) / len(actual) if actual else 0,
        "recall": len(matched) / len(expected),
        "missing": sorted(expected - actual),
        "extra": sorted(actual - expected),
        "scope": (
            "Relações explícitas normalizadas por documento; "
            "não avalia descrições livres nem qualidade de respostas"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with GraphConnection(Settings.from_env()) as connection:
        report = audit(connection.graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
