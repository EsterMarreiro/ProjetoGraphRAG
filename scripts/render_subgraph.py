"""Figura vetorial do subgrafo declarado como apoio em uma resposta registrada."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
ROW_ORDER = [
    "Atividade",
    "Subgrupo",
    "Grupo",
    "Checklist",
    "Faixa",
    "DocumentoExigido",
    "Porte",
    "LetraCusto",
    "Modalidade",
    "Orgao",
    "Estudo",
    "Norma",
    "Sistema",
]
SHADES = ["#d9d9d9", "#ececec", "#f5f5f5"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--question-contains", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/artigo")
    args = parser.parse_args()
    rows = [
        json.loads(line)["result"]
        for line in (args.run / f"{args.mode}-answers.jsonl").read_text().splitlines()
    ]
    result = next(
        r for r in rows if args.question_contains in r["retrieval"]["question"]
    )
    retrieval = result["retrieval"]
    declared = {int(item[1:]) - 1 for item in result["supporting_edges"]}
    edges = [retrieval["edges"][i] for i in sorted(declared)]
    names = {node["id"]: node for node in retrieval["nodes"]}
    used = {end for edge in edges for end in (edge["source"], edge["target"])}
    used_labels = [
        lab for lab in ROW_ORDER if any(names[n]["label"] == lab for n in used)
    ]
    rows_of_nodes: dict[int, list[str]] = {}
    for node_id in sorted(used, key=lambda n: names[n]["name"]):
        row = used_labels.index(names[node_id]["label"])
        rows_of_nodes.setdefault(row, []).append(node_id)
    width = max(6, max(len(v) for v in rows_of_nodes.values())) * 215
    position = {}
    for row, ids in rows_of_nodes.items():
        gap = width / (len(ids) + 1)
        for i, node_id in enumerate(ids, 1):
            position[node_id] = (gap * i, 55 + row * 150)
    height = 55 + (max(rows_of_nodes) + 1) * 150
    shade = {lab: SHADES[i % 3] for i, lab in enumerate(used_labels)}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" '
        'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#333"/></marker></defs>',
    ]
    labels = []
    for edge in edges:
        (x1, y1), (x2, y2) = position[edge["source"]], position[edge["target"]]
        dx, dy = x2 - x1, y2 - y1
        scale = min(
            85 / abs(dx) if dx else float("inf"),
            18 / abs(dy) if dy else float("inf"),
        )
        sx, sy = x1 + dx * scale, y1 + dy * scale
        ex, ey = x2 - dx * scale, y2 - dy * scale
        length = (dx * dx + dy * dy) ** 0.5 or 1
        ux, uy = dx / length, dy / length
        base = (ex - 10 * ux, ey - 10 * uy)
        head = (
            f"{ex:.1f},{ey:.1f} {base[0] - 4 * uy:.1f},{base[1] + 4 * ux:.1f} "
            f"{base[0] + 4 * uy:.1f},{base[1] - 4 * ux:.1f}"
        )
        svg.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{base[0]:.1f}" y2="{base[1]:.1f}" '
            'stroke="#333" stroke-width="1.4"/>'
            f'<polygon points="{head}" fill="#333"/>'
        )
        labels.append(
            (sx + (ex - sx) * 0.45, sy + (ey - sy) * 0.45, edge["relationship_type"])
        )
    for mx, my, label in labels:
        svg.append(
            f'<rect x="{mx - len(label) * 3.1:.1f}" y="{my - 7:.1f}" '
            f'width="{len(label) * 6.2:.1f}" height="13" fill="white"/>'
            f'<text x="{mx:.1f}" y="{my + 3:.1f}" font-size="10" text-anchor="middle" '
            f'font-family="Helvetica" fill="#111">{label}</text>'
        )
    for node_id, (x, y) in position.items():
        node = names[node_id]
        fill = shade.get(node["label"], "#ffffff")
        svg.append(
            f'<rect x="{x - 85}" y="{y - 18}" width="170" height="36" rx="5" '
            f'fill="{fill}" stroke="#333"/>'
        )
        svg.append(
            f'<text x="{x}" y="{y + 4}" font-size="12" text-anchor="middle" '
            f'font-family="Helvetica">{node["name"][:28]}</text>'
        )
    svg.append("</svg>")
    with pymupdf.open(stream="".join(svg).encode(), filetype="svg") as document:
        (args.output / "subgrafo.pdf").write_bytes(document.convert_to_pdf())
    print(f"{len(edges)} relações, {len(position)} nós")


if __name__ == "__main__":
    main()
