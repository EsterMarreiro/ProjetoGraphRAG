"""Gráfico de barras em tons de cinza com a acurácia do juiz por configuração."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CONFIGS = [
    ("Pipeline inicial\n(gpt-4o-mini)", "175843", "180018", "hybrid"),
    ("Final sem verificador\n(gpt-4.1)", "181931", "181934", "hybrid"),
    ("Final\n(gpt-4.1)", "181919", "181922", "hybrid"),
    ("Baseline vetorial\n(gpt-4.1)", "181919", "181922", "vector"),
]


def accuracy(run: str, mode: str) -> float:
    path = next(ART.glob(f"run-*{run}Z")) / f"{mode}-judge.json"
    return json.loads(path.read_text())["summary"]["accuracy"]


def main() -> None:
    width, height, base = 760, 330, 270
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<line x1="60" y1="{base}" x2="{width - 20}" y2="{base}" stroke="#333"/>',
    ]
    for tick in (0, 25, 50, 75, 100):
        y = base - tick * 2.2
        svg.append(
            f'<line x1="56" y1="{y}" x2="{width - 20}" y2="{y}" stroke="#ddd"/>'
            f'<text x="50" y="{y + 4}" font-size="11" text-anchor="end" '
            f'font-family="Helvetica">{tick}%</text>'
        )
    group = (width - 90) / len(CONFIGS)
    for i, (label, dev, held, mode) in enumerate(CONFIGS):
        for j, (run, shade) in enumerate(((dev, "#9a9a9a"), (held, "#333333"))):
            value = accuracy(run, mode) * 100
            x = 80 + i * group + j * 62
            svg.append(
                f'<rect x="{x}" y="{base - value * 2.2}" width="56" '
                f'height="{value * 2.2}" fill="{shade}"/>'
                f'<text x="{x + 28}" y="{base - value * 2.2 - 5}" font-size="11" '
                f'text-anchor="middle" font-family="Helvetica">{value:.0f}%</text>'
            )
        for k, line in enumerate(label.split("\n")):
            svg.append(
                f'<text x="{80 + i * group + 59}" y="{base + 18 + k * 13}" '
                f'font-size="11" text-anchor="middle" '
                f'font-family="Helvetica">{line}</text>'
            )
    svg.append(
        '<rect x="520" y="8" width="12" height="12" fill="#9a9a9a"/>'
        '<text x="538" y="18" font-size="11" font-family="Helvetica">'
        "Desenvolvimento</text>"
        '<rect x="640" y="8" width="12" height="12" fill="#333333"/>'
        '<text x="658" y="18" font-size="11" font-family="Helvetica">Retido</text>'
    )
    svg.append("</svg>")
    with pymupdf.open(stream="".join(svg).encode(), filetype="svg") as document:
        (ROOT / "docs/artigo/resultados.pdf").write_bytes(document.convert_to_pdf())


if __name__ == "__main__":
    main()
