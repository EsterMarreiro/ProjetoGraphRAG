"""Gráfico de barras em tons de cinza com a acurácia do juiz por configuração."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CONFIGS = [
    ("LicenciaGraph\n(híbrido)", "dev-hybrid-v3", "heldout-hybrid-v3", "hybrid"),
    ("Baseline vetorial", "dev-vector", "heldout-vector", "vector"),
]


def accuracy(run: str, mode: str) -> float:
    path = ART / f"run-{run}" / f"{mode}-judge.json"
    return json.loads(path.read_text())["summary"]["accuracy"]


def resolution_chart() -> None:
    """Barras da identificação da atividade em fala coloquial (média de 3 rodadas)."""
    runs = [
        json.loads((ART / f"resolution-v3-run{i}.json").read_text()) for i in (1, 2, 3)
    ]
    total = len(runs[0])
    values = [
        ("Método anterior\n(nome ou vetor)", "baseline_hit", "#9a9a9a"),
        ("Resolvedor\nde atividades", "resolver_hit", "#333333"),
    ]
    width, height, base = 420, 300, 240
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
    for i, (label, key, shade) in enumerate(values):
        mean = sum(sum(row[key] for row in run) for run in runs) / len(runs)
        value = mean / total * 100
        x = 100 + i * 150
        svg.append(
            f'<rect x="{x}" y="{base - value * 2.2}" width="90" '
            f'height="{value * 2.2}" fill="{shade}"/>'
            f'<text x="{x + 45}" y="{base - value * 2.2 - 5}" font-size="11" '
            f'text-anchor="middle" font-family="Helvetica">'
            f"{mean:.0f}/{total} ({value:.0f}%)</text>"
        )
        for k, line in enumerate(label.split("\n")):
            svg.append(
                f'<text x="{x + 45}" y="{base + 18 + k * 13}" font-size="11" '
                f'text-anchor="middle" font-family="Helvetica">{line}</text>'
            )
    svg.append("</svg>")
    with pymupdf.open(stream="".join(svg).encode(), filetype="svg") as document:
        (ROOT / "docs/artigo/resolucao.pdf").write_bytes(document.convert_to_pdf())


def main() -> None:
    resolution_chart()
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
