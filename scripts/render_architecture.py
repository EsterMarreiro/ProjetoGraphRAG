"""Figura vetorial dos quatro pilares implementados (não é dado experimental)."""

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
BOXES = [
    ("Fontes → dados", "tabelas: leitor determinístico", "texto: LLM com vocabulário"),
    (
        "FalkorDB LPG",
        "atividades, checklists, porte",
        "custos em UFR-PB e proveniência",
    ),
    (
        "Recuperação e geração",
        "vetor + texto + Cypher + grafo",
        "caminho, Louvain, PageRank, verificação",
    ),
    (
        "Interface Streamlit",
        "chat com fontes e subgrafo",
        "assistente de enquadramento",
    ),
]
parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1000" '
    'height="175" viewBox="0 0 1000 175">',
    '<rect width="1000" height="175" fill="white"/>',
]
for i, (title, detail, detail2) in enumerate(BOXES):
    x = 8 + i * 252
    parts.append(
        f'<rect x="{x}" y="20" width="228" height="120" rx="5" '
        'fill="#f2f2f2" stroke="#333"/>'
    )
    for y, text, size in [(50, title, 17), (83, detail, 14), (107, detail2, 14)]:
        parts.append(
            f'<text x="{x + 114}" y="{y}" text-anchor="middle" '
            f'font-size="{size}" font-family="sans-serif">{text}</text>'
        )
    if i < 3:
        end = x + 249
        parts.append(
            f'<path d="M{x + 228},80 H{end - 5} M{end - 10},76 '
            f'L{end - 5},80 L{end - 10},84" '
            'stroke="#333" stroke-width="2" fill="none"/>'
        )
parts.append("</svg>")
svg = "\n".join(parts)
directory = ROOT / "docs/artigo"
(directory / "arquitetura.svg").write_text(svg)
with pymupdf.open(stream=svg.encode(), filetype="svg") as document:
    (directory / "arquitetura.pdf").write_bytes(document.convert_to_pdf())
