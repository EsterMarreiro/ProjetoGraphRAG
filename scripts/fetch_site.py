"""Baixa páginas públicas da SUDEMA e grava o texto em data/curated/site.json."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from licenciagraph.sources.prose import chunk_blocks, site_text

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://sudema.pb.gov.br"
PAGES = {
    "unidades-regionais": (
        "Unidades regionais da SUDEMA",
        "/institucional/unidades-regionais",
    ),
    "copam": ("COPAM: Conselho de Proteção Ambiental", "/institucional/copam-1"),
    "licenciamento-municipal": (
        "Licenciamento ambiental municipal",
        "/institucional/copam-1/licenciamento-ambiental-municipal",
    ),
    "legislacao-pertinente": (
        "Legislação pertinente ao licenciamento",
        "/institucional/copam-1/legislacao-pertinente",
    ),
    "normas-administrativas": (
        "Normas administrativas do COPAM",
        "/institucional/copam-1/normas-administrativas",
    ),
    "eia-rima": ("EIA/RIMA e audiências públicas", "/eia-rima"),
    "fiscalizacao": ("Divisão de Fiscalização", "/fiscalizacao-e-controle"),
    "servicos-ao-publico": ("Serviços ao público", "/servicos/servicos-ao-publico"),
    "formularios": (
        "Formulários para requerimento",
        "/servicos/servicos-ao-publico/formularios-para-requerimento/formularios-para-requerimento",
    ),
    "legislacao-ambiental": (
        "Legislação ambiental",
        "/servicos/servicos-ao-publico/legislacao-ambienta",
    ),
}
MAX_LINES = {"eia-rima": 6}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/curated/site.json")
    args = parser.parse_args()
    accessed = datetime.date.today().isoformat()
    documents = []
    for key, (title, path) in PAGES.items():
        lines = site_text(BASE + path)
        lines = lines[: MAX_LINES.get(key, len(lines))]
        chunks = chunk_blocks([(0, line) for line in lines])
        documents.append(
            {
                "id": f"site-{key}",
                "title": title,
                "url": BASE + path,
                "accessed": accessed,
                "chunks": chunks,
            }
        )
        print(key, len(lines), "linhas,", len(chunks), "trechos")
    args.output.write_text(
        json.dumps({"documents": documents}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
