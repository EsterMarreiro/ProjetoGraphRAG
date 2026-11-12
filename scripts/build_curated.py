"""Gera os JSONs curados em data/curated a partir das fontes em data/sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from licenciagraph.sources import na101
from licenciagraph.sources.curate import curate_manual, curate_na101
from licenciagraph.sources.manual import parse_manual
from licenciagraph.sources.prose import chunk_blocks, pdf_blocks

ROOT = Path(__file__).resolve().parents[1]


def _dispensa_note(sources: Path) -> list[tuple[int, str]]:
    blocks = pdf_blocks(sources / "manual-licenciamento-5a-edicao.pdf", 36, 36)
    start = next(i for i, b in enumerate(blocks) if b[1].startswith("Nota explicativa"))
    return blocks[start:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "data/sources")
    parser.add_argument("--output", type=Path, default=ROOT / "data/curated")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manual = curate_manual(
        parse_manual(args.sources / "manual-licenciamento-5a-edicao.pdf")
    )
    norm = args.sources / "na-101-5a-edicao-2026.pdf"
    na_data = curate_na101(
        na101.parse_porte_tables(norm),
        na101.parse_cost_letters(norm),
        na101.parse_base_criteria(norm),
        na101.parse_activity_lists(norm),
        manual,
        na101.parse_master_list(norm),
    )
    prose = {
        "documents": [
            {
                "id": "manual-capitulos",
                "title": "Manual de Licenciamento Ambiental, 5ª edição: capítulos I a IV",
                "file": "manual-licenciamento-5a-edicao.pdf",
                "chunks": chunk_blocks(
                    pdf_blocks(
                        args.sources / "manual-licenciamento-5a-edicao.pdf", 9, 30
                    )
                ),
            },
            {
                "id": "manual-nota-dispensa",
                "title": "Manual de Licenciamento Ambiental, 5ª edição: nota sobre a dispensa",
                "file": "manual-licenciamento-5a-edicao.pdf",
                "chunks": chunk_blocks(_dispensa_note(args.sources)),
            },
            {
                "id": "na101-texto",
                "title": "NA-101, 5ª edição (2026): disposições gerais e atividades específicas",
                "file": "na-101-5a-edicao-2026.pdf",
                "chunks": chunk_blocks(pdf_blocks(norm, 5, 17)),
            },
            {
                "id": "na101-observacoes-custos",
                "title": "NA-101, 5ª edição (2026): observações sobre custos de análise",
                "file": "na-101-5a-edicao-2026.pdf",
                "chunks": chunk_blocks(pdf_blocks(norm, 383, 389)),
            },
        ]
    }
    for name, data in (("manual", manual), ("na101", na_data)):
        target = args.output / f"{name}.json"
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(name, json.dumps(data["report"], ensure_ascii=False, indent=1))
    (args.output / "prosa.json").write_text(
        json.dumps(prose, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print({d["id"]: len(d["chunks"]) for d in prose["documents"]})


if __name__ == "__main__":
    main()
