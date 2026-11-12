"""Gera os JSONs curados em data/curated a partir das fontes em data/sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from licenciagraph.sources import na101
from licenciagraph.sources.curate import curate_manual, curate_na101
from licenciagraph.sources.manual import parse_manual

ROOT = Path(__file__).resolve().parents[1]


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
    )
    for name, data in (("manual", manual), ("na101", na_data)):
        target = args.output / f"{name}.json"
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(name, json.dumps(data["report"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
