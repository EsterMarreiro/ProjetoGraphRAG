"""Normaliza os itens dos checklists em documentos canônicos (usa a API)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from licenciagraph.config import Settings
from licenciagraph.ingestion.llm import OpenAIClient
from licenciagraph.sources.documents import normalize_items, unique_items

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", type=Path, default=ROOT / "data/curated")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    settings = Settings.from_env()
    manual = json.loads((args.curated / "manual.json").read_text(encoding="utf-8"))
    items = unique_items(manual)
    if args.limit:
        items = items[: args.limit]
    llm = OpenAIClient.from_settings(settings)
    documents = normalize_items(items, llm, model=settings.chat_model)
    target = args.curated / "documentos.json"
    target.write_text(
        json.dumps(documents, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    names = {d["name"] for d in documents.values()}
    print(f"{len(items)} itens únicos, {len(names)} documentos canônicos")


if __name__ == "__main__":
    main()
