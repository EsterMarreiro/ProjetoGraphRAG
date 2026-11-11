"""Leitura deterministica de documentos TXT."""

from __future__ import annotations

import hashlib
from pathlib import Path

from trilhagraph.ingestion.canonicalization import slugify
from trilhagraph.ingestion.models import Document


def load_documents(data_dir: Path) -> list[Document]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Diretorio de dados nao encontrado: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Caminho de dados nao e um diretorio: {data_dir}")

    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.txt")):
        raw = path.read_bytes()
        try:
            content = raw.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"Arquivo deve estar em UTF-8: {path}") from exc
        if not content:
            continue
        title = next(
            (line.strip() for line in content.splitlines() if line.strip()), path.stem
        )
        documents.append(
            Document(
                id=f"documento:{slugify(path.stem)}",
                title=title.removeprefix("#")
                .removeprefix("Título:")
                .removeprefix("Titulo:")
                .strip(),
                path=path,
                checksum=hashlib.sha256(raw).hexdigest(),
                content=content,
            )
        )
    return documents
