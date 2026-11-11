"""Segmentacao de texto preservando limites de paragrafos."""

from __future__ import annotations

import re

from trilhagraph.ingestion.models import Chunk, Document


def split_document(
    document: Document, *, chunk_size: int = 1200, overlap: int = 200
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap deve estar entre zero e chunk_size - 1")

    paragraphs = [
        re.sub(r"[ \t]+", " ", item).strip()
        for item in re.split(r"\n\s*\n", document.content)
        if item.strip()
    ]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            pieces.append(paragraph)
            continue
        step = chunk_size - overlap
        pieces.extend(
            paragraph[start : start + chunk_size].strip()
            for start in range(0, len(paragraph), step)
            if paragraph[start : start + chunk_size].strip()
        )

    texts: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}".strip() if current else piece
        if current and len(candidate) > chunk_size:
            texts.append(current)
            prefix_capacity = max(0, chunk_size - len(piece) - 2)
            prefix_length = min(overlap, prefix_capacity)
            prefix = current[-prefix_length:].lstrip() if prefix_length else ""
            current = f"{prefix}\n\n{piece}".strip() if prefix else piece
        else:
            current = candidate
    if current:
        texts.append(current)

    return [
        Chunk(
            id=f"{document.id}:trecho:{position:04d}",
            document_id=document.id,
            position=position,
            text=text,
        )
        for position, text in enumerate(texts, start=1)
    ]
