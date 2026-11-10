"""Normalizacao e resolucao simples de entidades."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from rotagraph.ingestion.models import ExtractedEntity


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_text).strip().casefold()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_text(value)).strip("-")
    if not slug:
        raise ValueError("Nao foi possivel gerar um identificador canonico")
    return slug


class EntityCanonicalizer:
    def __init__(self) -> None:
        self._known: dict[tuple[str, str, tuple[str, ...]], str] = {}
        self._aliases: dict[tuple[str, str], str] = {}

    def canonicalize(self, entity: ExtractedEntity) -> str:
        context = self._location_context(entity.properties)
        key = (entity.label, normalize_text(entity.name), context)
        if key in self._known:
            entity.canonical_id = self._known[key]
            return entity.canonical_id
        alias_key = (entity.label, normalize_text(entity.name))
        if not context and alias_key in self._aliases:
            entity.canonical_id = self._aliases[alias_key]
            return entity.canonical_id

        base = slugify(entity.name)
        suffix = "-".join(slugify(value) for value in context if value)
        canonical_id = f"{entity.label.casefold()}:{base}"
        if suffix and suffix not in base:
            canonical_id = f"{canonical_id}-{suffix}"
        entity.canonical_id = canonical_id
        self._known[key] = canonical_id

        aliases = entity.properties.get("aliases", [])
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    self._aliases[(entity.label, normalize_text(alias))] = canonical_id
        return canonical_id

    @staticmethod
    def _location_context(properties: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            normalize_text(str(properties.get(key, "")))
            for key in ("municipio", "uf")
            if properties.get(key)
        )
