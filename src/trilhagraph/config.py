"""Configuracao central da aplicacao."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuracoes carregadas de variaveis de ambiente."""

    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: str | None = None
    falkordb_graph: str = "trilhagraph_aprendizagem"
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    chat_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 600
    chunk_size: int = 1200
    chunk_overlap: int = 200

    @classmethod
    def from_env(cls, *, load_file: bool = True) -> Settings:
        if load_file:
            from dotenv import load_dotenv

            load_dotenv()

        password = os.getenv("FALKORDB_PASSWORD", "").strip() or None
        settings = cls(
            falkordb_host=os.getenv("FALKORDB_HOST", "localhost").strip(),
            falkordb_port=_positive_int("FALKORDB_PORT", 6379),
            falkordb_password=password,
            falkordb_graph=os.getenv(
                "FALKORDB_GRAPH", "trilhagraph_aprendizagem"
            ).strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text").strip(),
            embedding_dimension=_positive_int("EMBEDDING_DIMENSION", 768),
            chat_model=os.getenv("CHAT_MODEL", "qwen2.5:3b").strip(),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            ollama_timeout=_positive_int("OLLAMA_TIMEOUT", 600),
            chunk_size=_positive_int("CHUNK_SIZE", 1200),
            chunk_overlap=_non_negative_int("CHUNK_OVERLAP", 200),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        required = {
            "FALKORDB_HOST": self.falkordb_host,
            "FALKORDB_GRAPH": self.falkordb_graph,
            "EMBEDDING_MODEL": self.embedding_model,
            "CHAT_MODEL": self.chat_model,
            "OLLAMA_BASE_URL": self.ollama_base_url,
        }
        empty = [name for name, value in required.items() if not value]
        if empty:
            names = ", ".join(empty)
            raise ValueError(f"Configuracoes obrigatorias vazias: {names}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE")


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} deve ser um numero inteiro") from exc
    if value <= 0:
        raise ValueError(f"{name} deve ser maior que zero")
    return value


def _non_negative_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} deve ser um numero inteiro") from exc
    if value < 0:
        raise ValueError(f"{name} nao pode ser negativo")
    return value
