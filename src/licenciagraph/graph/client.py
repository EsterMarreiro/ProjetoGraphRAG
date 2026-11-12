"""Conexao com o FalkorDB."""

from __future__ import annotations

from typing import Any

from licenciagraph.config import Settings


class GraphConnection:
    """Mantem o cliente e o grafo selecionado em um unico lugar."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._database: Any | None = None
        self._graph: Any | None = None

    def connect(self) -> Any:
        from falkordb import FalkorDB

        kwargs: dict[str, Any] = {
            "host": self.settings.falkordb_host,
            "port": self.settings.falkordb_port,
        }
        if self.settings.falkordb_password:
            kwargs["password"] = self.settings.falkordb_password

        self._database = FalkorDB(**kwargs)
        self._graph = self._database.select_graph(self.settings.falkordb_graph)
        return self._graph

    @property
    def graph(self) -> Any:
        if self._graph is None:
            return self.connect()
        return self._graph

    def healthcheck(self) -> dict[str, str | int]:
        if self._database is None:
            self.connect()
        if self._database is None or not self._database.connection.ping():
            raise RuntimeError("FalkorDB nao respondeu ao PING")
        return {
            "status": "ok",
            "host": self.settings.falkordb_host,
            "port": self.settings.falkordb_port,
            "graph": self.settings.falkordb_graph,
        }

    def close(self) -> None:
        if self._database is None:
            return
        connection = getattr(self._database, "connection", None)
        if connection is not None:
            connection.close()
        self._database = None
        self._graph = None

    def __enter__(self) -> GraphConnection:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
