from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from rotagraph.config import Settings
from rotagraph.graph.client import GraphConnection


class FakeGraph:
    pass


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def ping(self) -> bool:
        return True


class FakeFalkorDB:
    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs
        self.connection = FakeConnection()

    def select_graph(self, name: str) -> FakeGraph:
        self.graph_name = name
        return FakeGraph()


class GraphConnectionTests(unittest.TestCase):
    def test_connects_and_checks_graph_health(self) -> None:
        fake_module = types.ModuleType("falkordb")
        fake_module.FalkorDB = FakeFalkorDB  # type: ignore[attr-defined]
        settings = Settings(
            falkordb_host="graph.local",
            falkordb_port=6380,
            falkordb_password="secret",
            falkordb_graph="tourism",
        )

        with patch.dict(sys.modules, {"falkordb": fake_module}):
            connection = GraphConnection(settings)
            result = connection.healthcheck()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["graph"], "tourism")
        self.assertEqual(
            FakeFalkorDB.last_kwargs,
            {"host": "graph.local", "port": 6380, "password": "secret"},
        )


if __name__ == "__main__":
    unittest.main()
