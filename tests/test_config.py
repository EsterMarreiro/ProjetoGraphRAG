from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from licenciagraph.config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env(load_file=False)

        self.assertEqual(settings.falkordb_port, 6379)
        self.assertEqual(settings.falkordb_graph, "licenciagraph")
        self.assertEqual(settings.embedding_dimension, 768)
        self.assertIsNone(settings.falkordb_password)

    def test_environment_overrides_defaults(self) -> None:
        environment = {
            "FALKORDB_HOST": "database",
            "FALKORDB_PORT": "6380",
            "FALKORDB_PASSWORD": "secret",
            "FALKORDB_GRAPH": "tourism",
            "EMBEDDING_MODEL": "custom-embed",
            "EMBEDDING_DIMENSION": "1024",
            "CHAT_MODEL": "custom-chat",
            "OPENAI_BASE_URL": "https://proxy.local/v1/",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_env(load_file=False)

        self.assertEqual(settings.falkordb_host, "database")
        self.assertEqual(settings.falkordb_port, 6380)
        self.assertEqual(settings.falkordb_password, "secret")
        self.assertEqual(settings.embedding_dimension, 1024)
        self.assertEqual(settings.openai_base_url, "https://proxy.local/v1")

    def test_rejects_invalid_dimension(self) -> None:
        with (
            patch.dict(os.environ, {"EMBEDDING_DIMENSION": "0"}, clear=True),
            self.assertRaisesRegex(ValueError, "maior que zero"),
        ):
            Settings.from_env(load_file=False)


if __name__ == "__main__":
    unittest.main()
