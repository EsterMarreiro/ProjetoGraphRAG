"""FalkorDB nativo opcional para ambientes sem Docker; persiste em .runtime."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

from redislite.falkordb_client import FalkorDB


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6381)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / ".runtime"
    directory.mkdir(exist_ok=True)
    database = FalkorDB(
        str(directory / "falkordb.rdb"),
        serverconfig={"port": str(args.port), "bind": "127.0.0.1", "appendonly": "yes"},
    )
    print(f"FalkorDB nativo ativo em 127.0.0.1:{args.port}", flush=True)

    def stop(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        signal.pause()
    finally:
        database.client.shutdown(save=True)


if __name__ == "__main__":
    main()
