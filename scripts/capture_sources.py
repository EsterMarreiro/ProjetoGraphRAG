"""Captura as fontes e dependências do experimento sem incluir segredos/modelos."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def capture(directory: Path) -> None:
    files = sorted(
        [
            *ROOT.glob("src/**/*.py"),
            *ROOT.glob("scripts/*.py"),
            *ROOT.glob("data/curated/*.json"),
            *ROOT.glob("data/evaluation/*.json"),
            ROOT / "pyproject.toml",
            ROOT / ".env.example",
            ROOT / "streamlit_app.py",
        ]
    )
    manifest = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    (directory / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    with tarfile.open(directory / "source-snapshot.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    (directory / "requirements-environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )


if __name__ == "__main__":
    capture(Path(sys.argv[1]))
