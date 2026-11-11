"""Resume relatórios automáticos e do juiz de uma rodada."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    run = Path(sys.argv[1])
    for mode in ("hybrid", "vector"):
        report = load(run / f"{mode}-report.json")
        if not report:
            continue
        judge = load(run / f"{mode}-judge.json")
        neg = load(run / f"{mode}-judge-negatives.json")
        line = (
            f"{run.name} {mode}: auto {report['passed']}/{report['total']}, "
            f"conteúdo {report['content_check_rate']:.0%}, "
            f"recall {report['mean_source_recall']:.0%}, "
            f"precisão {report['mean_source_precision']:.0%}"
        )
        if judge:
            s = judge["summary"]
            line += (
                f" | juiz: correto {s['correct']}/{s['total']}, "
                f"completo {s['complete']}, alucinação {s['with_false_claims']}"
            )
        if neg:
            s = neg["summary"]
            line += f" | negativas corretas {s['correct']}/{s['total']}"
        print(line)
        if judge:
            for case in judge["cases"]:
                if not case["correct"]:
                    print(
                        "   ✗",
                        case["id"],
                        "falsas:",
                        case["false_claims"],
                        "| faltou:",
                        [
                            f["fact"]
                            for f in case["detail"]["facts"]
                            if not f["stated_correctly"]
                        ],
                    )


if __name__ == "__main__":
    main()
