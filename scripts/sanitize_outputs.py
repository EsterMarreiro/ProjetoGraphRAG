"""Produz a apresentação servida sem alterar afirmações/IDs da execução bruta."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from licenciagraph.retrieval.generation import SCOPE_CAVEAT


def main():
    directory = Path(sys.argv[1])
    counts = {}
    for mode in ("hybrid", "vector"):
        source = directory / f"{mode}-answers.jsonl"
        if not source.exists():
            continue
        output = []
        for line in source.read_text().splitlines():
            record = json.loads(line)
            original = record["result"].copy()
            record["result"]["caveats"] = [SCOPE_CAVEAT]
            assert {k: v for k, v in original.items() if k != "caveats"} == {
                k: v for k, v in record["result"].items() if k != "caveats"
            }
            output.append(json.dumps(record, ensure_ascii=False))
        (directory / f"{mode}-served-answers.jsonl").write_text(
            "\n".join(output) + "\n"
        )
        counts[mode] = len(output)
    (directory / "presentation-policy.json").write_text(
        json.dumps(
            {
                "processed_at": datetime.now(UTC).isoformat(),
                "change": (
                    "Only free-form caveats replaced by a known catalogue scope notice"
                ),
                "claims_citations_paths_abstention": "unchanged",
                "raw_outputs": "original *-answers.jsonl and llm-*.json retained",
                "records": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(counts)


if __name__ == "__main__":
    main()
