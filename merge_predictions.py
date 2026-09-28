"""Replace some question types of one prediction file with those of another.

    python merge_predictions.py --base outputs/predictions_hf/LCOT/human_dev.jsonl \\
        --override outputs/predictions_hf/LCOTFR/human_dev.jsonl --q-types FR \\
        --output outputs/predictions_hf/LCOTX/human_dev.jsonl

Used by experiment E1: L-CoT answers with FR taken from the checklist run.
Keys missing from the override file keep the base prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spartqa.predictions import read_predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, required=True)
    parser.add_argument("--q-types", default="FR")
    parser.add_argument("--source", help="Source name written into the merged records.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    types = {t.strip().upper() for t in args.q_types.split(",")}
    base, override = read_predictions(args.base), read_predictions(args.override)
    replaced = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for key, record in base.items():
            if record["q_type"] in types and key in override:
                record = override[key]
                replaced += 1
            if args.source:
                record = {**record, "source": args.source}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{args.output}: {len(base)} records, {replaced} {sorted(types)} replaced from {args.override}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
