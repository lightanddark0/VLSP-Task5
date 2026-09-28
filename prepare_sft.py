"""Build JSONL training data for branch F (Qwen3 + LoRA).

Stage 1 (Auto): a type-balanced sample of Auto train questions.
    python prepare_sft.py --stage auto --n-samples 50000 --output-dir outputs/sft/auto
Stage 2 (Human): every Human train question (optionally repeated) plus some
Auto questions so the model does not forget Auto.
    python prepare_sft.py --stage human --auto-mix 2000 --output-dir outputs/sft/human

Only the story-level train splits are used for training; dev.jsonl comes from
the dev splits and is used for eval loss and checkpoint selection. Each line
holds chat messages and the target answer text.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from spartqa.data import TASKS, read_json, write_json
from spartqa.predictions import iter_questions, payload_of
from spartqa.prompting import build_messages, format_answer, prompt_version


def records_from(path: Path, dataset: str) -> list[dict[str, Any]]:
    records = []
    for key, item, question in iter_questions(read_json(path)):
        payload = payload_of(item, question)
        records.append({"key": f"{dataset}:{key}", "dataset": dataset, "q_type": question["q_type"],
                        "messages": build_messages(payload, dataset),
                        "target": format_answer(question["answer"], payload)})
    return records


def parse_ratio(text: str) -> dict[str, float]:
    ratio = {task: float(value) for task, value in (part.split("=") for part in text.split(","))}
    if set(ratio) != set(TASKS):
        raise argparse.ArgumentTypeError(f"--type-ratio needs all of {TASKS}")
    total = sum(ratio.values())
    return {task: value / total for task, value in ratio.items()}


def balanced_sample(records: list[dict[str, Any]], count: int, ratio: dict[str, float],
                    rng: random.Random) -> list[dict[str, Any]]:
    """Sample ``count`` records with the given question-type shares; short types are taken whole."""
    by_task = {task: [r for r in records if r["q_type"] == task] for task in TASKS}
    if count >= len(records):
        return list(records)
    wanted = {task: round(count * ratio[task]) for task in TASKS}
    chosen = []
    for task in TASKS:
        pool = by_task[task]
        chosen.extend(rng.sample(pool, min(wanted[task], len(pool))))
    missing = count - len(chosen)
    if missing > 0:
        taken = {r["key"] for r in chosen}
        rest = [r for r in records if r["key"] not in taken]
        chosen.extend(rng.sample(rest, min(missing, len(rest))))
    return chosen


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("auto", "human"), required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("Data/splits"))
    parser.add_argument("--n-samples", type=int, default=50000, help="Auto questions for stage auto.")
    parser.add_argument("--type-ratio", type=parse_ratio, default=parse_ratio("YN=0.25,FR=0.30,FB=0.25,CO=0.20"))
    parser.add_argument("--human-repeat", type=int, default=1, help="Copies of each Human question (stage human).")
    parser.add_argument("--auto-mix", type=int, default=2000, help="Auto questions mixed into stage human.")
    parser.add_argument("--dev-samples", type=int, default=1000, help="Auto dev questions kept for eval loss.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Keep only N training and N dev records (smoke test).")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)

    auto_train = records_from(args.splits_dir / "auto_train.json", "auto")
    auto_dev = records_from(args.splits_dir / "auto_dev.json", "auto")
    if args.stage == "auto":
        train = balanced_sample(auto_train, args.n_samples, args.type_ratio, rng)
        dev = balanced_sample(auto_dev, args.dev_samples, args.type_ratio, rng)
    else:
        human_train = records_from(args.splits_dir / "human_train.json", "human")
        train = human_train * args.human_repeat
        train += balanced_sample(auto_train, args.auto_mix, args.type_ratio, rng) if args.auto_mix else []
        dev = records_from(args.splits_dir / "human_dev.json", "human")
    rng.shuffle(train)
    if args.limit:
        train, dev = train[:args.limit], dev[:max(1, args.limit // 5)]

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "dev.jsonl", dev)
    stats = {
        "stage": args.stage, "seed": args.seed, "prompt_version": prompt_version("F"),
        "train": {"count": len(train), "by_dataset": Counter(r["dataset"] for r in train),
                  "by_task": Counter(r["q_type"] for r in train)},
        "dev": {"count": len(dev), "by_dataset": Counter(r["dataset"] for r in dev),
                "by_task": Counter(r["q_type"] for r in dev)},
        "args": {key: str(value) for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "stats.json", stats)
    print(json.dumps({k: stats[k] for k in ("train", "dev")}, ensure_ascii=False))
    print(f"Wrote {args.output_dir}/train.jsonl and dev.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
