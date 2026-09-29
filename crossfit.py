"""A1: out-of-fold predictions on Human train, so ensembles can be tuned on train + dev (613 questions).

    # 1. story-level folds of Human train; each fold dir is a splits dir for prepare_sft.py
    python crossfit.py make-folds --folds 5 --output Data/crossfit
    # 2. after training one adapter per fold and predicting its held-out stories:
    python crossfit.py merge --folds-dir Data/crossfit \\
        --pattern "outputs/pred/F2C_folds/fold{k}.jsonl" --source F2C --output outputs/pred/F2C/human_train.jsonl
    # 3. train + dev as one labeled file, with every source's predictions renumbered to match
    python crossfit.py combine --pred-dir outputs/pred --name trdev

Fold k's human_train.json leaves out fold k's stories (heldout.json) and keeps
the global human_dev.json for eval loss, so an adapter trained on fold k never
sees the stories it later predicts. Keys are "<story index>_<q_id>" as in
spartqa.predictions; merge maps held-out indices back to Human train indices,
and combine appends the dev stories after the train stories.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

from spartqa.data import read_json, write_json
from spartqa.predictions import PredictionWriter, read_predictions
from spartqa.splits import subset_data


def story_folds(count: int, folds: int, seed: int) -> list[list[int]]:
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return [sorted(order[k::folds]) for k in range(folds)]


def link_or_copy(source: Path, target: Path) -> None:
    target.unlink(missing_ok=True)
    try:
        os.symlink(source.resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def make_folds(args: argparse.Namespace) -> int:
    train = read_json(args.splits_dir / "human_train.json")
    folds = story_folds(len(train["data"]), args.folds, args.seed)
    for k, held in enumerate(folds):
        fold_dir = args.output / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        kept = [i for i in range(len(train["data"])) if i not in set(held)]
        write_json(fold_dir / "human_train.json", subset_data(train, kept))
        write_json(fold_dir / "heldout.json", subset_data(train, held))
        for name in ("human_dev.json", "auto_train.json", "auto_dev.json"):
            link_or_copy(args.splits_dir / name, fold_dir / name)
        questions = sum(len(train["data"][i]["questions"]) for i in held)
        print(f"fold{k}: train {len(kept)} stories, held out {len(held)} stories / {questions} questions")
    write_json(args.output / "folds.json", {"seed": args.seed, "folds": folds,
                                            "source": (args.splits_dir / "human_train.json").as_posix()})
    return 0


def renumber(record: dict[str, Any], story: int, source: str | None) -> dict[str, Any]:
    q_id = record["key"].split("_", 1)[1]
    return {**record, "key": f"{story}_{q_id}", **({"source": source} if source else {})}


def merge(args: argparse.Namespace) -> int:
    folds = read_json(args.folds_dir / "folds.json")["folds"]
    records = []
    for k, held in enumerate(folds):
        path = Path(args.pattern.format(k=k))
        fold_records = read_predictions(path)
        if not fold_records:
            raise SystemExit(f"missing predictions for fold {k}: {path}")
        for record in fold_records.values():
            records.append(renumber(record, held[int(record["key"].split("_")[0])], args.source))
    args.output.unlink(missing_ok=True)
    PredictionWriter(args.output).write_many(sorted(records, key=lambda r: [int(p) for p in r["key"].split("_")]))
    print(f"Wrote {len(records)} out-of-fold predictions to {args.output}")
    return 0


def combine(args: argparse.Namespace) -> int:
    train = read_json(args.splits_dir / f"{args.dataset}_train.json")
    dev = read_json(args.splits_dir / f"{args.dataset}_dev.json")
    offset = len(train["data"])
    write_json(args.splits_dir / f"{args.dataset}_{args.name}.json",
               {**{k: v for k, v in train.items() if k != "data"}, "data": train["data"] + dev["data"]})
    for source_dir in sorted(p for p in args.pred_dir.iterdir() if p.is_dir()):
        train_path, dev_path = source_dir / f"{args.dataset}_train.jsonl", source_dir / f"{args.dataset}_dev.jsonl"
        if not (train_path.exists() and dev_path.exists()):
            continue
        records = list(read_predictions(train_path).values())
        records += [renumber(r, int(r["key"].split("_")[0]) + offset, None) for r in read_predictions(dev_path).values()]
        target = source_dir / f"{args.dataset}_{args.name}.jsonl"
        target.unlink(missing_ok=True)
        PredictionWriter(target).write_many(records)
        print(f"{source_dir.name}: {len(records)} predictions -> {target.name}")
    print(f"Wrote {args.splits_dir / f'{args.dataset}_{args.name}.json'} "
          f"({offset} train + {len(dev['data'])} dev stories)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-folds")
    make.add_argument("--splits-dir", type=Path, default=Path("Data/splits"))
    make.add_argument("--folds", type=int, default=5)
    make.add_argument("--seed", type=int, default=42)
    make.add_argument("--output", type=Path, default=Path("Data/crossfit"))
    make.set_defaults(func=make_folds)
    join = commands.add_parser("merge")
    join.add_argument("--folds-dir", type=Path, default=Path("Data/crossfit"))
    join.add_argument("--pattern", required=True, help="Held-out prediction file per fold, with {k}.")
    join.add_argument("--source", help="Source name written into the merged records.")
    join.add_argument("--output", type=Path, required=True)
    join.set_defaults(func=merge)
    both = commands.add_parser("combine")
    both.add_argument("--splits-dir", type=Path, default=Path("Data/splits"))
    both.add_argument("--pred-dir", type=Path, required=True)
    both.add_argument("--dataset", default="human")
    both.add_argument("--name", default="trdev")
    both.set_defaults(func=combine)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
