"""Branch LP (E3), CPU part: answer questions from saved LLM parses with the S reasoner.

llm_parse.py samples worlds (per story) and forms (per question) and saves them
in <output>.parses.jsonl. This script re-solves those parses under any answering
convention, without the GPU:

    # compare every convention on a labeled file (e.g. Human train)
    python solve_parsed.py --parses outputs/predictions/LP/human_train.parses.jsonl \\
        --input Data/splits/human_train.json --grid
    # write predictions with a chosen convention
    python solve_parsed.py --parses outputs/predictions/LP/human_dev.parses.jsonl \\
        --input Data/splits/human_dev.json --set cross_block_far=true \\
        --output outputs/predictions/LP/human_dev.jsonl

Each (world, form) pair gives one answer; answers are voted as in L-CoT
(YN/CO majority, FR/FB labels with at least half of the votes). A question
with no usable pair is abstained (answer null).
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from spartqa.data import TASKS, read_json, write_json
from spartqa.metrics import PRIMARY_METRIC, answer_matches
from spartqa.predictions import PredictionWriter, iter_questions
from spartqa.symbolic.structured import LPConfig, LPSolver, StructureError, parse_form, world_from_json
from spartqa.voting import vote_samples

GRID = {"yn_mode": ["closed", "open"], "fr_multi": ["all", "any"], "fb_scope": ["global", "block"],
        "far_chain": [True, False], "cross_block_far": [False, True]}


def read_parses(path: Path) -> tuple[dict[int, list[Any]], dict[str, list[Any]]]:
    """Worlds by story index and forms by question key (later lines add samples)."""
    worlds: dict[int, list[Any]] = {}
    forms: dict[str, list[Any]] = {}
    if not path.exists():
        return worlds, forms
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["kind"] == "world":
                worlds[record["story"]] = record["worlds"]
            else:
                forms[record["key"]] = record["forms"]
    return worlds, forms


def answer_question(worlds: list[Any], forms: list[Any], question: dict[str, Any],
                    config: LPConfig) -> tuple[list[Any] | None, dict[str, float], int]:
    """Vote over every usable (world, form) pair; returns answer, vote shares, number of votes."""
    solvers = []
    for world in worlds:
        try:
            solvers.append(LPSolver(world_from_json(world, config.rules()), config))
        except (StructureError, KeyError, TypeError):
            continue
    parsed = []
    for form in forms:
        try:
            parsed.append(parse_form(form, question["q_type"]))
        except (StructureError, KeyError, TypeError):
            continue
    samples = []
    for solver, form in itertools.product(solvers, parsed):
        try:
            samples.append(solver.answer(form, question))
        except (StructureError, KeyError, TypeError, IndexError):
            samples.append(None)
    answer, scores = vote_samples(question["q_type"], samples)
    return answer, scores, sum(sample is not None for sample in samples)


def solve_all(data: dict[str, Any], worlds: dict[int, list[Any]], forms: dict[str, list[Any]],
              config: LPConfig, source: str = "LP") -> dict[str, dict[str, Any]]:
    records = {}
    for key, _, question in iter_questions(data):
        story = int(key.split("_")[0])
        if story not in worlds or key not in forms:
            continue
        answer, scores, votes = answer_question(worlds[story], forms[key], question, config)
        records[key] = {"key": key, "q_type": question["q_type"], "answer": answer, "scores": scores,
                        "source": source, "votes": votes}
    return records


def coverage_report(data: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per type: questions parsed, answered (not abstained), and accuracy on the answered ones."""
    stats = {task: {"parsed": 0, "answered": 0, "correct": 0} for task in TASKS}
    for key, _, question in iter_questions(data):
        record = records.get(key)
        if record is None or "answer" not in question:
            continue
        task_stats = stats[question["q_type"]]
        task_stats["parsed"] += 1
        if record["answer"] is not None:
            task_stats["answered"] += 1
            task_stats["correct"] += answer_matches(question["q_type"], record["answer"], question["answer"])
    return {task: {**values, "coverage": values["answered"] / max(values["parsed"], 1),
                   "precision": values["correct"] / max(values["answered"], 1),
                   "score_if_abstain_wrong": values["correct"] / max(values["parsed"], 1)}
            for task, values in stats.items() if values["parsed"]}


def parse_setting(text: str) -> tuple[str, Any]:
    name, _, value = text.partition("=")
    kinds = {f.name: f.type for f in fields(LPConfig)}
    if name not in kinds:
        raise argparse.ArgumentTypeError(f"unknown setting {name}; known: {sorted(kinds)}")
    if value.lower() in ("true", "false"):
        return name, value.lower() == "true"
    return name, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parses", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--set", type=parse_setting, action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--grid", action="store_true", help="Score every convention (needs gold answers).")
    parser.add_argument("--best-json", type=Path, help="With --grid: write the best convention here (abstain = wrong).")
    parser.add_argument("--output", type=Path, help="Write predictions (JSONL) for the --set convention.")
    parser.add_argument("--source", default="LP")
    args = parser.parse_args(argv)

    data = read_json(args.input)
    worlds, forms = read_parses(args.parses)
    config = replace(LPConfig(), **dict(args.set))
    if args.grid:
        rows = []
        for values in itertools.product(*GRID.values()):
            candidate = replace(config, **dict(zip(GRID, values)))
            report = coverage_report(data, solve_all(data, worlds, forms, candidate))
            if not report:
                parser.error("no labeled questions with parses")
            rows.append((candidate, report))
        rows.sort(key=lambda row: -sum(r["score_if_abstain_wrong"] for r in row[1].values()) / len(row[1]))
        print("Per type: precision on answered / coverage (abstentions are left to other branches).")
        for candidate, report in rows[:12]:
            cells = "  ".join(f"{t} {r['precision']:.3f}/{r['coverage']:.2f}" for t, r in report.items())
            changed = {k: getattr(candidate, k) for k in GRID if getattr(candidate, k) != getattr(LPConfig(), k)}
            print(f"  {cells}   {changed or 'defaults'}")
        best = {task: max(rows, key=lambda row: row[1].get(task, {}).get("score_if_abstain_wrong", -1))
                for task in TASKS if task in rows[0][1]}
        for task, (candidate, report) in best.items():
            changed = {k: getattr(candidate, k) for k in GRID if getattr(candidate, k) != getattr(LPConfig(), k)}
            print(f"best for {task} ({PRIMARY_METRIC[task]}, abstain = wrong): "
                  f"{report[task]['score_if_abstain_wrong']:.3f}  {changed or 'defaults'}")
        if args.best_json:
            best_config, best_report = rows[0]
            write_json(args.best_json, {"config": {k: getattr(best_config, k) for k in GRID}, "report": best_report,
                                        "settings": [f"{k}={str(getattr(best_config, k)).lower()}" for k in GRID]})
            print(f"Best overall convention written to {args.best_json}")
    records = solve_all(data, worlds, forms, config, args.source)
    if any("answer" in q for _, _, q in iter_questions(data)):
        for task, values in coverage_report(data, records).items():
            print(f"{task}: answered {values['answered']}/{values['parsed']}, "
                  f"precision {values['precision']:.3f}, abstain = wrong {values['score_if_abstain_wrong']:.3f}")
    if args.output:
        if args.output.exists():
            args.output.unlink()
        PredictionWriter(args.output).write_many(list(records.values()))
        print(f"Wrote {len(records)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
