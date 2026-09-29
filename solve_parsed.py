"""Branch LP (E3), CPU part: answer questions from saved LLM parses with the S reasoner.

llm_parse.py samples worlds (per story) and forms (per question) and saves them
in <output>.parses.jsonl. This script re-solves those parses under any answering
convention, without the GPU:

    # compare conventions on a labeled file (e.g. Human train); --grid-keys limits the search
    python solve_parsed.py --parses outputs/predictions/LP/human_train.parses.jsonl \\
        --input Data/splits/human_train.json --grid --best-json best.json
    # learn the Human near/far habits for FR (C1) with a chosen convention
    python solve_parsed.py --parses ... --input Data/splits/human_train.json --set cross_block_far=true \\
        --fit-fr-calibration fr_calibration.json
    # write predictions with a chosen convention (and optionally the calibration)
    python solve_parsed.py --parses outputs/predictions/LP/human_dev.parses.jsonl \\
        --input Data/splits/human_dev.json --set cross_block_far=true --fr-calibration fr_calibration.json \\
        --output outputs/predictions/LP/human_dev.jsonl

Each (world, form) pair gives one answer; answers are voted as in L-CoT
(YN/CO majority, FR/FB labels with at least half of the votes). A question
with no usable pair is abstained (answer null). Conventions (spartqa.symbolic.
structured.LPConfig) also choose whether sampled worlds are merged fact by fact
(world_merge) and whether the rule-based Human question parser is used (forms_mode).
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
from spartqa.symbolic.human_forms import rule_form
from spartqa.symbolic.structured import LPConfig, LPSolver, StructureError, merge_worlds, parse_form, world_from_json
from spartqa.voting import vote_samples

GRID = {"yn_mode": ["closed", "open"], "fr_multi": ["all", "any"], "fb_scope": ["global", "block"],
        "far_chain": [True, False], "cross_block_far": [False, True],
        "forms_mode": ["llm", "rule_first", "rule_plus"], "relax": ["none", "size", "size_color"],
        "definite": ["any", "all"], "fr_block_share": [False, True], "world_merge": ["none", "merged", "both"]}
FIRST_GRID = ("yn_mode", "fr_multi", "fb_scope", "far_chain", "cross_block_far")   # the E3 search


def read_parses(path: Path) -> tuple[dict[int, list[Any]], dict[str, list[Any]]]:
    """Worlds by story index and forms by question key (later lines replace earlier ones)."""
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


def build_solvers(worlds: list[Any], config: LPConfig, calibration: dict | None = None) -> list[LPSolver]:
    candidates = list(worlds) if config.world_merge != "merged" else []
    if config.world_merge in ("merged", "both"):
        merged = merge_worlds(worlds)
        if merged is not None:
            candidates.append(merged)
    solvers = []
    for world in candidates:
        try:
            solvers.append(LPSolver(world_from_json(world, config.rules()), config, calibration))
        except (StructureError, KeyError, TypeError):
            continue
    return solvers


def question_forms(forms: list[Any], question: dict[str, Any], config: LPConfig) -> list[dict[str, Any] | None]:
    """Parsed forms to vote over; the rule form (B1) counts as many votes as the LLM forms."""
    parsed = []
    for form in forms:
        try:
            parsed.append(parse_form(form, question["q_type"]))
        except (StructureError, KeyError, TypeError):
            continue
    if config.forms_mode == "llm" or "question" not in question:
        return parsed
    try:
        rule = rule_form(question)
        rule = parse_form(rule, question["q_type"]) if rule is not None else None
    except (StructureError, KeyError, TypeError):
        rule = None
    if rule is None:
        return parsed
    if config.forms_mode == "rule_first":
        return [rule]
    return parsed + [rule] * max(len(parsed), 1)


def answer_question(worlds: list[Any] | list[LPSolver], forms: list[Any], question: dict[str, Any],
                    config: LPConfig) -> tuple[list[Any] | None, dict[str, float], int]:
    """Vote over every usable (world, form) pair; returns answer, vote shares, number of votes."""
    solvers = worlds if worlds and isinstance(worlds[0], LPSolver) else build_solvers(worlds, config)
    samples = []
    for solver, form in itertools.product(solvers, question_forms(forms, question, config)):
        try:
            samples.append(solver.answer(form, question))
        except (StructureError, KeyError, TypeError, IndexError):
            samples.append(None)
    answer, scores = vote_samples(question["q_type"], samples)
    return answer, scores, sum(sample is not None for sample in samples)


def solve_all(data: dict[str, Any], worlds: dict[int, list[Any]], forms: dict[str, list[Any]],
              config: LPConfig, source: str = "LP", calibration: dict | None = None,
              stories: set[int] | None = None) -> dict[str, dict[str, Any]]:
    records, solvers = {}, {}
    for key, _, question in iter_questions(data):
        story = int(key.split("_")[0])
        if story not in worlds or key not in forms or (stories is not None and story not in stories):
            continue
        if story not in solvers:
            solvers[story] = build_solvers(worlds[story], config, calibration)
        answer, scores, votes = answer_question(solvers[story], forms[key], question, config)
        records[key] = {"key": key, "q_type": question["q_type"], "answer": answer, "scores": scores,
                        "source": source, "votes": votes}
    return records


def fit_fr_calibration(data: dict[str, Any], worlds: dict[int, list[Any]], forms: dict[str, list[Any]],
                       config: LPConfig, min_support: float = 3.0,
                       stories: set[int] | None = None) -> dict[str, dict[str, float]]:
    """C1: for each FR situation (same/cross block, direction known, near/far stated or derived), the
    share of gold answers containing near (4) and far (5). Each question counts once, split over its samples."""
    totals: dict[str, dict[str, float]] = {}
    for key, _, question in iter_questions(data):
        story = int(key.split("_")[0])
        if question["q_type"] != "FR" or "answer" not in question or story not in worlds or key not in forms:
            continue
        if stories is not None and story not in stories:
            continue
        keys = []
        usable = [form for form in question_forms(forms[key], question, config) if form is not None]
        for solver, form in itertools.product(build_solvers(worlds[story], config), usable):
            try:
                pairs = solver.fr_pairs(form)
                if pairs:
                    keys.append(solver.fr_key(solver.fr(form, question) or [], pairs))
            except (StructureError, KeyError, TypeError, IndexError):
                continue
        for situation in keys:
            entry = totals.setdefault(situation, {"n": 0.0, "4": 0.0, "5": 0.0})
            entry["n"] += 1 / len(keys)
            for label in ("4", "5"):
                entry[label] += (int(label) in question["answer"]) / len(keys)
    return {situation: {"n": entry["n"], "4": entry["4"] / entry["n"], "5": entry["5"] / entry["n"]}
            for situation, entry in totals.items() if entry["n"] >= min_support}


def oof_calibrated(data: dict[str, Any], worlds: dict[int, list[Any]], forms: dict[str, list[Any]],
                   config: LPConfig, folds: int, source: str) -> dict[str, dict[str, Any]]:
    """Answers where each story's FR table is learned on the other folds (fair scores on a training file)."""
    stories = sorted(worlds)
    records = {}
    for k in range(folds):
        held = set(stories[k::folds])
        table = fit_fr_calibration(data, worlds, forms, config, stories=set(stories) - held)
        records.update(solve_all(data, worlds, forms, config, source, table, stories=held))
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


def mean_score(report: dict[str, dict[str, float]]) -> float:
    return sum(r["score_if_abstain_wrong"] for r in report.values()) / max(len(report), 1)


def parse_setting(text: str) -> tuple[str, Any]:
    name, _, value = text.partition("=")
    kinds = {f.name: f.type for f in fields(LPConfig)}
    if name not in kinds:
        raise argparse.ArgumentTypeError(f"unknown setting {name}; known: {sorted(kinds)}")
    if value.lower() in ("true", "false"):
        return name, value.lower() == "true"
    return name, value


def settings_of(config: LPConfig) -> list[str]:
    return [f"{f.name}={str(getattr(config, f.name)).lower()}" for f in fields(LPConfig)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parses", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--set", type=parse_setting, action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--grid", action="store_true", help="Score conventions (needs gold answers).")
    parser.add_argument("--grid-keys", default=",".join(FIRST_GRID),
                        help=f"Comma-separated switches searched by --grid (others keep --set); all: {','.join(GRID)}.")
    parser.add_argument("--best-json", type=Path, help="With --grid: write the best convention here (abstain = wrong).")
    parser.add_argument("--fit-fr-calibration", type=Path, help="Learn the FR near/far table (C1) on --input.")
    parser.add_argument("--fr-calibration", type=Path, help="Apply a learned FR near/far table when answering.")
    parser.add_argument("--oof-calibration", type=int, default=0, metavar="K",
                        help="Answer --input with FR tables learned out of fold (K story folds of --input itself).")
    parser.add_argument("--output", type=Path, help="Write predictions (JSONL) for the --set convention.")
    parser.add_argument("--source", default="LP")
    args = parser.parse_args(argv)

    data = read_json(args.input)
    worlds, forms = read_parses(args.parses)
    config = replace(LPConfig(), **dict(args.set))
    calibration = read_json(args.fr_calibration) if args.fr_calibration else None
    if args.grid:
        keys = [key.strip() for key in args.grid_keys.split(",") if key.strip()]
        unknown = [key for key in keys if key not in GRID]
        if unknown:
            parser.error(f"unknown grid keys {unknown}")
        rows = []
        for values in itertools.product(*(GRID[key] for key in keys)):
            candidate = replace(config, **dict(zip(keys, values)))
            report = coverage_report(data, solve_all(data, worlds, forms, candidate, calibration=calibration))
            if not report:
                parser.error("no labeled questions with parses")
            rows.append((candidate, report))
        rows.sort(key=lambda row: -mean_score(row[1]))
        print(f"{len(rows)} conventions. Per type: precision on answered / coverage; mean = abstain counted wrong.")
        for candidate, report in rows[:12]:
            cells = "  ".join(f"{t} {r['precision']:.3f}/{r['coverage']:.2f}" for t, r in report.items())
            changed = {k: getattr(candidate, k) for k in keys if getattr(candidate, k) != getattr(config, k)}
            print(f"  mean {mean_score(report):.4f}  {cells}   {changed or 'as --set'}")
        for task in TASKS:
            if task in rows[0][1]:
                candidate, report = max(rows, key=lambda row: row[1].get(task, {}).get("score_if_abstain_wrong", -1))
                changed = {k: getattr(candidate, k) for k in keys if getattr(candidate, k) != getattr(config, k)}
                print(f"best for {task} ({PRIMARY_METRIC[task]}, abstain = wrong): "
                      f"{report[task]['score_if_abstain_wrong']:.3f}  {changed or 'as --set'}")
        if args.best_json:
            best_config, best_report = rows[0]
            write_json(args.best_json, {"config": {f.name: getattr(best_config, f.name) for f in fields(LPConfig)},
                                        "report": best_report, "mean": mean_score(best_report),
                                        "settings": settings_of(best_config)})
            print(f"Best convention written to {args.best_json}")
    if args.fit_fr_calibration:
        table = fit_fr_calibration(data, worlds, forms, config)
        write_json(args.fit_fr_calibration, table)
        print(f"FR near/far table ({len(table)} situations) written to {args.fit_fr_calibration}")
        for situation, entry in sorted(table.items(), key=lambda item: -item[1]["n"]):
            print(f"  {situation:<28} n={entry['n']:5.1f}  P(near)={entry['4']:.2f}  P(far)={entry['5']:.2f}")
    if args.oof_calibration:
        records = oof_calibrated(data, worlds, forms, config, args.oof_calibration, args.source)
    else:
        records = solve_all(data, worlds, forms, config, args.source, calibration)
    if any("answer" in q for _, _, q in iter_questions(data)):
        report = coverage_report(data, records)
        for task, values in report.items():
            print(f"{task}: answered {values['answered']}/{values['parsed']}, "
                  f"precision {values['precision']:.3f}, abstain = wrong {values['score_if_abstain_wrong']:.3f}")
        print(f"mean (abstain = wrong): {mean_score(report):.4f}")
    if args.output:
        if args.output.exists():
            args.output.unlink()
        PredictionWriter(args.output).write_many(list(records.values()))
        print(f"Wrote {len(records)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
