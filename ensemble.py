"""Combine branch predictions, tune per question type on dev, and write the submission.

Backup submission from a single source:
    python ensemble.py --dataset auto --only F1 \\
        --source F1,outputs/pred/F1/auto_dev.jsonl,outputs/pred/F1/auto_test.jsonl
Tuned ensemble (source order = fallback priority):
    python ensemble.py --dataset human \\
        --source F2,outputs/pred/F2/human_dev.jsonl,outputs/pred/F2/human_test.jsonl \\
        --source LCOT,outputs/pred/LCOT/human_dev.jsonl,outputs/pred/LCOT/human_test.jsonl

For each question type the script tries every source alone and weighted votes
(weights from --weights, FR/FB thresholds from --thresholds) and keeps the
best dev configuration by the primary metric (ties: fewer sources). A source
that abstains is skipped; if all do, the first source with an answer is used,
then the training majority. Post-processing (FR constraints, Human YN DK
policy, and the optional ``indifinite`` rules) is applied during tuning and
on test. The submission keeps the test file's structure and is validated.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

from spartqa import hub, tracking
from spartqa.data import TASKS, read_json, write_json
from spartqa.inference import fallback_answers
from spartqa.metrics import PRIMARY_METRIC, answer_matches, detailed_metrics, score_records
from spartqa.postprocess import PostprocessOptions, finalize_answer
from spartqa.predictions import fill_answers, iter_questions, read_predictions, restore_field_order
from spartqa.submission import answer_problems, compare_structure
from spartqa.voting import combine_scores, decide


def parse_source(text: str) -> tuple[str, Path, Path]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3 or not parts[0]:
        raise argparse.ArgumentTypeError(f"Expected NAME,DEV.jsonl,TEST.jsonl but got {text!r}")
    return parts[0], Path(parts[1]), Path(parts[2])


def load_source(path: Path, name: str, split: str) -> dict[str, dict[str, Any]]:
    records = read_predictions(path)
    if not records:
        print(f"Warning: no {split} predictions for {name} at {path}")
    return records


def predict(key: str, question: dict[str, Any], sources: dict[str, dict[str, dict[str, Any]]],
            config: dict[str, Any], dataset: str, options: PostprocessOptions,
            fallback: dict[str, list[Any]]) -> list[Any]:
    task = question["q_type"]
    scores = combine_scores(task, [(config["weights"].get(name, 0), records.get(key))
                                   for name, records in sources.items()])
    if scores is None:
        for records in sources.values():
            record = records.get(key)
            if record is not None and record.get("answer") is not None:
                scores = record.get("scores") or {}
                break
    answer = decide(task, scores, config["threshold"]) if scores else None
    return finalize_answer(question, answer, scores, dataset, options, fallback)


def candidate_configs(names: list[str], task: str, grid: list[float], thresholds: list[float]) -> list[dict]:
    """Every weight vector and threshold, simplest first (fewer sources, default threshold 0.5)."""
    configs = []
    task_thresholds = thresholds if task in {"FR", "FB"} else [0.5]
    for weights in itertools.product(grid, repeat=len(names)):
        if not any(weights):
            continue
        active = [(name, weight) for name, weight in zip(names, weights) if weight > 0]
        if len(active) == 1 and active[0][1] != max(grid):
            continue  # a lone source gives the same answers at any positive weight
        for threshold in task_thresholds:
            complexity = (len(active), threshold != 0.5, sum(weights),
                          [names.index(name) for name, _ in active])
            configs.append({"weights": dict(active), "threshold": threshold, "_order": complexity})
    configs.sort(key=lambda config: config["_order"])
    for config in configs:
        del config["_order"]
    return configs


def score_config(questions: list[tuple[str, dict[str, Any]]], sources: dict, config: dict, dataset: str,
                 options: PostprocessOptions, fallback: dict) -> float:
    if not questions:
        return 0.0
    correct = sum(answer_matches(q["q_type"], predict(key, q, sources, config, dataset, options, fallback),
                                 q["answer"]) for key, q in questions)
    return correct / len(questions)


def full_report(data: dict[str, Any], sources: dict, configs: dict[str, dict], dataset: str,
                options: PostprocessOptions, fallback: dict) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    rows, answers = [], {}
    for key, _, question in iter_questions(data):
        answer = predict(key, question, sources, configs[question["q_type"]], dataset, options, fallback)
        answers[key] = answer
        if "answer" in question:
            rows.append({"story_index": int(key.split("_")[0]), "task": question["q_type"],
                         "reasoning_types": list(question.get("reasoning_type") or []),
                         "prediction": answer, "gold": question["answer"]})
    report = score_records(rows) if rows else {}
    if rows:
        report["details"] = detailed_metrics(rows)
    return report, answers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("human", "auto"), required=True)
    parser.add_argument("--source", type=parse_source, action="append", required=True,
                        help="NAME,DEV.jsonl,TEST.jsonl; order is the fallback priority.")
    parser.add_argument("--dev-gold", type=Path, help="Default Data/splits/<dataset>_dev.json.")
    parser.add_argument("--test-input", type=Path, help="Default Data/<dataset>_public_test.json.")
    parser.add_argument("--splits-dir", type=Path, default=Path("Data/splits"))
    parser.add_argument("--only", help="Use this one source for every type (no tuning), e.g. a backup.")
    parser.add_argument("--weights", default="0,1,2", help="Weight grid per source.")
    parser.add_argument("--thresholds", default="0.4,0.5,0.6", help="FR/FB label thresholds.")
    parser.add_argument("--use-indifinite", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply the indifinite -> DK rules (only if the organizers allow the field).")
    parser.add_argument("--human-yn-dk", choices=("keep", "no"), default="keep",
                        help="Human YN: keep DK answers, or map them to the likelier of Yes/No.")
    parser.add_argument("--fr-constraints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, help="Default outputs/submission/<dataset>.")
    parser.add_argument("--submission-name", help="Default <dataset>_submission.json.")
    parser.add_argument("--results-repo", help="Private Hub dataset repo for the submission and config.")
    parser.add_argument("--results-prefix", default="submissions")
    parser.add_argument("--final", action="store_true", help="Tag the W&B run as the final submission.")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)

    dataset = args.dataset
    dev_path = args.dev_gold or args.splits_dir / f"{dataset}_dev.json"
    test_path = args.test_input or Path("Data") / f"{dataset}_public_test.json"
    output_dir = args.output_dir or Path("outputs/submission") / dataset
    submission_name = args.submission_name or f"{dataset}_submission.json"
    names = [name for name, _, _ in args.source]
    if args.only and args.only not in names:
        parser.error(f"--only {args.only} is not one of the sources {names}")
    options = PostprocessOptions(use_indifinite=args.use_indifinite, human_yn_dk=args.human_yn_dk,
                                 fr_constraints=args.fr_constraints)
    fallback = fallback_answers(args.splits_dir, dataset)
    dev_sources = {name: load_source(dev, name, "dev") for name, dev, _ in args.source}
    test_sources = {name: load_source(test, name, "test") for name, _, test in args.source}
    dev_data, test_data = read_json(dev_path), read_json(test_path)

    name = args.run_name or tracking.run_name("ensemble", "ensemble", dataset, "test",
                                              args.only or "+".join(names))
    tracking.init_run("ensemble", "ensemble", name, [dataset, "test"] + (["final"] if args.final else []), {
        "sources": {n: {"dev": str(d), "test": str(t)} for n, d, t in args.source}, "only": args.only,
        "postprocess": options.as_dict(), "weights_grid": args.weights, "thresholds": args.thresholds,
        "dev_gold": str(dev_path), "test_input": str(test_path)})

    # Tune only on dev questions that every active source predicted, so partial runs compare fairly.
    dev_questions = {task: [] for task in TASKS}
    for key, _, question in iter_questions(dev_data):
        if "answer" in question and all(key in records for records in dev_sources.values()):
            dev_questions[question["q_type"]].append((key, question))
    grid = [float(value) for value in args.weights.split(",")]
    thresholds = [float(value) for value in args.thresholds.split(",")]
    chosen: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        if args.only:
            chosen[task] = {"weights": {args.only: 1.0}, "threshold": 0.5}
            continue
        best, best_score = None, -1.0
        for config in candidate_configs(names, task, grid, thresholds):
            value = score_config(dev_questions[task], dev_sources, config, dataset, options, fallback)
            if value > best_score + 1e-12:
                best, best_score = config, value
        chosen[task] = best
        print(f"{task}: {best} -> dev {PRIMARY_METRIC[task]} {best_score:.4f} (n={len(dev_questions[task])})")

    dev_report, dev_answers = full_report(dev_data, dev_sources, chosen, dataset, options, fallback)
    ablation = []
    variants = [(f"only {n}", {t: {"weights": {n: 1.0}, "threshold": 0.5} for t in TASKS}, options) for n in names]
    variants.append(("chosen", chosen, options))
    variants.append(("chosen, no FR constraints", chosen,
                     PostprocessOptions(options.use_indifinite, options.human_yn_dk, False)))
    variants.append((f"chosen, use_indifinite={not options.use_indifinite}", chosen,
                     PostprocessOptions(not options.use_indifinite, options.human_yn_dk, options.fr_constraints)))
    if dataset == "human":
        other = "no" if options.human_yn_dk == "keep" else "keep"
        variants.append((f"chosen, human_yn_dk={other}", chosen,
                         PostprocessOptions(options.use_indifinite, other, options.fr_constraints)))
    for label, configs, variant_options in variants:
        report, _ = full_report(dev_data, dev_sources, configs, dataset, variant_options, fallback)
        if report:
            ablation.append({"variant": label, "primary_macro": report["primary_macro_unofficial"],
                             **{f"{task}/{metric}": value for task, values in report["by_task"].items()
                                for metric, value in values.items() if metric != "count"}})
    print("Dev ablation:")
    for row in ablation:
        print(f"  {row['variant']:<40} mean primary {row['primary_macro']:.4f}")

    _, test_answers = full_report(test_data, test_sources, chosen, dataset, options, fallback)
    submission = restore_field_order(fill_answers(test_data, test_answers), test_data)
    problems = compare_structure(submission, test_data)
    answer_errors = answer_problems(submission)
    if dataset == "human" and options.human_yn_dk == "no":
        answer_errors += [(s, q, "Human YN answered DK") for s, item in enumerate(submission["data"])
                          for q, question in enumerate(item["questions"]) if question.get("answer") == ["DK"]]
    missing = {n: sum(key not in records for key, _, _ in iter_questions(test_data))
               for n, records in test_sources.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{dataset}_dev_predictions.json",
               restore_field_order(fill_answers(dev_data, dev_answers), dev_data))
    write_json(output_dir / f"{dataset}_dev_eval.json", dev_report)
    write_json(output_dir / "ensemble_config.json", {
        "dataset": dataset, "sources": names, "only": args.only, "per_type": chosen,
        "postprocess": options.as_dict(), "test_missing_by_source": missing,
        "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    write_json(output_dir / "ablation.json", ablation)
    write_json(output_dir / submission_name, submission)

    print(f"Test questions without a prediction, by source: {missing}")
    if dev_report:
        print(f"Dev mean primary (unofficial): {dev_report['primary_macro_unofficial']:.4f}")
    tracking.update_config({"per_type_config": chosen})
    if dev_report:
        tracking.log_eval(dev_report, f"eval/{dataset}/dev")
    if ablation:
        columns = list(ablation[0])
        tracking.log_table(f"ablation/{dataset}", [[row.get(c) for c in columns] for row in ablation], columns)
    agree = {}
    for n, records in test_sources.items():
        answered = [(key, question) for key, _, question in iter_questions(test_data) if key in records]
        agree[f"test/{dataset}/agree_with_final/{n}"] = (
            sum(records[key].get("answer") is not None and
                answer_matches(question["q_type"], records[key]["answer"], test_answers[key])
                for key, question in answered) / len(answered) if answered else None)
    tracking.log(agree)

    if problems or answer_errors:
        print(f"INVALID submission: {len(problems)} structure problems, {len(answer_errors)} answer problems")
        for problem in (problems + [str(error) for error in answer_errors])[:20]:
            print(f"  {problem}")
        tracking.finish()
        return 1
    total = sum(len(item["questions"]) for item in submission["data"])
    print(f"OK: {output_dir / submission_name} ({total} questions, structure preserved, answers valid)")
    revision = hub.upload_folder(output_dir, hub.resolve_repo(args.results_repo) if args.results_repo
                                 and hub.hub_enabled() else None,
                                 f"{args.results_prefix}/{dataset}/{time.strftime('%Y%m%d-%H%M')}", "dataset",
                                 f"Submission {dataset}")
    if revision:
        tracking.log_hf_link(hub.resolve_repo(args.results_repo), revision, f"submission_{dataset}", "dataset")
        print(f"Uploaded to {args.results_repo} (commit {revision})")
    tracking.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
