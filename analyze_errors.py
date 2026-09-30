"""Step 1: where the Human ensemble still fails, and why (CPU only, no model is run).

    python analyze_errors.py --pred-dir outputs/predictions_e4 --split trdev \\
        --branches "F=F2C,F1 S=S L=LCOT P=LPX" --lp-source LPX --lp-settings outputs/e4/lp_choice.json \\
        --output-dir outputs/analysis

Uses the out-of-fold ensemble (same folds as compare_branches "cv") and every
source's own answers to report, per question type:
  1. accuracy, errors, and how many errors some single source got right
     (recoverable by a better combination) versus none (needs a new idea);
  2. FR error kinds: wrong direction, missing/extra near-far, near<->far, touch, DK;
  3. YN and CO confusion matrices, and accuracy by reasoning_type;
  4. for the LP source, whether its wrong answers come from voting (some
     (world, form) pair was right), from the rule form, or are systematic.
Writes report.md and errors.csv (every wrong question with the story, gold and
each source's answer) for manual review, e.g. gold labels that contradict the story.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from solve_parsed import build_solvers, parse_setting, question_forms, read_parses
from spartqa.analysis import load_setup, oof_ensemble, source_answer, story_of
from spartqa.data import TASKS
from spartqa.metrics import PRIMARY_METRIC, answer_matches
from spartqa.postprocess import PostprocessOptions
from spartqa.symbolic.human_forms import rule_form
from spartqa.symbolic.structured import LPConfig, StructureError, parse_form

DIRECTIONS, DISTANCE = {0, 1, 2, 3}, {4, 5}


def fr_error_kind(prediction: list[int] | None, gold: list[int]) -> str:
    if prediction is None:
        return "abstain"
    pred, true = set(prediction), set(gold)
    if pred == true:
        return "đúng"
    if 7 in pred or 7 in true:
        return "DK (7)"
    if pred & DIRECTIONS != true & DIRECTIONS:
        return "sai hướng"
    pd, gd = pred & DISTANCE, true & DISTANCE
    if pd != gd:
        if pd and gd:
            return "đổi gần↔xa"
        return "thiếu gần/xa" if gd else "thừa gần/xa"
    return "chạm (6)"


def lp_diagnosis(worlds: list[Any], forms: list[Any], question: dict[str, Any], config: LPConfig) -> str:
    """Why LP is wrong on a question: which (world, form) pairs would have been right."""
    solvers = build_solvers(worlds, config)
    llm = question_forms(forms, question, replace(config, forms_mode="llm"))
    try:
        rule = rule_form(question)
        rule = parse_form(rule, question["q_type"]) if rule is not None else None
    except (StructureError, KeyError, TypeError):
        rule = None

    def correct_share(candidates: list[Any]) -> float | None:
        answers = []
        for solver, form in itertools.product(solvers, candidates):
            try:
                answers.append(solver.answer(form, question))
            except (StructureError, KeyError, TypeError, IndexError):
                answers.append(None)
        usable = [a for a in answers if a is not None]
        if not usable:
            return None
        return sum(answer_matches(question["q_type"], a, question["answer"]) for a in usable) / len(usable)

    if not solvers:
        return "không đọc được world"
    llm_share, rule_share = correct_share(llm), correct_share([rule]) if rule is not None else None
    if llm_share is None and rule_share is None:
        return "không đọc được câu hỏi"
    if rule_share == 1.0 and (llm_share or 0) < 0.5:
        return "dạng luật đúng, LLM sai"
    if (llm_share or 0) > 0 or (rule_share or 0) > 0:
        return "có cặp đúng (bỏ phiếu/chép không ổn định)"
    return "mọi cặp sai (quy ước, chép sai nhất quán, hoặc gold)"


def fmt_answer(answer: Any) -> str:
    return "–" if answer is None else json.dumps(answer, ensure_ascii=False)


def table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("Data/splits"))
    parser.add_argument("--split", default="trdev")
    parser.add_argument("--branches", default="F=F2C,F1 S=S L=LCOT P=LPX")
    parser.add_argument("--lp-source", default="LPX", help="LP source to diagnose with its saved parses ('' to skip).")
    parser.add_argument("--lp-parses", type=Path, help="Default <pred-dir>/LP/human_<split>.parses.jsonl.")
    parser.add_argument("--lp-settings", type=Path, help="JSON with a 'settings' list (e.g. outputs/e4/lp_choice.json).")
    parser.add_argument("--set", type=parse_setting, action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--human-yn-dk", choices=("keep", "no"), default="keep")
    parser.add_argument("--human-fr-dk", choices=("keep", "avoid"), default="keep",
                        help="avoid: Human FR [7] becomes the relations some source scored.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    args = parser.parse_args(argv)

    options = PostprocessOptions(human_yn_dk=args.human_yn_dk, human_fr_dk=args.human_fr_dk)
    setup = load_setup(args.pred_dir, args.splits_dir, args.split, args.branches)
    questions, sources, names = setup["questions"], setup["sources"], setup["names"]
    dataset, fallback = setup["dataset"], setup["fallback"]
    print(f"{len(questions)} labeled questions; sources {names}")
    ensemble = oof_ensemble(questions, sources, names, dataset, options, fallback)
    own = {name: {key: source_answer(sources[name].get(key), q, dataset, options, fallback)
                  for key, q in questions} for name in names}

    # LP diagnostics from the saved parses (split "trdev" = train stories then dev stories)
    lp_notes: dict[str, str] = {}
    if args.lp_source and args.lp_source in names:
        config = replace(LPConfig(), **dict(args.set))
        if args.lp_settings:
            config = replace(config, **dict(parse_setting(s) for s in json.loads(args.lp_settings.read_text())["settings"]))
        worlds, forms = {}, {}
        offset = 0
        for part in (["train", "dev"] if args.split == "trdev" else [args.split]):
            path = args.lp_parses if args.lp_parses and args.split != "trdev" else \
                args.pred_dir / "LP" / f"{dataset}_{part}.parses.jsonl"
            part_worlds, part_forms = read_parses(path)
            worlds.update({story + offset: value for story, value in part_worlds.items()})
            forms.update({f"{story_of(k) + offset}_{k.split('_', 1)[1]}": v for k, v in part_forms.items()})
            if part == "train":
                offset = len(json.loads((args.splits_dir / f"{dataset}_train.json").read_text(encoding="utf-8"))["data"])
        for key, q in questions:
            answer = own[args.lp_source][key]
            if answer is not None and answer_matches(q["q_type"], answer, q["answer"]):
                continue
            story = story_of(key)
            lp_notes[key] = (lp_diagnosis(worlds[story], forms[key], q, config)
                             if story in worlds and key in forms else "không có parses (story ví dụ trong prompt hoặc thiếu file)")

    report = [f"# Phân tích lỗi Human ({args.split}, {len(questions)} câu)\n",
              f"Ensemble ngoài fold (cùng fold với cv của compare_branches), nguồn: {', '.join(names)}.\n"]

    # 1. overview and recoverability
    rows, oracle_rows, rescue = [], [], defaultdict(Counter)
    for task in TASKS:
        subset = [(k, q) for k, q in questions if q["q_type"] == task]
        if not subset:
            continue
        right = [answer_matches(task, ensemble[k], q["answer"]) for k, q in subset]
        wrong = [(k, q) for (k, q), ok in zip(subset, right) if not ok]
        recoverable = 0
        for key, q in wrong:
            winners = [n for n in names if own[n][key] is not None and answer_matches(task, own[n][key], q["answer"])]
            recoverable += bool(winners)
            for name in winners:
                rescue[task][name] += 1
        rows.append([task, len(subset), f"{sum(right) / len(subset):.3f}", len(wrong), recoverable,
                     len(wrong) - recoverable, f"{(sum(right) + recoverable) / len(subset):.3f}"])
    total_wrong = sum(r[3] for r in rows)
    report += ["## 1. Tổng quan: ensemble sai ở đâu, có cứu được bằng cách gộp tốt hơn không\n",
               table(["Loại", "Số câu", "Ensemble", "Sai", "Có nguồn đúng", "Không nguồn nào đúng",
                      "Trần nếu luôn chọn đúng nguồn"], rows),
               f"\nTổng số câu sai: {total_wrong}. \"Có nguồn đúng\" là phần stacking/gộp tốt hơn có thể sửa; "
               "\"không nguồn nào đúng\" cần ý tưởng mới (hoặc là gold mâu thuẫn với câu chuyện).\n",
               table(["Loại"] + names, [[t] + [rescue[t][n] for n in names] for t in TASKS if t in rescue]),
               "\n(Số câu ensemble sai mà nguồn đó đúng.)\n"]

    # 1b. the two question batches
    from spartqa.analysis import group_of
    rows = []
    for group in ("A", "B"):
        subset = [(k, q) for k, q in questions if group_of(q) == group]
        cells = []
        for task in TASKS:
            part = [(k, q) for k, q in subset if q["q_type"] == task]
            cells.append(f"{sum(answer_matches(task, ensemble[k], q['answer']) for k, q in part) / max(len(part), 1):.3f}"
                         f" ({len(part)})")
        wrong = sum(not answer_matches(q["q_type"], ensemble[k], q["answer"]) for k, q in subset)
        rows.append([group, len(subset), *cells, wrong])
    report += ["## 1b. Theo nhóm câu hỏi (A: có reasoning_type, B: không có — chỉ để phân tích)\n",
               table(["Nhóm", "Số câu", *TASKS, "Sai"], rows), ""]

    # 2. each source alone
    rows = []
    for name in names:
        cells = []
        for task in TASKS:
            subset = [(k, q) for k, q in questions if q["q_type"] == task]
            answered = [(k, q) for k, q in subset if own[name][k] is not None]
            correct = sum(answer_matches(task, own[name][k], q["answer"]) for k, q in answered)
            cells.append(f"{correct / max(len(subset), 1):.3f} ({len(answered) / max(len(subset), 1):.0%})")
        rows.append([name] + cells)
    report += ["## 2. Từng nguồn một mình (abstain = sai; trong ngoặc: độ phủ)\n",
               table(["Nguồn"] + list(TASKS), rows), ""]

    # 3. FR error kinds
    fr = [(k, q) for k, q in questions if q["q_type"] == "FR"]
    kinds = ["đúng", "sai hướng", "thiếu gần/xa", "thừa gần/xa", "đổi gần↔xa", "chạm (6)", "DK (7)", "abstain"]
    rows = []
    for name, answers in [("Ensemble", ensemble)] + [(n, own[n]) for n in names if n != "S"]:
        counts = Counter(fr_error_kind(answers[k], q["answer"]) for k, q in fr)
        rows.append([name] + [counts[kind] for kind in kinds])
    report += [f"## 3. FR: kiểu lỗi ({len(fr)} câu)\n", table(["Nguồn"] + kinds, rows), ""]

    # 4. confusion matrices
    for task, labels in (("YN", ["Yes", "No", "DK"]), ("CO", [0, 1, 2, 3])):
        subset = [(k, q) for k, q in questions if q["q_type"] == task]
        matrix = Counter((q["answer"][0], ensemble[k][0] if ensemble[k] else None) for k, q in subset)
        report += [f"## 4. {task}: ma trận nhầm lẫn của ensemble (hàng = gold, cột = dự đoán)\n",
                   table(["gold \\ dự đoán"] + [str(l) for l in labels],
                         [[str(g)] + [matrix[(g, p)] for p in labels] for g in labels]), ""]

    # 5. reasoning types
    by_type = defaultdict(lambda: [0, 0])
    for key, q in questions:
        for rtype in q.get("reasoning_type") or ["(none)"]:
            by_type[(q["q_type"], rtype)][0] += 1
            by_type[(q["q_type"], rtype)][1] += answer_matches(q["q_type"], ensemble[key], q["answer"])
    rows = [[task, rtype, n, f"{ok / n:.3f}", n - ok] for (task, rtype), (n, ok) in
            sorted(by_type.items(), key=lambda item: (item[0][0], -(item[1][0] - item[1][1]))) if n >= 5]
    report += ["## 5. Theo reasoning_type (chỉ để phân tích, ≥ 5 câu)\n",
               table(["Loại", "reasoning_type", "Số câu", "Ensemble", "Sai"], rows), ""]

    # 6. LP diagnostics
    if lp_notes:
        rows = []
        for task in TASKS:
            counts = Counter(note for key, note in lp_notes.items()
                             if dict(questions)[key]["q_type"] == task)
            for note, count in counts.most_common():
                rows.append([task, note, count])
        report += [f"## 6. {args.lp_source} sai vì đâu (từ parses đã lưu)\n", table(["Loại", "Chẩn đoán", "Số câu"], rows), ""]

    # 7. CSV of every wrong question
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["key", "q_type", "question", "gold", "ensemble", *names, "fr_kind", "sources_right",
                         "lp_diagnosis", "reasoning_type", "story"])
        for key, q in questions:
            if answer_matches(q["q_type"], ensemble[key], q["answer"]):
                continue
            right = [n for n in names if own[n][key] is not None and answer_matches(q["q_type"], own[n][key], q["answer"])]
            writer.writerow([key, q["q_type"], q["question"], fmt_answer(q["answer"]), fmt_answer(ensemble[key]),
                             *[fmt_answer(own[n][key]) for n in names],
                             fr_error_kind(ensemble[key], q["answer"]) if q["q_type"] == "FR" else "",
                             ",".join(right) or "không", lp_notes.get(key, ""), ",".join(q.get("reasoning_type") or []),
                             " ".join(setup["stories"][story_of(key)]["story"])])
    mean = sum(answer_matches(q["q_type"], ensemble[k], q["answer"]) for k, q in questions) / len(questions)
    by_task = {t: sum(answer_matches(t, ensemble[k], q["answer"]) for k, q in questions if q["q_type"] == t)
               / max(sum(q["q_type"] == t for _, q in questions), 1) for t in TASKS}
    report.insert(2, f"**Ensemble (cv): trung bình 4 loại {sum(by_task.values()) / 4:.4f}** "
                     f"({', '.join(f'{t} {v:.3f}' for t, v in by_task.items())}); tỷ lệ đúng trên mọi câu {mean:.4f}.\n")
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"Wrote {args.output_dir / 'report.md'} and {args.output_dir / 'errors.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
