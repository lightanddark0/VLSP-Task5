"""Check a prediction file before submission, optionally filling invalid answers.

    python validate_submission.py outputs/run/human_public_test_predictions.json \
        --reference Data/human_public_test.json
    python validate_submission.py PRED --reference REF \
        --fill-from Data/human_train.json --output outputs/run/human_submission.json

Structure changes cannot be repaired. Missing or invalid answers can be filled
with the most frequent training answer of each question type; the filled
question positions are printed so that the fallback rate is visible.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from spartqa.data import read_json, write_json
from spartqa.submission import answer_problems, compare_structure, fill_invalid_answers, majority_answers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--reference", type=Path, required=True, help="The original test (or gold) JSON file.")
    parser.add_argument("--fill-from", type=Path, help="Labeled training file used for fallback answers.")
    parser.add_argument("--output", type=Path, help="Where to write the filled file (required with --fill-from).")
    args = parser.parse_args(argv)
    if args.fill_from and not args.output:
        parser.error("--fill-from requires --output")
    if args.output and args.output.resolve() in {args.prediction.resolve(), args.reference.resolve()}:
        parser.error("--output must not overwrite the prediction or reference file")

    prediction, reference = read_json(args.prediction), read_json(args.reference)
    structure = compare_structure(prediction, reference)
    if structure:
        print(f"INVALID: {len(structure)} structural differences from {args.reference}:")
        for problem in structure[:20]:
            print(f"  {problem}")
        return 1

    problems = answer_problems(prediction)
    total = sum(len(story["questions"]) for story in prediction["data"])
    if not problems:
        print(f"OK: {total} questions, structure preserved, every answer valid.")
        return 0
    print(f"{len(problems)}/{total} questions have missing or invalid answers:")
    for reason, count in Counter(problem for _, _, problem in problems).most_common():
        print(f"  {count:>6}  {reason}")
    if not args.fill_from:
        print("INVALID: pass --fill-from TRAIN --output OUT to fill them with fallback answers.")
        return 1

    fallback = majority_answers(read_json(args.fill_from))
    filled, keys = fill_invalid_answers(prediction, fallback)
    remaining = answer_problems(filled)
    if remaining:
        print(f"INVALID: {len(remaining)} answers still invalid after filling.")
        return 1
    write_json(args.output, filled)
    print(f"Fallback answers: {fallback}")
    print(f"Filled {len(keys)} questions (story:question): {', '.join(keys[:30])}{' ...' if len(keys) > 30 else ''}")
    print(f"OK: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
