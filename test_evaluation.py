import copy
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import evaluate
import make_splits
import validate_submission
from spartqa.data import read_json, write_json
from spartqa.metrics import bootstrap_intervals, final_scores, question_records, score_records
from spartqa.repro import run_metadata, set_seed
from spartqa.splits import split_stories
from spartqa.submission import answer_problems, compare_structure, fill_invalid_answers, majority_answers


def story(index):
    return {
        "story": [f"Story {index}: block A contains a circle."],
        "questions": [
            {"q_id": 1, "q_type": "YN", "reasoning_type": ["Transitivity"], "indifinite": False,
             "question": "Is the circle in A?", "candidate_answers": [], "answer": ["Yes" if index % 3 else "No"]},
            {"q_id": 2, "q_type": "FR", "reasoning_type": [], "indifinite": False,
             "question": "Where is the circle?", "candidate_answers": list(range(8)), "answer": [0, 5]},
            {"q_id": 3, "q_type": "FB", "reasoning_type": ["Quantifier"], "indifinite": False,
             "question": "Which block has a square?", "candidate_answers": ["A", "B"], "answer": []},
            {"q_id": 4, "q_type": "CO", "reasoning_type": [], "indifinite": False,
             "question": "Circle or square?", "candidate_answers": ["circle", "square", "both", "none"],
             "answer": [index % 4]},
        ],
    }


def dataset(count=10):
    return {"name": "SPaRTQA", "data": [story(index) for index in range(count)]}


def without_answers(data):
    output = copy.deepcopy(data)
    for item in output["data"]:
        for question in item["questions"]:
            question.pop("answer")
    return output


def run_quietly(main, argv):
    with redirect_stdout(StringIO()) as stream:
        code = main(argv)
    return code, stream.getvalue()


class SplitTests(unittest.TestCase):
    def test_story_split_is_disjoint_complete_and_seeded(self):
        data = dataset(20)
        train, dev = split_stories(data, dev_ratio=0.25, seed=7)
        self.assertEqual(len(dev), 5)
        self.assertEqual(sorted(train + dev), list(range(20)))
        self.assertFalse(set(train) & set(dev))
        self.assertEqual((train, dev), split_stories(data, dev_ratio=0.25, seed=7))
        self.assertNotEqual(dev, split_stories(data, dev_ratio=0.25, seed=8)[1])
        self.assertEqual(len(split_stories(data, dev_stories=3)[1]), 3)
        with self.assertRaises(ValueError):
            split_stories(dataset(1))

    def test_manifest_rebuilds_identical_files_and_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "human.json", dataset(12))
            write_json(root / "auto.json", dataset(30))
            out = root / "splits"
            argv = ["--human", str(root / "human.json"), "--auto", str(root / "auto.json"), "--output-dir", str(out)]
            self.assertEqual(run_quietly(make_splits.main, argv)[0], 0)
            manifest = read_json(out / "manifest.json")
            first = {path.name: path.read_bytes() for path in out.glob("*_*.json")}
            self.assertEqual(set(first), {"human_train.json", "human_dev.json", "auto_train.json", "auto_dev.json"})
            dev = read_json(out / "human_dev.json")
            self.assertEqual(len(dev["data"]), len(manifest["datasets"]["human"]["dev_story_indices"]))
            self.assertEqual(manifest["datasets"]["human"]["summary"]["dev"]["questions"], 4 * len(dev["data"]))

            with self.assertRaises(SystemExit), redirect_stdout(StringIO()), patch("sys.stderr"):
                make_splits.main(argv)
            for path in out.glob("*_*.json"):
                path.unlink()
            self.assertEqual(run_quietly(make_splits.main, ["--output-dir", str(out), "--from-manifest"])[0], 0)
            self.assertEqual({path.name: path.read_bytes() for path in out.glob("*_*.json")}, first)

            write_json(root / "human.json", dataset(13))
            with self.assertRaises(SystemExit), redirect_stdout(StringIO()), patch("sys.stderr"):
                make_splits.main(["--output-dir", str(out), "--from-manifest"])


class MetricTests(unittest.TestCase):
    def test_missing_and_invalid_predictions_score_zero(self):
        gold = dataset(1)
        prediction = copy.deepcopy(gold)
        questions = prediction["data"][0]["questions"]
        questions[0]["answer"] = ["yes"]  # wrong case is invalid
        questions[1]["answer"] = [5, 0]   # order does not matter
        del questions[2]["answer"]        # missing FB counts as wrong even though gold is []
        questions[3]["answer"] = ["0"]    # CO must be an integer
        report = score_records(question_records(prediction, gold))
        self.assertEqual(report["answered_valid"], 1)
        self.assertEqual(report["by_task"]["YN"]["accuracy"], 0.0)
        self.assertEqual(report["by_task"]["FR"], {"exact_match": 1.0, "jaccard": 1.0, "count": 1.0})
        self.assertEqual(report["by_task"]["FB"], {"exact_match": 0.0, "jaccard": 0.0, "count": 1.0})
        self.assertEqual(report["primary_macro_unofficial"], 0.25)
        self.assertEqual(set(report["by_reasoning_type"]), {"(none)", "Quantifier", "Transitivity"})

    def test_gold_as_prediction_is_perfect_and_final_score_averages_datasets(self):
        gold = dataset(4)
        perfect = score_records(question_records(gold, gold))
        self.assertEqual(perfect["primary_macro_unofficial"], 1.0)
        self.assertEqual(perfect["by_task"]["FB"]["jaccard"], 1.0)
        empty = score_records(question_records(without_answers(gold), gold))
        final = final_scores(perfect, empty)
        self.assertEqual(final["by_task"]["FR"], {"exact_match": 0.5, "jaccard": 0.5})
        self.assertEqual(final["primary_macro_unofficial"], 0.5)

    def test_bootstrap_is_seeded_and_bounded(self):
        gold = dataset(12)
        prediction = copy.deepcopy(gold)
        rng = random.Random(0)
        for item in prediction["data"]:
            item["questions"][0]["answer"] = [rng.choice(["Yes", "No"])]
        records = question_records(prediction, gold)
        first = bootstrap_intervals(records, samples=200, seed=3)
        self.assertEqual(first, bootstrap_intervals(records, samples=200, seed=3))
        low, high = first["YN"]
        self.assertTrue(0.0 <= low <= high <= 1.0)

    def test_evaluate_cli_rejects_modified_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = dataset(3)
            write_json(root / "gold.json", gold)
            write_json(root / "pred.json", gold)
            code, output = run_quietly(evaluate.main, ["--human", str(root / "pred.json"), str(root / "gold.json"),
                                                       "--auto", str(root / "pred.json"), str(root / "gold.json"),
                                                       "--output", str(root / "eval.json")])
            self.assertEqual(code, 0)
            self.assertIn("final = (human + auto) / 2", output)
            self.assertEqual(read_json(root / "eval.json")["final"]["primary_macro_unofficial"], 1.0)

            changed = copy.deepcopy(gold)
            changed["data"][1]["questions"][0]["question"] = "Edited?"
            write_json(root / "pred.json", changed)
            with self.assertRaises(SystemExit) as context, patch("sys.stderr"):
                evaluate.main(["--human", str(root / "pred.json"), str(root / "gold.json")])
            self.assertEqual(context.exception.code, 1)


class SubmissionTests(unittest.TestCase):
    def test_structure_ignores_answers_but_reports_other_changes(self):
        reference = without_answers(dataset(2))
        self.assertEqual(compare_structure(dataset(2), reference), [])
        changed = dataset(2)
        changed["data"][0]["questions"][1]["q_type"] = "YN"
        changed["data"][1]["story"] = ["Different"]
        problems = compare_structure(changed, reference)
        self.assertEqual(len(problems), 2)
        self.assertEqual(compare_structure(dataset(1), reference), ["expected 2 stories, found 1"])
        self.assertEqual(compare_structure([], reference), ["prediction must be a JSON object with a data list"])

    def test_answer_formats_and_fallback_filling(self):
        prediction = dataset(1)
        questions = prediction["data"][0]["questions"]
        questions[0]["answer"] = "Yes"      # not a list
        questions[1]["answer"] = [7, 0]     # DK mixed with a relation
        questions[2]["answer"] = ["C"]      # not a candidate block
        del questions[3]["answer"]
        self.assertEqual([problem[:2] for problem in answer_problems(prediction)], [(0, 0), (0, 1), (0, 2), (0, 3)])

        train = dataset(9)
        train["data"][0]["questions"][2]["answer"] = ["C"]
        train["data"][1]["questions"][2]["answer"] = ["C"]
        fallback = majority_answers(train)
        self.assertEqual(fallback["YN"], ["Yes"])
        self.assertEqual(fallback["FR"], [0, 5])
        fallback["FB"] = ["A", "C"]
        filled, keys = fill_invalid_answers(prediction, fallback)
        self.assertEqual(keys, ["0:0", "0:1", "0:2", "0:3"])
        self.assertEqual(filled["data"][0]["questions"][2]["answer"], ["A"])
        self.assertEqual(answer_problems(filled), [])
        self.assertIn("answer", questions[0])  # the input is not modified

    def test_validate_cli_fills_and_writes_a_valid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = without_answers(dataset(3))
            prediction = dataset(3)
            del prediction["data"][2]["questions"][0]["answer"]
            write_json(root / "ref.json", reference)
            write_json(root / "pred.json", prediction)
            write_json(root / "train.json", dataset(5))
            argv = [str(root / "pred.json"), "--reference", str(root / "ref.json")]
            self.assertEqual(run_quietly(validate_submission.main, argv)[0], 1)
            code, output = run_quietly(validate_submission.main, argv + [
                "--fill-from", str(root / "train.json"), "--output", str(root / "out.json")])
            self.assertEqual(code, 0)
            self.assertIn("Filled 1 questions", output)
            self.assertEqual(run_quietly(validate_submission.main, [str(root / "out.json"), "--reference",
                                                                    str(root / "ref.json")])[0], 0)


class ReproTests(unittest.TestCase):
    def test_seed_repeats_random_draws_and_metadata_is_recorded(self):
        set_seed(123)
        first = [random.random() for _ in range(3)]
        set_seed(123)
        self.assertEqual(first, [random.random() for _ in range(3)])
        info = run_metadata(seed=123)
        self.assertEqual(info["seed"], 123)
        self.assertIn("git_commit", info)
        self.assertIn("timestamp_utc", info)


if __name__ == "__main__":
    unittest.main()
