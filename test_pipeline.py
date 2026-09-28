import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import ensemble
import explore
import prepare_sft
from spartqa.data import read_json, write_json
from spartqa.inference import parse_job, pending, score_predictions
from spartqa.postprocess import EMPTY, PostprocessOptions, finalize_answer
from spartqa.predictions import (PredictionWriter, fill_answers, iter_questions, one_hot_scores, question_key,
                                 read_predictions, restore_field_order)
from spartqa.prompting import (build_cot_messages, build_messages, option_lines, parse_answer, parse_cot_answer,
                               prompt_version)
from spartqa.submission import answer_problems, compare_structure
from spartqa.voting import combine_scores, decide, vote_samples

FR_CANDIDATES = ["bên trái", "bên phải", "bên trên", "bên dưới", "gần", "xa", "chạm vào", "DK"]
FALLBACK = {"YN": ["Yes"], "FR": [3], "FB": ["A"], "CO": [2]}


def question(q_id, q_type, answer=None, indifinite=False, candidates=None):
    default = {"YN": [], "FR": FR_CANDIDATES, "FB": ["A", "B", "C"], "CO": ["x", "y", "cả hai", "không"]}
    entry = {"q_id": q_id, "q_type": q_type, "reasoning_type": [], "indifinite": indifinite,
             "question": f"Câu hỏi {q_id}?", "candidate_answers": candidates or default[q_type]}
    if answer is not None:
        entry["answer"] = answer
    return entry


def story(index, labeled=True):
    answers = {"YN": ["Yes"] if index % 2 else ["No"], "FR": [0, 5], "FB": [] if index % 3 == 0 else ["B"],
               "CO": [index % 4]}
    return {"story": [f"Có ba khối A, B, C. Câu chuyện {index}."], "questions": [
        question(1, "YN", answers["YN"] if labeled else None),
        question(2, "YN", ["DK"] if labeled else None, indifinite=True),
        question(3, "FR", answers["FR"] if labeled else None),
        question(4, "FB", answers["FB"] if labeled else None),
        question(5, "CO", answers["CO"] if labeled else None),
    ]}


def dataset(count, labeled=True):
    return {"name": "SPaRTQA", "data": [story(index, labeled) for index in range(count)]}


class PromptTests(unittest.TestCase):
    def test_messages_list_options_per_type(self):
        payload = {"story": ["A ở bên trái B."], "question": "Q?", "q_type": "FR", "candidate_answers": FR_CANDIDATES}
        messages = build_messages(payload, "auto")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("0: bên trái | 1: bên phải", messages[1]["content"])
        self.assertIn("Bộ dữ liệu: auto", messages[1]["content"])
        self.assertEqual(option_lines({"q_type": "CO", "candidate_answers": ["x", "y"]}),
                         "0: x | 1: y | 2: cả hai | 3: không cái nào")
        self.assertEqual(option_lines({"q_type": "FB", "candidate_answers": ["A", "B"]}), "A, B")
        self.assertNotIn("indifinite", json.dumps(messages, ensure_ascii=False))
        self.assertTrue(prompt_version("F").startswith("F-"))

    def test_parse_answer_is_lenient_but_validates(self):
        fr = {"q_type": "FR", "candidate_answers": FR_CANDIDATES}
        fb = {"q_type": "FB", "candidate_answers": ["A", "B", "C"]}
        yn = {"q_type": "YN", "candidate_answers": []}
        self.assertEqual(parse_answer("[5, 0]", fr), [0, 5])
        self.assertEqual(parse_answer('Đáp án: ["2"]', fr), [2])
        self.assertEqual(parse_answer('["c", "a"]', fb), ["A", "C"])
        self.assertEqual(parse_answer("[]", fb), [])
        self.assertEqual(parse_answer('["yes"]', yn), ["Yes"])
        self.assertIsNone(parse_answer("[7, 1]", fr))
        self.assertIsNone(parse_answer('["D"]', fb))
        self.assertIsNone(parse_answer("không biết", yn))

    def test_cot_answer_uses_last_marker_after_thinking(self):
        yn = {"q_type": "YN", "candidate_answers": []}
        text = '<think>ĐÁP ÁN: ["No"] hmm</think>\nSuy luận...\nĐÁP ÁN: ["Yes"]'
        self.assertEqual(parse_cot_answer(text, yn), ["Yes"])
        self.assertIsNone(parse_cot_answer('<think>ĐÁP ÁN: ["No"]', yn))
        self.assertIsNone(parse_cot_answer("</think> không có đáp án", yn))
        payload = {"story": ["S"], "question": "Q?", "q_type": "YN", "candidate_answers": []}
        messages = build_cot_messages(payload, "human", [(payload, ["No"])])
        self.assertIn('ĐÁP ÁN: ["No"]', messages[1]["content"])


class PostprocessTests(unittest.TestCase):
    def test_fr_conflicts_and_dk_follow_scores(self):
        q = question(1, "FR")
        options = PostprocessOptions()
        self.assertEqual(finalize_answer(q, [0, 1, 5], {"0": 0.4, "1": 0.9, "5": 0.8}, "auto", options, FALLBACK),
                         [1, 5])
        self.assertEqual(finalize_answer(q, [7, 2], {"7": 0.9, "2": 0.6}, "auto", options, FALLBACK), [7])
        self.assertEqual(finalize_answer(q, [7, 2], {"7": 0.3, "2": 0.6}, "auto", options, FALLBACK), [2])
        self.assertEqual(finalize_answer(q, [], {"4": 0.2, "6": 0.3}, "auto", options, FALLBACK), [6])
        self.assertEqual(finalize_answer(q, None, None, "auto", options, FALLBACK), [3])
        unconstrained = PostprocessOptions(fr_constraints=False)
        self.assertEqual(finalize_answer(q, [0, 1], None, "auto", unconstrained, FALLBACK), [0, 1])

    def test_indifinite_rules_only_when_enabled(self):
        on, off = PostprocessOptions(use_indifinite=True), PostprocessOptions()
        dk_yn, known_yn = question(1, "YN", indifinite=True), question(2, "YN")
        self.assertEqual(finalize_answer(dk_yn, ["Yes"], None, "auto", on, FALLBACK), ["DK"])
        self.assertEqual(finalize_answer(dk_yn, ["Yes"], None, "auto", off, FALLBACK), ["Yes"])
        self.assertEqual(finalize_answer(known_yn, ["DK"], {"DK": 0.6, "No": 0.4}, "auto", on, FALLBACK), ["No"])
        self.assertEqual(finalize_answer(known_yn, ["DK"], None, "auto", off, FALLBACK), ["DK"])
        dk_fr, known_fr = question(3, "FR", indifinite=True), question(4, "FR")
        self.assertEqual(finalize_answer(dk_fr, [2], None, "auto", on, FALLBACK), [7])
        self.assertEqual(finalize_answer(known_fr, [7], {"7": 0.7, "0": 0.2}, "auto", on, FALLBACK), [0])
        self.assertEqual(finalize_answer(known_fr, [7], None, "auto", off, FALLBACK), [7])

    def test_human_yn_dk_policy_and_fb_order(self):
        known = question(1, "YN")
        self.assertEqual(finalize_answer(known, ["DK"], None, "human", PostprocessOptions(), FALLBACK), ["DK"])
        mapped = PostprocessOptions(human_yn_dk="no")
        self.assertEqual(finalize_answer(known, ["DK"], {"No": 0.3, "Yes": 0.1}, "human", mapped, FALLBACK), ["No"])
        self.assertEqual(finalize_answer(known, ["DK"], None, "auto", mapped, FALLBACK), ["DK"])
        fb = question(2, "FB", candidates=["A", "B"])
        self.assertEqual(finalize_answer(fb, ["B", "C", "A"], None, "auto", PostprocessOptions(), FALLBACK),
                         ["A", "B"])
        with self.assertRaises(ValueError):
            PostprocessOptions(human_yn_dk="maybe")


class VotingTests(unittest.TestCase):
    def test_vote_samples(self):
        answer, scores = vote_samples("YN", [["Yes"], ["No"], ["Yes"], None])
        self.assertEqual((answer, scores["Yes"]), (["Yes"], 2 / 3))
        answer, scores = vote_samples("FR", [[0, 5], [0], [0, 5], [1]])
        self.assertEqual(answer, [0, 5])
        answer, scores = vote_samples("FB", [[], [], ["A"]])
        self.assertEqual((answer, scores[EMPTY]), ([], 2 / 3))
        self.assertEqual(vote_samples("CO", [None, None]), (None, {}))

    def test_combine_skips_abstaining_sources(self):
        first = {"answer": [0], "scores": {"0": 1.0}}
        second = {"answer": [0, 5], "scores": {"0": 1.0, "5": 1.0}}
        scores = combine_scores("FR", [(1, first), (1, second), (2, {"answer": None, "scores": {}})])
        self.assertEqual(scores, {"0": 1.0, "5": 0.5})
        self.assertEqual(decide("FR", scores, 0.5), [0, 5])
        self.assertEqual(decide("FR", scores, 0.6), [0])
        self.assertIsNone(combine_scores("YN", [(0, first), (1, None)]))
        self.assertIsNone(decide("YN", {}, 0.5))
        self.assertEqual(decide("FB", {"B": 0.4, EMPTY: 0.6}, 0.5), [])


class PredictionFileTests(unittest.TestCase):
    def test_keys_resume_and_fill(self):
        data = dataset(2)
        keys = [key for key, _, _ in iter_questions(data)]
        self.assertEqual(keys[:2], [question_key(0, 1), "0_2"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.jsonl"
            PredictionWriter(path).write_many([{"key": "0_1", "answer": ["No"]}, {"key": "0_2", "answer": None}])
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"key": "0_3", "ans')
            with redirect_stdout(StringIO()):
                records = read_predictions(path)
                todo, done = pending(data, path, limit=4)
            self.assertEqual(set(records), {"0_1", "0_2"})
            self.assertEqual((done, [key for key, _, _ in todo]), (2, ["0_3", "0_4"]))
        filled = fill_answers(data, {"0_1": ["Yes"]})
        self.assertEqual(filled["data"][0]["questions"][0]["answer"], ["Yes"])
        self.assertNotIn("answer", filled["data"][0]["questions"][1])
        self.assertIn("answer", data["data"][0]["questions"][1])
        self.assertEqual(one_hot_scores("FB", []), {EMPTY: 1.0})
        ordered = restore_field_order(fill_answers(data, {"0_1": ["Yes"]}), data)
        self.assertEqual(list(ordered["data"][0]["questions"][0]), list(data["data"][0]["questions"][0]))

    def test_parse_job_and_scoring(self):
        job = parse_job("human,Data/splits/human_dev.json,out/h.jsonl")
        self.assertEqual((job.dataset, job.split), ("human", "dev"))
        self.assertEqual(parse_job("auto,Data/auto_public_test.json,o.jsonl").split, "test")
        data = dataset(1)
        records = {"0_1": {"answer": ["No"]}, "0_3": {"answer": [0, 1, 5], "scores": {"0": 1, "1": 0.2, "5": 1}}}
        report = score_predictions(data, records, "auto", PostprocessOptions(), FALLBACK)
        self.assertEqual(report["questions"], 2)
        self.assertEqual(report["by_task"]["FR"]["exact_match"], 1.0)
        self.assertIsNone(score_predictions(dataset(1, labeled=False), records, "auto", PostprocessOptions(),
                                            FALLBACK))


class ScriptTests(unittest.TestCase):
    def write_splits(self, root: Path) -> None:
        splits = root / "splits"
        for name, count in (("human_train", 6), ("human_dev", 3), ("auto_train", 12), ("auto_dev", 4)):
            write_json(splits / f"{name}.json", dataset(count))
        write_json(root / "human_public_test.json", dataset(3, labeled=False))

    def test_prepare_sft_writes_balanced_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_splits(root)
            with redirect_stdout(StringIO()):
                prepare_sft.main(["--stage", "auto", "--splits-dir", str(root / "splits"), "--n-samples", "20",
                                  "--dev-samples", "5", "--output-dir", str(root / "sft")])
                prepare_sft.main(["--stage", "human", "--splits-dir", str(root / "splits"), "--auto-mix", "4",
                                  "--human-repeat", "2", "--output-dir", str(root / "sft_h")])
            lines = (root / "sft/train.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 20)
            record = json.loads(lines[0])
            self.assertEqual(set(record), {"key", "dataset", "q_type", "messages", "target"})
            human = [json.loads(line) for line in (root / "sft_h/train.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual(sum(r["dataset"] == "human" for r in human), 2 * 6 * 5)
            self.assertEqual(sum(r["dataset"] == "auto" for r in human), 4)

    def test_explore_assumption_checks(self):
        checks = explore.check_assumptions({"human_train": dataset(4)})["human_train"]
        self.assertEqual(checks["fr_dk_with_other_labels"], 0)
        self.assertEqual(checks["indifinite_vs_dk_mismatches"], 0)
        self.assertFalse(checks["q_id_unique_in_file"])
        self.assertTrue(checks["q_id_unique_in_story"])

    def test_ensemble_tunes_on_dev_and_writes_valid_submission(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"WANDB_MODE": "disabled"}):
            root = Path(directory)
            self.write_splits(root)
            dev, test = read_json(root / "splits/human_dev.json"), read_json(root / "human_public_test.json")
            good = [{"key": k, "q_type": q["q_type"], "answer": q["answer"],
                     "scores": one_hot_scores(q["q_type"], q["answer"])} for k, _, q in iter_questions(dev)]
            wrong = {"YN": ["Yes"], "FR": [2], "FB": ["C"], "CO": None}
            bad = [{"key": k, "q_type": q["q_type"],
                    "answer": wrong[q["q_type"]] or [(q["answer"][0] + 1) % 4],
                    "scores": one_hot_scores(q["q_type"], wrong[q["q_type"]] or [(q["answer"][0] + 1) % 4])}
                   for k, _, q in iter_questions(dev)]
            test_records = [{"key": k, "q_type": q["q_type"], "answer": ["DK"] if q["q_type"] == "YN" else None,
                             "scores": {"DK": 1.0} if q["q_type"] == "YN" else {}}
                            for k, _, q in iter_questions(test)]
            for name, records in (("good_dev", good), ("bad_dev", bad), ("test", test_records)):
                PredictionWriter(root / f"{name}.jsonl").write_many(records)
            arguments = ["--dataset", "human", "--splits-dir", str(root / "splits"),
                         "--test-input", str(root / "human_public_test.json"),
                         "--source", f"BAD,{root / 'bad_dev.jsonl'},{root / 'test.jsonl'}",
                         "--source", f"GOOD,{root / 'good_dev.jsonl'},{root / 'test.jsonl'}",
                         "--output-dir", str(root / "out")]
            with redirect_stdout(StringIO()):
                self.assertEqual(ensemble.main(arguments + ["--use-indifinite"]), 0)
            config = read_json(root / "out/ensemble_config.json")
            self.assertEqual(config["per_type"]["CO"]["weights"], {"GOOD": 2.0})
            self.assertEqual(read_json(root / "out/human_dev_eval.json")["by_task"]["CO"]["accuracy"], 1.0)
            submission = read_json(root / "out/human_submission.json")
            self.assertEqual(compare_structure(submission, test), [])
            self.assertEqual(answer_problems(submission), [])
            answers = [q["answer"] for _, _, q in iter_questions(submission) if q["q_type"] == "YN"]
            self.assertEqual(answers[:2], [["Yes"], ["DK"]])  # indifinite=False forbids DK; True forces it
            with redirect_stdout(StringIO()):
                self.assertEqual(ensemble.main(arguments + ["--only", "GOOD", "--human-yn-dk", "no",
                                                            "--no-use-indifinite"]), 0)
            submission = read_json(root / "out/human_submission.json")
            self.assertNotIn(["DK"], [q["answer"] for _, _, q in iter_questions(submission)])


if __name__ == "__main__":
    unittest.main()


class HubVisibilityTests(unittest.TestCase):
    def test_visibility_is_set_only_when_requested(self):
        from unittest.mock import MagicMock
        from spartqa import hub
        api = MagicMock()
        with patch.dict("os.environ", {"HF_TOKEN": "x"}), patch("huggingface_hub.HfApi", return_value=api):
            hub.ensure_repo("user/adapter", "model", private=False)
            api.create_repo.assert_called_with("user/adapter", private=False, exist_ok=True, repo_type="model")
            api.update_repo_settings.assert_called_with("user/adapter", private=False, repo_type="model")
            api.reset_mock()
            hub.ensure_repo("user/results", "dataset")
            api.create_repo.assert_called_with("user/results", private=True, exist_ok=True, repo_type="dataset")
            api.update_repo_settings.assert_not_called()
