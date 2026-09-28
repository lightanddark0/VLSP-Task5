import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from spartqa.data import read_json, validate_answer, write_json
from spartqa.metrics import answer_matches, finalize_metrics, set_jaccard, update_metric_state
from spartqa.qwen_data import (
    assign_splits,
    build_manifest,
    build_prompt_messages,
    canonical_answer,
    completion_text,
    legal_answers,
    load_source_examples,
    repair_zero_coverage,
    story_fingerprint,
)


ROOT = Path(__file__).resolve().parent


class SharedUtilitiesTests(unittest.TestCase):
    def test_json_roundtrip_creates_parent_and_replaces_atomically(self):
        original = {"name": "SPaRTQA", "data": [{"story": ["Ti\u1ebfng Vi\u1ec7t"], "questions": []}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/data.json"
            write_json(path, original)
            self.assertEqual(read_json(path), original)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)
            write_json(path, {"data": []})
            self.assertEqual(read_json(path), {"data": []})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_encoder_metrics_keep_original_format(self):
        state = {}
        for task, prediction, gold in [
            ("YN", ["Yes"], ["Yes"]), ("YN", ["DK"], ["No"]),
            ("CO", [2], [2]), ("FR", [5, 2], [2, 5]), ("FR", [2], [2, 5]),
            ("FB", [], []), ("FB", ["A"], ["A", "B"]),
        ]:
            update_metric_state(state, task, prediction, gold)
        self.assertEqual(finalize_metrics(state), {
            "YN": {"accuracy": 0.5, "count": 2.0},
            "CO": {"accuracy": 1.0, "count": 1.0},
            "FR": {"exact_match": 0.5, "jaccard": 0.75, "count": 2.0},
            "FB": {"exact_match": 0.5, "jaccard": 0.75, "count": 2.0},
        })
        self.assertEqual(finalize_metrics({}), {})

    def test_per_question_scores(self):
        self.assertEqual(set_jaccard([], []), 1.0)
        self.assertEqual(set_jaccard([], ["A"]), 0.0)
        self.assertEqual(set_jaccard([0, 2], [2, 5]), 1 / 3)
        self.assertTrue(answer_matches("FR", [2, 0], [0, 2]))
        self.assertTrue(answer_matches("FB", ["B", "A"], ["A", "B"]))
        self.assertFalse(answer_matches("CO", [0], [1]))


def make_qwen_data_document(story_text: str, questions: list[dict]) -> dict:
    return {"name": "SPaRTQA", "data": [{"story": [story_text], "questions": questions}]}


def make_question(q_id: int, q_type: str, question: str, candidate_answers: list, answer: list) -> dict:
    return {"q_id": q_id, "q_type": q_type, "question": question, "candidate_answers": candidate_answers, "answer": answer}


FR_CANDIDATES = ["bên trái", "bên phải", "bên trên", "bên dưới", "gần", "xa", "chạm vào", "DK"]


class QwenDataContractTests(unittest.TestCase):
    def test_source_qualified_keys_are_unique_and_fingerprints_group_duplicates(self):
        human_data = make_qwen_data_document("Câu chuyện chung giống hệt nhau.", [
            make_question(1, "YN", "Có phải không?", [], ["DK"]),
        ])
        auto_data = make_qwen_data_document("Câu chuyện chung   giống hệt   nhau.", [
            make_question(1, "FR", "Quan hệ là gì?", FR_CANDIDATES, [0, 5]),
        ])
        human_examples = load_source_examples("human", human_data)
        auto_examples = load_source_examples("auto", auto_data)
        self.assertEqual(human_examples[0]["key"], "human:0:0")
        self.assertEqual(auto_examples[0]["key"], "auto:0:0")
        self.assertEqual(human_examples[0]["story_fingerprint"], auto_examples[0]["story_fingerprint"])
        self.assertEqual(story_fingerprint(["a  b"]), story_fingerprint(["a", "b"]))

    def test_split_is_deterministic_and_keeps_duplicate_groups_together(self):
        examples = []
        for index in range(6):
            shared_story = "Câu chuyện chung số không." if index < 2 else f"Câu chuyện riêng số {index}."
            source = "human" if index % 2 == 0 else "auto"
            data = make_qwen_data_document(shared_story, [make_question(1, "YN", "Hỏi?", [], ["Yes"])])
            examples.extend(load_source_examples(source, data))
        first = assign_splits(examples, seed=42, val_ratio=0.2)
        second = assign_splits(examples, seed=42, val_ratio=0.2)
        self.assertEqual(first, second)
        shared_keys = [example["key"] for example in examples if example["story_fingerprint"] == examples[0]["story_fingerprint"]]
        self.assertTrue(len(shared_keys) >= 2)
        self.assertEqual(len({first[key] for key in shared_keys}), 1)

    def test_repair_zero_coverage_prefers_safe_group_and_reports_unsafe_cells(self):
        human_data = {"name": "SPaRTQA", "data": [
            {"story": ["Câu chuyện một hai."], "questions": [
                make_question(1, "YN", "Hỏi một?", [], ["Yes"]),
                make_question(2, "FR", "Quan hệ một?", FR_CANDIDATES, [0]),
            ]},
            {"story": ["Câu chuyện ba."], "questions": [
                make_question(1, "YN", "Hỏi ba?", [], ["No"]),
            ]},
        ]}
        auto_data = make_qwen_data_document("Câu chuyện bốn.", [
            make_question(1, "YN", "Hỏi bốn?", [], ["DK"]),
        ])
        examples = load_source_examples("human", human_data) + load_source_examples("auto", auto_data)
        assignment = {example["key"]: "train" for example in examples}
        assignment["auto:0:0"] = "val"
        warnings = repair_zero_coverage(examples, assignment)
        self.assertEqual(assignment["human:1:0"], "val")
        self.assertIn("No validation coverage available for source=human task=FR", warnings)

    def test_gold_and_reasoning_metadata_are_absent_from_prompt(self):
        payload = {
            "story": ["Câu chuyện có tam giác."], "question": "Tam giác ở đâu?",
            "q_type": "FR", "candidate_answers": FR_CANDIDATES,
        }
        messages = build_prompt_messages(payload)
        rendered = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("reasoning_type", rendered)
        self.assertNotIn("indifinite", rendered)
        self.assertIn("JSON", rendered)

    def test_canonical_answer_orders_by_task_contract(self):
        payload = {"story": ["s"], "question": "q", "q_type": "FR", "candidate_answers": FR_CANDIDATES}
        self.assertEqual(canonical_answer([5, 0], payload), [0, 5])
        fb_payload = {"story": ["s"], "question": "q", "q_type": "FB", "candidate_answers": ["A", "B", "C"]}
        self.assertEqual(canonical_answer(["C", "A"], fb_payload), ["A", "C"])
        self.assertEqual(json.loads(completion_text([], fb_payload)), {"answer": []})
        yn_payload = {"story": ["s"], "question": "q", "q_type": "YN", "candidate_answers": []}
        self.assertEqual(json.loads(completion_text(["DK"], yn_payload)), {"answer": ["DK"]})
        with self.assertRaises(ValueError):
            canonical_answer([7, 0], payload)

    def test_legal_answers_are_all_valid_and_have_expected_counts(self):
        yn_payload = {"story": ["s"], "question": "q", "q_type": "YN", "candidate_answers": []}
        co_payload = {"story": ["s"], "question": "q", "q_type": "CO", "candidate_answers": []}
        fr_payload = {"story": ["s"], "question": "q", "q_type": "FR", "candidate_answers": FR_CANDIDATES}
        fb_payload = {"story": ["s"], "question": "q", "q_type": "FB", "candidate_answers": ["A", "B"]}
        for payload, expected_count in ((yn_payload, 3), (co_payload, 4), (fr_payload, 128), (fb_payload, 4)):
            answers = legal_answers(payload)
            self.assertEqual(len(answers), expected_count)
            self.assertEqual(len(answers), len({tuple(answer) for answer in answers}))
            for answer in answers:
                validate_answer(answer, payload)

    def test_build_manifest_is_deterministic_and_detects_conflicts(self):
        human_data = make_qwen_data_document("Câu chuyện xung đột.", [
            make_question(1, "YN", "Hỏi giống hệt?", [], ["Yes"]),
        ])
        auto_data = make_qwen_data_document("Câu chuyện xung đột.", [
            make_question(1, "YN", "Hỏi giống hệt?", [], ["No"]),
        ])
        with tempfile.TemporaryDirectory() as directory:
            human_path = Path(directory) / "human_train.json"
            auto_path = Path(directory) / "auto_train.json"
            write_json(human_path, human_data)
            write_json(auto_path, auto_data)
            first = build_manifest({"human": human_path, "auto": auto_path}, seed=42, val_ratio=0.2)
            second = build_manifest({"human": human_path, "auto": auto_path}, seed=42, val_ratio=0.2)
            self.assertEqual(first["assignment"], second["assignment"])
            self.assertEqual(len(first["label_conflicts"]), 1)
            self.assertEqual(first["sources"]["human"]["count"], 1)


@unittest.skipUnless(shutil.which("git"), "Git is needed to validate publication rules")
class PublicationTests(unittest.TestCase):
    def test_gitignore_excludes_private_artifacts_but_keeps_source(self):
        ignored = [
            ".env", ".env.production", "nested/.env", "outputs/run/config.json",
            "outputs/run/metrics.json", "outputs/run/responses.jsonl", "outputs/model.pt",
            "Data/human_train.json", "Data/auto_train.json", "Data/human_public_test.json",
            "__pycache__/module.pyc", ".venv/pyvenv.cfg", "Untitled.ipynb",
            "weights/model.safetensors", ".jupyter/settings.json", "Docs/reference.pdf",
        ]
        included = [
            ".env.example", ".gitignore", "readme.md", "gpt_experiment.py", "XLNER.py",
            "spartqa/api.py", "spartqa/data.py", "spartqa/metrics.py", "spartqa/qwen_data.py",
            "spartqa/qwen_model.py", "spartqa/qwen_dataset.py", "spartqa/qwen_eval.py", "spartqa/qwen_train.py",
            "qwen_finetune.py", "modal_app.py", "requirements-qlora.txt", "requirements-modal.txt",
            "Docs/qwen_qlora.md", "test_spartqa.py",
            "test_gpt_experiment.py", "Data/README.md", "Docs/spartqa_cot.txt",
            ".github/workflows/tests.yml", "requirements-api.txt", "requirements.txt",
        ]
        with tempfile.TemporaryDirectory() as directory:
            shutil.copyfile(ROOT / ".gitignore", Path(directory) / ".gitignore")
            subprocess.run(["git", "init", "--quiet", directory], check=True, capture_output=True)
            result = subprocess.run(
                ["git", "-C", directory, "check-ignore", "--no-index", "-z", "--stdin"],
                input=("\0".join(ignored + included) + "\0").encode("utf-8"), capture_output=True, check=True,
            )
        self.assertEqual(set(result.stdout.decode("utf-8").rstrip("\0").split("\0")), set(ignored))


if __name__ == "__main__":
    unittest.main()