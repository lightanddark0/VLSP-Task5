import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from spartqa.data import read_json, write_json
from spartqa.metrics import answer_matches, finalize_metrics, set_jaccard, update_metric_state


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


@unittest.skipUnless(shutil.which("git"), "Git is needed to validate publication rules")
class PublicationTests(unittest.TestCase):
    def test_gitignore_excludes_private_artifacts_but_keeps_source(self):
        ignored = [
            ".env", ".env.production", "nested/.env", "outputs/run/config.json",
            "outputs/run/metrics.json", "outputs/run/responses.jsonl", "outputs/model.pt",
            "Data/human_train.json", "Data/auto_train.json", "Data/human_public_test.json",
            "__pycache__/module.pyc", ".venv/pyvenv.cfg", "Untitled.ipynb",
            "weights/model.safetensors", ".jupyter/settings.json", "Docs/reference.pdf",
            "Data/splits/human_train.json", "Data/splits/auto_dev.json", ".lh/.lhignore",
        ]
        included = [
            ".env.example", ".gitignore", "readme.md", "gpt_experiment.py", "XLNER.py",
            "spartqa/api.py", "spartqa/data.py", "spartqa/metrics.py", "test_spartqa.py",
            "test_gpt_experiment.py", "Data/README.md", "Docs/spartqa_cot.txt",
            "Data/splits/manifest.json", "make_splits.py", "evaluate.py", "validate_submission.py",
            "spartqa/splits.py", "spartqa/submission.py", "spartqa/repro.py", "test_evaluation.py",
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