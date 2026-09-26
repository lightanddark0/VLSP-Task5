import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gpt_experiment as experiment


def sample_data():
    return {
        "name": "SPaRTQA",
        "data": [{
            "story": ["Block A contains a circle. Block B contains a square."],
            "questions": [
                {"q_id": 1, "q_type": "YN", "question": "Is there a circle in A?",
                 "candidate_answers": [], "answer": ["Yes"], "reasoning_type": ["Quantifier"], "indifinite": False},
                {"q_id": 2, "q_type": "FR", "question": "Where is A relative to B?",
                 "candidate_answers": list(range(8)), "answer": [7]},
                {"q_id": 3, "q_type": "FB", "question": "Which blocks contain a triangle?",
                 "candidate_answers": ["A", "B"], "answer": []},
                {"q_id": 4, "q_type": "CO", "question": "Which is in A: the circle or square?",
                 "candidate_answers": ["circle", "square", "both", "neither"], "answer": [0]},
            ],
        }],
    }


def fake_response(answer, finish_reason="stop"):
    return SimpleNamespace(
        id="test-response", model="fake-model",
        usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(refusal=None, content=json.dumps({"reasoning": "Evidence.", "answer": answer})),
        )],
    )


class GPTExperimentTests(unittest.TestCase):
    def test_api_errors_preserve_parameter_details_but_redact_secrets(self):
        error = ValueError("This exception text must not be logged")
        error.status_code = 400
        error.body = {"error": {
            "message": "Unsupported max_completion_tokens: test-private-value sk-test-value Bearer bearer-value",
            "code": "unsupported_parameter", "param": "max_completion_tokens",
            "headers": {"Authorization": "private"},
        }}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-private-value"}):
            details = experiment.api_error_details(error)
        self.assertEqual(details["status_code"], 400)
        self.assertEqual(details["param"], "max_completion_tokens")
        self.assertIn("Unsupported", details["message"])
        for secret in ("test-private-value", "sk-test-value", "bearer-value"):
            self.assertNotIn(secret, json.dumps(details))
        self.assertNotIn("headers", details)

    def test_payload_omits_all_gold_and_reasoning_metadata(self):
        for example in experiment.examples_from_data(sample_data()):
            self.assertEqual(set(example["payload"]), {"story", "question", "q_type", "candidate_answers"})
            self.assertNotIn("answer", example["payload"])
            self.assertNotIn("reasoning_type", example["payload"])
            self.assertNotIn("indifinite", example["payload"])

    def test_answer_validation(self):
        examples = experiment.examples_from_data(sample_data())
        invalid_answers = {"YN": [["yes"], [], ["Yes", "No"]], "FR": [[], [7, 1], [1, 1], ["1"], [True]],
                           "FB": [["C"], ["A", "A"]], "CO": [[4], ["0"], []]}
        for example in examples:
            for answer in invalid_answers[example["payload"]["q_type"]]:
                with self.subTest(answer=answer, task=example["payload"]["q_type"]):
                    with self.assertRaises(ValueError):
                        experiment.validate_answer(answer, example["payload"])

    def test_request_uses_typed_structured_output(self):
        config = {"model": "fake-model", "prompt": "Instructions", "temperature": 0, "max_output_tokens": 100}
        client = MagicMock()
        for example in experiment.examples_from_data(sample_data()):
            client.chat.completions.create.return_value = fake_response(example["gold"])
            record = experiment.request_answer(client, example, config)
            self.assertEqual(record["answer"], example["gold"])
            kwargs = client.chat.completions.create.call_args.kwargs
            self.assertEqual(json.loads(kwargs["messages"][1]["content"]), example["payload"])
            self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])
            self.assertEqual(record["usage"]["total_tokens"], 15)

    def test_incomplete_and_invalid_responses_are_not_fake_answers(self):
        example = experiment.examples_from_data(sample_data())[0]
        config = {"model": "fake", "prompt": "Instructions", "temperature": 0, "max_output_tokens": 100}
        client = MagicMock()
        for response in (fake_response(["Yes"], "length"), fake_response(["invalid"])):
            client.chat.completions.create.return_value = response
            record = experiment.request_answer(client, example, config)
            self.assertIn("error", record)
            self.assertNotIn("answer", record)
            self.assertEqual(record["usage"]["total_tokens"], 15)

    def test_metrics_and_prediction_export_preserve_source(self):
        data = sample_data()
        data["data"][0]["questions"][1]["answer"] = [2, 5]
        before = copy.deepcopy(data)
        examples = experiment.examples_from_data(data)
        records = [{"key": "0:0", "answer": ["Yes"]}, {"key": "0:1", "answer": [2]},
                   {"key": "0:2", "answer": []}, {"key": "0:3", "error": "APIError"}]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            report = experiment.export_results(data, examples, records, output_dir, "sample")
            output = experiment.read_json(output_dir / "sample_predictions.json")
        self.assertEqual(data, before)
        self.assertEqual(report["coverage"], 0.75)
        self.assertEqual(report["by_task"]["YN"]["accuracy"], 1.0)
        self.assertEqual(report["by_task"]["FR"]["exact_match"], 0.0)
        self.assertEqual(report["by_task"]["FR"]["jaccard"], 0.5)
        self.assertEqual(report["by_task"]["FB"]["jaccard"], 1.0)
        self.assertEqual(report["by_task"]["CO"]["accuracy"], 0.0)
        self.assertNotIn("answer", output["data"][0]["questions"][3])
        self.assertEqual(output["data"][0]["story"], data["data"][0]["story"])
        for original, prediction in zip(data["data"][0]["questions"], output["data"][0]["questions"]):
            self.assertEqual({key: value for key, value in original.items() if key != "answer"},
                             {key: value for key, value in prediction.items() if key != "answer"})

    def test_missing_fb_prediction_is_not_an_empty_set_match(self):
        data = sample_data()
        examples = experiment.examples_from_data(data)
        with tempfile.TemporaryDirectory() as directory:
            report = experiment.export_results(data, examples, [], Path(directory), "sample")
        self.assertEqual(report["by_task"]["FB"]["exact_match"], 0)
        self.assertEqual(report["by_task"]["FB"]["jaccard"], 0)

    def test_resume_skips_successful_calls_and_rejects_config_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment.write_json(root / "data.json", sample_data())
            args = ["gpt_experiment.py", "--input", str(root / "data.json"), "--output-dir", str(root / "run"), "--env-file", str(root / "absent.env")]
            client = MagicMock()
            client.__enter__.return_value = client
            client.chat.completions.create.side_effect = [fake_response(answer) for answer in (["Yes"], [7], [], [0])]
            fake_sdk = SimpleNamespace(OpenAI=MagicMock(return_value=client))
            with patch.dict("sys.modules", {"openai": fake_sdk}), patch.dict("os.environ", {"OPENAI_API_KEY": "test-only", "OPENAI_BASE_URL": ""}), redirect_stdout(StringIO()):
                with patch("sys.argv", args + ["--max-questions", "2"]):
                    self.assertEqual(experiment.main(), 0)
                self.assertEqual(client.chat.completions.create.call_count, 2)
                with patch("sys.argv", args):
                    self.assertEqual(experiment.main(), 0)
                    self.assertEqual(experiment.main(), 0)
                self.assertEqual(client.chat.completions.create.call_count, 4)
                with patch("sys.argv", args + ["--temperature", "0.2"]):
                    with self.assertRaises(SystemExit):
                        experiment.main()
                report = experiment.read_json(root / "run/metrics.json")
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["usage_all_attempts"]["total_tokens"], 60)

    def test_dotenv_gateway_configuration_is_used_without_saving_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment.write_json(root / "data.json", sample_data())
            env_path = root / ".env"
            env_path.touch()
            args = ["gpt_experiment.py", "--input", str(root / "data.json"), "--env-file", str(env_path),
                    "--output-dir", str(root / "run"), "--max-questions", "1"]
            def load_environment(path, override):
                self.assertEqual(path, env_path)
                self.assertFalse(override)
                experiment.os.environ.update({"OPENAI_API_KEY": "test-gateway-key", "OPENAI_BASE_URL": "https://gateway.example/v1/",
                                              "BASE_MODEL_ID": "gpt-4.1-mini", "OPENAI_MODEL_2": "gpt-4o"})
            client = MagicMock()
            client.__enter__.return_value = client
            client.chat.completions.create.return_value = fake_response(["Yes"])
            factory = MagicMock(return_value=client)
            with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=factory), "dotenv": SimpleNamespace(load_dotenv=load_environment)}), patch.dict("os.environ", {}, clear=True), patch("sys.argv", args), redirect_stdout(StringIO()):
                self.assertEqual(experiment.main(), 0)
            self.assertEqual(factory.call_args.kwargs["base_url"], "https://gateway.example/v1/")
            self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "gpt-4.1-mini")
            config = experiment.read_json(root / "run/config.json")
            self.assertEqual(config["base_url"], "https://gateway.example/v1/")
            self.assertNotIn("test-gateway-key", json.dumps(config))


if __name__ == "__main__":
    unittest.main()