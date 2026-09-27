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
from spartqa.pot import REASONING_INSTRUCTIONS, aggregate_fr, graph_response_format, identify_paths, identify_query_paths, path_statements, request_pot_answer, validate_graph


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


class PathOfThoughtsTests(unittest.TestCase):
    def graph(self):
        return validate_graph({
            "entities": [{"id": name, "description": name} for name in ("A", "B", "C", "D", "X", "Y")],
            "triples": [
                {"head": "B", "relation": "right", "tail": "A"},
                {"head": "B", "relation": "left", "tail": "C"},
                {"head": "A", "relation": "above", "tail": "D"},
                {"head": "D", "relation": "above", "tail": "C"},
                {"head": "X", "relation": "touch", "tail": "Y"},
            ],
            "query": {"source": "A", "target": "C"},
        })

    def test_paths_preserve_reverse_edges_and_exclude_disconnected_noise(self):
        graph = self.graph()
        result = identify_paths(graph)
        self.assertEqual(result["paths"], [[0, 1], [2, 3]])
        self.assertEqual(result["limits_hit"], [])
        self.assertEqual(path_statements(graph, result["paths"][0]), ["B --right--> A", "B --left--> C"])

    def test_parallel_relations_and_cycles(self):
        graph = self.graph()
        graph["triples"].append({"head": "B", "relation": "above", "tail": "A"})
        self.assertEqual(identify_paths(graph)["paths"], [[0, 1], [2, 3], [5, 1]])

    def test_limits_are_explicit_and_disconnected_search_is_empty(self):
        graph = self.graph()
        self.assertEqual(identify_paths(graph, max_paths=1)["limits_hit"], ["max_paths"])
        self.assertEqual(identify_paths(graph, max_paths=2)["limits_hit"], [])
        self.assertEqual(identify_paths(graph, max_hops=1)["paths"], [])
        self.assertIn("max_expansions", identify_paths(graph, max_expansions=1)["limits_hit"])
        graph["query"]["target"] = "X"
        self.assertEqual(identify_paths(graph)["no_path_reason"], "disconnected")
        graph["query"]["target"] = None
        self.assertEqual(identify_paths(graph)["no_path_reason"], "unresolved_query")

    def test_graph_validation_and_union(self):
        graph = self.graph()
        graph["triples"].append(dict(graph["triples"][0]))
        self.assertEqual(len(validate_graph(graph)["triples"]), 5)
        graph["triples"][0]["head"] = "missing"
        with self.assertRaises(ValueError):
            validate_graph(graph)
        self.assertEqual(aggregate_fr([[0], [2], [7], [0]]), [0, 2])
        self.assertEqual(aggregate_fr([[7], [7]]), [7])
        with self.assertRaises(ValueError):
            aggregate_fr([])

    def config(self, method="pot"):
        return {"model": "fake", "prompt": "Base FR instructions", "temperature": 0,
                "max_output_tokens": 100, "max_extraction_tokens": 1000,
                "method": method, "extraction_prompt": "Extract graph", "max_paths": 16,
                "max_hops": 0, "max_expansions": 10000, "reasoning_instructions": REASONING_INSTRUCTIONS}

    def graph_response(self):
        response = fake_response([0])
        response.choices[0].message.content = json.dumps(self.graph())
        return response

    def test_all_tasks_reason_jointly_over_paths(self):
        for example in experiment.examples_from_data(sample_data()):
            with self.subTest(task=example["payload"]["q_type"]):
                client = MagicMock()
                client.chat.completions.create.return_value = fake_response(example["gold"])
                stages = {"extract": {"graph": self.graph()}}
                record = request_pot_answer(client, example, self.config(), stages, stages.__setitem__)
                self.assertEqual(record["answer"], example["gold"])
                self.assertEqual(client.chat.completions.create.call_count, 1)
                payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
                self.assertEqual(len(payload["reasoning_paths"]), 2)
                self.assertEqual(payload["story"], example["payload"]["story"])
                self.assertNotIn("answer", payload)
                self.assertNotIn("gold", payload)
                self.assertEqual(request_pot_answer(client, example, self.config(), stages, stages.__setitem__)["answer"],
                                 example["gold"])
                self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_joint_call_no_gold_and_cached_stages(self):
        example = experiment.examples_from_data(sample_data())[1]
        client = MagicMock()
        client.chat.completions.create.side_effect = [self.graph_response(), fake_response([0, 2])]
        stages = {}
        record = request_pot_answer(client, example, self.config(), stages, stages.__setitem__)
        self.assertEqual(record["answer"], [0, 2])
        self.assertIsNone(record["path_disagreement"])
        self.assertEqual(client.chat.completions.create.call_count, 2)
        calls = client.chat.completions.create.call_args_list
        for call in calls:
            payload = json.loads(call.kwargs["messages"][1]["content"])
            self.assertNotIn("gold", payload)
            self.assertNotIn("answer", payload)
        self.assertEqual(json.loads(calls[1].kwargs["messages"][1]["content"])["reasoning_paths"],
                 [["B --right--> A", "B --left--> C"], ["A --above--> D", "D --above--> C"]])
        self.assertEqual(request_pot_answer(client, example, self.config(), stages, stages.__setitem__)["answer"], [0, 2])
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_invalid_graph_or_failed_path_is_not_a_final_answer(self):
        example = experiment.examples_from_data(sample_data())[1]
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response([0])
        stages = {}
        record = request_pot_answer(client, example, self.config(), {}, stages.__setitem__)
        self.assertEqual(record["error"], "InvalidGraph")
        self.assertEqual(record["validation_error"], {
            "type": "ValueError", "message": "Graph requires entities, triples and query",
        })
        self.assertEqual(json.loads(stages["extract"]["raw_graph_response"])["answer"], [0])
        self.assertNotIn("answer", record)
        client.chat.completions.create.side_effect = [self.graph_response(), fake_response([2], "length")]
        record = request_pot_answer(client, example, self.config(), {}, lambda *args: None)
        self.assertEqual(record["error_stage"], "reason:joint")
        self.assertNotIn("answer", record)

    def test_no_path_fallback_and_full_graph_ablation(self):
        example = experiment.examples_from_data(sample_data())[1]
        for method in ("pot", "pot-no-path"):
            with self.subTest(method=method):
                graph = self.graph()
                graph["query"]["target"] = "X"
                client = MagicMock()
                client.chat.completions.create.return_value = fake_response([7])
                record = request_pot_answer(client, example, self.config(method), {"extract": {"graph": graph}}, lambda *args: None)
                payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
                self.assertEqual(record["answer"], [7])
                self.assertEqual(record["fallback"], method == "pot")
                self.assertEqual(payload["evidence_mode"], "fallback_graph" if method == "pot" else "full_graph")
                self.assertEqual(len(payload["reasoning_chain"]), 5)

    def test_invalid_graph_diagnostics_distinguish_json_and_reference_errors(self):
        example = experiment.examples_from_data(sample_data())[0]
        unknown_reference = self.graph()
        unknown_reference["triples"][0]["tail"] = "missing"
        duplicate_id = self.graph()
        duplicate_id["entities"].append(dict(duplicate_id["entities"][0]))
        cases = [
            ("not JSON", "JSONDecodeError", "Expecting value"),
            (json.dumps(unknown_reference), "ValueError", "Triple references an unknown entity"),
            (json.dumps(duplicate_id), "ValueError", "Duplicate entity id"),
        ]
        for content, error_type, message in cases:
            with self.subTest(error_type=error_type, message=message):
                client = MagicMock()
                response = self.graph_response()
                response.choices[0].message.content = content
                client.chat.completions.create.return_value = response
                stages = {}
                record = request_pot_answer(client, example, self.config(), {}, stages.__setitem__)
                self.assertEqual(record["error"], "InvalidGraph")
                self.assertEqual(record["validation_error"]["type"], error_type)
                self.assertIn(message, record["validation_error"]["message"])
                self.assertEqual(stages["extract"]["raw_graph_response"], content)
                self.assertNotIn("answer", record)
                self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_cli_stops_repeated_invalid_graphs_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment.write_json(root / "data.json", sample_data())
            args = ["gpt_experiment.py", "--input", str(root / "data.json"), "--method", "pot",
                    "--output-dir", str(root / "run"), "--env-file", str(root / "absent.env")]
            client = MagicMock()
            client.__enter__.return_value = client
            client.chat.completions.create.return_value = fake_response([0])
            output = StringIO()
            with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=MagicMock(return_value=client))}), patch.dict("os.environ", {"OPENAI_API_KEY": "test-only", "OPENAI_BASE_URL": ""}), patch("sys.argv", args), redirect_stdout(output):
                self.assertEqual(experiment.main(), 1)
                self.assertEqual(client.chat.completions.create.call_count, 3)
                report = experiment.read_json(root / "run/metrics.json")
                self.assertEqual(report["status"], "partial")
                self.assertEqual(report["answered_questions"], 0)
                self.assertIn("Stopped after 3 consecutive invalid graphs", output.getvalue())
                self.assertIn("Graph requires entities, triples and query", output.getvalue())
                responses = []
                for example in experiment.examples_from_data(sample_data()):
                    responses.extend([self.graph_response(), fake_response(example["gold"])])
                client.chat.completions.create.side_effect = responses
                self.assertEqual(experiment.main(), 0)
            report = experiment.read_json(root / "run/metrics.json")
            self.assertEqual(report["answered_questions"], 4)
            self.assertEqual(report["usage_all_attempts"]["total_tokens"], 165)

    def test_multi_pair_search_validation_and_global_budget(self):
        graph = self.graph()
        graph["query"] = {"source": None, "target": None, "pairs": [
            {"source": "A", "target": "C"}, {"source": "X", "target": "Y"},
            {"source": "A", "target": "C"},
        ], "focus": ["A", "C", "X", "Y"]}
        graph = validate_graph(graph)
        result = identify_query_paths(graph, max_paths=1)
        self.assertEqual(result["paths"], [[0, 1], [4]])
        self.assertEqual(len(result["pair_searches"]), 2)
        self.assertIn("max_paths", result["limits_hit"])
        limited = identify_query_paths(graph, max_expansions=1)
        self.assertEqual(limited["expansions"], 1)
        self.assertTrue(limited["incomplete"])
        self.assertEqual(limited["pair_searches"][1]["no_path_reason"], "budget_exhausted")
        schema = graph_response_format()["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]["query"]["required"]), {"source", "target", "pairs", "focus"})
        for field, value in (("pairs", [{"source": "missing", "target": "A"}]),
                             ("focus", ["missing"]), ("pairs", "A"), ("focus", [None])):
            with self.subTest(field=field, value=value):
                invalid = copy.deepcopy(graph)
                invalid["query"][field] = value
                with self.assertRaises(ValueError):
                    validate_graph(invalid)

    def test_candidate_queries_keep_inventory_and_fallback_for_missing_evidence(self):
        graph = self.graph()
        graph["triples"].append({"head": "X", "relation": "in", "tail": "Y"})
        graph["query"] = {"source": None, "target": None, "pairs": [
            {"source": "A", "target": "C"}, {"source": "X", "target": "C"},
        ], "focus": ["A", "X", "C", "Y"]}
        example = experiment.examples_from_data(sample_data())[3]
        client = MagicMock()
        client.chat.completions.create.return_value = fake_response([0])
        record = request_pot_answer(client, example, self.config(), {"extract": {"graph": graph}}, lambda *args: None)
        payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertTrue(record["fallback"])
        self.assertEqual(len(payload["reasoning_paths"]), 2)
        self.assertEqual(len(payload["reasoning_chain"]), 6)
        self.assertEqual(payload["containment_facts"], ["X --in--> Y"])
        self.assertIn("X --in--> Y", payload["query_facts"])
        self.assertEqual(payload["query_plan"], graph["query"])
        self.assertEqual(payload["graph_entities"], graph["entities"])

    def test_containment_path_keeps_direction_and_boundary_contact_separate(self):
        graph = validate_graph({
            "entities": [{"id": name, "description": name} for name in ("P", "Q", "circle", "square", "P_bottom")],
            "triples": [
                {"head": "circle", "relation": "in", "tail": "P"},
                {"head": "P", "relation": "above", "tail": "Q"},
                {"head": "square", "relation": "in", "tail": "Q"},
                {"head": "circle", "relation": "covered_by", "tail": "P"},
                {"head": "circle", "relation": "touch", "tail": "P_bottom"},
            ],
            "query": {"source": "circle", "target": "square", "pairs": [], "focus": ["circle", "square"]},
        })
        search = identify_query_paths(graph)
        self.assertIn([0, 1, 2], search["paths"])
        self.assertIn([3, 1, 2], search["paths"])
        self.assertFalse(any(4 in path for path in search["paths"]))
        self.assertEqual(path_statements(graph, [0, 1, 2]),
                         ["circle --in--> P", "P --above--> Q", "square --in--> Q"])

    def test_cli_pot_resume_preserves_stages_usage_and_fr_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment.write_json(root / "data.json", sample_data())
            args = ["gpt_experiment.py", "--input", str(root / "data.json"), "--method", "pot", "--tasks", "FR",
                    "--output-dir", str(root / "run"), "--env-file", str(root / "absent.env")]
            client = MagicMock()
            client.__enter__.return_value = client
            client.chat.completions.create.side_effect = [self.graph_response(), fake_response([2], "length"),
                                                        fake_response([2])]
            with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=MagicMock(return_value=client))}), patch.dict("os.environ", {"OPENAI_API_KEY": "test-only", "OPENAI_BASE_URL": ""}), patch("sys.argv", args), redirect_stdout(StringIO()):
                self.assertEqual(experiment.main(), 1)
                self.assertEqual(experiment.main(), 0)
                self.assertEqual(experiment.main(), 0)
                self.assertEqual(client.chat.completions.create.call_count, 3)
                with patch("sys.argv", args + ["--max-paths", "3"]), self.assertRaises(SystemExit):
                    experiment.main()
            report = experiment.read_json(root / "run/metrics.json")
            self.assertEqual(report["selected_questions"], 1)
            self.assertEqual(report["usage_all_attempts"]["total_tokens"], 45)
            self.assertEqual(report["pot_diagnostics"]["stage_attempts_all"], 3)
            self.assertEqual(set(report["by_task"]), {"FR"})

    def test_controlled_ablation_reuses_graph_without_double_counting_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment.write_json(root / "data.json", sample_data())
            common = ["gpt_experiment.py", "--input", str(root / "data.json"), "--tasks", "FR", "--env-file", str(root / "absent.env")]
            client = MagicMock()
            client.__enter__.return_value = client
            client.chat.completions.create.side_effect = [self.graph_response(), fake_response([0, 2]),
                                                        fake_response([0, 2])]
            with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=MagicMock(return_value=client))}), patch.dict("os.environ", {"OPENAI_API_KEY": "test-only", "OPENAI_BASE_URL": ""}), redirect_stdout(StringIO()):
                with patch("sys.argv", common + ["--method", "pot", "--output-dir", str(root / "source")]):
                    self.assertEqual(experiment.main(), 0)
                args = common + ["--method", "pot-no-path", "--graph-cache-dir", str(root / "source"), "--output-dir", str(root / "ablation")]
                with patch("sys.argv", args):
                    self.assertEqual(experiment.main(), 0)
                    self.assertEqual(experiment.main(), 0)
                with patch("sys.argv", args + ["--temperature", "0.3"]), self.assertRaises(SystemExit):
                    experiment.main()
            self.assertEqual(client.chat.completions.create.call_count, 3)
            report = experiment.read_json(root / "ablation/metrics.json")
            self.assertEqual(report["usage_all_attempts"]["total_tokens"], 15)
            self.assertEqual(report["pot_diagnostics"]["reused_extractions"], 1)
            self.assertEqual(report["pot_diagnostics"]["stage_attempts_all"], 1)

    def test_cli_pot_defaults_to_all_tasks_and_rejects_old_graph_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment.write_json(root / "data.json", sample_data())
            args = ["gpt_experiment.py", "--input", str(root / "data.json"), "--method", "pot",
                    "--output-dir", str(root / "run"), "--env-file", str(root / "absent.env")]
            client = MagicMock()
            client.__enter__.return_value = client
            responses = []
            for example in experiment.examples_from_data(sample_data()):
                responses.extend([self.graph_response(), fake_response(example["gold"])])
            client.chat.completions.create.side_effect = responses
            with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=MagicMock(return_value=client))}), patch.dict("os.environ", {"OPENAI_API_KEY": "test-only", "OPENAI_BASE_URL": ""}), patch("sys.argv", args), redirect_stdout(StringIO()):
                self.assertEqual(experiment.main(), 0)
                self.assertEqual(experiment.main(), 0)
            self.assertEqual(client.chat.completions.create.call_count, 8)
            report = experiment.read_json(root / "run/metrics.json")
            self.assertEqual(report["selected_questions"], 4)
            self.assertEqual(report["answered_questions"], 4)
            self.assertEqual(set(report["by_task"]), {"YN", "FR", "FB", "CO"})
            self.assertIsNone(report["pot_diagnostics"]["path_disagreement_questions"])
            self.assertEqual(report["usage_all_attempts"]["total_tokens"], 120)
            config = experiment.read_json(root / "run/config.json")
            self.assertEqual(config["pot_version"], 2)
            self.assertEqual(config["aggregation"], "joint_evidence_by_question_type")
            experiment.write_json(root / "run/config.json", {**config, "pot_version": 1})
            with self.assertRaises(ValueError):
                experiment.load_graph_cache(root / "run", config)

    def test_extraction_prompt_examples_match_graph_schema(self):
        prompt = (experiment.ROOT / "Docs/spartqa_pot_extract.txt").read_text(encoding="utf-8")
        examples = [json.loads(line) for line in prompt.splitlines() if line.startswith('{"entities":')]
        self.assertEqual(len(examples), 2)
        required = set(graph_response_format()["json_schema"]["schema"]["properties"]["query"]["required"])
        for graph in examples:
            self.assertEqual(set(validate_graph(graph)["query"]), required)
        contact_graph = examples[0]
        self.assertIn({"head": "P_circle", "relation": "touch", "tail": "P_bottom"}, contact_graph["triples"])
        self.assertNotIn({"head": "P_square", "relation": "above", "tail": "P_circle"}, contact_graph["triples"])


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
        self.assertEqual(report["by_task"]["FR"]["hit_accuracy"], 1.0)
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