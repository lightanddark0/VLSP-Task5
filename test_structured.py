import json
import unittest
from pathlib import Path

from solve_parsed import answer_question, read_parses
from spartqa.lp_prompting import ANNOTATIONS, extract_json, load_examples, question_messages, world_messages
from spartqa.symbolic.structured import LPConfig, LPSolver, StructureError, parse_form, solve, world_from_json
from spartqa.symbolic.world import RulesConfig

ROOT = Path(__file__).resolve().parent


def obj(ident, block, shape=None, size=None, color=None):
    return {"id": ident, "block": block, "shape": shape, "size": size, "color": color}


LINE = {"blocks": ["A", "B"],
        "objects": [obj("p", "A", "circle"), obj("q", "A", "square"), obj("r", "A", "triangle"), obj("s", "B", "circle")],
        "facts": [["p", "left", "q"], ["q", "far", "r"], ["q", "left", "r"], ["A", "left", "B"]],
        "edges": [["r", "right"]]}


def task_of(form):
    if "first" in form:
        return "FR"
    if "condition" in form:
        return "FB"
    return "CO" if "options" in form else "YN"


class WorldTests(unittest.TestCase):
    def test_invalid_worlds_raise(self):
        for bad in ([], {"objects": []}, {"objects": [obj("x", "A"), obj("x", "A")]},
                    {"objects": [obj("x", "A")], "facts": [["x", "beside", "x"]]},
                    {"objects": [obj("x", "A")], "facts": [["x", "left", "A"]]},
                    {"objects": [obj("x", "A")], "facts": [["x", "left", "y"]]},
                    {"objects": [obj("x", "A")], "edges": [["x", "middle"]]}):
            with self.assertRaises(StructureError):
                world_from_json(bad)

    def test_far_rules_are_off_by_default_and_extend_along_one_direction(self):
        plain = world_from_json(LINE)
        self.assertTrue(plain.holds("LEFT", 0, 2) and plain.holds("RIGHT", 2, 0))
        self.assertFalse(plain.holds("FAR", 0, 2))
        chained = world_from_json(LINE, RulesConfig(far_chain=True))
        self.assertTrue(chained.holds("FAR", 0, 2) and chained.holds("FAR", 2, 0))   # p left of q, q far left of r
        self.assertTrue(chained.holds("FAR", 3, 1))                                  # s in B, right of r's block
        self.assertFalse(chained.holds("FAR", 0, 1))                                 # p-q stays unknown
        crossed = world_from_json(LINE, RulesConfig(cross_block_far=True))
        self.assertTrue(crossed.holds("FAR", 0, 3) and not crossed.holds("FAR", 0, 1))


class SolverTests(unittest.TestCase):
    def solver(self, **config):
        cfg = LPConfig(**config)
        return LPSolver(world_from_json(LINE, cfg.rules()), cfg)

    def test_yn_closed_and_open_world(self):
        form = parse_form({"subject": {"shape": "circle", "block": "A"}, "quant": "the",
                           "predicate": {"rels": [{"rel": ["above"], "target": {"shape": "square"}}]}}, "YN")
        self.assertEqual(self.solver().answer(form, {"q_type": "YN"}), ["No"])
        self.assertEqual(self.solver(yn_mode="open").answer(form, {"q_type": "YN"}), ["DK"])
        left = parse_form({"subject": {"shape": "circle", "block": "A"},
                           "predicate": {"rels": [{"rel": "right", "target": {"shape": "square"}}]}}, "YN")
        self.assertEqual(self.solver(yn_mode="open").answer(left, {"q_type": "YN"}), ["No"])   # refuted
        every = parse_form({"subject": {"shape": "circle"}, "quant": "all",
                            "predicate": {"rels": [{"rel": ["left"], "target": {"shape": "triangle"}}]}}, "YN")
        self.assertEqual(self.solver().answer(every, {"q_type": "YN"}), ["No"])    # s is right of r
        some = dict(every, quant="some")
        self.assertEqual(self.solver().answer(some, {"q_type": "YN"}), ["Yes"])
        missing = parse_form({"subject": {"color": "red"}, "predicate": {}}, "YN")
        self.assertIsNone(self.solver().answer(missing, {"q_type": "YN"}))

    def test_fr_fb_co(self):
        fr = parse_form({"first": {"shape": "circle"}, "second": {"shape": "triangle"}}, "FR")
        self.assertEqual(self.solver(far_chain=False).answer(fr, {"q_type": "FR"}), [7])     # p left, s right
        self.assertEqual(self.solver(fr_multi="any", far_chain=False).answer(fr, {"q_type": "FR"}), [0, 1])
        fr_one = parse_form({"first": {"shape": "circle", "block": "A"}, "second": {"shape": "triangle"}}, "FR")
        self.assertEqual(self.solver().answer(fr_one, {"q_type": "FR"}), [0, 5])
        question = {"q_type": "FB", "candidate_answers": ["A", "B"]}
        self.assertEqual(self.solver().answer(parse_form({"condition": "has_not", "object": {"edge": "any"}}, "FB"),
                                              question), ["B"])
        self.assertEqual(self.solver().answer(parse_form({"condition": "has_all", "object": {"shape": "circle"}}, "FB"),
                                              question), [])
        self.assertEqual(self.solver().answer(parse_form({"condition": "has", "object": {"shape": "square"}}, "FB"),
                                              question), ["A"])
        co = parse_form({"options": [{"shape": "circle"}, {"shape": "triangle"}],
                         "predicate": {"rels": [{"rel": ["left"], "target": {"shape": "square"}}]}}, "CO")
        self.assertEqual(self.solver().answer(co, {"q_type": "CO"}), [0])
        self.assertEqual(self.solver().answer(dict(co, negated=True), {"q_type": "CO"}), [1])
        self.assertIsNone(parse_form({"unsupported": True}, "CO"))
        with self.assertRaises(StructureError):
            parse_form({"options": [{"shape": "circle"}]}, "CO")


class ParsingAndVotingTests(unittest.TestCase):
    def test_extract_json_takes_the_last_object_after_thinking(self):
        self.assertEqual(extract_json('<think>{"a": 1}</think>ok ```json\n{"b": {"c": 2}}\n```'), {"b": {"c": 2}})
        self.assertEqual(extract_json('x {"a": 1} y {"b": 2} {bad'), {"b": 2})
        self.assertIsNone(extract_json('<think>{"a": 1} never closed'))
        self.assertIsNone(extract_json("no json here"))

    def test_votes_skip_unusable_samples(self):
        worlds = [LINE, None, {"objects": "broken"}]
        forms = [{"first": {"shape": "circle", "block": "A"}, "second": {"shape": "square"}}, None,
                 {"first": {"shape": "circle", "block": "A"}, "second": {"shape": "square"}}]
        answer, scores, votes = answer_question(worlds, forms, {"q_type": "FR"}, LPConfig())
        self.assertEqual((answer, votes), ([0], 2))
        self.assertEqual(answer_question([None], forms, {"q_type": "FR"}, LPConfig())[:2], (None, {}))

    def test_read_parses_keeps_the_latest_line(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.parses.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in [
                {"kind": "world", "story": 0, "worlds": [None]}, {"kind": "form", "key": "0_1", "forms": [{}]},
                {"kind": "world", "story": 0, "worlds": [LINE]}]) + "\n", encoding="utf-8")
            worlds, forms = read_parses(path)
        self.assertEqual((worlds, forms), ({0: [LINE]}, {"0_1": [{}]}))


class OracleTests(unittest.TestCase):
    """The hand annotations give the hand-derived answers (no data files needed)."""

    def test_annotated_stories(self):
        spec = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
        for story in spec["stories"]:
            for qid, form in story["forms"].items():
                question = {"q_type": task_of(form), "candidate_answers": story["world"]["blocks"]}
                with self.subTest(story=story["index"], q=qid):
                    self.assertEqual(solve(story["world"], form, question, LPConfig()), story["expected"][qid])

    @unittest.skipUnless((ROOT / "Data/splits/human_train.json").exists(), "needs Data/splits (local only)")
    def test_prompts_use_the_annotated_story_text(self):
        example = load_examples()[0]
        self.assertEqual(len(example["questions"]), 16)
        world_prompt = world_messages(["Có một khối A."], [example])[1]["content"]
        self.assertIn('"a1"', world_prompt)
        self.assertTrue(world_prompt.endswith("Có một khối A."))
        payload = {"story": ["Có một khối A."], "question": "Khối nào có hình tròn?", "q_type": "FB",
                   "candidate_answers": ["A", "B"]}
        prompt = question_messages(payload, [example])[1]["content"]
        self.assertIn('"condition": "has_not"', prompt)
        self.assertTrue(prompt.rstrip().endswith("Lựa chọn: A, B"))


if __name__ == "__main__":
    unittest.main()
