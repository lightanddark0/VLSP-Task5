import json
import tempfile
import unittest
from pathlib import Path

import crossfit
from solve_parsed import fit_fr_calibration, oof_calibrated, question_forms, solve_all
from spartqa.data import write_json
from spartqa.symbolic.human_forms import rule_form
from spartqa.symbolic.structured import LPConfig, LPSolver, merge_worlds, parse_form, world_from_json

WORLD = {"blocks": ["A", "B"],
         "objects": [{"id": "a1", "block": "A", "shape": "circle", "size": "small", "color": "blue"},
                     {"id": "a2", "block": "A", "shape": "square", "size": "large", "color": "black"},
                     {"id": "b1", "block": "B", "shape": "circle", "size": None, "color": "yellow"}],
         "facts": [["a1", "left", "a2"], ["a1", "near", "a2"], ["B", "right", "A"]],
         "edges": [["a2", "right"]]}


def question(text, task, candidates=None):
    return {"question": text, "q_type": task, "candidate_answers": candidates or []}


class RuleFormTests(unittest.TestCase):
    def test_common_shapes(self):
        self.assertEqual(rule_form(question("Mối quan hệ giữa hình tròn nhỏ màu xanh lam ở A và vật lớn là gì?", "FR")),
                         {"first": {"shape": "circle", "size": "small", "color": "blue", "block": "A"},
                          "second": {"size": "large"}})
        self.assertEqual(rule_form(question("Khối nào không có bất kỳ vật thể cỡ trung bình nào bên trong nó?", "FB")),
                         {"condition": "has_not", "object": {"size": "medium"}})
        self.assertEqual(rule_form(question("Khối nào có tất cả các hình tròn bên trong?", "FB")),
                         {"condition": "has_all", "object": {"shape": "circle"}})
        self.assertEqual(rule_form(question("Khối nào không có vật chạm vào cạnh của nó?", "FB"))["object"],
                         {"edge": "any"})
        yn = rule_form(question("Có phải tất cả các vật màu vàng đều ở xa phía trên một vật màu xanh lam không?", "YN"))
        self.assertEqual((yn["quant"], yn["predicate"]["rels"][0]["rel"]), ("all", ["far", "above"]))
        some = rule_form(question("Có vật nhỏ nào ở bên trái hình vuông kia không?", "YN"))
        self.assertEqual((some["quant"], some["subject"]), ("some", {"size": "small"}))
        co = rule_form(question("Vật nào không ở gần vật lớn màu đen, hình tròn nhỏ hay hình vuông lớn?", "CO",
                                ["hình tròn nhỏ", "một hình vuông lớn", "cả hai", "không có"]))
        self.assertTrue(co["negated"])
        self.assertEqual(co["options"], [{"shape": "circle", "size": "small"}, {"shape": "square", "size": "large"}])
        touch = rule_form(question("Bên dưới hình vuông là gì: hình tròn hay tam giác?", "CO", ["hình tròn", "tam giác"]))
        self.assertEqual(touch["predicate"]["rels"][0]["rel"], ["below"])
        for form, task in ((yn, "YN"), (co, "CO"), (touch, "CO")):
            parse_form(form, task)

    def test_unrecognised_shapes_are_refused(self):
        self.assertIsNone(rule_form(question("Hình tròn sắp xếp lại vật thể ở đâu?", "FR")))
        self.assertIsNone(rule_form(question("Có phải tất cả vật nhỏ đều ở trên bất kỳ tam giác lớn nào không?", "YN")))
        self.assertIsNone(rule_form(question("Vật nào ở bên trái hình tròn?", "CO", ["a", "b"])))


class NewSwitchTests(unittest.TestCase):
    def solver(self, **config):
        cfg = LPConfig(**config)
        return LPSolver(world_from_json(WORLD, cfg.rules()), cfg)

    def test_relax_definite_and_block_share(self):
        fr = parse_form({"first": {"shape": "circle", "size": "small", "color": "yellow"}, "second": {"shape": "square"}}, "FR")
        self.assertIsNone(self.solver().answer(fr, {"q_type": "FR"}))            # b1 has no size
        self.assertEqual(self.solver(relax="size").answer(fr, {"q_type": "FR"}), [1])
        yn = parse_form({"subject": {"shape": "circle"}, "predicate": {"rels": [{"rel": "left", "target": {"shape": "square"}}]}}, "YN")
        self.assertEqual(self.solver().answer(yn, {"q_type": "YN"}), ["Yes"])     # a1 is left of a2
        self.assertEqual(self.solver(definite="all").answer(yn, {"q_type": "YN"}), ["No"])   # b1 is not
        share = parse_form({"first": {"shape": "circle"}, "second": {"shape": "square", "block": "A"}}, "FR")
        self.assertEqual(self.solver().answer(share, {"q_type": "FR"}), [7])      # a1 left, b1 right: nothing common
        self.assertEqual(self.solver(fr_block_share=True).answer(share, {"q_type": "FR"}), [0, 4])

    def test_merge_worlds_keeps_majority_facts_across_different_ids(self):
        samples = []
        for k, dropped in enumerate((0, 1, None)):
            world = json.loads(json.dumps(WORLD))
            if dropped is not None:
                del world["facts"][dropped]
            rename = {o["id"]: f"s{k}_{o['id']}" for o in world["objects"]}
            for o in world["objects"]:
                o["id"] = rename[o["id"]]
            world["facts"] = [[rename.get(a, a), r, rename.get(b, b)] for a, r, b in world["facts"]]
            world["edges"] = [[rename[o], side] for o, side in world["edges"]]
            samples.append(world)
        merged = merge_worlds(samples + [None])
        self.assertEqual((len(merged["objects"]), len(merged["facts"]), len(merged["edges"])), (3, 3, 1))
        self.assertIsNone(merge_worlds([WORLD, None]))

    def test_rule_forms_join_the_vote(self):
        q = {**question("Khối nào có hình vuông?", "FB", ["A", "B"])}
        llm = [{"condition": "has", "object": {"shape": "circle"}}] * 2
        self.assertEqual(len(question_forms(llm, q, LPConfig())), 2)
        self.assertEqual(len(question_forms(llm, q, LPConfig(forms_mode="rule_plus"))), 4)
        self.assertEqual(question_forms(llm, q, LPConfig(forms_mode="rule_first"))[0]["condition"], "has")


class CalibrationTests(unittest.TestCase):
    def test_fit_and_oof(self):
        stories, worlds, forms = [], {}, {}
        for index in range(6):
            stories.append({"story": ["x"], "questions": [{"q_id": 1, "q_type": "FR", "question": "q",
                                                           "candidate_answers": [], "answer": [0, 5]}]})
            worlds[index] = [WORLD]
            forms[f"{index}_1"] = [{"first": {"shape": "circle", "block": "A"}, "second": {"shape": "circle", "block": "B"}}]
        data = {"data": stories}
        table = fit_fr_calibration(data, worlds, forms, LPConfig())
        self.assertEqual(table["cross|dir|-|-"]["5"], 1.0)                       # gold always adds "far"
        plain = solve_all(data, worlds, forms, LPConfig())
        calibrated = solve_all(data, worlds, forms, LPConfig(), calibration=table)
        self.assertEqual((plain["0_1"]["answer"], calibrated["0_1"]["answer"]), ([0], [0, 5]))
        self.assertEqual(oof_calibrated(data, worlds, forms, LPConfig(), 3, "LP")["0_1"]["answer"], [0, 5])


class CrossfitTests(unittest.TestCase):
    def test_folds_merge_and_combine(self):
        folds = crossfit.story_folds(11, 5, 42)
        self.assertEqual(sorted(i for fold in folds for i in fold), list(range(11)))
        self.assertEqual(folds, crossfit.story_folds(11, 5, 42))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "folds.json", {"folds": [[3, 0], [1, 2]]})
            for k, rows in enumerate(([("0_1", "a"), ("1_2", "b")], [("0_1", "c"), ("1_1", "d")])):
                (root / f"f{k}.jsonl").write_text("\n".join(json.dumps({"key": key, "answer": [v]}) for key, v in rows),
                                                  encoding="utf-8")
            crossfit.main(["merge", "--folds-dir", str(root), "--pattern", str(root / "f{k}.jsonl"),
                           "--source", "F2C", "--output", str(root / "out.jsonl")])
            merged = [json.loads(line) for line in (root / "out.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([(r["key"], r["answer"][0], r["source"]) for r in merged],
                             [("0_2", "b", "F2C"), ("1_1", "c", "F2C"), ("2_1", "d", "F2C"), ("3_1", "a", "F2C")])

            splits, pred = root / "splits", root / "pred" / "X"
            pred.mkdir(parents=True)
            story = {"story": ["s"], "questions": [{"q_id": 1, "q_type": "YN", "question": "q",
                                                    "candidate_answers": [], "answer": ["Yes"]}]}
            write_json(splits / "human_train.json", {"name": "t", "data": [story, story]})
            write_json(splits / "human_dev.json", {"name": "t", "data": [story]})
            (pred / "human_train.jsonl").write_text('{"key": "1_1", "answer": ["No"]}\n', encoding="utf-8")
            (pred / "human_dev.jsonl").write_text('{"key": "0_1", "answer": ["Yes"]}\n', encoding="utf-8")
            crossfit.main(["combine", "--splits-dir", str(splits), "--pred-dir", str(root / "pred")])
            keys = [json.loads(line)["key"] for line in (pred / "human_trdev.jsonl").read_text().splitlines()]
            self.assertEqual(keys, ["1_1", "2_1"])
            self.assertEqual(len(json.loads((splits / "human_trdev.json").read_text())["data"]), 3)


if __name__ == "__main__":
    unittest.main()
