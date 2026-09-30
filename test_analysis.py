import importlib.util
import unittest

from analyze_errors import fr_error_kind
from spartqa.analysis import story_folds

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


class ErrorKindTests(unittest.TestCase):
    def test_fr_error_kinds(self):
        self.assertEqual(fr_error_kind([0, 5], [5, 0]), "đúng")
        self.assertEqual(fr_error_kind([1, 5], [0, 5]), "sai hướng")
        self.assertEqual(fr_error_kind([0], [0, 5]), "thiếu gần/xa")
        self.assertEqual(fr_error_kind([0, 4], [0]), "thừa gần/xa")
        self.assertEqual(fr_error_kind([0, 4], [0, 5]), "đổi gần↔xa")
        self.assertEqual(fr_error_kind([0, 6], [0]), "chạm (6)")
        self.assertEqual(fr_error_kind([7], [0]), "DK (7)")
        self.assertEqual(fr_error_kind(None, [0]), "abstain")

    def test_folds_match_compare_branches(self):
        questions = [(f"{s}_1", {}) for s in (4, 0, 2, 9, 7, 1)]
        self.assertEqual(story_folds(questions, 5), {0: 0, 1: 1, 2: 2, 4: 3, 7: 4, 9: 0})


@unittest.skipUnless(HAS_SKLEARN, "stacking needs scikit-learn")
class StackingTests(unittest.TestCase):
    def test_learns_to_trust_the_reliable_source(self):
        from stack_ensemble import candidates, decide, features, fit_models, labels_of, stacked_answer
        from spartqa.postprocess import PostprocessOptions
        questions, good, bad = [], {}, {}
        for i in range(60):
            gold = ["Yes"] if i % 2 else ["No"]
            key = f"{i}_1"
            questions.append((key, {"q_type": "YN", "answer": gold, "candidate_answers": []}))
            good[key] = {"answer": gold, "scores": {gold[0]: 1.0}}
            other = "No" if gold == ["Yes"] else "Yes"
            bad[key] = {"answer": [other], "scores": {other: 1.0}}
        sources, names = {"G": good, "B": bad}, ["B", "G"]
        q = questions[0][1]
        self.assertEqual(len(features(questions[0][0], q, sources, names)), len(candidates(q)))
        self.assertEqual(labels_of(q), [0, 1, 0])
        models = fit_models(questions, sources, names, 1.0)
        fallback = {"YN": ["Yes"], "FR": [7], "FB": [], "CO": [3]}
        right = sum(stacked_answer(models, k, q, sources, names, "human", PostprocessOptions(), fallback) == q["answer"]
                    for k, q in questions)
        self.assertEqual(right, 60)
        self.assertEqual(decide({"q_type": "FR"}, [0.1, 0.9, 0, 0, 0, 0.6, 0, 0.95]), [1, 5])
        self.assertEqual(decide({"q_type": "FR"}, [0.1, 0.2, 0, 0, 0, 0, 0, 0.3]), [7])
        self.assertEqual(decide({"q_type": "FB", "candidate_answers": ["A", "B"]}, [0.2, 0.7]), ["B"])


if __name__ == "__main__":
    unittest.main()


class E6Tests(unittest.TestCase):
    def test_human_fr_dk_avoid(self):
        from spartqa.postprocess import PostprocessOptions, finalize_answer
        q = {"q_type": "FR", "candidate_answers": []}
        fallback = {"YN": ["Yes"], "FR": [0], "FB": [], "CO": [3]}
        scores = {"7": 0.6, "1": 0.4, "5": 0.3, "4": 0.1}
        keep = finalize_answer(q, [7], scores, "human", PostprocessOptions(), fallback)
        avoid = finalize_answer(q, [7], scores, "human", PostprocessOptions(human_fr_dk="avoid"), fallback)
        auto = finalize_answer(q, [7], scores, "auto", PostprocessOptions(human_fr_dk="avoid"), fallback)
        only_dk = finalize_answer(q, [7], {"7": 1.0}, "human", PostprocessOptions(human_fr_dk="avoid"), fallback)
        self.assertEqual((keep, avoid, auto, only_dk), ([7], [1, 5], [7], [7]))

    def test_lenient_world_fr_unknown_between_and_nested_rule_forms(self):
        from spartqa.symbolic.human_forms import rule_form
        from spartqa.symbolic.structured import LPConfig, LPSolver, StructureError, parse_form, world_from_json
        world = {"blocks": ["A"], "objects": [{"id": "l", "block": "A", "shape": "circle"},
                                              {"id": "m", "block": "A", "shape": "triangle"},
                                              {"id": "r", "block": "A", "shape": "circle", "size": "small"},
                                              {"id": "bad"}],
                 "facts": [["l", "left", "m"], ["m", "left", "r"], ["m", "inside", "l"], ["m", "left", "ghost"]]}
        with self.assertRaises(StructureError):
            world_from_json(world)
        lenient = world_from_json(world, lenient=True)
        self.assertEqual((len(lenient.objects), lenient.holds("LEFT", 0, 2)), (3, True))
        solver = LPSolver(lenient, LPConfig(lenient=True))
        between = rule_form({"q_type": "FB", "question": "Khối nào có hình tam giác nằm giữa hai hình tròn?",
                             "candidate_answers": ["A"]})
        self.assertEqual(solver.answer(parse_form(between, "FB"), {"q_type": "FB", "candidate_answers": ["A"]}), ["A"])
        fr = parse_form({"first": {"shape": "circle", "size": "small"}, "second": {"shape": "circle", "size": None,
                                                                                   "rels": [{"rel": "left", "target": {"shape": "triangle"}}]}}, "FR")
        self.assertEqual(solver.answer(fr, {"q_type": "FR"}), [1])
        unknown = parse_form({"first": {"shape": "triangle"}, "second": {"shape": "square"}}, "FR")
        self.assertIsNone(solver.answer(unknown, {"q_type": "FR"}))                    # no square: no pair
        nested = rule_form({"q_type": "FR", "candidate_answers": [],
                            "question": "Mối quan hệ giữa hình tròn nhỏ và hình tròn bên trái hình tam giác là gì?"})
        self.assertEqual(nested["second"]["rels"][0]["rel"], ["left"])
        self.assertEqual(solver.answer(parse_form(nested, "FR"), {"q_type": "FR"}), [1])
        self.assertIsNone(LPSolver(world_from_json({"objects": [{"id": "a", "block": "A"}, {"id": "b", "block": "A"}]}),
                                   LPConfig(fr_unknown="abstain")).answer(parse_form({"first": {"block": "A"}, "second": {"block": "A"}}, "FR"),
                                                                           {"q_type": "FR"}))

    def test_world_report_and_groups(self):
        from solve_parsed import world_report
        from spartqa.analysis import group_of
        rows, reasons = world_report({3: [None, {"objects": [{"id": "a", "block": "A"}], "facts": [["a", "x", "a"]]}]})
        self.assertEqual(rows, [(3, 2, 0, 1)])
        self.assertEqual(sum(reasons.values()), 2)
        self.assertEqual([group_of({"reasoning_type": r}) for r in ([], ["None"], ["Converse"])], ["B", "B", "A"])
