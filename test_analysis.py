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
