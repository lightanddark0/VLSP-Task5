import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import solve_symbolic
from spartqa.data import write_json
from spartqa.predictions import read_predictions
from spartqa.symbolic import RulesConfig, Solver, read_story

# Synthetic stories in the style of the generated data (not copied from it).
STORY = ("Chúng ta có ba khối, A, B và C. Khối B ở bên phải khối A. Khối C ở phía dưới B. "
         "Khối A chứa một hình vuông lớn màu đen và một hình tròn nhỏ màu vàng. "
         "Hình tròn nhỏ màu vàng chạm vào cạnh dưới của khối này. Nó ở phía dưới hình vuông lớn màu đen. "
         "Khối B chứa hai hình tam giác vừa màu xanh. Hình tam giác vừa màu xanh số 1 ở bên trái "
         "hình tam giác vừa màu xanh số 2. Ở gần hình tam giác vừa màu xanh số 2 có một hình tròn lớn màu vàng. "
         "Khối C chứa một hình vuông nhỏ màu xanh.")


def question(q_type, text, candidates=()):
    return {"q_id": 1, "q_type": q_type, "question": text, "candidate_answers": list(candidates)}


class StoryTests(unittest.TestCase):
    def setUp(self):
        self.world = read_story(STORY)

    def test_objects_blocks_numbers_and_pronouns(self):
        objects = [(o.block, o.size, o.color, o.shape, o.number) for o in self.world.objects]
        self.assertEqual(objects, [
            ("A", "large", "black", "square", None), ("A", "small", "yellow", "circle", None),
            ("B", "medium", "blue", "triangle", 1), ("B", "medium", "blue", "triangle", 2),
            ("B", "large", "yellow", "circle", None), ("C", "small", "blue", "square", None),
        ])
        self.assertIn((1, "bottom"), self.world.edges)
        self.assertIn(("BELOW", 1, 0), self.world.facts)          # "Nó" = the small yellow circle
        self.assertIn(("NEAR", 4, 3), self.world.facts)           # inverted "Ở gần X có một Y"

    def test_closure_converse_transitivity_and_inheritance(self):
        world = self.world
        self.assertTrue(world.holds("ABOVE", 0, 1))               # converse of BELOW
        self.assertTrue(world.holds("RIGHT", 3, 2))               # converse of LEFT
        self.assertTrue(world.holds("LEFT", 0, 2))                # block A left of block B
        self.assertTrue(world.holds("ABOVE", 2, 5))               # C below B
        self.assertFalse(world.holds("LEFT", 0, 5))               # A vs C unknown
        no_inherit = read_story(STORY, RulesConfig(inherit=()))
        self.assertFalse(no_inherit.holds("LEFT", 0, 2))


class QuestionTests(unittest.TestCase):
    def setUp(self):
        self.solver = Solver(read_story(STORY))

    def ask(self, q_type, text, candidates=()):
        return self.solver.answer(question(q_type, text, candidates))

    def test_yes_no_dk(self):
        self.assertEqual(self.ask("YN", "Hình vuông lớn màu đen có ở bên trái hình tròn lớn màu vàng không?"), ["Yes"])
        self.assertEqual(self.ask("YN", "Hình vuông lớn màu đen có ở bên phải hình tròn lớn màu vàng không?"), ["No"])
        self.assertEqual(self.ask("YN", "Hình vuông lớn màu đen có ở gần hình vuông nhỏ màu xanh không?"), ["DK"])

    def test_existential_clauses_after_nao_describe_the_subject(self):
        text = "Có hình tam giác ở bên trái một hình tam giác nào ở phía trên một hình vuông nhỏ không?"
        self.assertEqual(self.ask("YN", text), ["Yes"])

    def test_universal_is_irreflexive(self):
        text = "Có phải tất cả hình tam giác màu xanh đều ở bên trái tất cả hình màu xanh không?"
        self.assertEqual(self.ask("YN", text), ["No"])

    def test_fr_co_fb(self):
        self.assertEqual(self.ask("FR", "Mối quan hệ giữa hình tròn nhỏ màu vàng và hình tam giác vừa màu xanh "
                                        "số 1 là gì?"), [0])
        self.assertEqual(self.ask("FR", "Mối quan hệ giữa hình vuông lớn màu đen và hình vuông nhỏ màu xanh là gì?"),
                         [7])
        self.assertEqual(self.ask("CO", "Vật nào ở bên trái hình tròn lớn màu vàng: hình vuông lớn màu đen hay "
                                        "hình vuông nhỏ màu xanh?", ["hình vuông lớn màu đen", "hình vuông nhỏ màu xanh",
                                                                    "cả hai", "không"]), [0])
        blocks = ["A", "B", "C"]
        self.assertEqual(self.ask("FB", "Khối nào chứa một hình màu vàng?", blocks), ["A", "B"])
        self.assertEqual(self.ask("FB", "Khối nào không chứa bất kỳ hình tròn nào?", blocks), ["C"])
        self.assertEqual(self.ask("FB", "Khối nào chứa một vật thể ở phía trên một hình vuông nhỏ màu xanh?", blocks),
                         ["B"])

    def test_unknown_forms_abstain(self):
        self.assertIsNone(self.ask("YN", "Câu hỏi lạ không theo mẫu?"))
        self.assertIsNone(Solver(read_story("Có ba khối.")).answer(question("FR", "Mối quan hệ giữa X và Y là gì?")))


class ScriptTests(unittest.TestCase):
    def test_solve_symbolic_writes_predictions_and_abstains_on_unparsed_story(self):
        data = {"name": "SPaRTQA", "data": [
            {"story": [STORY], "questions": [
                {"q_id": 1, "q_type": "YN", "reasoning_type": [], "indifinite": False, "candidate_answers": [],
                 "question": "Hình vuông lớn màu đen có ở bên trái hình tròn lớn màu vàng không?", "answer": ["Yes"]}]},
            {"story": ["Câu chuyện viết tự do mà parser không hiểu."], "questions": [
                {"q_id": 1, "q_type": "YN", "reasoning_type": [], "indifinite": False, "candidate_answers": [],
                 "question": "Có không?", "answer": ["No"]}]},
        ]}
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"WANDB_MODE": "disabled"}):
            root = Path(directory)
            write_json(root / "auto_dev.json", data)
            with redirect_stdout(StringIO()):
                solve_symbolic.main(["--job", f"auto,{root / 'auto_dev.json'},{root / 'out.jsonl'}"])
            records = read_predictions(root / "out.jsonl")
        self.assertEqual(records["0_1"]["answer"], ["Yes"])
        self.assertIsNone(records["1_1"]["answer"])
        self.assertFalse(records["1_1"]["story_parsed"])


if __name__ == "__main__":
    unittest.main()
