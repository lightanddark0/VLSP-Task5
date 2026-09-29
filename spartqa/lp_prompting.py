"""Prompts for branch LP (E3): story -> world JSON, question -> form JSON.

The worked examples are the hand annotations in spartqa/symbolic/lp_annotations.json;
their story and question text is read from the data file at run time, so no
organizer text is stored in the repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from spartqa.data import read_json
from spartqa.prompting import option_lines, story_text

ANNOTATIONS = Path(__file__).resolve().parent / "symbolic" / "lp_annotations.json"

VOCABULARY = """Từ vựng (luôn dùng giá trị tiếng Anh này):
- shape: hình vuông → "square", hình tròn → "circle", hình tam giác/tam giác → "triangle", hình chữ nhật → "rectangle"; "vật", "vật thể", "thứ", "hình" không rõ loại → null.
- size: nhỏ → "small", trung bình/vừa → "medium", lớn/to → "large"; không nói → null.
- color: xanh lam/xanh dương/xanh → "blue", vàng → "yellow", đen → "black", đỏ → "red", xanh lá/xanh lá cây → "green", trắng → "white"; không nói → null.
- quan hệ: bên trái → "left", bên phải → "right", phía trên/bên trên/trên → "above", phía dưới/bên dưới/dưới → "below", gần → "near", xa → "far", chạm vào → "touch".
- cạnh của khối: trái "left", phải "right", trên/đỉnh "top", dưới/đáy "bottom"."""

WORLD_PROMPT = f"""Bạn chuyển một câu chuyện tiếng Việt mô tả các khối và vật thể thành JSON. Không suy luận, không trả lời câu hỏi: chỉ ghi lại đúng những gì câu chuyện nói.

Định dạng:
{{"blocks": ["A", "B"],
 "objects": [{{"id": "a1", "block": "A", "shape": "square", "size": "small", "color": "blue"}}],
 "facts": [["a1", "far", "a2"], ["a1", "above", "a2"], ["B", "right", "A"]],
 "edges": [["a1", "right"]]}}

{VOCABULARY}

Quy tắc:
1. Mỗi vật được giới thiệu là một object riêng với id duy nhất (a1, a2... cho khối A; b1... cho khối B). Khi câu sau nhắc lại một vật đã có ("hình vuông màu xanh lam", "nó", "tam giác màu đen"), dùng lại id cũ, không tạo vật mới. Hai vật giống hệt nhau vẫn là hai object.
2. Thuộc tính không được nêu thì để null, không đoán.
3. Mỗi fact ["X", rel, "Y"] nghĩa là X nằm ở vị trí rel so với Y. "X ở xa phía trên Y" là hai fact: [X far Y], [X above Y]. "Gần và bên phải của Y là X" nghĩa là X near Y, X right Y (vật được giới thiệu là chủ ngữ). "X ở bên phải cả hai Y và Z" cho fact với từng vật.
4. Quan hệ giữa hai khối ghi bằng tên khối: "Trong B, ở bên phải của A" → ["B", "right", "A"]; "C nằm bên phải B" → ["C", "right", "B"].
5. "X chạm vào cạnh phải của khối A" → edges ["X", "right"]. "X chạm vào đỉnh của vật Y" → [X touch Y] và [X above Y]; "chạm vào đáy của Y" → [X touch Y], [X below Y]; "chạm vào cạnh trái của Y" → [X touch Y], [X left Y].
6. Chỉ ghi fact được nói rõ. Không tự thêm quan hệ đảo chiều, bắc cầu, hay "xa" giữa các khối; bộ suy luận sẽ làm việc đó.

Nghĩ ngắn gọn, rồi in kết quả cuối cùng trong một khối ```json ... ```."""

QUESTION_PROMPT = f"""Bạn chuyển một câu hỏi tiếng Việt về vị trí các vật thành dạng logic JSON. Không trả lời câu hỏi.

Mô tả D (một tập vật): {{"shape", "size", "color", "block", "edge", "rels"}}; chỉ ghi khoá cần thiết, bỏ qua hoặc null nghĩa là "bất kỳ".
- "block": khối chứa vật, ví dụ "vật màu đen ở A" → {{"color": "black", "block": "A"}}.
- "edge": vật chạm vào cạnh của khối nó nằm trong: "left"/"right"/"top"/"bottom", hoặc "any" nếu không nói cạnh nào.
- "rels": [{{"rel": ["near", "above"], "quant": "a" | "all", "target": D}}] — vật phải có TẤT CẢ quan hệ trong "rel" với một vật ("a") hoặc với mọi vật ("all") thuộc target. Dùng "all" khi câu nói "tất cả các ...".

Dạng theo loại câu hỏi:
- YN: {{"subject": D, "quant": "the" | "some" | "all", "predicate": P, "negated": false}}. "Có phải tất cả X đều ..." → "all"; "Có X nào ..." → "some"; "X có ... không" → "the". P giống D nhưng không có shape/size/color: điều chủ ngữ phải thoả. "negated": true nếu câu hỏi phủ định ("X không ở bên trái Y phải không").
- FR: {{"first": D, "second": D}} — quan hệ của first so với second. "Mối quan hệ giữa X và Y" và "X ở đâu so với Y" đều là first = X, second = Y.
- FB: {{"condition": "has" | "has_not" | "has_all", "object": D}}. "Khối nào có X" → "has"; "Khối nào không có X" → "has_not"; "Khối nào có tất cả các X bên trong" → "has_all". "Khối nào không có vật chạm vào cạnh của nó" → has_not {{"edge": "any"}}.
- CO: {{"options": [D1, D2], "predicate": P, "negated": false}} — D1, D2 là hai vật đầu tiên trong danh sách lựa chọn; "Vật nào không ở gần Y" → "negated": true.
- Câu không diễn đạt được bằng dạng này (ví dụ "nằm giữa hai vật") → {{"unsupported": true}}.

{VOCABULARY}

Dùng câu chuyện để hiểu câu hỏi nhắc đến vật nào, nhưng chỉ chép lại mô tả như câu hỏi viết (thuộc tính, khối), không thêm thuộc tính mà câu hỏi không nói.

Nghĩ ngắn gọn, rồi in kết quả cuối cùng trong một khối ```json ... ```."""


def story_sha(story: Any) -> str:
    return hashlib.sha256(" ".join(story).encode() if isinstance(story, list) else story.encode()).hexdigest()[:16]


def load_examples(annotations: Path = ANNOTATIONS, data_path: str | Path | None = None,
                  count: int = 1) -> list[dict[str, Any]]:
    """Annotated stories with their text: {"story", "sha", "world", "questions": [(question, form)]}."""
    spec = read_json(annotations)
    data = read_json(data_path or spec["source"])
    examples = []
    for entry in spec["stories"][:count]:
        item = data["data"][entry["index"]]
        if story_sha(item["story"]) != entry["story_sha256"]:
            raise ValueError(f"{data_path or spec['source']} story {entry['index']} does not match the annotation")
        questions = {str(q["q_id"]): q for q in item["questions"]}
        examples.append({"story": item["story"], "sha": entry["story_sha256"], "world": entry["world"],
                         "questions": [(questions[qid], form) for qid, form in entry["forms"].items()]})
    return examples


def _json(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False) + "\n```"


def world_messages(story: Any, examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    parts = [f"### Ví dụ\nCâu chuyện: {story_text(e['story'])}\nKết quả:\n{_json(e['world'])}" for e in examples]
    parts.append(f"### Câu chuyện cần chuyển\n{story_text(story)}")
    return [{"role": "system", "content": WORLD_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


def question_block(payload: dict[str, Any]) -> str:
    lines = [f"Loại câu hỏi: {payload['q_type']}", f"Câu hỏi: {payload['question'].strip()}"]
    if payload["q_type"] in {"CO", "FB"}:
        lines.append(f"Lựa chọn: {option_lines(payload)}")
    return "\n".join(lines)


def question_messages(payload: dict[str, Any], examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Examples: every annotated question of the example stories, same type first."""
    parts = []
    for example in examples:
        shots = sorted(example["questions"], key=lambda pair: pair[0]["q_type"] != payload["q_type"])
        lines = [f"Câu chuyện: {story_text(example['story'])}"]
        for question, form in shots:
            shot = {"q_type": question["q_type"], "question": question["question"],
                    "candidate_answers": question["candidate_answers"]}
            lines.append(f"{question_block(shot)}\nKết quả: {json.dumps(form, ensure_ascii=False)}")
        parts.append("### Ví dụ\n" + "\n\n".join(lines))
    parts.append(f"### Câu hỏi cần chuyển\nCâu chuyện: {story_text(payload['story'])}\n{question_block(payload)}")
    return [{"role": "system", "content": QUESTION_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


def extract_json(text: str) -> Any:
    """The last JSON object in a generation, after any thinking block; None if there is none."""
    if "<think>" in text and "</think>" not in text:
        return None
    visible = text.rsplit("</think>", 1)[-1]
    decoder = json.JSONDecoder()
    found, index = None, visible.find("{")
    while index != -1:
        try:
            value, end = decoder.raw_decode(visible, index)
        except json.JSONDecodeError:
            index = visible.find("{", index + 1)
            continue
        if isinstance(value, dict):
            found = value
        index = visible.find("{", end)
    return found


def prompt_version() -> str:
    text = WORLD_PROMPT + QUESTION_PROMPT + ANNOTATIONS.read_text(encoding="utf-8")
    return f"LP-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:10]}"
