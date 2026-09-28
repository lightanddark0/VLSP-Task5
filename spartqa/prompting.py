"""Prompt text and answer parsing for the generative (LLM) branches.

The fine-tuned model (branch F) and the chain-of-thought model (branch L-CoT)
both read the story, question, and options as chat messages, and must print
the answer in the task's JSON list format. The same messages are used at
training and inference time. Metadata fields such as reasoning_type and
indifinite are never shown to a model; see spartqa.postprocess instead.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from spartqa.data import validate_answer

FR_LABELS = ["bên trái", "bên phải", "bên trên", "bên dưới", "gần", "xa", "chạm vào", "DK"]
CO_EXTRA = ["cả hai", "không cái nào"]

SYSTEM_PROMPT = (
    "Bạn là hệ thống trả lời câu hỏi suy luận không gian bằng tiếng Việt. "
    "Đọc câu chuyện mô tả các khối và vật thể, rồi trả lời câu hỏi. "
    "Chỉ in đáp án dưới dạng JSON list, không giải thích."
)

ANSWER_FORMATS = {
    "YN": '["Yes"], ["No"] hoặc ["DK"] (DK: không xác định được từ câu chuyện)',
    "FR": "JSON list các chỉ số quan hệ đúng, tăng dần, ví dụ [0, 5]; [7] nếu không xác định được",
    "FB": 'JSON list tên các khối thoả mãn, ví dụ ["A", "C"]; [] nếu không có khối nào',
    "CO": "[0] vật thứ nhất, [1] vật thứ hai, [2] cả hai, [3] không cái nào",
}

COT_SYSTEM_PROMPT = """Bạn là chuyên gia suy luận không gian. Mỗi câu chuyện mô tả các khối (A, B, C, ...) và các vật thể (hình vuông, hình tròn, hình tam giác) có màu và kích thước, cùng quan hệ giữa chúng.

Các quan hệ (chỉ số dùng cho câu hỏi FR):
0 bên trái, 1 bên phải, 2 bên trên, 3 bên dưới, 4 gần, 5 xa, 6 chạm vào, 7 DK (không xác định được).

Quy tắc suy luận:
- Đảo chiều: X bên trái Y ⇔ Y bên phải X; X bên trên Y ⇔ Y bên dưới X.
- Đối xứng: gần, xa, chạm vào.
- Bắc cầu: trái, phải, trên, dưới.
- Quan hệ giữa hai khối áp dụng cho mọi cặp vật thể nằm trong hai khối đó.
- Chỉ kết luận điều được nêu hoặc suy ra được từ câu chuyện.

Loại câu hỏi:
- YN: ["Yes"] nếu suy ra được là đúng, ["No"] nếu suy ra được là sai, ["DK"] nếu câu chuyện không đủ thông tin.
- FR: mọi quan hệ đúng giữa hai thực thể, ví dụ [0, 5]; [7] nếu không suy ra được quan hệ nào.
- FB: mọi khối thoả điều kiện, ví dụ ["A", "C"]; [] nếu không có khối nào.
- CO: [0] vật thứ nhất, [1] vật thứ hai, [2] cả hai, [3] không cái nào.

Cách làm: (1) liệt kê các khối và vật thể kèm thuộc tính, (2) vẽ sơ đồ vị trí bằng chữ, (3) suy luận từng bước, (4) dòng cuối cùng in đúng dạng
ĐÁP ÁN: <JSON list>"""

ANSWER_MARKER = "ĐÁP ÁN:"


def story_text(story: str | list[str]) -> str:
    if isinstance(story, list):
        return " ".join(part.strip() for part in story if part.strip())
    return story.strip()


def option_lines(payload: dict[str, Any]) -> str:
    task = payload["q_type"]
    candidates = list(payload.get("candidate_answers") or [])
    if task == "FR":
        labels = candidates if len(candidates) == len(FR_LABELS) else FR_LABELS
        return " | ".join(f"{index}: {label}" for index, label in enumerate(labels))
    if task == "CO":
        if len(candidates) == 2:
            candidates += CO_EXTRA
        return " | ".join(f"{index}: {label}" for index, label in enumerate(candidates))
    if task == "FB":
        return ", ".join(str(block) for block in candidates)
    return "Yes | No | DK"


def question_block(payload: dict[str, Any], dataset: str) -> str:
    return "\n".join([
        f"Bộ dữ liệu: {dataset}",
        f"Câu chuyện: {story_text(payload['story'])}",
        f"Loại câu hỏi: {payload['q_type']}",
        f"Câu hỏi: {payload['question'].strip()}",
        f"Lựa chọn: {option_lines(payload)}",
        f"Định dạng đáp án: {ANSWER_FORMATS[payload['q_type']]}",
    ])


def build_messages(payload: dict[str, Any], dataset: str) -> list[dict[str, str]]:
    """Chat messages for the fine-tuned model (branch F)."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_block(payload, dataset)}]


def build_cot_messages(payload: dict[str, Any], dataset: str,
                       examples: list[tuple[dict[str, Any], list[Any]]] = ()) -> list[dict[str, str]]:
    """Chat messages for the thinking model (branch L-CoT), with optional solved examples."""
    parts = []
    for index, (example, answer) in enumerate(examples, 1):
        parts.append(f"### Ví dụ {index}\n{question_block(example, dataset)}\n"
                     f"{ANSWER_MARKER} {format_answer(answer, example)}")
    parts.append(f"### Câu hỏi cần trả lời\n{question_block(payload, dataset)}")
    return [{"role": "system", "content": COT_SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
    """Prompt string ending where the assistant's answer starts (Qwen3 chat template)."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=enable_thinking)


def prompt_version(kind: str = "F") -> str:
    """Short content hash that changes whenever the prompt wording changes."""
    text = SYSTEM_PROMPT + json.dumps(ANSWER_FORMATS, ensure_ascii=False) if kind == "F" else COT_SYSTEM_PROMPT
    return f"{kind}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:10]}"


def canonical_answer(answer: list[Any], payload: dict[str, Any]) -> list[Any]:
    if payload["q_type"] == "FR":
        return sorted(answer)
    if payload["q_type"] == "FB":
        order = {block: index for index, block in enumerate(payload["candidate_answers"])}
        return sorted(answer, key=lambda block: order.get(block, len(order)))
    return list(answer)


def format_answer(answer: list[Any], payload: dict[str, Any]) -> str:
    return json.dumps(canonical_answer(answer, payload), ensure_ascii=False)


_LIST = re.compile(r"\[[^\[\]]*\]")
_YN = {"yes": "Yes", "có": "Yes", "no": "No", "không": "No", "dk": "DK"}


def _coerce(values: list[Any], task: str) -> list[Any]:
    if task in {"FR", "CO"}:
        return [int(value) if isinstance(value, str) and value.strip().isdigit() else value for value in values]
    if task == "YN":
        return [_YN.get(value.strip().lower(), value) if isinstance(value, str) else value for value in values]
    return [value.strip().upper() if isinstance(value, str) else value for value in values]


def parse_answer(text: str, payload: dict[str, Any]) -> list[Any] | None:
    """Return a valid answer from the first JSON list in a generation, or None."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    match = _LIST.search(text)
    if match is None:
        return None
    try:
        answer = _coerce(json.loads(match.group(0)), payload["q_type"])
        answer = list(dict.fromkeys(answer))
        validate_answer(answer, payload)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return canonical_answer(answer, payload)


def parse_cot_answer(text: str, payload: dict[str, Any]) -> list[Any] | None:
    """Answer after the last 'ĐÁP ÁN:' outside the thinking block; None if missing or truncated."""
    if "<think>" in text and "</think>" not in text:
        return None
    visible = text.rsplit("</think>", 1)[-1]
    if ANSWER_MARKER not in visible:
        return None
    return parse_answer(visible.rsplit(ANSWER_MARKER, 1)[1], payload)


# --- E1: FR as a checklist of four independent axis questions ----------------------------------
FR_AXES = {
    "horizontal": ("chiều ngang (trái/phải)", (0, 1),
                   "[0] nếu vật thứ nhất ở bên trái vật thứ hai, [1] nếu ở bên phải, [] nếu không suy ra được"),
    "vertical": ("chiều dọc (trên/dưới)", (2, 3),
                 "[2] nếu vật thứ nhất ở bên trên vật thứ hai, [3] nếu ở bên dưới, [] nếu không suy ra được"),
    "distance": ("khoảng cách (gần/xa)", (4, 5),
                 "[4] nếu vật thứ nhất ở gần vật thứ hai, [5] nếu ở xa, [] nếu không suy ra được"),
    "touch": ("tiếp xúc (chạm)", (6,), "[6] nếu vật thứ nhất chạm vào vật thứ hai, [] nếu không suy ra được"),
}


def axis_answer(answer: list[int], axis: str) -> list[int]:
    labels = FR_AXES[axis][1]
    return sorted(label for label in answer if label in labels)


def _axis_block(payload: dict[str, Any], dataset: str, axis: str) -> str:
    name, _, answer_format = FR_AXES[axis]
    return "\n".join([
        f"Bộ dữ liệu: {dataset}",
        f"Câu chuyện: {story_text(payload['story'])}",
        f"Câu hỏi gốc: {payload['question'].strip()}",
        f"Chỉ xét quan hệ theo {name} giữa vật thứ nhất và vật thứ hai trong câu hỏi gốc.",
        f"Định dạng đáp án: {answer_format}",
    ])


def build_fr_axis_messages(payload: dict[str, Any], dataset: str, axis: str,
                           examples: list[tuple[dict[str, Any], list[Any]]] = ()) -> list[dict[str, str]]:
    """One axis of an FR question; solved examples show only that axis of their answers."""
    parts = []
    for index, (example, answer) in enumerate(examples, 1):
        parts.append(f"### Ví dụ {index}\n{_axis_block(example, dataset, axis)}\n"
                     f"{ANSWER_MARKER} {json.dumps(axis_answer(answer, axis))}")
    parts.append(f"### Câu hỏi cần trả lời\n{_axis_block(payload, dataset, axis)}")
    return [{"role": "system", "content": COT_SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


def parse_axis_answer(text: str, axis: str) -> list[int] | None:
    """Labels of one axis after the last 'ĐÁP ÁN:'; [] allowed; None if missing or contradictory."""
    if "<think>" in text and "</think>" not in text:
        return None
    visible = text.rsplit("</think>", 1)[-1]
    if ANSWER_MARKER not in visible:
        return None
    match = _LIST.search(visible.rsplit(ANSWER_MARKER, 1)[1])
    if match is None:
        return None
    try:
        values = _coerce(json.loads(match.group(0)), "FR")
    except (ValueError, TypeError):
        return None
    labels = FR_AXES[axis][1]
    if any(value not in labels for value in values) or len(set(values)) > 1:
        return None
    return sorted(set(values))


def merge_axis_samples(per_axis: dict[str, list[list[int] | None]]) -> list[list[int] | None]:
    """Sample i of the checklist = union over axes of sample i; [7] when no axis holds; None if all failed."""
    count = max((len(samples) for samples in per_axis.values()), default=0)
    merged = []
    for index in range(count):
        parts = [samples[index] if index < len(samples) else None for samples in per_axis.values()]
        if all(part is None for part in parts):
            merged.append(None)
            continue
        labels = sorted({label for part in parts if part for label in part})
        merged.append(labels or [7])
    return merged
