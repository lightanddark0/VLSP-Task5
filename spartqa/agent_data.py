"""Prompt/target contracts for the two-agent (extraction + reasoning) pipeline.

Agent 1 (extraction): story+question -> structured facts JSON (entities, triples, query).
Agent 2 (reasoning): story+question[+facts] -> answer JSON, same contract as the
single-agent path in qwen_data.py, so legal_answers/canonical_answer/completion_text
are reused unchanged.

Agent 1 has no gold labels in Data/; its training targets are distilled from a
completed `gpt_experiment.py --method pot` run (GPT-generated graphs), parsed by
graphs_from_pot_log(). Independent of torch/transformers/PEFT so it stays covered
by the offline test suite, like qwen_data.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spartqa.pot import RELATIONS, validate_graph
from spartqa.qwen_data import SYSTEM_INSTRUCTIONS, build_user_content

EXTRACTION_SYSTEM_INSTRUCTIONS = (
    "Bạn là một hệ thống trích xuất tri thức không gian bằng tiếng Việt.\n"
    "Đọc câu chuyện và câu hỏi, rồi trả về DUY NHẤT một đối tượng JSON có đúng 3 khóa:\n"
    '- "entities": danh sách {"id": <mã ngắn duy nhất>, "description": <mô tả bằng tiếng Việt>}.\n'
    '- "triples": danh sách {"head": <id>, "relation": <một trong ' + ", ".join(RELATIONS) + '>, "tail": <id>}, '
    "mỗi phần tử mô tả một quan hệ không gian giữa hai thực thể.\n"
    '- "query": {"source": <id hoặc null>, "target": <id hoặc null>} là hai thực thể chính mà câu hỏi cần đối chiếu.\n'
    "Không giải thích thêm, không thêm khóa nào khác, không lặp lại câu chuyện."
)

REASONING_SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS + (
    "\n\nBạn có thể nhận thêm một khối \"Dữ kiện trích xuất\" tóm tắt các thực thể và quan hệ không gian "
    "trong câu chuyện. Đây là hỗ trợ suy luận, không thay thế; nếu dữ kiện trích xuất thiếu hoặc mâu thuẫn "
    "với câu chuyện gốc, câu chuyện gốc luôn là căn cứ chính xác nhất."
)


def build_extraction_prompt_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": EXTRACTION_SYSTEM_INSTRUCTIONS}]},
        {"role": "user", "content": [{"type": "text", "text": build_user_content(payload)}]},
    ]


def canonical_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Validates a graph and drops pot.py's path-search-only query.pairs/focus fields.

    This pipeline reasons over the full entity/triple set in one shot (no local
    path search), so only the query's source/target focus is kept.
    """
    validated = validate_graph(graph)
    query = validated["query"]
    return {
        "entities": sorted(validated["entities"], key=lambda entity: entity["id"]),
        "triples": validated["triples"],
        "query": {"source": query.get("source"), "target": query.get("target")},
    }


def graph_completion_text(graph: dict[str, Any]) -> str:
    return json.dumps(canonical_graph(graph), ensure_ascii=False)


def graphs_from_pot_log(log_path: str | Path) -> dict[str, dict[str, Any]]:
    """Last validated extraction graph per key from a gpt_experiment.py --method pot/pot-no-path run.

    Reads the run's responses.jsonl checkpoint log directly; entries with an invalid
    or missing graph are skipped rather than raising, since a partial/interrupted run
    is still usable for whatever it did extract successfully.
    """
    graphs: dict[str, dict[str, Any]] = {}
    with Path(log_path).open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("stage") != "extract":
                continue
            result = event.get("result", {})
            if "graph" not in result:
                continue
            try:
                graphs[event["key"]] = canonical_graph(result["graph"])
            except (ValueError, KeyError, TypeError):
                continue
    return graphs


def format_graph_facts(graph: dict[str, Any]) -> str:
    canonical = canonical_graph(graph)
    lines = ["Dữ kiện trích xuất:"]
    for entity in canonical["entities"]:
        lines.append(f"- {entity['id']}: {entity['description']}")
    for triple in canonical["triples"]:
        lines.append(f"- {triple['head']} --{triple['relation']}--> {triple['tail']}")
    query = canonical["query"]
    if query["source"] or query["target"]:
        lines.append(f"- Trọng tâm câu hỏi: source={query['source']}, target={query['target']}")
    return "\n".join(lines)


def build_reasoning_user_content(payload: dict[str, Any], graph: dict[str, Any] | None) -> str:
    base = build_user_content(payload)
    if graph is None:
        return base
    return base + "\n" + format_graph_facts(graph)


def build_reasoning_prompt_messages(payload: dict[str, Any], graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": REASONING_SYSTEM_INSTRUCTIONS}]},
        {"role": "user", "content": [{"type": "text", "text": build_reasoning_user_content(payload, graph)}]},
    ]
