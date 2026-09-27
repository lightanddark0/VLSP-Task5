"""Path-of-Thoughts graph processing for all four SPARTQA question types."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable

from spartqa.api import api_error_details, request_answer


RELATIONS = (
    "far", "in", "touch", "has", "covered_by", "right", "overlap", "front",
    "behind", "cover", "left", "disconnected_from", "below", "above", "near",
)

REASONING_INSTRUCTIONS = """
Path-of-Thoughts evidence:
The payload contains query_entities, query_plan, graph_entities, reasoning_paths and
reasoning_chain. Each statement reads HEAD --relation--> TAIL, meaning HEAD
has that relation relative to TAIL. Traversal does not reverse the statement.
reasoning_paths are evidence paths, NOT independent answers. reasoning_chain
collects their distinct facts. Evaluate all evidence together in ONE answer.
Connectivity alone is not proof of a relation. Never add unit grid offsets or
infer alignment on an unstated axis. near/far/touch are not transitive.
The original story is authoritative if extraction is incomplete or inconsistent.
query_plan.pairs lists the ordered entity pairs to inspect, not true relations.
query_plan.focus lists relevant entities, not the selected answer. Missing or
ambiguous references must be resolved against the question and original story.
If evidence_mode is full_graph or fallback_graph, the chain is the entire graph,
not an ordered path. fallback_graph means some paths were missing or search was
limited; do not treat those gaps as counterexamples. query_facts includes facts
incident to the focused entities, including intermediate reference descriptions.
graph_entities lists even isolated objects; containment_facts lists membership
and boundary contact. Check the full story inventory for quantifiers and absence;
missing paths or missing triples do NOT establish a negative spatial fact.
Internal in/has and covered_by/cover relations describe containment; covered_by
also includes contact with the container boundary. Internal overlap is NOT
touching. Boundary contact is not contact with other objects in that block.
Entities described as block boundaries are not inventory objects. Contact with
a named side of an object can establish direction relative to that object;
contact with a block side does not establish position relative to every member.
Follow q_type, not the internal relation names:
YN: evaluate the complete proposition, including negation and quantifiers, using
Yes/No/DK. An unknown relation is not false. For all, one counterexample refutes
the claim; for any, one witness proves it. Unresolved cases can require DK.
FB: inspect EVERY candidate block and return all qualifying identifiers, or [].
For absence, use an exhaustive story inventory, not a missing extracted edge.
CO: evaluate only candidate 0 and 1 against the complete condition, then return
[0], [1], [2] (both), or [3] (neither established). Negation requires evidence;
do not count an unknown positive relation as a proven negative one.
FR: return all supported labels 0..6, or [7] only if none is supported. Verify
direction and scope against the story before including conflicting labels from
different paths. Do not blindly union predictions for different possible objects.
Give a concise Vietnamese evidence summary and the typed answer list.
"""


def graph_response_format() -> dict[str, Any]:
    def object_schema(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    text = {"type": "string"}
    pair = object_schema({"source": text, "target": text})
    schema = object_schema({
        "entities": {"type": "array", "items": object_schema({"id": text, "description": text})},
        "triples": {"type": "array", "items": object_schema({
            "head": text, "relation": {"type": "string", "enum": list(RELATIONS)}, "tail": text,
        })},
        "query": object_schema({"source": {"type": ["string", "null"]},
                                "target": {"type": ["string", "null"]},
                                "pairs": {"type": "array", "items": pair},
                                "focus": {"type": "array", "items": text}}),
    })
    return {"type": "json_schema", "json_schema": {"name": "spatial_graph", "strict": True, "schema": schema}}


def request_graph(client: Any, example: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": config["extraction_prompt"]},
            {"role": "user", "content": json.dumps(example["payload"], ensure_ascii=False)},
        ],
        temperature=config["temperature"],
        max_completion_tokens=config["max_extraction_tokens"],
        response_format=graph_response_format(),
    )
    record = {"response_id": response.id, "response_model": response.model,
              "usage": response.usage.model_dump() if response.usage else {}}
    if not response.choices:
        return {**record, "error": "EmptyGraphResponse"}
    choice = response.choices[0]
    if choice.finish_reason != "stop" or choice.message.refusal:
        return {**record, "error": "RefusedOrIncompleteGraph"}
    try:
        graph = validate_graph(json.loads(choice.message.content or ""))
    except (ValueError, KeyError, TypeError) as error:
        return {**record, "error": "InvalidGraph",
                "validation_error": {"type": type(error).__name__, "message": str(error)},
                "raw_graph_response": choice.message.content}
    return {**record, "graph": graph}


def request_pot_answer(
    client: Any, example: dict[str, Any], config: dict[str, Any],
    cached: dict[str, dict[str, Any]], checkpoint: Callable[[str, dict[str, Any]], None],
) -> dict[str, Any]:
    def stage(name, operation):
        if name in cached:
            return cached[name]
        try:
            result = operation()
        except Exception as error:
            result = {"error": type(error).__name__, "error_details": api_error_details(error)}
        checkpoint(name, result)
        return result

    record: dict[str, Any] = {"key": example["key"], "method": config["method"]}
    extraction = stage("extract", lambda: request_graph(client, example, config))
    if "error" in extraction:
        return {**record, "error": extraction["error"], "error_stage": "extract",
                **({"validation_error": extraction["validation_error"]} if "validation_error" in extraction else {}),
                **({"error_details": extraction["error_details"]} if "error_details" in extraction else {})}
    graph = extraction["graph"]
    record["graph"] = graph
    if config["method"] == "pot-no-path":
        search = {"paths": [list(range(len(graph["triples"])))], "limits_hit": [], "expansions": 0}
        evidence_mode = "full_graph"
    else:
        search = identify_query_paths(graph, config["max_paths"], config["max_hops"], config["max_expansions"])
        evidence_mode = "path"
    record["path_search"] = search
    record["fallback"] = evidence_mode == "path" and (
        not search["paths"] or bool(search["limits_hit"]) or search.get("incomplete", False)
    )
    if record["fallback"]:
        evidence_mode = "fallback_graph"
    reasoning_config = {**config, "prompt": config["prompt"] + config["reasoning_instructions"]}
    descriptions = {entity["id"]: entity["description"] for entity in graph["entities"]}
    edge_indices = list(dict.fromkeys(edge for path in search["paths"] for edge in path))
    if evidence_mode != "path":
        edge_indices = list(range(len(graph["triples"])))
    focus = set(graph["query"].get("focus", []))
    payload = {
        **example["payload"], "evidence_mode": evidence_mode,
        "query_entities": {name: descriptions.get(graph["query"][name]) for name in ("source", "target")},
        "query_plan": graph["query"],
        "graph_entities": graph["entities"],
        "reasoning_paths": [path_statements(graph, path) for path in search["paths"]]
                           if config["method"] == "pot" else [],
        "reasoning_chain": path_statements(graph, edge_indices),
        "query_facts": path_statements(graph, [
            index for index, triple in enumerate(graph["triples"])
            if triple["head"] in focus or triple["tail"] in focus
        ]),
        "containment_facts": path_statements(graph, [
            index for index, triple in enumerate(graph["triples"])
            if triple["relation"] in {"in", "has", "covered_by", "cover"}
        ]),
        "search_limits_hit": search["limits_hit"],
    }
    result = stage("reason:joint", lambda: request_answer(
        client, {"key": example["key"], "payload": payload}, reasoning_config,
    ))
    record["path_results"] = [result]
    record["path_disagreement"] = None
    if "error" in result:
        return {**record, "error": result["error"], "error_stage": "reason:joint",
                **({"error_details": result["error_details"]} if "error_details" in result else {})}
    record["answer"] = result["answer"]
    return record


def validate_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"entities", "triples", "query"}:
        raise ValueError("Graph requires entities, triples and query")
    if not isinstance(value["entities"], list) or not isinstance(value["triples"], list):
        raise ValueError("Entities and triples must be lists")
    entities = {}
    for entity in value["entities"]:
        if not isinstance(entity, dict) or set(entity) != {"id", "description"}:
            raise ValueError("Entity requires id and description")
        if any(not isinstance(field, str) or not field.strip() for field in entity.values()):
            raise ValueError("Entity fields must be nonempty strings")
        if entity["id"] in entities:
            raise ValueError("Duplicate entity id")
        entities[entity["id"]] = entity["description"]
    triples = []
    seen = set()
    for triple in value["triples"]:
        if not isinstance(triple, dict) or set(triple) != {"head", "relation", "tail"}:
            raise ValueError("Triple requires head, relation and tail")
        if any(not isinstance(field, str) for field in triple.values()):
            raise ValueError("Triple fields must be strings")
        if triple["head"] not in entities or triple["tail"] not in entities:
            raise ValueError("Triple references an unknown entity")
        if triple["relation"] not in RELATIONS:
            raise ValueError("Unknown spatial relation")
        identity = (triple["head"], triple["relation"], triple["tail"])
        if identity not in seen:
            triples.append(dict(triple))
            seen.add(identity)
    query = value["query"]
    if not isinstance(query, dict) or set(query) not in (
        {"source", "target"}, {"source", "target", "pairs", "focus"},
    ):
        raise ValueError("Query requires source/target and optionally pairs/focus")
    for endpoint in (query["source"], query["target"]):
        if endpoint is not None and (not isinstance(endpoint, str) or endpoint not in entities):
            raise ValueError("Query references an unknown entity")
    if "pairs" in query:
        if not isinstance(query["pairs"], list) or not isinstance(query["focus"], list):
            raise ValueError("Query pairs and focus must be lists")
        for pair in query["pairs"]:
            if not isinstance(pair, dict) or set(pair) != {"source", "target"}:
                raise ValueError("Query pair requires source and target")
            if any(not isinstance(endpoint, str) or endpoint not in entities for endpoint in pair.values()):
                raise ValueError("Query pair references an unknown entity")
        if any(not isinstance(endpoint, str) or endpoint not in entities for endpoint in query["focus"]):
            raise ValueError("Query focus references an unknown entity")
    return {"entities": list(value["entities"]), "triples": triples, "query": dict(query)}


def identify_query_paths(
    graph: dict[str, Any], max_paths: int = 16, max_hops: int = 0,
    max_expansions: int = 10000,
) -> dict[str, Any]:
    if max_paths < 0 or max_hops < 0 or max_expansions < 1:
        raise ValueError("Invalid path search limits")
    query = graph["query"]
    pairs = list(query.get("pairs", []))
    if query["source"] is not None and query["target"] is not None:
        pairs.insert(0, {"source": query["source"], "target": query["target"]})
    endpoints = list(dict.fromkeys((pair["source"], pair["target"]) for pair in pairs))
    result: dict[str, Any] = {"paths": [], "pair_searches": [], "limits_hit": [],
                              "expansions": 0, "incomplete": not bool(endpoints)}
    if not endpoints:
        result["no_path_reason"] = "unresolved_query"
        return result
    seen_paths = set()
    for source, target in endpoints:
        remaining = max_expansions - result["expansions"]
        if remaining <= 0:
            result["incomplete"] = True
            if "max_expansions" not in result["limits_hit"]:
                result["limits_hit"].append("max_expansions")
            result["pair_searches"].append({"source": source, "target": target,
                                            "paths": [], "no_path_reason": "budget_exhausted"})
            continue
        search = identify_paths({**graph, "query": {"source": source, "target": target}},
                                max_paths, max_hops, remaining)
        result["pair_searches"].append({"source": source, "target": target, **search})
        result["expansions"] += search["expansions"]
        result["incomplete"] |= not bool(search["paths"])
        for limit in search["limits_hit"]:
            if limit not in result["limits_hit"]:
                result["limits_hit"].append(limit)
        for path in search["paths"]:
            identity = tuple(path)
            if identity not in seen_paths:
                result["paths"].append(path)
                seen_paths.add(identity)
    return result


def identify_paths(
    graph: dict[str, Any], max_paths: int = 16, max_hops: int = 0,
    max_expansions: int = 10000,
) -> dict[str, Any]:
    if max_paths < 0 or max_hops < 0 or max_expansions < 1:
        raise ValueError("Invalid path search limits")
    source, target = graph["query"]["source"], graph["query"]["target"]
    result: dict[str, Any] = {"paths": [], "limits_hit": [], "expansions": 0}
    if source is None or target is None:
        result["no_path_reason"] = "unresolved_query"
        return result
    if source == target:
        result["paths"] = [[]]
        return result
    adjacency = defaultdict(list)
    for index, triple in enumerate(graph["triples"]):
        head, tail = triple["head"], triple["tail"]
        adjacency[head].append((tail, index))
        adjacency[tail].append((head, index))
    reachable = {target}
    pending = [target]
    while pending:
        for neighbor, _ in adjacency[pending.pop()]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                pending.append(neighbor)
    if source not in reachable:
        result["no_path_reason"] = "disconnected"
        return result
    hop_limit = max_hops or max(len(graph["entities"]) - 1, 1)
    stack = [(source, {source}, [], iter(adjacency[source]))]
    while stack:
        current, visited, path, neighbors = stack[-1]
        next_edge = next(neighbors, None)
        if next_edge is None:
            stack.pop()
            continue
        if result["expansions"] >= max_expansions:
            result["limits_hit"].append("max_expansions")
            break
        result["expansions"] += 1
        neighbor, edge_index = next_edge
        if neighbor in visited:
            continue
        next_path = path + [edge_index]
        if neighbor == target:
            if max_paths and len(result["paths"]) >= max_paths:
                result["limits_hit"].append("max_paths")
                break
            result["paths"].append(next_path)
        elif len(next_path) < hop_limit:
            stack.append((neighbor, visited | {neighbor}, next_path, iter(adjacency[neighbor])))
        elif max_hops and "max_hops" not in result["limits_hit"]:
            result["limits_hit"].append("max_hops")
    if not result["paths"]:
        result["no_path_reason"] = "search_limited"
    return result


def path_statements(graph: dict[str, Any], path: list[int]) -> list[str]:
    descriptions = {entity["id"]: entity["description"] for entity in graph["entities"]}
    return [
        f"{descriptions[triple['head']]} --{triple['relation']}--> {descriptions[triple['tail']]}"
        for triple in (graph["triples"][index] for index in path)
    ]


def aggregate_fr(answers: list[list[int]]) -> list[int]:
    if not answers:
        raise ValueError("Cannot aggregate missing predictions")
    known = sorted({label for answer in answers for label in answer if label != 7})
    return known or [7]