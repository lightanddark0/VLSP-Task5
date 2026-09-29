"""E3 (branch LP): worlds and questions written as JSON by an LLM, answered by the S reasoner.

The LLM only transcribes what the text says; every inference (converse,
transitivity, block-to-object inheritance, optional FAR rules) is done by
World.closure. JSON that does not fit the schema raises StructureError, and
the caller abstains for that sample.

World JSON (one per story):
    {"blocks": ["A", "B"],
     "objects": [{"id": "a1", "block": "A", "shape": "square", "size": "small", "color": "blue"}],
     "facts": [["a1", "far", "a2"], ["a1", "above", "a2"], ["B", "right", "A"]],
     "edges": [["a1", "right"]]}
shape/size/color may be null. A fact links two objects or two blocks and
states the first one's position relative to the second.

Description D (a set of objects): {"shape", "size", "color", "block", "edge",
"rels": [{"rel": [...], "quant": "a"|"all", "target": D}]}; missing keys and
nulls mean "any". "edge" is a side or "any" (touches that edge of its block).

Question forms, one per q_type:
    YN {"subject": D, "quant": "the"|"some"|"all", "predicate": P, "negated": bool}
    FR {"first": D, "second": D}
    FB {"condition": "has"|"has_not"|"has_all", "object": D}
    CO {"options": [D, D], "predicate": P, "negated": bool}
P is a description without shape/size/color: what the subject must satisfy.
{"unsupported": true} is a valid form meaning "cannot be expressed": abstain.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import product
from typing import Any

from spartqa.symbolic.grammar import ParseError
from spartqa.symbolic.lexicon import FR_INDEX, OPPOSITE
from spartqa.symbolic.world import Obj, RulesConfig, World

RELATIONS = {"left": "LEFT", "right": "RIGHT", "above": "ABOVE", "below": "BELOW",
             "near": "NEAR", "far": "FAR", "touch": "TOUCH"}
SIDES = ("left", "right", "top", "bottom")
ATTRIBUTES = ("shape", "size", "color")


class StructureError(ParseError):
    pass


@dataclass
class LPConfig:
    """Answering conventions; each is a switch so it can be tuned on Human train."""
    yn_mode: str = "closed"          # "closed": not provable -> No; "open": No only if refuted, else DK
    fr_multi: str = "all"            # several object pairs: labels holding for "all" pairs or "any" pair
    fb_scope: str = "global"         # clause targets anywhere ("global") or in the candidate block ("block")
    far_chain: bool = True
    cross_block_far: bool = False

    def rules(self) -> RulesConfig:
        return RulesConfig(far_chain=self.far_chain, cross_block_far=self.cross_block_far)

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


def _text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise StructureError(f"{field} must be a nonempty string or null")
    return value.strip().lower()


def _block(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructureError("block names must be strings")
    return value.strip().upper()


def world_from_json(data: Any, rules: RulesConfig | None = None) -> World:
    if not isinstance(data, dict):
        raise StructureError("world must be an object")
    world = World(rules=rules or RulesConfig())
    world.blocks = [_block(name) for name in data.get("blocks") or []]
    ids: dict[str, int] = {}
    for item in data.get("objects") or []:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] in ids:
            raise StructureError("each object needs a unique string id")
        block = _block(item.get("block"))
        if block not in world.blocks:
            world.blocks.append(block)
        ids[item["id"]] = len(world.objects)
        world.objects.append(Obj(len(world.objects), block, _text(item.get("shape"), "shape"),
                                 _text(item.get("size"), "size"), _text(item.get("color"), "color")))

    def node(name: Any) -> int | str:
        if isinstance(name, str) and name in ids:
            return ids[name]
        if isinstance(name, str) and name.strip().upper() in world.blocks:
            return name.strip().upper()
        raise StructureError(f"unknown object or block {name!r}")

    for fact in data.get("facts") or []:
        if not isinstance(fact, list) or len(fact) != 3 or fact[1] not in RELATIONS:
            raise StructureError(f"bad fact {fact!r}")
        a, b = node(fact[0]), node(fact[2])
        if isinstance(a, str) != isinstance(b, str):
            raise StructureError("a fact must link two objects or two blocks")
        world.add_fact(RELATIONS[fact[1]], a, b)
    for edge in data.get("edges") or []:
        if not isinstance(edge, list) or len(edge) != 2 or edge[1] not in SIDES:
            raise StructureError(f"bad edge {edge!r}")
        target = node(edge[0])
        if isinstance(target, str):
            raise StructureError("an edge contact needs an object")
        world.edges.add((target, edge[1]))
    if not world.objects:
        raise StructureError("world without objects")
    return world


@dataclass
class Description:
    shape: str | None = None
    size: str | None = None
    color: str | None = None
    block: str | None = None
    edge: str | None = None
    rels: list[tuple[tuple[str, ...], str, "Description"]] | None = None


def description(data: Any, predicate: bool = False) -> Description:
    if data is None and predicate:
        return Description()
    if not isinstance(data, dict):
        raise StructureError("description must be an object")
    known = {"rels", "block", "edge"} | (set() if predicate else set(ATTRIBUTES))
    if set(data) - known:
        raise StructureError(f"unknown description keys {sorted(set(data) - known)}")
    result = Description(**{name: _text(data.get(name), name) for name in ATTRIBUTES if not predicate})
    if data.get("block") is not None:
        result.block = _block(data["block"])
    edge = data.get("edge")
    if edge is not None and edge not in SIDES + ("any",):
        raise StructureError(f"bad edge {edge!r}")
    result.edge = edge
    result.rels = []
    for clause in data.get("rels") or []:
        if not isinstance(clause, dict):
            raise StructureError("relation clause must be an object")
        rels = clause.get("rel")
        rels = [rels] if isinstance(rels, str) else rels
        if not rels or any(rel not in RELATIONS for rel in rels):
            raise StructureError(f"bad relation {clause.get('rel')!r}")
        quant = clause.get("quant", "a")
        if quant not in ("a", "all"):
            raise StructureError(f"bad quantifier {quant!r}")
        result.rels.append((tuple(RELATIONS[rel] for rel in rels), quant, description(clause.get("target"))))
    return result


def parse_form(data: Any, task: str) -> dict[str, Any] | None:
    """Validated form, or None for {"unsupported": true}."""
    if not isinstance(data, dict):
        raise StructureError("form must be an object")
    if data.get("unsupported") is True:
        return None
    negated = data.get("negated", False)
    if not isinstance(negated, bool):
        raise StructureError("negated must be true or false")
    if task == "YN":
        quant = data.get("quant", "the")
        if quant not in ("the", "some", "all"):
            raise StructureError(f"bad YN quantifier {quant!r}")
        return {"subject": description(data.get("subject")), "quant": quant,
                "predicate": description(data.get("predicate"), predicate=True), "negated": negated}
    if task == "FR":
        return {"first": description(data.get("first")), "second": description(data.get("second"))}
    if task == "FB":
        if data.get("condition") not in ("has", "has_not", "has_all"):
            raise StructureError(f"bad FB condition {data.get('condition')!r}")
        return {"condition": data["condition"], "object": description(data.get("object"))}
    if task == "CO":
        options = data.get("options")
        if not isinstance(options, list) or len(options) != 2:
            raise StructureError("CO needs two options")
        return {"options": [description(option) for option in options],
                "predicate": description(data.get("predicate"), predicate=True), "negated": negated}
    raise StructureError(f"unknown q_type {task}")


class LPSolver:
    def __init__(self, world: World, config: LPConfig | None = None) -> None:
        self.world = world
        self.config = config or LPConfig()

    def match(self, desc: Description, within: str | None = None) -> set[int]:
        world = self.world
        found = {o.id for o in world.objects
                 if all(getattr(desc, name) is None or getattr(o, name) == getattr(desc, name) for name in ATTRIBUTES)
                 and (desc.block is None or o.block == desc.block) and (within is None or o.block == within)}
        return {x for x in found if self.satisfies(x, desc, within)}

    def satisfies(self, x: int, desc: Description, within: str | None = None) -> bool:
        if desc.block is not None and self.world.objects[x].block != desc.block:
            return False
        if desc.edge is not None:
            sides = SIDES if desc.edge == "any" else (desc.edge,)
            if not any((x, side) in self.world.edges for side in sides):
                return False
        return all(self.related(x, rels, self.match(target, within), quant)
                   for rels, quant, target in desc.rels or [])

    def related(self, x: int, rels: tuple[str, ...], targets: set[int], quant: str) -> bool:
        targets = targets - {x}
        if not targets:
            return False
        check = all if quant == "all" else any
        return check(all(self.world.holds(rel, x, y) for rel in rels) for y in targets)

    def refuted(self, x: int, desc: Description) -> bool:
        """Some relation the predicate asks for is contradicted for every target."""
        for rels, _, target in desc.rels or []:
            opposite = [OPPOSITE[rel] for rel in rels if rel in OPPOSITE]
            targets = self.match(target) - {x}
            if opposite and targets and all(any(self.world.holds(o, x, y) for o in opposite) for y in targets):
                return True
        return False

    def answer(self, form: dict[str, Any] | None, question: dict[str, Any]) -> list[Any] | None:
        if form is None:
            return None
        return getattr(self, question["q_type"].lower())(form, question)

    def yn(self, form: dict[str, Any], question: dict[str, Any]) -> list[str] | None:
        xs = self.match(form["subject"])
        if not xs:
            return None
        results = [self.satisfies(x, form["predicate"]) for x in xs]
        truth = all(results) if form["quant"] == "all" else any(results)
        if form["negated"]:
            truth = not truth
        if truth:
            return ["Yes"]
        if self.config.yn_mode == "open" and not form["negated"]:
            refuted = [self.refuted(x, form["predicate"]) for x in xs]
            if not (any(refuted) if form["quant"] == "all" else all(refuted)):
                return ["DK"]
        return ["No"]

    def fr(self, form: dict[str, Any], question: dict[str, Any]) -> list[int] | None:
        xs, ys = self.match(form["first"]), self.match(form["second"])
        pairs = [(x, y) for x, y in product(xs, ys) if x != y]
        if not pairs:
            return None
        check = all if self.config.fr_multi == "all" else any
        labels = [index for rel, index in FR_INDEX.items() if check(self.world.holds(rel, x, y) for x, y in pairs)]
        return labels or [7]

    def fb(self, form: dict[str, Any], question: dict[str, Any]) -> list[str]:
        blocks = list(question["candidate_answers"])
        condition, desc = form["condition"], form["object"]
        if condition == "has_all":
            xs = self.match(desc)
            return [b for b in blocks if xs and all(self.world.objects[x].block == b for x in xs)]
        if self.config.fb_scope == "block":
            found = {b for b in blocks if self.match(desc, within=b)}
        else:
            found = {self.world.objects[x].block for x in self.match(desc)}
        return [b for b in blocks if (b in found) != (condition == "has_not")]

    def co(self, form: dict[str, Any], question: dict[str, Any]) -> list[int]:
        satisfied = []
        for option in form["options"]:
            xs = self.match(option)
            truth = any(self.satisfies(x, form["predicate"]) for x in xs)
            satisfied.append(bool(xs) and (truth != form["negated"]))
        return [{(True, False): 0, (False, True): 1, (True, True): 2, (False, False): 3}[tuple(satisfied)]]


def solve(world_json: Any, form_json: Any, question: dict[str, Any], config: LPConfig) -> list[Any] | None:
    """One sampled world and one sampled form -> answer, or None if either is unusable."""
    try:
        world = world_from_json(world_json, config.rules())
        form = parse_form(form_json, question["q_type"])
        return LPSolver(world, config).answer(form, question)
    except (StructureError, KeyError, TypeError, IndexError):
        return None


__all__ = ["LPConfig", "LPSolver", "StructureError", "parse_form", "solve", "world_from_json"]
