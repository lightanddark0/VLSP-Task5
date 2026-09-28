"""Question parsing and answering over a World.

Semantics (checked against Auto training answers with explore/solve_symbolic):
  YN  Yes if provable; No if the opposite relation is provable; otherwise DK.
  FR  every relation provable between the two objects; [7] if none.
  CO  closed world: a candidate counts only if the relation is provable.
  FB  blocks containing a provable match; "không chứa bất kỳ" is the complement.
Anything the parser does not recognise returns None (abstain).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spartqa.symbolic.grammar import Desc, ParseError, Parser
from spartqa.symbolic.lexicon import FR_INDEX, OPPOSITE, tokenize
from spartqa.symbolic.world import World


@dataclass
class SolverConfig:
    yn_multi_subject: str = "any"     # several objects fit the YN subject: "any" or "all" must hold
    yn_no_rule: bool = True           # answer No when the opposite relation is provable
    fr_multi: str = "abstain"         # FR with several matching objects: "abstain" or "intersection"
    fr_empty: str = "abstain"         # FR with no matching object: "abstain" or "dk"
    abstain_ambiguous_story: bool = False
    definite_as_all: bool = True      # "hình X" (no article) as a target: every matching object
    fb_negative_within_block: bool = False  # tested: False matches the data (98% vs 76%)


class Unsupported(ValueError):
    pass


def parse_tail_desc(parser: Parser, end: tuple[str, ...] = ()) -> Desc:
    desc = parser.desc()
    if desc is None:
        raise Unsupported("description expected")
    for word in end:
        if not parser.accept((word,)):
            raise Unsupported(f"expected '{word}'")
    parser.expect_end()
    return desc


class Solver:
    def __init__(self, world: World, config: SolverConfig | None = None) -> None:
        self.world = world
        self.config = config or SolverConfig()

    # --- helpers -------------------------------------------------------
    def related(self, x: int, rels: tuple[str, ...], targets: set[int], quant: str) -> bool:
        if quant == "def" and self.config.definite_as_all:
            quant = "all"
        if quant == "all":
            # "tất cả Y" includes x itself when x fits Y; no object relates to itself.
            return bool(targets) and x not in targets and all(
                self.world.holds(rel, x, y) for rel in rels for y in targets)
        targets = targets - {x}
        if not targets:
            return False
        return any(all(self.world.holds(rel, x, y) for rel in rels) for y in targets)

    def refuted(self, x: int, rels: tuple[str, ...], targets: set[int], quant: str) -> bool:
        """The opposite of some asked relation is provable (for every target, or for one under "all")."""
        if quant == "def" and self.config.definite_as_all:
            quant = "all"
        if quant == "all" and x in targets:
            return True
        targets = targets - {x}
        if not targets:
            return False
        opposite = [OPPOSITE[r] for r in rels if r in OPPOSITE]
        if not opposite:
            return False
        if quant == "all":
            return any(self.world.holds(o, x, y) for o in opposite for y in targets)
        return all(any(self.world.holds(o, x, y) for o in opposite) for y in targets)

    # --- question types ------------------------------------------------
    def answer(self, question: dict[str, Any]) -> list[Any] | None:
        if self.config.abstain_ambiguous_story and self.world.ambiguous:
            return None
        tokens = tokenize(question["question"])
        try:
            return getattr(self, question["q_type"].lower())(tokens, question)
        except (ParseError, Unsupported, IndexError, KeyError):
            return None

    def yn(self, tokens: list[str], question: dict[str, Any]) -> list[str] | None:
        world = self.world
        # "Có phải tất cả X đều <rel> Y không?"
        if tokens[:4] == ["có", "phải", "tất", "cả"]:
            parser = Parser(tokens[2:])
            subject = parser.desc()
            if subject is None or not parser.accept(("đều",)):
                raise Unsupported("universal YN")
            rels = parser.rel_list()
            target = parse_tail_desc(parser, ("không",)) if rels else None
            if target is None:
                raise Unsupported("universal YN relation")
            xs, ys = world.evaluate(subject), world.evaluate(target)
            if not xs or not ys:
                return None
            if all(self.related(x, rels, ys, target.quant) for x in xs):
                return ["Yes"]
            if self.config.yn_no_rule and any(self.refuted(x, rels, ys, target.quant) for x in xs):
                return ["No"]
            return ["DK"]
        # "Có X <rel> Y nào <rel> Z không?": clauses after "nào" also describe X.
        if tokens[0] == "có":
            if tokens[-1] != "không":
                raise Unsupported("existential YN end")
            body = tokens[1:-1]
            head, tail = (body[:body.index("nào")], body[body.index("nào") + 1:]) if "nào" in body else (body, [])
            desc = parse_tail_desc(Parser(head))
            parser = Parser(tail)
            while not parser.done():
                clause = parser.clause()
                if clause is None:
                    raise Unsupported("existential YN clause")
                desc.clauses.append(clause)
            if world.evaluate(desc):
                return ["Yes"]
            if not desc.clauses or desc.clauses[0].kind != "rel":
                return None
            core = Desc(shape=desc.shape, size=desc.size, color=desc.color, number=desc.number)
            xs = world.evaluate(core)
            rel_clauses = [c for c in desc.clauses if c.kind == "rel"]
            if self.config.yn_no_rule and xs and all(
                    any(self.refuted(x, c.rels, world.evaluate(c.sub), c.sub.quant) for c in rel_clauses)
                    for x in xs):
                return ["No"]
            return ["DK"]
        # "X có <rel> Y không?"
        if "có" not in tokens:
            raise Unsupported("YN form")
        parser = Parser(tokens)
        subject = parser.desc()
        if subject is None or not parser.accept(("có",)):
            raise Unsupported("YN subject")
        rels = parser.rel_list()
        if not rels:
            raise Unsupported("YN relation")
        target = parse_tail_desc(parser, ("không",))
        xs, ys = world.evaluate(subject), world.evaluate(target)
        if not xs or not ys:
            return None
        holds = [self.related(x, rels, ys, target.quant) for x in xs]
        if (any(holds) if self.config.yn_multi_subject == "any" else all(holds)):
            return ["Yes"]
        if self.config.yn_no_rule and all(self.refuted(x, rels, ys, target.quant) for x in xs):
            return ["No"]
        return ["DK"]

    def fr(self, tokens: list[str], question: dict[str, Any]) -> list[int] | None:
        prefix, suffix = ["mối", "quan", "hệ", "giữa"], ["là", "gì"]
        if tokens[:4] != prefix or tokens[-2:] != suffix:
            raise Unsupported("FR form")
        body = tokens[4:-2]
        splits = []
        for index, word in enumerate(body):
            if word != "và" or index == 0:
                continue
            try:
                left = Parser(body[:index])
                first = parse_tail_desc(left)
                right = Parser(body[index + 1:])
                second = parse_tail_desc(right)
            except (ParseError, Unsupported):
                continue
            splits.append((first, second))
        if len(splits) != 1:
            raise Unsupported("FR split")
        first, second = splits[0]
        xs, ys = self.world.evaluate(first), self.world.evaluate(second)
        if not xs or not ys:
            return [7] if self.config.fr_empty == "dk" else None
        pairs = [(x, y) for x in xs for y in ys if x != y]
        if not pairs:
            return None
        if len(pairs) > 1 and self.config.fr_multi == "abstain":
            return None
        rels = [index for rel, index in FR_INDEX.items()
                if all(self.world.holds(rel, x, y) for x, y in pairs)]
        return rels or [7]

    def co(self, tokens: list[str], question: dict[str, Any]) -> list[int] | None:
        if tokens[:2] != ["vật", "nào"] or ":" not in tokens:
            raise Unsupported("CO form")
        colon = tokens.index(":")
        parser = Parser(tokens[2:colon])
        rels = parser.rel_list()
        if not rels:
            raise Unsupported("CO relation")
        target = parse_tail_desc(parser)
        ys = self.world.evaluate(target)
        options = question["candidate_answers"][:2]
        satisfied = []
        for option in options:
            desc = parse_tail_desc(Parser(tokenize(option)))
            xs = self.world.evaluate(desc)
            satisfied.append(any(self.related(x, rels, ys, target.quant) for x in xs))
        return [{(True, False): 0, (False, True): 1, (True, True): 2, (False, False): 3}[tuple(satisfied)]]

    def fb(self, tokens: list[str], question: dict[str, Any]) -> list[str] | None:
        if tokens[:2] != ["khối", "nào"]:
            raise Unsupported("FB form")
        rest = tokens[2:]
        negative = False
        if rest[:2] == ["không", "chứa"]:
            negative, rest = True, rest[2:]
        elif rest[:1] in (["chứa"], ["có"]):
            rest = rest[1:]
        else:
            raise Unsupported("FB verb")
        if rest and rest[-1] == "nào":
            rest = rest[:-1]
        desc = parse_tail_desc(Parser(rest))
        blocks = list(question["candidate_answers"])
        if negative and self.config.fb_negative_within_block:
            return [b for b in blocks if not self.world.evaluate(desc, within=b)]
        found = {self.world.objects[x].block for x in self.world.evaluate(desc)}
        return [b for b in blocks if (b in found) != negative]
