"""Story parsing into objects and facts, and the spatial reasoner (closure).

Conventions learned from the generated (Auto) stories:
  - "một X" / "thêm" introduces a new object in the block being described;
    "X" without an article refers to an existing object (current block first).
  - "hai X" introduces objects numbered 1 and 2 ("X số 1", "X số 2").
  - "Nó" / "Cái này" is the subject of the previous sentence; "Nó chứa ..."
    after a block introduction refers to that block.
  - Relation clauses inside a description state facts about the inner noun.
Unknown sentence shapes raise ParseError, and the caller abstains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product

from spartqa.symbolic.grammar import Clause, Desc, ParseError, Parser
from spartqa.symbolic.lexicon import CONVERSE, DIRECTIONS, block_names, split_sentences, tokenize


@dataclass
class Obj:
    id: int
    block: str
    shape: str | None
    size: str | None
    color: str | None
    number: int | None = None


@dataclass
class RulesConfig:
    """Reasoning rules; each is a switch so it can be checked against training data."""
    inherit: tuple[str, ...] = ("LEFT", "RIGHT", "ABOVE", "BELOW", "FAR")
    transitive: tuple[str, ...] = DIRECTIONS
    # Objects touching opposite edges of the same block: left edge LEFT of right edge, top ABOVE bottom.
    edge_pairs: bool = True


@dataclass
class World:
    blocks: list[str] = field(default_factory=list)
    objects: list[Obj] = field(default_factory=list)
    facts: set[tuple[str, object, object]] = field(default_factory=set)
    edges: set[tuple[int, str]] = field(default_factory=set)
    ambiguous: int = 0
    rules: RulesConfig = field(default_factory=RulesConfig)
    _closure: set | None = None
    _closure_size: int = -1

    # --- objects -------------------------------------------------------
    def new_object(self, block: str, desc: Desc) -> Obj:
        same = [o for o in self.objects if o.block == block
                and (o.shape, o.size, o.color) == (desc.shape, desc.size, desc.color)]
        number = desc.number
        if number is None and same:
            if same[0].number is None:
                same[0].number = 1
            number = max(o.number or 1 for o in same) + 1
        obj = Obj(len(self.objects), block, desc.shape, desc.size, desc.color, number)
        self.objects.append(obj)
        return obj

    def add_fact(self, rel: str, a: object, b: object) -> None:
        if a != b:
            self.facts.add((rel, a, b))

    # --- reasoning -----------------------------------------------------
    def closure(self) -> set[tuple[str, object, object]]:
        if self._closure is not None and self._closure_size == len(self.facts) + len(self.objects):
            return self._closure
        facts = set(self.facts)
        facts |= {(CONVERSE[r], b, a) for r, a, b in facts}
        blocks = set(self.blocks) | {a for _, a, _ in facts if isinstance(a, str)}
        block_facts = {f for f in facts if isinstance(f[1], str)}
        block_facts = transitive(block_facts, self.rules.transitive, sorted(blocks))
        object_facts = {f for f in facts if not isinstance(f[1], str)}
        if self.rules.edge_pairs:
            for (x, side_x), (y, side_y) in product(self.edges, self.edges):
                if x != y and self.objects[x].block == self.objects[y].block:
                    if (side_x, side_y) == ("left", "right"):
                        object_facts |= {("LEFT", x, y), ("RIGHT", y, x)}
                    if (side_x, side_y) == ("top", "bottom"):
                        object_facts |= {("ABOVE", x, y), ("BELOW", y, x)}
        members: dict[str, list[int]] = {}
        for obj in self.objects:
            members.setdefault(obj.block, []).append(obj.id)
        for rel, a, b in block_facts:
            if rel in self.rules.inherit:
                for x, y in product(members.get(a, []), members.get(b, [])):
                    object_facts.add((rel, x, y))
        object_facts = transitive(object_facts, self.rules.transitive, [o.id for o in self.objects])
        self._closure = object_facts | block_facts
        self._closure_size = len(self.facts) + len(self.objects)
        return self._closure

    def holds(self, rel: str, a: object, b: object) -> bool:
        return (rel, a, b) in self.closure()

    # --- descriptions --------------------------------------------------
    def matches_core(self, obj: Obj, desc: Desc) -> bool:
        return ((desc.shape is None or obj.shape == desc.shape) and (desc.size is None or obj.size == desc.size)
                and (desc.color is None or obj.color == desc.color)
                and (desc.number is None or obj.number == desc.number))

    def evaluate(self, desc: Desc, block: str | None = None, within: str | None = None) -> set[int]:
        """Objects that provably fit the description (clauses checked against the closure).

        ``within`` restricts every object, including clause targets, to one block.
        """
        found = {o.id for o in self.objects if self.matches_core(o, desc) and (within is None or o.block == within)}
        if within is not None:
            return {x for x in found if all(self.satisfies(x, clause, block, within) for clause in desc.clauses)}
        for clause in desc.clauses:
            found = {x for x in found if self.satisfies(x, clause, block)}
        return found

    def satisfies(self, x: int, clause: Clause, block: str | None = None, within: str | None = None) -> bool:
        obj = self.objects[x]
        if clause.kind == "edge":
            return (x, clause.edge) in self.edges and (clause.block in (None, "this") or obj.block == clause.block)
        if clause.kind == "in":
            return obj.block == (block if clause.block == "this" else clause.block)
        targets = self.evaluate(clause.sub, block, within)
        if clause.sub.quant == "all":
            return bool(targets) and x not in targets and all(
                self.holds(rel, x, y) for rel in clause.rels for y in targets)
        targets = targets - {x}
        if not targets:
            return False
        return any(all(self.holds(rel, x, y) for rel in clause.rels) for y in targets)


def transitive(facts: set, rels: tuple[str, ...], nodes: list) -> set:
    facts = set(facts)
    for rel in rels:
        succ: dict[object, set] = {}
        for r, a, b in facts:
            if r == rel:
                succ.setdefault(a, set()).add(b)
        changed = True
        while changed:
            changed = False
            for a in list(succ):
                reach = set(succ[a])
                for b in list(succ[a]):
                    reach |= succ.get(b, set())
                reach.discard(a)
                if reach != succ[a]:
                    succ[a] = reach
                    changed = True
        for a, bs in succ.items():
            facts |= {(rel, a, b) for b in bs}
    return facts


class StoryReader:
    """Reads a story sentence by sentence into a World."""

    def __init__(self, rules: RulesConfig | None = None) -> None:
        self.world = World(rules=rules or RulesConfig())
        self.block: str | None = None          # block being described
        self.last_block: str | None = None     # block last introduced (for "Nó chứa")
        self.subject: int | None = None        # object subject of the previous sentence

    # --- resolution ----------------------------------------------------
    def resolve(self, desc: Desc, create: bool = True) -> list[int]:
        """Objects a story description refers to, creating new ones for "một"/"thêm"/"hai"."""
        world = self.world
        if desc.pronoun:
            if self.subject is None:
                raise ParseError("pronoun without antecedent")
            return [self.subject]
        for clause in desc.clauses:
            if clause.kind == "in" and clause.block not in (None, "this"):
                self.add_block(clause.block)
                self.set_block(clause.block)
        if self.block is None:
            self.set_block(self.implicit_block())
        if desc.quant == "one" or desc.extra or desc.count > 1:
            if desc.count > 1:
                created = []
                for number in range(1, desc.count + 1):
                    clone = Desc(shape=desc.shape, size=desc.size, color=desc.color, number=number)
                    created.append(world.new_object(self.block, clone).id)
                ids = created
            else:
                ids = [world.new_object(self.block, desc).id]
        else:
            ids = self.find(desc)
            if not ids:
                if not create:
                    raise ParseError("reference to an unknown object")
                ids = [world.new_object(self.block, desc).id]
        for obj_id in ids:
            self.apply_clauses(obj_id, desc.clauses)
        return ids

    def find(self, desc: Desc) -> list[int]:
        world = self.world
        for scope in ("block", "all"):
            candidates = [o.id for o in world.objects if world.matches_core(o, desc)
                          and (scope == "all" or o.block == self.block)]
            if len(candidates) > 1 and desc.clauses:
                narrowed = [c for c in candidates
                            if all(world.satisfies(c, clause, self.block) for clause in desc.clauses)]
                candidates = narrowed or candidates
            if len(candidates) > 1:
                world.ambiguous += 1
                return candidates[:1]
            if candidates:
                return candidates
        return []

    def apply_clauses(self, obj_id: int, clauses: list[Clause]) -> None:
        for clause in clauses:
            if clause.kind == "edge":
                self.world.edges.add((obj_id, clause.edge))
            elif clause.kind == "rel":
                for target in self.resolve(clause.sub):
                    for rel in clause.rels:
                        self.world.add_fact(rel, obj_id, target)

    # --- sentences -----------------------------------------------------
    DISCOURSE = (("sau", "đó"), ("cuối", "cùng"), ("ngoài", "ra"), ("tiếp", "theo"), ("đầu", "tiên"),
                 ("bên", "cạnh", "đó"), ("thêm", "vào", "đó"), ("hơn", "nữa"))

    def read(self, story: str) -> World:
        for sentence in split_sentences(story):
            self.sentence(sentence)
        return self.world

    def sentence(self, original: str) -> None:
        tokens = tokenize(original)
        while tokens and tokens[0] == "và":
            tokens = tokens[1:]
        # Generator glitch: "Ngoài ra, có" pasted inside a sentence ("... số 2 và Ngoài ra, có một X ...").
        glitch = ["ngoài", "ra", ",", "có"]
        index = 1
        while index < len(tokens):
            if tokens[index:index + 4] == glitch:
                tokens = tokens[:index] + tokens[index + 4:]
            else:
                index += 1
        if tokens[:2] == ["có", "thêm"] and any(tuple(tokens[2:2 + len(w)]) == w for w in self.DISCOURSE):
            tokens = tokens[2:]
        for words in self.DISCOURSE:
            if tuple(tokens[:len(words)]) == words:
                tokens = tokens[len(words):]
                if tokens[:1] == [","]:
                    tokens = tokens[1:]
                break
        if not tokens:
            return
        text = " ".join(tokens)

        # A single unnamed block: "(Chúng ta) có một khối (chứa ...)."
        match = re.match(r"^(chúng ta )?có một khối( tên là [a-h]| gọi là [a-h])?( chứa )?", text)
        if match:
            names = block_names(original)
            self.set_block(names[0] if names else self.implicit_block())
            rest = Parser(tokens[len(match.group(0).split()):])
            if match.group(3):
                self.object_list(rest)
            elif rest.accept((",", "trong", "đó")):
                descs = self.desc_list(rest)
                if not (rest.accept(("ở", "trong", "nó")) or rest.accept(("trong", "nó"))):
                    raise ParseError("'trong đó' without 'ở trong nó'")
                for desc in descs:
                    desc.extra = True
                    self.resolve(desc)
            rest.expect_end()
            return
        # Block declarations: "Chúng ta có ba khối, A, B và C." / "Chúng ta gọi chúng là A và B."
        if re.match(r"^(chúng ta )?(có (một|hai|ba|bốn) khối|gọi chúng)\b", text) and "chứa" not in tokens:
            names = block_names(original)
            for name in names:
                self.add_block(name)
            if len(names) == 1:
                self.set_block(names[0])
            return

        parser = Parser(tokens)
        if parser.at(("khối",)) and parser.peek(1) not in (None, "này") and parser.peek(2) == "ở":
            self.block_relations(parser)
            return
        if parser.at(("khối",)) and parser.peek(2) == "và" and parser.peek(4) == "đều":
            subjects = [self.block_name(parser)]
            parser.accept(("và",))
            subjects.append(self.block_name(parser))
            parser.accept(("đều",))
            rels = parser.rel_list()
            anchor = self.block_name(parser)
            parser.expect_end()
            for subject in subjects:
                for rel in rels:
                    self.world.add_fact(rel, subject, anchor)
            return
        if parser.at(("nó", "ở")) and self.last_block is not None and "khối" in tokens \
                and not any(word in tokens for word in ("hình", "vật")):
            self.block_relations(parser)
            return
        if self.block_introduction(parser):
            return
        if self.containment(parser):
            return
        if parser.at(("có", "thêm")) or parser.at(("có", "một")) or parser.at(("có", "hai")):
            # "Có thêm một X chạm vào cạnh dưới của khối này." / "Có một X trong khối này."
            parser.accept(("có",))
            parser.accept(("thêm",))
            descs = self.desc_list(parser)
            for desc in descs:
                desc.extra = True
            self.location_or_predicates(parser, descs)
            return
        if parser.at(("ở",)) or (parser.at(("chạm", "vào")) and parser.peek(2) != "cạnh"):
            self.inverted(parser)
            return
        # Subject-first: "X (và Y) ở bên trái Z." / "Nó chạm vào cạnh phải của khối này." / "Một X trong khối A."
        first = parser.desc(allow_pronoun=True, clauses=False)
        if first is None:
            raise ParseError(f"unknown sentence: {text}")
        descs = [first]
        start = parser.pos
        if (parser.accept(("và",)) or parser.accept((",",))) and parser.starts_desc():
            descs += self.desc_list(parser)
        else:
            parser.pos = start
        self.location_or_predicates(parser, descs)

    def implicit_block(self) -> str:
        """Block for "ở trong một khối" when the story has not named one."""
        if self.block is not None:
            return self.block
        empty = [b for b in self.world.blocks if not any(o.block == b for o in self.world.objects)]
        name = (empty or self.world.blocks or ["A"])[0]
        self.add_block(name)
        return name

    def add_block(self, name: str) -> None:
        if name and name not in self.world.blocks:
            self.world.blocks.append(name)

    def block_name(self, parser: Parser) -> str:
        """A block reference: "khối A", "A", "khối này", "nó" (the block being described)."""
        parser.accept(("khối",))
        word = parser.peek()
        if word is None:
            raise ParseError("block name missing")
        parser.pos += 1
        if word in ("này", "nó", "đó"):
            return self.block if self.block is not None else self.implicit_block()
        if not re.fullmatch(r"[a-h]", word):
            raise ParseError(f"block name: {word}")
        name = word.upper()
        self.add_block(name)
        return name

    def set_block(self, name: str) -> None:
        self.block = self.last_block = name

    def block_relations(self, parser: Parser) -> None:
        """"Khối A ở bên trái khối C và B." / "Khối B ở bên phải và khối C ở phía trên A." /
        "Khối A ở bên trái khối C và ở bên phải khối B." / "... và khối A ở bên trái C." """
        pending: list[tuple[str, tuple[str, ...]]] = []
        subject = None
        while True:
            if parser.at(("khối",)):
                subject = self.block_name(parser)
            elif parser.accept(("nó",)):
                subject = subject or self.last_block
            rels = parser.rel_list()
            if not rels or subject is None:
                raise ParseError("block relation")
            pending.append((subject, rels))
            # "Khối B ở bên phải và khối C ở phía trên A": the anchor comes later.
            if parser.at(("và", "khối")) and parser.peek(3) == "ở":
                parser.pos += 1
                continue
            parser.accept(("cả", "hai"))
            anchor_pronoun = parser.peek() in ("nó", "đó") or parser.at(("khối", "này"))
            anchors = [self.block_name(parser)]
            while True:
                start = parser.pos
                if (parser.accept(("và",)) or parser.accept((",",))) and not parser.at(("ở",)):
                    if parser.at(("khối",)) and parser.peek(2) == "ở":
                        parser.pos = start
                        break
                    if parser.peek() == "khối" or re.fullmatch(r"[a-h]", parser.peek() or ""):
                        anchors.append(self.block_name(parser))
                        continue
                parser.pos = start
                break
            for (name, relations), anchor in product(pending, anchors):
                for rel in relations:
                    self.world.add_fact(rel, name, anchor)
            # "Khối B ở phía dưới khối A. Nó chứa ..." -> B, but "Khối A ở phía trên nó. Nó chứa" keeps "nó".
            if not anchor_pronoun:
                self.last_block = pending[0][0]
            pending = []
            start = parser.pos
            if parser.accept(("và",)) or parser.accept((",",)):
                parser.accept(("và",))
                if parser.at(("ở",)) or parser.at(("khối",)) or parser.at(("nó", "ở")):
                    continue
            parser.pos = start
            break
        parser.expect_end()

    def relation_anchors(self, parser: Parser, subject: str) -> None:
        """"ở bên phải khối A (và ở phía dưới khối B)*" with a known subject block."""
        while True:
            rels = parser.rel_list()
            if not rels:
                raise ParseError("block relation expected")
            parser.accept(("cả", "hai"))
            anchors = [self.block_name(parser)]
            while parser.at(("và",)) and re.fullmatch(r"[a-h]", parser.peek(1) or ""):
                parser.pos += 1
                anchors.append(self.block_name(parser))
            for anchor in anchors:
                for rel in rels:
                    self.world.add_fact(rel, subject, anchor)
            if parser.at(("và", "ở")) or parser.at((",", "ở")):
                parser.pos += 1
                continue
            return

    def block_introduction(self, parser: Parser) -> bool:
        """"(Ở bên phải khối A (và ở ... khối B)) (chúng ta) có/là khối C (ở ... khối B)
        (và khối A ở phía dưới nó) (chứa ... | , trong đó ... ở trong nó)." """
        start = parser.pos
        before: list[tuple[tuple[str, ...], str]] = []
        while parser.at(("ở",)):
            rels = parser.rel_list()
            parser.accept(("cả", "hai"))
            if not rels or not parser.at(("khối",)):
                parser.pos = start
                return False
            before.append((rels, self.block_name(parser)))
            while parser.at(("và",)) and re.fullmatch(r"[a-h]", parser.peek(1) or ""):
                parser.pos += 1
                before.append((rels, self.block_name(parser)))
            if parser.at(("và", "ở")) or parser.at((",", "ở")):
                parser.pos += 1
        parser.accept(("chúng", "ta"))
        if not (parser.accept(("có",)) or parser.accept(("là",))) or not parser.at(("khối",)) \
                or parser.peek(1) in (None, "này"):
            parser.pos = start
            return False
        name = self.block_name(parser)
        for rels, anchor in before:
            for rel in rels:
                self.world.add_fact(rel, name, anchor)
        self.set_block(name)
        if parser.at(("ở",)):
            self.relation_anchors(parser, name)
        if parser.at(("và", "khối")) and parser.peek(3) == "ở":
            parser.pos += 1
            other = self.block_name(parser)
            self.relation_anchors(parser, other)
            self.set_block(name)
        if parser.accept(("chứa",)) or parser.accept(("có",)):
            self.object_list(parser)
        elif parser.accept((",", "trong", "đó")):
            descs = self.desc_list(parser)
            if not (parser.accept(("ở", "trong", "nó")) or parser.accept(("trong", "nó"))):
                raise ParseError("'trong đó' without 'ở trong nó'")
            for desc in descs:
                desc.extra = True
                self.resolve(desc)
        parser.expect_end()
        return True

    def containment(self, parser: Parser) -> bool:
        """"Khối A chứa ..." / "Nó chứa ..." / "Khối này (cũng) có/chứa ..." """
        start = parser.pos
        if parser.at(("khối",)) and parser.peek(1) is not None:
            name = self.block_name(parser)
        elif parser.accept(("nó",)):
            if self.last_block is None:
                parser.pos = start
                return False
            name = self.last_block
        else:
            return False
        parser.accept(("cũng",))
        if not (parser.accept(("chứa",)) or parser.accept(("có",))):
            parser.pos = start
            return False
        self.set_block(name)
        self.object_list(parser)
        parser.expect_end()
        return True

    def location_or_predicates(self, parser: Parser, descs: list[Desc]) -> None:
        """After the subject(s): "ở trong khối X" / "trong khối này" / relations / edge contact."""
        for desc in descs:
            for clause in desc.clauses:
                if clause.kind == "in" and clause.block not in (None, "this"):
                    self.add_block(clause.block)
                    self.set_block(clause.block)
        if parser.accept(("ở", "trong")) or parser.accept(("trong",)):
            if parser.accept(("một", "khối")):
                self.set_block(self.implicit_block())
            else:
                self.set_block(self.block_name(parser))
            subjects = [obj for desc in descs for obj in self.resolve(desc)]
            if not parser.done():
                self.predicates(parser, subjects)
            parser.expect_end()
            self.subject = subjects[0]
            return
        subjects = [obj for desc in descs for obj in self.resolve(desc)]
        if not parser.done():
            self.predicates(parser, subjects)
        parser.expect_end()
        self.subject = subjects[0]

    def inverted(self, parser: Parser) -> None:
        """"Ở bên trên X (và Y) có Z." """
        rels = parser.rel_list()
        if not rels:
            raise ParseError("inverted sentence without relation")
        anchors = self.desc_list(parser)
        if not (parser.accept(("có",)) or parser.accept(("là",))):
            raise ParseError("inverted sentence without 'có'")
        subject_descs = self.desc_list(parser)
        parser.expect_end()
        subjects = [obj for desc in subject_descs for obj in self.resolve(desc)]
        anchor_ids = [a for desc in anchors for a in self.resolve(desc)]
        for subject, anchor in product(subjects, anchor_ids):
            for rel in rels:
                self.world.add_fact(rel, subject, anchor)
        self.subject = subjects[0]

    def object_list(self, parser: Parser) -> None:
        for desc in self.desc_list(parser):
            desc.extra = True
            self.resolve(desc)

    def desc_list(self, parser: Parser) -> list[Desc]:
        descs = []
        while True:
            desc = parser.desc(allow_pronoun=not descs)
            if desc is None:
                raise ParseError("expected a description")
            descs.append(desc)
            start = parser.pos
            if parser.accept((",",)) or parser.accept(("và",)):
                parser.accept(("và",))
                if parser.starts_desc():
                    continue
            parser.pos = start
            return descs

    def predicates(self, parser: Parser, subjects: list[int]) -> None:
        stated = False
        while not parser.done():
            clause_start = parser.pos
            if parser.at(("chạm", "vào", "cạnh")):
                clause = parser.clause()
                for subject in subjects:
                    self.world.edges.add((subject, clause.edge))
            else:
                rels = parser.rel_list()
                if not rels:
                    raise ParseError(f"unknown predicate: {' '.join(parser.tokens[clause_start:])}")
                for anchor_desc in self.desc_list(parser):
                    for anchor in self.resolve(anchor_desc):
                        for subject, rel in product(subjects, rels):
                            self.world.add_fact(rel, subject, anchor)
            stated = True
            start = parser.pos
            if parser.accept((",",)) or parser.accept(("và",)):
                parser.accept(("và",))
                if parser.at(("ở",)) or parser.at(("chạm",)):
                    continue
            parser.pos = start
            break
        if not stated:
            raise ParseError("sentence without predicate")


def read_story(story: str | list[str], rules: RulesConfig | None = None) -> World:
    text = " ".join(story) if isinstance(story, list) else story
    return StoryReader(rules).read(text)
