"""Recursive-descent parser for object descriptions and relation phrases.

A description is a head noun with optional size, color, and number, followed
by clauses that attach to the nearest preceding noun (right-branching):

    hình tam giác vừa màu đen ở phía trên một hình tròn chạm vào cạnh dưới của một khối
    = triangle(medium, black) ABOVE circle(touching the bottom edge of its block)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spartqa.symbolic.lexicon import COLORS, COUNTS, EDGES, PRONOUNS, QUANTIFIERS, RELATIONS, SHAPES, SIZES, TOUCH


class ParseError(ValueError):
    pass


@dataclass
class Clause:
    kind: str                      # "rel", "edge", or "in"
    rels: tuple[str, ...] = ()
    sub: "Desc | None" = None
    edge: str | None = None
    block: str | None = None       # block name, "this", or None for "một khối"


@dataclass
class Desc:
    quant: str = "def"             # "def", "one", "all", "any"
    count: int = 1                 # "hai hình ..." -> 2
    shape: str | None = None
    size: str | None = None
    color: str | None = None
    number: int | None = None      # "số 1"
    extra: bool = False            # "thêm": a further object
    pronoun: bool = False          # "nó", "cái này"
    clauses: list[Clause] = field(default_factory=list)

    def core(self) -> tuple:
        return (self.shape, self.size, self.color, self.number)


class Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    # --- token helpers -------------------------------------------------
    def peek(self, offset: int = 0) -> str | None:
        index = self.pos + offset
        return self.tokens[index] if index < len(self.tokens) else None

    def at(self, words: tuple[str, ...] | list[str]) -> bool:
        return tuple(self.tokens[self.pos:self.pos + len(words)]) == tuple(words)

    def accept(self, words: tuple[str, ...] | list[str]) -> bool:
        if self.at(words):
            self.pos += len(words)
            return True
        return False

    def accept_any(self, table: list[tuple[tuple[str, ...], object]]) -> object | None:
        for words, value in table:
            if self.at(words):
                self.pos += len(words)
                return value if value is not None else "__none__"
        return None

    def done(self) -> bool:
        return self.pos >= len(self.tokens)

    def expect_end(self) -> None:
        if not self.done():
            raise ParseError(f"unexpected tokens: {' '.join(self.tokens[self.pos:])}")

    # --- relations -----------------------------------------------------
    def rel_item(self) -> str | None:
        start = self.pos
        self.accept(("ở",))
        if self.at(TOUCH) and self.peek(2) != "cạnh":
            self.pos += len(TOUCH)
            return "TOUCH"
        rel = self.accept_any(RELATIONS)
        if rel is None:
            self.pos = start
            return None
        self.accept(("khỏi",)) or self.accept(("tới",)) or self.accept(("với",))
        return rel

    def rel_list(self) -> tuple[str, ...]:
        rels = []
        rel = self.rel_item()
        if rel is None:
            return ()
        rels.append(rel)
        while True:
            start = self.pos
            if self.accept((",",)) or self.accept(("và",)):
                self.accept(("và",))
                rel = self.rel_item()
                if rel is not None:
                    rels.append(rel)
                    continue
            self.pos = start
            return tuple(rels)

    # --- descriptions --------------------------------------------------
    def starts_desc(self) -> bool:
        start = self.pos
        try:
            return self.desc_head(allow_pronoun=True) is not None
        finally:
            self.pos = start

    def desc_head(self, allow_pronoun: bool = False) -> Desc | None:
        start = self.pos
        desc = Desc()
        if allow_pronoun:
            for words in PRONOUNS:
                if self.accept(words):
                    desc.pronoun = True
                    return desc
        quant = self.accept_any([(words, value) for words, value in QUANTIFIERS])
        if quant is not None:
            desc.quant = quant
        elif self.peek() in COUNTS and self.peek(1) in ("hình", "vật"):
            desc.count = COUNTS[self.peek()]
            desc.quant = "one"
            self.pos += 1
        shape = self.accept_any(SHAPES)
        if shape is None:
            self.pos = start
            return None
        desc.shape = None if shape == "__none__" else shape
        for _ in range(4):
            size = self.accept_any(SIZES)
            if size is not None and desc.size is None:
                desc.size = size
                continue
            color = self.accept_any(COLORS)
            if color is not None and desc.color is None:
                desc.color = color
                continue
            if self.at(("số",)) and (self.peek(1) or "").isdigit():
                desc.number = int(self.peek(1))
                self.pos += 2
                continue
            if self.accept(("thêm",)):
                desc.extra = True
                continue
            if self.accept(("khác",)):
                continue
            break
        self.accept(("nào",))
        return desc

    def clause(self) -> Clause | None:
        start = self.pos
        if self.accept(TOUCH + ("cạnh",)):
            edge = EDGES.get(self.peek() or "")
            if edge is None:
                raise ParseError("unknown edge")
            self.pos += 1
            if not self.accept(("của",)):
                raise ParseError("edge without 'của'")
            if self.accept(("một", "khối")):
                block = None
            elif self.accept(("khối", "này")) or self.accept(("nó",)):
                block = "this"
            elif self.accept(("khối",)) and self.peek() is not None:
                block = self.peek().upper()
                self.pos += 1
            else:
                raise ParseError("edge without block")
            return Clause("edge", edge=edge, block=block)
        if self.accept(("ở", "trong", "khối")) or self.accept(("trong", "khối")):
            name = self.peek()
            if name is None:
                raise ParseError("block name missing")
            self.pos += 1
            return Clause("in", block="this" if name == "này" else name.upper())
        rels = self.rel_list()
        if not rels:
            self.pos = start
            return None
        sub = self.desc()
        if sub is None:
            self.pos = start
            return None
        return Clause("rel", rels=rels, sub=sub)

    def desc(self, allow_pronoun: bool = False, clauses: bool = True) -> Desc | None:
        desc = self.desc_head(allow_pronoun)
        if desc is None or desc.pronoun or not clauses:
            return desc
        while True:
            clause = self.clause()
            if clause is None:
                break
            desc.clauses.append(clause)
            self.accept(("nào",))
            if clause.kind == "rel":
                break  # the nested description consumed the rest of the chain
        return desc


def parse_desc_text(tokens: list[str]) -> Desc:
    parser = Parser(tokens)
    desc = parser.desc()
    if desc is None:
        raise ParseError(f"not a description: {' '.join(tokens)}")
    parser.expect_end()
    return desc
