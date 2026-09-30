"""Answer post-processing applied after a branch or the ensemble has decided.

Constraints verified on both training files (see explore.py):
  - FR never contains both labels of a pair (0,1), (2,3), (4,5).
  - FR label 7 (DK) never appears with another label.
  - The question field ``indifinite`` is True exactly when the YN answer is
    DK or the FR answer is [7], in Human and Auto.

The ``indifinite`` rules are off by default: the field is not listed as an
input in the task description, so it is used only if the organizers allow it.
Human YN is documented as binary, but Human train contains DK labels;
``human_yn_dk`` chooses between keeping DK ("keep") and mapping it to the
more likely of Yes/No ("no"). Human FR is [7] for only 3 of 149 training
questions; ``human_fr_dk="avoid"`` replaces a predicted [7] by the relations
that some source scored, keeping [7] only when no source supports any relation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

FR_CONFLICTS = ((0, 1), (2, 3), (4, 5))
FR_DK = 7
EMPTY = "__EMPTY__"


@dataclass(frozen=True)
class PostprocessOptions:
    use_indifinite: bool = False
    human_yn_dk: str = "keep"
    fr_constraints: bool = True
    human_fr_dk: str = "keep"     # "avoid": Human FR [7] becomes the best-scored relations when any source has one

    def __post_init__(self) -> None:
        if self.human_yn_dk not in {"keep", "no"}:
            raise ValueError("human_yn_dk must be 'keep' or 'no'")
        if self.human_fr_dk not in {"keep", "avoid"}:
            raise ValueError("human_fr_dk must be 'keep' or 'avoid'")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score(scores: dict[str, float] | None, label: Any) -> float:
    return float((scores or {}).get(str(label), 0.0))


def _best(labels: list[Any], scores: dict[str, float] | None, default: Any) -> Any:
    """Highest-scoring label; ties and all-zero scores prefer ``default``, then list order."""
    return max(labels, key=lambda label: (_score(scores, label), label == default, -labels.index(label)))


def finalize_answer(
    question: dict[str, Any], answer: list[Any] | None, scores: dict[str, float] | None,
    dataset: str, options: PostprocessOptions, fallback: dict[str, list[Any]],
) -> list[Any]:
    """Return a valid answer for ``question``; ``answer`` may be None when every source abstained."""
    task = question["q_type"]
    indifinite = question.get("indifinite") if options.use_indifinite else None

    if task == "YN":
        default = fallback["YN"][0] if fallback["YN"] and fallback["YN"][0] in {"Yes", "No"} else "Yes"
        label = answer[0] if answer else None
        if label not in {"Yes", "No", "DK"}:
            label = _best(["Yes", "No", "DK"], scores, fallback["YN"][0] if fallback["YN"] else "Yes")
        if indifinite is True:
            return ["DK"]
        if label == "DK" and (indifinite is False or (dataset == "human" and options.human_yn_dk == "no")):
            label = _best(["Yes", "No"], scores, default)
        return [label]

    if task == "CO":
        label = answer[0] if answer and answer[0] in range(4) else None
        if label is None:
            label = _best([0, 1, 2, 3], scores, fallback["CO"][0] if fallback["CO"] else 3)
        return [label]

    if task == "FB":
        candidates = list(question["candidate_answers"])
        if answer is None:
            answer = [block for block in candidates if _score(scores, block) >= 0.5] if scores else fallback["FB"]
        chosen = set(answer)
        return [block for block in candidates if block in chosen]

    # FR
    if indifinite is True:
        return [FR_DK]
    labels = {label for label in (answer or []) if label in range(8)}
    if indifinite is False:
        labels.discard(FR_DK)
    if dataset == "human" and options.human_fr_dk == "avoid" and labels == {FR_DK}:
        # Human FR gold is DK for only 2% of questions: prefer any relation some source supports.
        scored = {label: _score(scores, label) for label in range(7) if _score(scores, label) > 0}
        if scored:
            top = max(scored.values())
            labels = {label for label, value in scored.items() if value >= top / 2}
    if options.fr_constraints:
        for first, second in FR_CONFLICTS:
            if first in labels and second in labels:
                labels.discard(second if _score(scores, first) >= _score(scores, second) else first)
        if FR_DK in labels and len(labels) > 1:
            others = labels - {FR_DK}
            if _score(scores, FR_DK) > max(_score(scores, label) for label in others):
                labels = {FR_DK}
            else:
                labels = others
    if not labels:
        allowed = [label for label in range(8) if not (indifinite is False and label == FR_DK)]
        scored = [label for label in allowed if _score(scores, label) > 0]
        if scored:
            labels = {_best(scored, scores, scored[0])}
        else:
            labels = {label for label in fallback["FR"] if label in allowed} or {allowed[0]}
            if options.fr_constraints and FR_DK in labels and len(labels) > 1:
                labels.discard(FR_DK)
    return sorted(labels)
