from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from typing import Any

from ..search import CandidateModel, Deadline
from .common import circuit_analysis, primitive_names


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    context: Any = None,
) -> Iterator[CandidateModel]:
    """Boolean-max PROP interpretations, with smallest supports first.

    Every object pair is interpreted by the Boolean commutative monoid; both
    composition and tensor are ``or``, while identities and symmetries are
    false.  Thus the interpretation remains sound for arity-changing PROPs.
    """
    if bound < 1:
        return
    names = primitive_names(theory, target, deadline, context)
    if bound > len(names):
        return

    for selected_tuple in combinations(names, bound):
        deadline.check()
        selected = frozenset(selected_tuple)

        def evaluate(term: Any, selected: frozenset[str] = selected) -> bool:
            occurrences = circuit_analysis(term, context).occurrences
            return not selected.isdisjoint(name for name, _ in occurrences)

        label = ", ".join(selected_tuple)
        yield CandidateModel(
            description=f"presence of {{{label}}}",
            parameters={"generators": list(selected_tuple)},
            evaluate=evaluate,
        )
