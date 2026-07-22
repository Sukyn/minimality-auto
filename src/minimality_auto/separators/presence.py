from __future__ import annotations

from itertools import combinations
from typing import Any, Iterator

from ..core import primitive_occurrences
from ..search import CandidateModel, Deadline, relevant_equations


def _primitive_names(theory: Any, target: Any, deadline: Deadline) -> list[str]:
    """Names that can occur in the supplied syntax, including inline primitives."""
    names: set[str] = set()
    for equation in [*relevant_equations(theory, target), target]:
        deadline.check()
        names.update(primitive_occurrences(equation.lhs, theory.macros))
        names.update(primitive_occurrences(equation.rhs, theory.macros))
    return sorted(names)


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    **_: Any,
) -> Iterator[CandidateModel]:
    """Boolean-max PROP interpretations, with smallest supports first.

    Every object pair is interpreted by the Boolean commutative monoid; both
    composition and tensor are ``or``, while identities and symmetries are
    false.  Thus the interpretation remains sound for arity-changing PROPs.
    """
    if bound < 1:
        return
    names = _primitive_names(theory, target, deadline)
    if bound > len(names):
        return
    for selected_tuple in combinations(names, bound):
        deadline.check()
        selected = frozenset(selected_tuple)

        def evaluate(term: Any, selected: frozenset[str] = selected) -> bool:
            counts = primitive_occurrences(term, theory.macros)
            return any(counts.get(name, 0) > 0 for name in selected)

        label = ", ".join(selected_tuple)
        yield CandidateModel(
            kind="presence",
            description=f"presence of {{{label}}}",
            parameters={"generators": list(selected_tuple)},
            evaluator=evaluate,
            key=selected_tuple,
        )
