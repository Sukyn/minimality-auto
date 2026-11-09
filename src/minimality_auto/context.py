"""Cached circuit analyses and the retained equations for each target."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .analysis import CircuitAnalysis, analyze_circuit, contains_arity_change, iter_theory_terms
from .search_types import Deadline
from .syntax import Circuit, Equation, Theory


@dataclass(slots=True)
class TargetContext:
    equations: tuple[Equation, ...]
    primitive_names: tuple[str, ...]
    fixed_arity_presentation: Any = None


@dataclass(slots=True)
class SearchContext:
    targets: dict[Equation, TargetContext]
    active_primitive_count: int
    endomorphic: bool
    analyses: dict[Circuit, CircuitAnalysis]

    @classmethod
    def build(
        cls,
        theory: Theory,
        targets: Iterable[Equation],
        deadline: Deadline | None = None,
    ) -> SearchContext:
        targets = tuple(targets)
        if deadline is not None:
            deadline.check()
        checkpoint = deadline.check if deadline is not None else None
        terms = dict.fromkeys(iter_theory_terms(theory))
        analyses = {term: analyze_circuit(term, checkpoint) for term in terms}
        endomorphic = _theory_is_endomorphic(theory, deadline, analyses)
        target_contexts: dict[Equation, TargetContext] = {}
        active_names: set[str] = set()
        for target in targets:
            if deadline is not None:
                deadline.check()
            retained = tuple(
                _relevant_equations(
                    theory,
                    target,
                    endomorphic=endomorphic,
                    deadline=deadline,
                    analyses=analyses,
                )
            )
            names: set[str] = set()
            for equation in (*retained, target):
                for term in (equation.lhs, equation.rhs):
                    if deadline is not None:
                        deadline.check()
                    names.update(name for name, _ in analyses[term].occurrences)
            target_contexts[target] = TargetContext(retained, tuple(sorted(names)))
            active_names.update(names)
        return cls(
            target_contexts,
            len(active_names),
            endomorphic,
            analyses,
        )

    def for_target(self, target: Equation) -> TargetContext:
        return self.targets[target]

    def analysis_for(self, term: Circuit) -> CircuitAnalysis:
        analysis = self.analyses.get(term)
        if analysis is None:
            analysis = analyze_circuit(term)
            self.analyses[term] = analysis
        return analysis


def _changes_arity(
    term: Circuit,
    analyses: Mapping[Circuit, CircuitAnalysis] | None,
    deadline: Deadline | None,
) -> bool:
    if analyses is not None:
        return analyses[term].changes_arity
    return contains_arity_change(term, deadline.check if deadline is not None else None)


def _theory_is_endomorphic(
    theory: Theory,
    deadline: Deadline | None = None,
    analyses: Mapping[Circuit, CircuitAnalysis] | None = None,
) -> bool:
    for generator in theory.signature.values():
        if deadline is not None:
            deadline.check()
        if generator.inputs != generator.outputs:
            return False
    return all(not _changes_arity(term, analyses, deadline) for term in iter_theory_terms(theory))


def _relevant_equations(
    theory: Theory,
    target: Equation,
    *,
    endomorphic: bool,
    deadline: Deadline | None = None,
    analyses: Mapping[Circuit, CircuitAnalysis] | None = None,
) -> list[Equation]:
    prune = endomorphic and not any(
        _changes_arity(term, analyses, deadline) for term in (target.lhs, target.rhs)
    )
    return [
        equation
        for equation in theory.equations
        if equation.id != target.id and (not prune or equation.arity <= target.arity)
    ]


def relevant_equations(theory: Theory, target: Equation) -> list[Equation]:
    """Select the retained equations required to validate a candidate.

    Endomorphic targets in endomorphic theories need equations up to the
    target arity; other cases need every retained equation.
    """
    return _relevant_equations(theory, target, endomorphic=_theory_is_endomorphic(theory))


def retained_equations(
    theory: Any,
    target: Any,
    context: Any = None,
) -> tuple[Any, ...]:
    if context is not None:
        return context.for_target(target).equations
    return tuple(relevant_equations(theory, target))
