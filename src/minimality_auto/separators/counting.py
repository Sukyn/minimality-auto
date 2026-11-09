"""Modular invariants of primitive counts and wire-permutation signs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..algebra.abelian import separating_weights
from ..search_types import CandidateModel, Deadline
from .common import (
    circuit_analysis,
    primitive_names,
    retained_equations,
    wire_parity_is_preserved,
)


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_modulus: int = 8,
    context: Any = None,
) -> Iterator[CandidateModel]:
    """Solve for an additive ``Z/m`` PROP separator, optionally using the swap sign."""
    modulus = bound
    if modulus < 2 or modulus > max_modulus:
        return
    names = primitive_names(theory, target, deadline, context)
    use_swap = modulus % 2 == 0 and wire_parity_is_preserved(theory, target.lhs, target.rhs)

    def counts(term: Any) -> tuple[int, ...]:
        deadline.check()
        analysis = circuit_analysis(term, context)
        occurrences = dict(analysis.occurrences)
        row = tuple(occurrences.get(name, 0) for name in names)
        return row + (analysis.permutation_parity,) if use_swap else row

    def difference(equation: Any) -> tuple[int, ...]:
        return tuple(a - b for a, b in zip(counts(equation.lhs), counts(equation.rhs), strict=True))

    rows = [difference(eq) for eq in retained_equations(theory, target, context)]
    if use_swap:
        rows.append((0,) * len(names) + (2,))  # Symmetry is an involution: 2 * swap = 0.
    values = separating_weights(rows, difference(target), modulus, deadline)
    if values is None:
        return

    def evaluate(term: Any) -> int:
        return (
            sum(count * weight for count, weight in zip(counts(term), values, strict=True))
            % modulus
        )

    weights = {
        name: value for name, value in zip(names, values[: len(names)], strict=True) if value
    }
    swap = values[-1] if use_swap else 0
    description = f"count mod {modulus}: {weights}"
    if swap:
        description += f", swap={swap}"
    yield CandidateModel(
        description=description,
        parameters={"modulus": modulus, "weights": weights, "swap": swap},
        evaluate=evaluate,
    )
