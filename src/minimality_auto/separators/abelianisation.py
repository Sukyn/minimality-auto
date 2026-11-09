"""The full additive quotient of primitive counts and the symmetry sign."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from ..algebra.abelian import smith_basis
from ..search_types import CandidateModel, Deadline
from .common import (
    circuit_analysis,
    primitive_names,
    retained_equations,
    wire_parity_is_preserved,
)


def candidates(
    theory: Any, target: Any, *, bound: int, deadline: Deadline, context: Any = None
) -> Iterator[CandidateModel]:
    if bound != 0:
        return
    names = primitive_names(theory, target, deadline, context)
    use_swap = wire_parity_is_preserved(theory, target.lhs, target.rhs)
    columns = len(names) + int(use_swap)

    def counts(term: Any) -> tuple[int, ...]:
        analysis = circuit_analysis(term, context)
        occurrences = dict(analysis.occurrences)
        result = tuple(occurrences.get(name, 0) for name in names)
        return result + (analysis.permutation_parity,) if use_swap else result

    relations = [
        tuple(a - b for a, b in zip(counts(eq.lhs), counts(eq.rhs), strict=True))
        for eq in retained_equations(theory, target, context)
    ]
    if use_swap:
        relations.append((0,) * len(names) + (2,))
    diagonal, basis = smith_basis(relations, columns, deadline)
    coordinates = tuple(i for i, modulus in enumerate(diagonal) if modulus != 1)
    moduli = tuple(diagonal[i] for i in coordinates)

    def reduce(values: Iterable[int]) -> tuple[int, ...]:
        return tuple(
            value % modulus if modulus else value
            for value, modulus in zip(values, moduli, strict=True)
        )

    images = tuple(reduce(row[i] for i in coordinates) for row in basis)

    def evaluate(term: Any) -> tuple[int, ...]:
        deadline.check()
        row = counts(term)
        return reduce(
            sum(count * image[i] for count, image in zip(row, images, strict=True))
            for i in range(len(moduli))
        )

    if evaluate(target.lhs) == evaluate(target.rhs):
        return
    weights = dict(zip(names, images[: len(names)], strict=True))
    swap = images[-1] if use_swap else (0,) * len(moduli)
    free_rank = moduli.count(0)
    torsion = tuple(modulus for modulus in moduli if modulus)
    yield CandidateModel(
        description=f"abelianisation: Z^{free_rank}, torsion {torsion}",
        parameters={
            "free_rank": free_rank,
            "torsion": torsion,
            "coordinate_moduli": moduli,
            "weights": weights,
            "swap": swap,
        },
        evaluate=evaluate,
    )
