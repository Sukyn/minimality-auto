"""Shared finite matrix synthesis; no supplied generator matrices are required."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import Any

import z3

from ..algebra.semiring_matrix import FiniteSemiring, SemiringMatrixSemantics, symbolic_equal
from ..search_types import CandidateModel, Deadline, NotApplicable, SearchTimeout
from ..syntax import Circuit
from .common import is_prime, primitive_names, retained_equations
from .synthesis import solve


def _synthesize(
    theory: Any,
    target: Any,
    kind: str,
    dimension: int,
    prime: int,
    deadline: Deadline,
    max_entries: int,
    context: Any,
    model_timeout_ms: int,
) -> CandidateModel | None:
    attempt = deadline.capped(model_timeout_ms / 1000)
    solver = z3.Solver()
    ring = FiniteSemiring(kind, prime, symbolic=True)
    interpretation = {}
    semantics = SemiringMatrixSemantics(dimension, ring, interpretation, attempt, max_entries)
    for index, name in enumerate(primitive_names(theory, target, deadline, context)):
        rows, columns = semantics.shape(Circuit.generator(theory.signature[name]))
        matrix = []
        for i in range(rows):
            attempt.check()
            matrix.append(tuple(ring.unknown(solver, f"g{index}_{i}_{j}") for j in range(columns)))
        interpretation[name] = tuple(matrix)

    def equal(equation: Any) -> z3.BoolRef:
        return symbolic_equal(
            semantics.evaluate(equation.lhs),
            semantics.evaluate(equation.rhs),
            dual=kind == "dual_numbers",
        )

    for equation in retained_equations(theory, target, context):
        solver.add(equal(equation))
    solver.add(z3.Not(equal(target)))
    solution = solve(solver, attempt)
    if solution is None:
        return None
    values = {
        name: tuple(tuple(ring.decode(solution, value) for value in row) for row in matrix)
        for name, matrix in interpretation.items()
    }
    exact = SemiringMatrixSemantics(
        dimension,
        FiniteSemiring(kind, prime),
        values,
        deadline,
        max_entries,
    )
    # The common verifier checks retained equations and separation with this exact evaluator.
    ring_name = {
        "relations": "Boolean",
        "finite_field_matrices": f"F_{prime}",
        "dual_numbers": f"F_{prime}[epsilon]/(epsilon^2)",
    }[kind]
    return CandidateModel(
        description=f"matrices over {ring_name}, wire dimension {dimension}",
        parameters={
            "ring": ring_name,
            "characteristic": None if kind == "relations" else prime,
            "dimension": dimension,
            "interpretation": values,
        },
        evaluate=exact.evaluate,
    )


def _candidates(
    theory: Any,
    target: Any,
    *,
    kind: str,
    bound: int,
    deadline: Deadline,
    max_matrix_dimension: int = 2,
    max_matrix_prime: int = 5,
    max_matrix_entries: int = 256,
    model_timeout_ms: int = 1000,
    context: Any = None,
) -> Iterator[CandidateModel]:
    if bound < 1 or bound > max(max_matrix_dimension, max_matrix_prime):
        return
    primes = (
        (2,)
        if kind == "relations"
        else (p for p in range(2, min(bound, max_matrix_prime) + 1) if is_prime(p, deadline))
    )
    for prime in primes:
        for dimension in range(1, min(bound, max_matrix_dimension) + 1):
            if (dimension if kind == "relations" else max(prime, dimension)) != bound:
                continue
            try:
                model = _synthesize(
                    theory,
                    target,
                    kind,
                    dimension,
                    prime,
                    deadline,
                    max_matrix_entries,
                    context,
                    model_timeout_ms,
                )
            except (NotApplicable, SearchTimeout):
                deadline.check()
                continue
            if model is not None:
                yield model


relations = partial(_candidates, kind="relations")
dual_numbers = partial(_candidates, kind="dual_numbers")
finite_field_matrices = partial(_candidates, kind="finite_field_matrices")
