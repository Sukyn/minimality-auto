from __future__ import annotations

from collections.abc import Iterable
from math import isqrt
from typing import Any

import numpy as np

from ..core import (
    Circuit,
    CircuitAnalysis,
    analyze_circuit,
    contains_arity_change,
    iter_circuit_nodes,
    iter_theory_terms,
)
from ..matrix import evaluate_matrix
from ..search import Deadline, NotApplicable, relevant_equations


def bounded_power(base: int, exponent: int, limit: int) -> int | None:
    if base < 1 or exponent < 0 or limit < 1:
        return None
    if base == 1:
        return 1
    result = 1
    for _ in range(exponent):
        if result > limit // base:
            return None
        result *= base
    return result


def is_prime(value: int, deadline: Deadline | None = None) -> bool:
    if value < 2:
        return False
    for divisor in range(2, isqrt(value) + 1):
        if deadline is not None and divisor % 1024 == 0:
            deadline.check()
        if value % divisor == 0:
            return False
    return True


def retained_equations(
    theory: Any,
    target: Any,
    context: Any = None,
) -> tuple[Any, ...]:
    if context is not None:
        return context.for_target(target).equations
    return tuple(relevant_equations(theory, target))


def circuit_analysis(term: Circuit, context: Any = None) -> CircuitAnalysis:
    return context.analysis_for(term) if context is not None else analyze_circuit(term)


def active_primitive_names(
    terms: Iterable[Circuit], context: Any = None
) -> frozenset[str]:
    return frozenset(
        name
        for term in terms
        for name, _ in circuit_analysis(term, context).occurrences
    )


def primitive_names(
    theory: Any,
    target: Any,
    deadline: Deadline,
    context: Any = None,
) -> tuple[str, ...]:
    if context is not None:
        deadline.check()
        return context.for_target(target).primitive_names
    names: set[str] = set()
    for equation in (*retained_equations(theory, target, context), target):
        deadline.check()
        for term in (equation.lhs, equation.rhs):
            names.update(name for name, _ in circuit_analysis(term, context).occurrences)
    return tuple(sorted(names))


def projectively_equal(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """Compare finite matrices up to a global phase and the given tolerance."""

    left = np.asarray(left, dtype=complex)
    right = np.asarray(right, dtype=complex)
    if (
        left.shape != right.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        return False
    if not left.size:
        return True
    left_scale = float(np.max(np.abs(left)))
    right_scale = float(np.max(np.abs(right)))
    if left_scale == 0.0 or right_scale == 0.0:
        return left_scale == right_scale

    scale = max(left_scale, right_scale)
    left = left / scale
    right = right / scale
    joint_scale = np.maximum(np.abs(left), np.abs(right))
    index = np.unravel_index(int(np.argmax(joint_scale)), right.shape)
    if abs(left[index]) <= tolerance or abs(right[index]) <= tolerance:
        return False
    phase = left[index] / right[index]
    phase /= abs(phase)
    return float(np.max(np.abs(left - phase * right))) <= tolerance


def numerical_matrix(theory: Any, term: Any, purpose: str) -> np.ndarray:
    """Evaluate a term as a finite complex matrix for numerical search."""

    try:
        value = np.asarray(
            evaluate_matrix(
                term,
                theory.signature,
                theory.wire_dimension,
            ),
            dtype=np.complex128,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise NotApplicable(f"{purpose} needs valid matrices") from exc
    if not np.all(np.isfinite(value)):
        raise NotApplicable(f"{purpose} needs finite matrix entries")
    return value


def wire_parity_is_preserved(theory: Any, *extra_terms: Circuit) -> bool:
    """Check the parity condition for the permutation sign character.

    Naturality around ``f: m -> n`` requires ``m % 2 == n % 2`` for every
    generator.
    """

    def preserves(root: Circuit) -> bool:
        return all(
            node.kind != "gen" or node.inputs % 2 == node.outputs % 2
            for node in iter_circuit_nodes(root)
        )

    if any(
        generator.inputs % 2 != generator.outputs % 2
        for generator in theory.signature.values()
    ):
        return False
    return all(preserves(term) for term in iter_theory_terms(theory, *extra_terms))


def require_endomorphic_theory(
    theory: Any,
    deadline: Deadline,
    context: Any = None,
) -> None:
    """Require equal input and output arities in all generators and terms."""

    if context is not None:
        deadline.check()
        if not context.endomorphic:
            raise NotApplicable(
                "fixed-arity PROP search requires an endomorphic theory"
            )
        return

    for generator in theory.signature.values():
        deadline.check()
        if generator.inputs != generator.outputs:
            raise NotApplicable(
                "fixed-arity PROP search requires an endomorphic signature"
            )

    for term in iter_theory_terms(theory):
        if contains_arity_change(term, deadline.check):
            raise NotApplicable(
                "fixed-arity PROP search requires endomorphic "
                "signature terms and macros"
            )
