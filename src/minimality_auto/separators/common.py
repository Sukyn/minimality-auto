from __future__ import annotations

from typing import Any

import numpy as np

from ..core import (
    Circuit,
    contains_arity_change,
    iter_circuit_nodes,
    iter_theory_terms,
    primitive_occurrences,
)
from ..matrix import evaluate_matrix
from ..search import Deadline, NotApplicable, relevant_equations


def retained_equations(
    theory: Any,
    target: Any,
    context: Any = None,
) -> tuple[Any, ...]:
    if context is not None:
        return context.equations_for(target)
    return tuple(relevant_equations(theory, target))


def primitive_names(
    theory: Any,
    target: Any,
    deadline: Deadline,
    context: Any = None,
) -> list[str]:
    names: set[str] = set()
    for equation in (*retained_equations(theory, target, context), target):
        deadline.check()
        for term in (equation.lhs, equation.rhs):
            counts = (
                context.occurrences_for(term)
                if context is not None
                else primitive_occurrences(term)
            )
            names.update(counts)
    return sorted(names)


def projectively_equal(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """Compare finite numerical matrices up to a phase and tolerance."""

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
    """Evaluate a finite matrix for a conservative numerical heuristic."""

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
    """Whether the sign character is natural around every supplied generator.

    Naturality around ``f: m -> n`` requires input and output wire counts to
    have the same parity. Requiring every generator to be an endomorphism would
    be unnecessarily restrictive.
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
    """Guard fixed-arity interpretations against non-endomorphic PROP syntax."""

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
