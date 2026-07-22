"""Spin-cover separators for high-arity commutation equations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Iterator, Sequence

import numpy as np

from ..core import (
    Circuit,
    evaluate_matrix,
    expand_macros,
    primitive_occurrences,
)
from ..search import (
    Deadline,
    NotApplicable,
    Separation,
    equation_id,
    relevant_equations,
)
from .finite_model import _require_endomorphic_theory


TOLERANCE = 1e-12
MAX_DENSE_DIMENSION = 1024


def _close(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, atol=TOLERANCE, rtol=0.0))


def _rank(value: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(value, tol=TOLERANCE))


def _matrix(theory: Any, term: Circuit) -> np.ndarray:
    try:
        value = np.asarray(
            evaluate_matrix(
                term,
                theory.signature,
                theory.macros,
                theory.wire_dimension,
            ),
            dtype=np.complex128,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise NotApplicable(
            "Spin-cover search needs valid matrices for every active primitive"
        ) from exc
    if not np.all(np.isfinite(value)):
        raise NotApplicable("Spin-cover search requires finite matrix entries")
    return value


def _parts(term: Circuit) -> tuple[Circuit, ...]:
    parts = term.parts if term.kind == "compose" else (term,)
    return tuple(part for part in parts if part.kind != "id")


def _check_reflection(name: str, value: np.ndarray, width: int) -> None:
    size = 1 << width
    if value.shape != (size, size) or not np.all(np.isfinite(value)):
        raise NotApplicable(f"{name} is not a real orthogonal reflection")
    identity = np.eye(size, dtype=np.complex128)
    if (
        not _close(value.imag, np.zeros_like(value.imag))
        or not _close(value.T, value)
        or not _close(value.T @ value, identity)
        or not _close(value @ value, identity)
        or _rank(identity - value) != 1
        or not np.isclose(
            np.linalg.det(value).real,
            -1.0,
            atol=TOLERANCE,
            rtol=0.0,
        )
    ):
        raise NotApplicable(f"{name} is not a real orthogonal reflection")


def _reflection_generators(
    theory: Any, active: frozenset[str], deadline: Deadline
) -> tuple[tuple[str, int], ...]:
    if theory.wire_dimension != 2:
        raise NotApplicable("Spin-cover search currently requires qubit generators")

    generators: list[tuple[str, int]] = []
    for generator in theory.signature.values():
        if generator.name not in active:
            continue
        deadline.check()
        if generator.matrix is None:
            raise NotApplicable(
                f"Spin-cover search needs a matrix for {generator.name!r}"
            )
        value = _matrix(theory, Circuit.generator(generator))
        _check_reflection(f"generator {generator.name!r}", value, generator.inputs)
        generators.append((generator.name, generator.inputs))

    missing = active.difference(theory.signature)
    if missing:
        name = min(missing)
        raise NotApplicable(
            f"Spin-cover search needs valid matrices for active primitive {name!r}"
        )

    swap = _matrix(theory, Circuit.perm((1, 0)))
    _check_reflection("structural SWAP", swap, 2)
    return tuple(generators)


def _active_generators(theory: Any, terms: Sequence[Circuit]) -> frozenset[str]:
    names: set[str] = set()
    for term in terms:
        names.update(primitive_occurrences(term, theory.macros))
    return frozenset(names)


def _commutator_data(
    left_first: np.ndarray,
    left_second: np.ndarray,
) -> tuple[int, int, int] | None:
    if left_first.shape != left_second.shape:
        return None
    identity = np.eye(left_first.shape[0], dtype=np.complex128)
    zero = np.zeros_like(left_first.real)
    for value in (left_first, left_second):
        if (
            not np.all(np.isfinite(value))
            or not _close(value.imag, zero)
            or not _close(value.T, value)
            or not _close(value.T @ value, identity)
            or not _close(value @ value, identity)
        ):
            return None
    if not _close(left_first @ left_second, left_second @ left_first):
        return None

    first_projector = (identity - left_first) / 2
    second_projector = (identity - left_second) / 2
    intersection = first_projector @ second_projector
    if (
        not _close(first_projector.T, first_projector)
        or not _close(second_projector.T, second_projector)
        or not _close(intersection.T, intersection)
        or not _close(intersection @ intersection, intersection)
    ):
        return None

    first_dim = _rank(first_projector)
    second_dim = _rank(second_projector)
    intersection_dim = _rank(intersection)
    if first_dim % 2 or second_dim % 2:
        return None
    return first_dim, second_dim, intersection_dim


def _commutator_sign(data: tuple[int, int, int]) -> int:
    first_dim, second_dim, intersection_dim = data
    return -1 if (first_dim * second_dim - intersection_dim) % 2 else 1


def _find_cyclic_reversal_lifts(
    theory: Any,
    equation: Any,
    deadline: Deadline,
    matrix: Callable[[Circuit], np.ndarray],
    *,
    required_sign: int,
) -> tuple[int, int, int] | None:
    left_parts = _parts(expand_macros(equation.lhs, theory.macros))
    right_parts = _parts(expand_macros(equation.rhs, theory.macros))

    for index in range(1, len(left_parts)):
        deadline.check()
        first_parts = left_parts[:index]
        second_parts = left_parts[index:]
        if right_parts != second_parts + first_parts:
            continue
        first = matrix(Circuit.compose(first_parts))
        second = matrix(Circuit.compose(second_parts))
        data = _commutator_data(first, second)
        if data is not None and _commutator_sign(data) == required_sign:
            return data
    return None


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_spin_matrix_dimension: int = MAX_DENSE_DIMENSION,
    **_: Any,
) -> Iterator[Separation]:
    if bound != 0:
        return
    if (
        target.lhs.type != target.rhs.type
        or target.lhs.inputs != target.lhs.outputs
    ):
        return
    _require_endomorphic_theory(theory, deadline)
    arity = target.lhs.inputs
    if arity < 5:
        return
    if max_spin_matrix_dimension < 1:
        raise NotApplicable("the Spin matrix-dimension bound must be positive")
    if (
        arity >= max_spin_matrix_dimension.bit_length()
        or 1 << arity > max_spin_matrix_dimension
    ):
        raise NotApplicable(
            f"ambient matrix dimension exceeds {max_spin_matrix_dimension}"
        )

    retained = relevant_equations(theory, target)
    active = _active_generators(
        theory,
        (
            target.lhs,
            target.rhs,
            *(side for equation in retained for side in (equation.lhs, equation.rhs)),
        ),
    )
    primitive_generators = _reflection_generators(theory, active, deadline)
    primitive_widths = sorted(
        {2} | {width for _, width in primitive_generators}
    )  # include the fixed adjacent-SWAP lift
    for width in primitive_widths:
        complement = arity - width
        if complement < 3:
            raise NotApplicable("too few complement wires for the Spin lift")
    for index, left_width in enumerate(primitive_widths):
        for right_width in primitive_widths[index:]:
            # The lift exponent is
            # 2**complement * (2**arity - 1), so it is odd exactly when
            # disjoint supports fill every wire.  Overlapping supports impose
            # no PROP-interchange equation.
            if left_width + right_width == arity:
                raise NotApplicable("too few wires for PROP interchange")

    matrices: dict[Circuit, np.ndarray] = {}

    def matrix(term: Circuit) -> np.ndarray:
        deadline.check()
        if term not in matrices:
            matrices[term] = _matrix(theory, term)
        return matrices[term]

    full_arity_reversals: list[dict[str, Any]] = []
    for equation in retained:
        deadline.check()
        if (
            equation.lhs.type != equation.rhs.type
            or equation.lhs.inputs != equation.lhs.outputs
        ):
            raise NotApplicable(
                "another retained equation is not an endomorphism"
            )
        complement = arity - equation.lhs.inputs
        if complement < 0:
            raise NotApplicable(
                "another retained equation is too large for this Spin certificate"
            )
        if complement >= 1:
            # With one outside wire, every local primitive is already even:
            # the ambient primitive guard gives
            # 2**((arity - 1) - primitive_width), a multiple of four, local
            # copies.  The same holds for adjacent SWAP.  Hence the two-block
            # lift is multiplicative for arbitrary compositions, tensors and
            # permutations; it also sends the two possible local lifts u and
            # -u to the same value.  Ordinary matrix equality is sufficient.
            # More outside wires are the usual block-lift case.
            left = matrix(equation.lhs)
            right = matrix(equation.rhs)
            deadline.check()
            if not _close(left, right):
                raise NotApplicable(
                    f"retained equation {equation_id(equation)!r} is not matrix-sound"
                )
        else:
            data = _find_cyclic_reversal_lifts(
                theory,
                equation,
                deadline,
                matrix,
                required_sign=1,
            )
            if data is None:
                raise NotApplicable(
                    f"retained equation {equation_id(equation)!r} has no "
                    "certified trivial block-lift sign"
                )
            full_arity_reversals.append(
                {
                    "equation": equation_id(equation),
                    "minus_dimensions": [data[0], data[1]],
                    "intersection_dimension": data[2],
                    "commutator_sign": 1,
                }
            )

    data = _find_cyclic_reversal_lifts(
        theory,
        target,
        deadline,
        matrix,
        required_sign=-1,
    )
    if data is None:
        return
    first_dim, second_dim, intersection_dim = data
    description = (
        f"ambient-{arity}-wire Spin-cover invariant: the commuting target "
        "factors lift with sign -1"
    )
    if full_arity_reversals:
        description += (
            f"; {len(full_arity_reversals)} full-arity retained "
            "reversal(s) have sign +1"
        )

    yield Separation(
        equation=equation_id(target),
        strategy="spin",
        description=description,
        parameters={
            "arity": arity,
            "primitive_reflections": [name for name, _ in primitive_generators],
            "structural_permutations": "fixed adjacent-SWAP decomposition",
            "matrix_tolerance": TOLERANCE,
            "minus_dimensions": [first_dim, second_dim],
            "intersection_dimension": intersection_dim,
            "commutator_sign": -1,
            "full_arity_retained_reversals": full_arity_reversals,
        },
        checked_equations=tuple(equation_id(equation) for equation in retained),
        lhs_value={"orthogonal_value": "same", "spin_lift": "u"},
        rhs_value={"orthogonal_value": "same", "spin_lift": "-u"},
    )
