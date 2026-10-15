from __future__ import annotations

import cmath
import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..core import (
    _wire_size_matches,
)
from ..search import CandidateModel, Deadline, NotApplicable
from .common import circuit_analysis, primitive_names, wire_parity_is_preserved

PHASE_TOLERANCE = 1e-8


def _phase(matrix: np.ndarray, dimension: int, arity: int) -> float:
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise NotApplicable("endomorphism generator matrix must be square")
    size = matrix.shape[0]
    if not _wire_size_matches(size, dimension, arity):
        raise NotApplicable(
            f"generator matrix dimension {size} is incompatible with arity {arity}"
        )
    if not np.all(np.isfinite(matrix)):
        raise NotApplicable("generator matrices must have finite entries")
    try:
        sign, log_absolute = np.linalg.slogdet(matrix)
    except np.linalg.LinAlgError as exc:
        raise NotApplicable("could not compute a generator determinant") from exc
    if not np.isfinite(log_absolute) or abs(sign) < 1e-10:
        raise NotApplicable("determinant model requires invertible generator matrices")
    return cmath.phase(complex(sign)) % math.tau


def _scale(dimension: int, exponent: int) -> float | None:
    """Return ``dimension**exponent`` as a float within the size bound."""
    if dimension == 1:
        return 1.0
    if exponent < 0:
        return float(dimension**exponent)
    if exponent * dimension.bit_length() > 1023:
        return None
    return float(dimension**exponent)


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    context: Any = None,
) -> Iterator[CandidateModel]:
    dimension = int(getattr(theory, "wire_dimension", 2))
    if dimension < 1:
        raise NotApplicable("wire dimension must be positive")
    k = bound
    if k < 2:
        return
    active = set(primitive_names(theory, target, deadline, context))
    weights: dict[str, float] = {}
    defaulted: list[str] = []
    for generator in theory.signature.values():
        deadline.check()
        matrix = getattr(generator, "matrix", None)
        source = int(generator.inputs)
        target_arity = int(generator.outputs)
        name = str(generator.name)
        if name not in active:
            continue
        if source != target_arity or matrix is None:
            # The additive model accepts arbitrary generator weights, so these
            # generators receive zero.
            weights[name] = 0.0
            defaulted.append(name)
            continue
        try:
            phase = _phase(np.asarray(matrix), dimension, source)
        except (NotApplicable, TypeError, ValueError):
            weights[name] = 0.0
            defaulted.append(name)
            continue
        scale = _scale(dimension, k - source)
        if scale is None:
            weights[name] = 0.0
            defaulted.append(name)
            continue
        weights[name] = (scale * phase) % math.tau

    base_swap_is_odd = (dimension * (dimension - 1) // 2) % 2
    padding_is_odd = k == 2 or dimension % 2 == 1
    epsilon = math.pi if base_swap_is_odd and padding_is_odd else 0.0
    if epsilon and not wire_parity_is_preserved(theory, target.lhs, target.rhs):
        # Parity-changing maps require the trivial symmetry for naturality.
        epsilon = 0.0

    def evaluate(term: Any) -> float:
        analysis = circuit_analysis(term, context)
        counts = analysis.occurrence_counter()
        value = sum(counts[name] * phase for name, phase in weights.items())
        value += analysis.permutation_parity * epsilon
        return value % math.tau

    def equal(left: float, right: float) -> bool:
        distance = (left - right + math.pi) % math.tau - math.pi
        return abs(distance) <= PHASE_TOLERANCE

    if not any(abs(value) > 1e-10 for value in weights.values()) and not epsilon:
        return
    compact = {
        name: round(value / math.pi, 10)
        for name, value in weights.items()
        if abs(value) > 1e-10
    }
    yield CandidateModel(
        description=(
            f"numerical scaled determinant phase (dimension={dimension}, k={k})"
        ),
        parameters={
            "wire_dimension": dimension,
            "k": k,
            "generator_phases_over_pi": compact,
            "swap_phase_over_pi": epsilon / math.pi,
            "zero_weight_generators": defaulted,
            "phase_tolerance": PHASE_TOLERANCE,
        },
        evaluate=evaluate,
        equal=equal,
        certified=False,
    )
