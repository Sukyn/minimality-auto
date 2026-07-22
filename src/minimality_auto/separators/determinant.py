from __future__ import annotations

import cmath
import math
from typing import Any, Iterator

import numpy as np

from ..core import primitive_occurrences, structural_permutation_parity
from ..search import CandidateModel, Deadline, NotApplicable
from .counting import _wire_parity_is_preserved
from .presence import _primitive_names


TAU = 2.0 * math.pi


def _phase(matrix: np.ndarray, dimension: int, arity: int) -> float:
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise NotApplicable("endomorphism generator matrix must be square")
    size = matrix.shape[0]
    if dimension == 1:
        shape_is_valid = size == 1
    else:
        # If d**arity equals the observed (machine-sized) dimension, arity is
        # at most its bit length.  This avoids constructing hostile huge ints.
        shape_is_valid = arity <= size.bit_length() and dimension**arity == size
    if not shape_is_valid:
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
    return cmath.phase(complex(sign)) % TAU


def _scale(dimension: int, exponent: int) -> float | None:
    """Return d**exponent when it has a useful finite float representation."""
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
    **_: Any,
) -> Iterator[CandidateModel]:
    dimension = int(getattr(theory, "wire_dimension", 2))
    if dimension < 1:
        raise NotApplicable("wire dimension must be positive")
    k = bound
    if k < 2:
        return
    active = set(_primitive_names(theory, target, deadline))
    weights: dict[str, float] = {}
    defaulted: list[str] = []
    for generator in theory.signature.values():
        deadline.check()
        matrix = getattr(generator, "matrix", None)
        source = int(generator.inputs)
        target_arity = int(generator.outputs)
        name = str(getattr(generator, "name"))
        if name not in active:
            continue
        if source != target_arity or matrix is None:
            # An arbitrary zero weight is a sound extension of the phase
            # character.  This lets known square matrices coexist with
            # rectangular or intentionally matrix-free generators.
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
        weights[name] = (scale * phase) % TAU

    base_swap_is_odd = (dimension * (dimension - 1) // 2) % 2
    padding_is_odd = k == 2 or dimension % 2 == 1
    epsilon = math.pi if base_swap_is_odd and padding_is_odd else 0.0
    if epsilon and not _wire_parity_is_preserved(theory, target.lhs, target.rhs):
        # A sign-valued symmetry is not natural around parity-changing maps.
        # The trivial symmetry still gives a valid additive PROP model.
        epsilon = 0.0

    def evaluate(term: Any) -> float:
        counts = primitive_occurrences(term, theory.macros)
        value = sum(counts.get(name, 0) * phase for name, phase in weights.items())
        value += structural_permutation_parity(term, theory.macros) * epsilon
        return value % TAU

    def equal(left: float, right: float) -> bool:
        distance = (left - right + math.pi) % TAU - math.pi
        return abs(distance) <= 1e-8

    if not any(abs(value) > 1e-10 for value in weights.values()) and not epsilon:
        return
    compact = {
        name: round(value / math.pi, 10)
        for name, value in weights.items()
        if abs(value) > 1e-10
    }
    yield CandidateModel(
        kind="determinant",
        description=f"scaled determinant phase (dimension={dimension}, k={k})",
        parameters={
            "wire_dimension": dimension,
            "k": k,
            "generator_phases_over_pi": compact,
            "swap_phase_over_pi": epsilon / math.pi,
            "zero_weight_generators": defaulted,
        },
        evaluator=evaluate,
        equality=equal,
        key=(k,),
    )
