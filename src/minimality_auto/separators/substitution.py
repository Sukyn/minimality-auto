from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress
from itertools import permutations
from typing import Any

import numpy as np

from ..matrix import MatrixSemantics, evaluate_matrix
from ..search_types import CandidateModel, Deadline, NotApplicable
from ..syntax import Circuit, _matrix_shape_matches
from .common import (
    bounded_power,
    circuit_analysis,
    projectively_equal,
    retained_equations,
)

DEFAULT_MAX_MATRIX_ENTRIES = 1_000_000
PROJECTIVE_TOLERANCE = 1e-8
_Operation = tuple[str, str, int, int]


def _fingerprint(matrix: np.ndarray) -> bytes:
    value = np.asarray(matrix, dtype=complex).copy()
    flat = value.ravel()
    pivot = int(np.argmax(np.abs(flat)))
    if abs(flat[pivot]) > 1e-10:
        value /= flat[pivot] / abs(flat[pivot])
    rounded = np.round(value.real, 9) + 1j * np.round(value.imag, 9)
    rounded.real[np.abs(rounded.real) < 1e-12] = 0.0
    rounded.imag[np.abs(rounded.imag) < 1e-12] = 0.0
    return rounded.tobytes()


def _operation_alphabet(
    wire_dimension: int,
    wires: int,
    operations: tuple[_Operation, ...],
    deadline: Deadline,
) -> Iterator[tuple[dict[str, Any], int]]:
    for kind, name, source, target in operations:
        deadline.check()
        if not 0 <= source <= wires:
            continue
        placements = (
            (tuple(range(source)),) if wire_dimension == 1 else permutations(range(wires), source)
        )
        for placement in placements:
            deadline.check()
            yield (
                {kind: name, "on": list(placement)},
                wires - source + target,
            )
    if wire_dimension > 1 and wires >= 2:
        for left in range(wires):
            for right in range(left + 1, wires):
                deadline.check()
                permutation = list(range(wires))
                permutation[left], permutation[right] = permutation[right], permutation[left]
                yield {"perm": permutation}, wires


def _matrix_budget_allows(
    wire_dimension: int,
    wires: int,
    max_matrix_entries: int,
) -> bool:
    return bounded_power(wire_dimension, 2 * wires, max_matrix_entries) is not None


def _replacement_terms(
    theory: Any,
    inputs: int,
    outputs: int,
    depth: int,
    deadline: Deadline,
    operation_records: tuple[_Operation, ...],
    max_matrix_entries: int,
) -> Iterator[tuple[Any, str]]:
    if depth == 0:
        if inputs == outputs:
            yield Circuit.identity(inputs), f"id_{inputs}"
        return

    wire_dimension = int(theory.wire_dimension)
    Frame = tuple[int, Iterator[tuple[dict[str, Any], int]]]
    stack: list[Frame] = [
        (
            depth,
            _operation_alphabet(
                wire_dimension,
                inputs,
                operation_records,
                deadline,
            ),
        )
    ]
    word: list[dict[str, Any]] = []
    while stack:
        deadline.check()
        remaining, operations = stack[-1]
        try:
            operation, next_wires = next(operations)
        except StopIteration:
            stack.pop()
            if stack:
                word.pop()
            continue
        if not _matrix_budget_allows(wire_dimension, next_wires, max_matrix_entries):
            continue
        word.append(operation)
        if remaining > 1:
            stack.append(
                (
                    remaining - 1,
                    _operation_alphabet(
                        wire_dimension,
                        next_wires,
                        operation_records,
                        deadline,
                    ),
                )
            )
            continue
        if next_wires != outputs:
            word.pop()
            continue
        data = {"wires": inputs, "ops": list(word)}
        with suppress(TypeError, ValueError):
            yield Circuit.from_json(data, theory.signature, theory.macros), _word_label(word)
        word.pop()


def _word_label(word: list[dict[str, Any]]) -> str:
    labels: list[str] = []
    for operation in word:
        if "gen" in operation:
            labels.append(f"{operation['gen']}@{operation['on']}")
        elif "macro" in operation:
            labels.append(f"{operation['macro']}@{operation['on']}")
        else:
            labels.append(f"perm{operation['perm']}")
    return "; ".join(labels)


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_depth: int = 3,
    max_substitution_matrix_entries: int = DEFAULT_MAX_MATRIX_ENTRIES,
    context: Any = None,
) -> Iterator[CandidateModel]:
    depth = bound
    if depth < 0 or depth > max_depth:
        return
    wire_dimension = theory.wire_dimension
    checked = [*retained_equations(theory, target, context), target]
    generators = sorted(theory.signature.values(), key=lambda item: item.name)
    if not generators:
        raise NotApplicable("the theory has no generators")

    usable: set[str] = set()
    for generator in generators:
        deadline.check()
        raw = generator.matrix
        if raw is None:
            continue
        source, target_arity = generator.type
        matrix = np.asarray(raw, dtype=complex)
        if _matrix_shape_matches(matrix.shape, wire_dimension, source, target_arity) and np.all(
            np.isfinite(matrix)
        ):
            usable.add(generator.name)
    usable_generators = frozenset(usable)

    operations: list[_Operation] = [
        ("gen", generator.name, *generator.type)
        for generator in generators
        if generator.name in usable_generators
    ]
    for name, body in sorted(theory.macros.items()):
        deadline.check()
        analysis = circuit_analysis(body, context)
        if {
            name for name, _ in analysis.occurrences
        } <= usable_generators and _matrix_budget_allows(
            wire_dimension,
            analysis.maximum_wires,
            max_substitution_matrix_entries,
        ):
            operations.append(("macro", name, *body.type))
    operation_records = tuple(operations)

    required: set[str] = set()
    for equation in checked:
        deadline.check()
        for term in (equation.lhs, equation.rhs):
            analysis = circuit_analysis(term, context)
            if not _matrix_budget_allows(
                wire_dimension,
                analysis.maximum_wires,
                max_substitution_matrix_entries,
            ):
                return
            required.update(name for name, _ in analysis.occurrences)
    target_names = {
        name
        for term in (target.lhs, target.rhs)
        for name, _ in circuit_analysis(term, context).occurrences
    }

    for generator in generators:
        source, target_arity = generator.type
        if generator.name not in target_names:
            continue
        # Every other required generator needs a usable matrix for evaluation.
        if not (required - {generator.name}) <= usable_generators:
            continue
        if not _matrix_budget_allows(
            wire_dimension, source, max_substitution_matrix_entries
        ) or not _matrix_budget_allows(
            wire_dimension, target_arity, max_substitution_matrix_entries
        ):
            continue
        seen_matrices: set[bytes] = set()
        for replacement, label in _replacement_terms(
            theory,
            source,
            target_arity,
            depth,
            deadline,
            operation_records,
            max_substitution_matrix_entries,
        ):
            deadline.check()
            replacement_matrix = evaluate_matrix(
                replacement,
                theory.signature,
                wire_dimension,
            )
            replacement_matrix = np.asarray(replacement_matrix, dtype=complex)
            if not _matrix_shape_matches(
                replacement_matrix.shape,
                wire_dimension,
                source,
                target_arity,
            ) or not np.all(np.isfinite(replacement_matrix)):
                continue
            fingerprint = _fingerprint(replacement_matrix)
            if fingerprint in seen_matrices:
                continue
            seen_matrices.add(fingerprint)
            semantics = MatrixSemantics(
                theory.signature,
                wire_dimension,
                overrides={generator.name: replacement_matrix},
            )

            yield CandidateModel(
                description=(f"numerical projective substitution {generator.name} := {label}"),
                parameters={
                    "generator": generator.name,
                    "replacement": label,
                    "depth": depth,
                    "matrix_tolerance": PROJECTIVE_TOLERANCE,
                },
                evaluate=semantics.evaluate,
                equal=lambda left, right: projectively_equal(left, right, PROJECTIVE_TOLERANCE),
                key=(generator.name, fingerprint),
                certified=False,
            )
