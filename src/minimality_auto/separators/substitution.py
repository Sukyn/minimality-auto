from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress
from itertools import permutations
from math import isqrt
from typing import Any

import numpy as np

from ..core import (
    Circuit,
    _wire_size_matches,
    analyze_circuit,
    primitive_occurrences,
)
from ..matrix import evaluate_matrix
from ..search import CandidateModel, Deadline, NotApplicable
from .common import projectively_equal, retained_equations

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


def _arity(item: Any) -> tuple[int, int]:
    return int(item.inputs), int(item.outputs)


def _placements(
    wires: int,
    width: int,
    deadline: Deadline,
) -> Iterator[tuple[int, ...]]:
    if width < 0 or width > wires:
        return
    for placement in permutations(range(wires), width):
        deadline.check()
        yield placement


def _operation_alphabet(
    wire_dimension: int,
    wires: int,
    operations: tuple[_Operation, ...],
    deadline: Deadline,
) -> Iterator[tuple[dict[str, Any], int]]:
    for kind, name, source, target in operations:
        deadline.check()
        if source > wires:
            continue
        for placement in _placements(wires, source, deadline):
            deadline.check()
            yield (
                {kind: name, "on": list(placement)},
                wires - source + target,
            )
            if wire_dimension == 1:
                break
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
    if max_matrix_entries < 1:
        return False
    limit = isqrt(max_matrix_entries)
    if wire_dimension == 1:
        return limit >= 1
    return wires <= limit.bit_length() and wire_dimension**wires <= limit


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
        if not _matrix_budget_allows(
            wire_dimension, next_wires, max_matrix_entries
        ):
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
    wire_dimension = int(theory.wire_dimension)
    if wire_dimension < 1:
        raise NotApplicable("wire dimension must be positive")
    checked = [*retained_equations(theory, target, context), target]
    generators = sorted(theory.signature.values(), key=lambda item: item.name)
    if not generators:
        raise NotApplicable("the theory has no generators")

    usable: set[str] = set()
    for generator in generators:
        deadline.check()
        raw = getattr(generator, "matrix", None)
        if raw is None:
            continue
        source, target_arity = _arity(generator)
        try:
            matrix = np.asarray(raw, dtype=complex)
        except (TypeError, ValueError):
            continue
        if (
            matrix.ndim == 2
            and _wire_size_matches(matrix.shape[0], wire_dimension, target_arity)
            and _wire_size_matches(matrix.shape[1], wire_dimension, source)
            and np.all(np.isfinite(matrix))
        ):
            usable.add(generator.name)
    usable_generators = frozenset(usable)

    operations: list[_Operation] = [
        ("gen", generator.name, *_arity(generator))
        for generator in sorted(
            theory.signature.values(), key=lambda item: item.name
        )
        if generator.name in usable_generators
    ]
    for name, body in sorted(theory.macros.items()):
        deadline.check()
        maximum_wires = analyze_circuit(body).maximum_wires
        if (
            set(primitive_occurrences(body))
            <= usable_generators
            and _matrix_budget_allows(
                wire_dimension,
                maximum_wires,
                max_substitution_matrix_entries,
            )
        ):
            operations.append(("macro", name, *_arity(body)))
    operation_records = tuple(operations)

    required: set[str] = set()
    for equation in checked:
        deadline.check()
        if any(
            not _matrix_budget_allows(
                wire_dimension,
                analyze_circuit(term).maximum_wires,
                max_substitution_matrix_entries,
            )
            for term in (equation.lhs, equation.rhs)
        ):
            return
        required.update(primitive_occurrences(equation.lhs))
        required.update(primitive_occurrences(equation.rhs))
    target_names = set(primitive_occurrences(target.lhs))
    target_names.update(primitive_occurrences(target.rhs))

    for generator in generators:
        source, target_arity = _arity(generator)
        if generator.name not in target_names:
            continue
        # A single override can repair at most this generator's missing matrix.
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
            try:
                replacement_matrix = evaluate_matrix(
                    replacement,
                    theory.signature,
                    wire_dimension,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            replacement_matrix = np.asarray(replacement_matrix, dtype=complex)
            expected = (wire_dimension**target_arity, wire_dimension**source)
            if replacement_matrix.shape != expected or not np.all(
                np.isfinite(replacement_matrix)
            ):
                continue
            fingerprint = _fingerprint(replacement_matrix)
            if fingerprint in seen_matrices:
                continue
            seen_matrices.add(fingerprint)
            override = {generator.name: replacement_matrix}

            def evaluate(term: Any, override: dict[str, np.ndarray] = override) -> np.ndarray:
                return evaluate_matrix(
                    term,
                    theory.signature,
                    wire_dimension,
                    overrides=override,
                )

            yield CandidateModel(
                description=(
                    f"numerical projective substitution "
                    f"{generator.name} := {label}"
                ),
                parameters={
                    "generator": generator.name,
                    "replacement": label,
                    "depth": depth,
                    "matrix_tolerance": PROJECTIVE_TOLERANCE,
                },
                evaluator=evaluate,
                equality=lambda left, right: projectively_equal(
                    left, right, PROJECTIVE_TOLERANCE
                ),
                key=(generator.name, fingerprint),
                certified=False,
            )
