from __future__ import annotations

from math import isqrt
from typing import Any, Iterator

import numpy as np

from ..core import Circuit, ValidationError, evaluate_matrix, primitive_occurrences
from ..search import CandidateModel, Deadline, NotApplicable, relevant_equations


DEFAULT_MAX_MATRIX_ENTRIES = 1_000_000


def _projectively_equal(left: np.ndarray, right: np.ndarray, tolerance: float = 1e-8) -> bool:
    left = np.asarray(left, dtype=complex)
    right = np.asarray(right, dtype=complex)
    if left.shape != right.shape:
        return False
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
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

    # Choose the pivot symmetrically so comparison is order-independent.
    joint_scale = np.maximum(np.abs(left), np.abs(right))
    index = np.unravel_index(int(np.argmax(joint_scale)), right.shape)
    if abs(left[index]) <= tolerance or abs(right[index]) <= tolerance:
        return False
    phase = left[index] / right[index]
    phase /= abs(phase)
    residual = float(np.max(np.abs(left - phase * right)))
    return residual <= tolerance


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


def _generator_catalog(
    theory: Any,
    equations: list[Any],
) -> list[Any]:
    """Return consistently typed primitives, including inline generators.

    ``Circuit.from_json`` permits an explicitly typed generator which is not
    declared in the top-level signature.  Matrix evaluation can still assign
    such a primitive through an override, so it is a legitimate substitution
    target.  A name used at two different types cannot receive one matrix and
    is therefore omitted conservatively.
    """
    catalog = dict(theory.signature.items())
    conflicts: set[str] = set()
    visited_macros: set[str] = set()

    def visit(term: Circuit, active: list[str]) -> None:
        if term.kind == "gen":
            name = term.name or ""
            existing = catalog.get(name)
            if existing is None:
                catalog[name] = term
            elif _arity(existing) != term.type:
                conflicts.add(name)
            return
        if term.kind == "macro":
            name = term.name or ""
            if name in active or name not in theory.macros:
                raise ValidationError(f"invalid recursive macro {name!r}")
            definition = theory.macros[name]
            body = getattr(definition, "body", definition)
            if body.type != term.type:
                raise ValidationError(f"macro {name!r} has inconsistent type")
            if name in visited_macros:
                return
            active.append(name)
            visit(body, active)
            active.pop()
            visited_macros.add(name)
            return
        for part in term.parts:
            visit(part, active)

    for equation in equations:
        visit(equation.lhs, [])
        visit(equation.rhs, [])
    return [catalog[name] for name in sorted(catalog) if name not in conflicts]


def _placements(
    wires: int,
    width: int,
    deadline: Deadline,
) -> Iterator[tuple[int, ...]]:
    """Generate ordered wire selections without materializing the ambient set."""
    if width < 0 or width > wires:
        return
    if width == 0:
        yield ()
        return
    chosen: list[int] = []
    used: set[int] = set()
    next_values = [0]
    while next_values:
        deadline.check()
        candidate = next_values[-1]
        scanned = 0
        while candidate < wires and candidate in used:
            candidate += 1
            scanned += 1
            if scanned % 1024 == 0:
                deadline.check()
        if candidate >= wires:
            next_values.pop()
            if chosen:
                used.remove(chosen.pop())
            continue
        next_values[-1] = candidate + 1
        chosen.append(candidate)
        used.add(candidate)
        if len(chosen) == width:
            yield tuple(chosen)
            used.remove(chosen.pop())
        else:
            next_values.append(0)


def _operation_alphabet(
    theory: Any,
    wires: int,
    usable_generators: frozenset[str],
    macro_wire_counts: dict[str, int],
    max_matrix_entries: int,
    deadline: Deadline,
) -> Iterator[tuple[dict[str, Any], int]]:
    wire_dimension = int(theory.wire_dimension)
    for generator in sorted(theory.signature.values(), key=lambda item: item.name):
        deadline.check()
        source, target = _arity(generator)
        if generator.name not in usable_generators or source > wires:
            continue
        for placement in _placements(wires, source, deadline):
            deadline.check()
            yield (
                {"gen": generator.name, "on": list(placement)},
                wires - source + target,
            )
            if wire_dimension == 1:
                break
    for macro in sorted(theory.macros.values(), key=lambda item: item.name):
        deadline.check()
        try:
            source, target = _arity(macro)
            name = macro.name
        except (AttributeError, TypeError):
            continue
        body = getattr(macro, "body", macro)
        required = set(primitive_occurrences(body, theory.macros))
        if (
            source > wires
            or not required <= usable_generators
            or not _matrix_budget_allows(
                int(theory.wire_dimension),
                macro_wire_counts[name],
                max_matrix_entries,
            )
        ):
            continue
        for placement in _placements(wires, source, deadline):
            deadline.check()
            yield {"macro": name, "on": list(placement)}, wires - source + target
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


def _dimension_matches(size: int, wire_dimension: int, wires: int) -> bool:
    if wire_dimension == 1:
        return size == 1
    return wires <= size.bit_length() and wire_dimension**wires == size


def _maximum_wires(
    term: Circuit,
    macros: Any,
    cache: dict[str, int],
    active: list[str],
) -> int:
    if term.kind == "macro":
        name = term.name or ""
        if name in active or name not in macros:
            raise ValidationError(f"invalid recursive macro {name!r}")
        definition = macros[name]
        body = getattr(definition, "body", definition)
        if body.type != term.type:
            raise ValidationError(f"macro {name!r} has inconsistent type")
        if name in cache:
            return cache[name]
        active.append(name)
        cache[name] = _maximum_wires(body, macros, cache, active)
        active.pop()
        return cache[name]
    return max(
        term.inputs,
        term.outputs,
        *(_maximum_wires(part, macros, cache, active) for part in term.parts),
    )


def _replacement_terms(
    theory: Any,
    inputs: int,
    outputs: int,
    depth: int,
    deadline: Deadline,
    usable_generators: frozenset[str],
    macro_wire_counts: dict[str, int],
    max_matrix_entries: int,
) -> Iterator[tuple[Any, str]]:
    if depth == 0:
        if inputs == outputs:
            yield (
                Circuit.from_json({"id": inputs}, theory.signature, theory.macros),
                f"id_{inputs}",
            )
        return

    Frame = tuple[int, int, Iterator[tuple[dict[str, Any], int]]]
    stack: list[Frame] = [
        (
            inputs,
            depth,
            _operation_alphabet(
                theory,
                inputs,
                usable_generators,
                macro_wire_counts,
                max_matrix_entries,
                deadline,
            ),
        )
    ]
    word: list[dict[str, Any]] = []
    while stack:
        deadline.check()
        wires, remaining, operations = stack[-1]
        try:
            operation, next_wires = next(operations)
        except StopIteration:
            stack.pop()
            if stack:
                word.pop()
            continue
        if not _matrix_budget_allows(
            int(theory.wire_dimension), next_wires, max_matrix_entries
        ):
            continue
        word.append(operation)
        if remaining > 1:
            stack.append(
                (
                    next_wires,
                    remaining - 1,
                    _operation_alphabet(
                        theory,
                        next_wires,
                        usable_generators,
                        macro_wire_counts,
                        max_matrix_entries,
                        deadline,
                    ),
                )
            )
            continue
        if next_wires != outputs:
            word.pop()
            continue
        data = {"wires": inputs, "ops": list(word)}
        try:
            yield Circuit.from_json(data, theory.signature, theory.macros), _word_label(word)
        except (TypeError, ValueError, ValidationError):
            pass
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
    **_: Any,
) -> Iterator[CandidateModel]:
    depth = bound
    if depth < 0 or depth > max_depth:
        return
    wire_dimension = int(theory.wire_dimension)
    if wire_dimension < 1:
        raise NotApplicable("wire dimension must be positive")
    checked = [*relevant_equations(theory, target), target]
    generators = _generator_catalog(theory, checked)
    if not generators:
        raise NotApplicable("the checked theory has no consistently typed generators")

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
            and _dimension_matches(matrix.shape[0], wire_dimension, target_arity)
            and _dimension_matches(matrix.shape[1], wire_dimension, source)
            and np.all(np.isfinite(matrix))
        ):
            usable.add(generator.name)
    usable_generators = frozenset(usable)

    macro_wire_counts: dict[str, int] = {}
    for macro in theory.macros.values():
        deadline.check()
        body = getattr(macro, "body", macro)
        macro_wire_counts[macro.name] = _maximum_wires(
            body, theory.macros, macro_wire_counts, [macro.name]
        )

    required: set[str] = set()
    for equation in checked:
        deadline.check()
        if any(
            not _matrix_budget_allows(
                wire_dimension,
                _maximum_wires(term, theory.macros, macro_wire_counts, []),
                max_substitution_matrix_entries,
            )
            for term in (equation.lhs, equation.rhs)
        ):
            return
        required.update(primitive_occurrences(equation.lhs, theory.macros))
        required.update(primitive_occurrences(equation.rhs, theory.macros))
    target_names = set(primitive_occurrences(target.lhs, theory.macros))
    target_names.update(primitive_occurrences(target.rhs, theory.macros))

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
            usable_generators,
            macro_wire_counts,
            max_substitution_matrix_entries,
        ):
            deadline.check()
            try:
                replacement_matrix = evaluate_matrix(
                    replacement,
                    theory.signature,
                    theory.macros,
                    wire_dimension,
                )
            except (TypeError, ValueError, ValidationError, OverflowError):
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
                    theory.macros,
                    wire_dimension,
                    overrides=override,
                )

            yield CandidateModel(
                kind="substitution",
                description=f"projective substitution {generator.name} := {label}",
                parameters={"generator": generator.name, "replacement": label, "depth": depth},
                evaluator=evaluate,
                equality=_projectively_equal,
                key=(generator.name, fingerprint),
            )
