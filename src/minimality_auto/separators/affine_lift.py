"""Exact finite affine lifts of supplied basis-permutation semantics.

At the target arity, each box keeps its supplied permutation as a linear action
on ``F_p^N`` and receives an unknown translation vector.  The retained PROP
equations become a homogeneous linear system; a kernel vector that breaks the
target is a finite, exact fixed-arity invariant for an endomorphic PROP.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ..core import Circuit
from ..matrix import basis_digits, basis_permutation, basis_wire_permutation
from ..search import CandidateModel, Deadline, NotApplicable
from . import fixed_arity
from .common import bounded_power, is_prime, require_endomorphic_theory
from .fixed_arity import Token, Word

Permutation = tuple[int, ...]
Vector = tuple[int, ...]
Rows = tuple[Vector, ...]
MAX_PRIME = 5
MAX_DIMENSION = 128
MAX_SYSTEM_ENTRIES = 2_000_000


def _matrix_permutation(matrix: Any, deadline: Deadline) -> Permutation:
    result = basis_permutation(matrix, deadline.check)
    if result is None:
        raise NotApplicable(
            "affine lifts need exact 0/1 basis-permutation matrices"
        )
    return result


def _base_assignment(
    theory: Any,
    presentation: fixed_arity.Presentation,
    deadline: Deadline,
) -> tuple[dict[Token, Permutation], dict[str, Permutation]]:
    dimension = theory.wire_dimension
    arity = presentation.arity
    atoms: dict[Token, Permutation] = {}
    native: dict[str, Permutation] = {}
    for name, width in presentation.generator_specs:
        deadline.check()
        matrix = theory.signature[name].matrix
        if matrix is None:
            raise NotApplicable(f"affine lift needs a matrix for {name!r}")
        value = _matrix_permutation(matrix, deadline)
        expected = dimension**width
        if len(value) != expected:
            raise NotApplicable(f"matrix for {name!r} has the wrong dimension")
        native[name] = value
        padding = dimension ** (arity - width)
        atoms[fixed_arity.generator(name)] = tuple(
            value[column // padding] * padding + column % padding
            for column in range(dimension**arity)
        )
    for index in range(arity - 1):
        order = list(range(arity))
        order[index], order[index + 1] = order[index + 1], order[index]
        atoms[fixed_arity.swap(index)] = basis_wire_permutation(
            tuple(order), dimension, deadline.check
        )
    return atoms, native


def _word_coefficients(
    word: Word,
    assignment: Mapping[Token, Permutation],
    offsets: Mapping[Token, int],
    columns: int,
    prime: int,
    deadline: Deadline,
) -> tuple[Permutation, Rows]:
    dimension = len(next(iter(assignment.values())))
    base = tuple(range(dimension))
    rows = [[0] * columns for _ in range(dimension)]
    for step, token in enumerate(word):
        if step % 64 == 0:
            deadline.check()
        action = assignment[token]
        moved = [[0] * columns for _ in range(dimension)]
        for source, target in enumerate(action):
            moved[target] = rows[source].copy()
        offset = offsets[token]
        for coordinate in range(dimension):
            moved[coordinate][offset + coordinate] = (
                moved[coordinate][offset + coordinate] + 1
            ) % prime
        rows = moved
        base = tuple(action[value] for value in base)
    return base, tuple(tuple(row) for row in rows)


def _difference(left: Rows, right: Rows, prime: int) -> Rows:
    return tuple(
        tuple((first - second) % prime for first, second in zip(a, b, strict=True))
        for a, b in zip(left, right, strict=True)
    )


def _dot(left: Sequence[int], right: Sequence[int], prime: int) -> int:
    return sum(a * b for a, b in zip(left, right, strict=True)) % prime


def _separating_vector(
    source: Sequence[Vector],
    targets: Sequence[Vector],
    columns: int,
    prime: int,
    deadline: Deadline,
) -> Vector | None:
    """Find a vector killed by every source row but detected by a target row."""

    rows = [list(row) for row in source if any(row)]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(columns):
        if column % 16 == 0:
            deadline.check()
        found = next(
            (
                row
                for row in range(pivot_row, len(rows))
                if rows[row][column] % prime
            ),
            None,
        )
        if found is None:
            continue
        rows[pivot_row], rows[found] = rows[found], rows[pivot_row]
        scale = pow(rows[pivot_row][column] % prime, -1, prime)
        rows[pivot_row] = [(entry * scale) % prime for entry in rows[pivot_row]]
        for row, values in enumerate(rows):
            if row == pivot_row:
                continue
            scale = values[column] % prime
            if scale:
                rows[row] = [
                    (entry - scale * pivot) % prime
                    for entry, pivot in zip(values, rows[pivot_row], strict=True)
                ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break

    pivot_set = frozenset(pivots)
    for free in range(columns):
        if free % 16 == 0:
            deadline.check()
        if free in pivot_set:
            continue
        vector = [0] * columns
        vector[free] = 1
        for row, pivot in zip(rows[: len(pivots)], pivots, strict=True):
            vector[pivot] = -row[free] % prime
        result = tuple(vector)
        if any(_dot(target, result, prime) for target in targets):
            return result
    return None


def _evaluate_word(
    word: Word,
    assignment: Mapping[Token, Permutation],
    translations: Mapping[Token, Vector],
    prime: int,
    deadline: Deadline,
) -> tuple[Permutation, Vector]:
    """Evaluate a resolved affine word independently of the symbolic solver."""

    dimension = len(next(iter(assignment.values())))
    base = tuple(range(dimension))
    translation = [0] * dimension
    for step, token in enumerate(word):
        if step % 64 == 0:
            deadline.check()
        action = assignment[token]
        moved = [0] * dimension
        for source, target in enumerate(action):
            moved[target] = translation[source]
        own = translations[token]
        translation = [
            (value + added) % prime
            for value, added in zip(moved, own, strict=True)
        ]
        base = tuple(action[value] for value in base)
    return base, tuple(translation)


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_affine_lift_prime: int = MAX_PRIME,
    max_affine_lift_dimension: int = MAX_DIMENSION,
    context: Any = None,
) -> Iterator[CandidateModel]:
    if (
        bound < 2
        or bound > max_affine_lift_prime
        or not is_prime(bound, deadline)
        or max_affine_lift_dimension < 1
    ):
        return
    deadline.check()
    require_endomorphic_theory(theory, deadline, context)
    arity = target.arity
    wire_dimension = theory.wire_dimension
    dimension = bounded_power(
        wire_dimension,
        arity,
        max_affine_lift_dimension,
    )
    if arity > max_affine_lift_dimension or dimension is None:
        raise NotApplicable(
            f"affine-lift dimension exceeds {max_affine_lift_dimension}"
        )
    presentation = fixed_arity.compile_presentation(
        theory,
        target,
        deadline,
        context,
    )

    bases, native = _base_assignment(theory, presentation, deadline)
    # A canonical variable order makes the chosen nullspace witness independent
    # of generator declaration order and usually keeps it sparse.
    variables = tuple(sorted(presentation.variables))
    if not variables:
        return
    columns = len(variables) * dimension
    if (
        columns > max_affine_lift_dimension**2
        or len(presentation.required_constraints) * dimension * columns
        > MAX_SYSTEM_ENTRIES
    ):
        raise NotApplicable("affine-lift linear system exceeds its safe dense bound")
    offsets = {token: index * dimension for index, token in enumerate(variables)}
    cache: dict[Word, tuple[Permutation, Rows]] = {}

    def coefficients(word: Word) -> tuple[Permutation, Rows]:
        value = cache.get(word)
        if value is None:
            value = _word_coefficients(
                word,
                bases,
                offsets,
                columns,
                bound,
                deadline,
            )
            cache[word] = value
        return value

    equations: list[Vector] = []
    for constraint in presentation.required_constraints:
        left_base, left_rows = coefficients(constraint.left)
        right_base, right_rows = coefficients(constraint.right)
        if left_base != right_base:
            return
        equations.extend(_difference(left_rows, right_rows, bound))

    left_base, left_rows = coefficients(presentation.target_pair[0])
    right_base, right_rows = coefficients(presentation.target_pair[1])
    target_rows = _difference(left_rows, right_rows, bound)
    if left_base != right_base:
        witness = (0,) * columns
    else:
        witness = _separating_vector(
            equations,
            target_rows,
            columns,
            bound,
            deadline,
        )
        if witness is None:
            return

    translations = {
        token: witness[offsets[token] : offsets[token] + dimension]
        for token in variables
    }
    cache.clear()
    equations.clear()
    sparse_generators: dict[str, list[list[Any]]] = {}
    sparse_swaps: dict[str, list[list[Any]]] = {}
    for token in variables:
        entries = [
            [list(basis_digits(index, presentation.arity, theory.wire_dimension)), value]
            for index, value in enumerate(translations[token])
            if value
        ]
        if not entries:
            continue
        if token[0] == "generator":
            sparse_generators[str(token[1])] = entries
        else:
            sparse_swaps[fixed_arity.display_name(token)] = entries
    swaps = {
        fixed_arity.display_name(token): list(bases[token])
        for token in variables
        if token[0] == "swap"
    }

    def evaluate(term: Circuit) -> tuple[Permutation, Vector]:
        if term.inputs != term.outputs or term.inputs > presentation.arity:
            raise NotApplicable(
                "affine lift supports endomorphisms through arity "
                f"{presentation.arity}"
            )
        word = fixed_arity.compile_word(
            term,
            presentation.signature,
            presentation.arity,
            deadline=deadline,
        )
        return _evaluate_word(word, bases, translations, bound, deadline)

    def render_entry(entry: list[Any]) -> str:
        coordinates, value = entry
        label = str(tuple(coordinates))
        return label if value == 1 else f"{label}:{value}"

    def render_group(label: str, values: Mapping[str, list[list[Any]]]) -> str:
        rendered_values = ", ".join(
            f"{name}={{{', '.join(map(render_entry, entries))}}}"
            for name, entries in values.items()
        )
        return f"{label} {{{rendered_values}}}"

    groups = []
    if sparse_generators:
        groups.append(render_group("generators", sparse_generators))
    if sparse_swaps:
        groups.append(render_group("structural swaps", sparse_swaps))
    rendered = "; ".join(groups) or "none"
    yield CandidateModel(
        description=(
            f"exact F_{bound}^{dimension} affine lift of the supplied basis "
            f"permutations; nonzero translations {{{rendered}}}"
        ),
        parameters={
            "arity": presentation.arity,
            "prime": bound,
            "module_dimension": dimension,
            "base_interpretation": {
                "generators": {name: list(value) for name, value in native.items()},
                "structural_swaps": swaps,
            },
            "translations": {
                "generators": sparse_generators,
                "structural_swaps": sparse_swaps,
            },
        },
        evaluate=evaluate,
    )
