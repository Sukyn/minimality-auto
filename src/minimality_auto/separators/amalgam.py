"""Search finite factors and separate equations in their free amalgam.

Generator matrices supply sparsity and scalar templates. The search enumerates
small prime-field assignments and factor partitions, derives shared subgroups
and conjugation actions, and checks the retained equations exactly.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from itertools import combinations, product
from typing import Any

from .. import presentation as fixed_arity
from ..algebra import finite_field
from ..algebra.amalgam import (
    _Amalgam,
    _close_shared_subgroup,
    _cosets,
    _DGroup,
)
from ..presentation import Token, Word
from ..search_types import CandidateModel, Deadline, NotApplicable
from ..syntax import Circuit
from .common import bounded_power, is_prime, require_endomorphic_theory

MAX_MATRIX_DIMENSION = 32


def _word_matrix(
    word: Word,
    atoms: Mapping[Token, finite_field.Mat],
    size: int,
    prime: int,
    deadline: Deadline,
) -> finite_field.Mat:
    result = finite_field.identity(size)
    for index, token in enumerate(word):
        result = finite_field.multiply(atoms[token], result, size, prime)
        if index % 128 == 0:
            deadline.check()
    return result


def _cancel(left: Word, right: Word, deadline: Deadline | None = None) -> tuple[Word, Word]:
    start = 0
    while start < min(len(left), len(right)) and left[start] == right[start]:
        start += 1
        if deadline is not None and start % 1024 == 0:
            deadline.check()
    left_end, right_end = len(left), len(right)
    while left_end > start and right_end > start and left[left_end - 1] == right[right_end - 1]:
        left_end -= 1
        right_end -= 1
        if deadline is not None and (len(left) - left_end) % 1024 == 0:
            deadline.check()
    return left[start:left_end], right[start:right_end]


def _blocks(
    word: Word,
    bridges: frozenset[Token],
    deadline: Deadline | None = None,
) -> tuple[Word, ...]:
    result: list[Word] = []
    current: list[Token] = []
    for index, token in enumerate(word):
        if token in bridges:
            if current:
                result.append(tuple(current))
                current = []
        else:
            current.append(token)
        if deadline is not None and index % 1024 == 0:
            deadline.check()
    if current:
        result.append(tuple(current))
    return tuple(result)


def _rows(matrix: finite_field.Mat, size: int) -> list[list[int]]:
    return [list(matrix[row * size : (row + 1) * size]) for row in range(size)]


def _candidate(
    theory: Any,
    presentation: fixed_arity.Presentation,
    atoms: Mapping[Token, finite_field.Mat],
    native: Mapping[str, finite_field.Mat],
    bridges: frozenset[Token],
    prime: int,
    assignment: Sequence[int],
    variables: Sequence[complex],
    max_order: int,
    deadline: Deadline,
) -> CandidateModel | None:
    arity = presentation.arity
    signature = presentation.signature
    target_words = presentation.target_pair
    size = theory.wire_dimension**arity
    base_atom_tokens = tuple(sorted(set(atoms) - bridges))
    bridge_tokens = tuple(sorted(bridges))
    base = finite_field.MatrixGroup(
        [atoms[token] for token in base_atom_tokens], size, prime, max_order, deadline
    )
    quotient = finite_field.MatrixGroup(
        [atoms[token] for token in bridge_tokens], size, prime, max_order, deadline
    )
    base_tokens = {token: base.element(atoms[token]) for token in base_atom_tokens}
    quotient_tokens = {token: quotient.element(atoms[token]) for token in bridge_tokens}

    seeds: list[int] = []
    for relation in presentation.required_constraints:
        left, right = _cancel(relation.left, relation.right, deadline)
        if not (bridges & {*left, *right}):
            continue
        for block in (
            *_blocks(left, bridges, deadline),
            *_blocks(right, bridges, deadline),
        ):
            value = 0
            for index, token in enumerate(block):
                value = base.multiply(value, base_tokens[token])
                if index % 128 == 0:
                    deadline.check()
            seeds.append(value)

    shared, action = _close_shared_subgroup(base, quotient, seeds, max_order, deadline)
    second = _DGroup(base, quotient, action)
    decomposition, representatives = _cosets(base, shared, deadline)
    tokens: dict[Token, tuple[str, int]] = {
        token: ("C", value) for token, value in base_tokens.items()
    }
    tokens.update(
        {token: ("D", second.element(0, value)) for token, value in quotient_tokens.items()}
    )
    amalgam = _Amalgam(
        base,
        second,
        decomposition,
        {value: index for index, value in enumerate(representatives)},
        {value: index for index, value in enumerate(sorted(shared))},
        tokens,
    )
    if any(
        amalgam.evaluate_word(relation.left, deadline)
        != amalgam.evaluate_word(relation.right, deadline)
        for relation in presentation.required_constraints
    ):
        raise finite_field.Reject("a retained equation fails after amalgam normalization")
    if amalgam.evaluate_word(target_words[0], deadline) == amalgam.evaluate_word(
        target_words[1], deadline
    ):
        return None

    def evaluate(term: Circuit) -> tuple[Any, ...]:
        return amalgam.evaluate_word(
            presentation.compile(term, deadline),
            deadline,
        )

    interpretation = {}
    for name in signature:
        bridge = fixed_arity.generator(name) in bridges
        interpretation[name] = {
            "factor": "D" if bridge else "C",
            "quotient_seed_matrix" if bridge else "matrix": _rows(
                native[name], theory.wire_dimension ** signature[name]
            ),
        }
    structural = {}
    for index in range(arity - 1):
        order = list(range(arity))
        order[index], order[index + 1] = order[index + 1], order[index]
        structural[fixed_arity.display_name(fixed_arity.swap(index))] = order
    bridge_names = [str(token[1]) for token in bridge_tokens]

    def variable_label(index: int) -> str:
        value = variables[index]
        if abs(value.imag) <= 1e-10:
            rendered = f"{value.real:.12g}"
        else:
            rendered = f"({value.real:.12g}{value.imag:+.12g}i)"
        return f"x{index}~{rendered}"

    return CandidateModel(
        description=(
            f"discovered F_{prime} amalgam with bridge "
            f"{{{', '.join(bridge_names)}}}: "
            f"|C|={base.order}, |A|={len(shared)}, "
            f"|D|={len(shared) * quotient.order}"
        ),
        parameters={
            "arity": arity,
            "prime": prime,
            "scalar_assignment": {
                variable_label(index): value for index, value in enumerate(assignment)
            },
            "bridge_generators": bridge_names,
            "factor_orders": {
                "C": base.order,
                "A": len(shared),
                "Q": quotient.order,
                "D": len(shared) * quotient.order,
            },
            "interpretation": interpretation,
            "structural_swaps": structural,
        },
        evaluate=evaluate,
    )


def _bridge_subsets(
    generators: Sequence[Token], maximum: int, deadline: Deadline
) -> Iterator[frozenset[Token]]:
    checked = 0
    for count in range(1, min(maximum, len(generators)) + 1):
        for values in combinations(generators, count):
            checked += 1
            if checked % 128 == 0:
                deadline.check()
            yield frozenset(values)


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_amalgam_prime: int = 11,
    max_amalgam_order: int = 4096,
    max_amalgam_bridge_generators: int = 1,
    max_amalgam_scalars: int = 3,
    max_amalgam_matrix_dimension: int = MAX_MATRIX_DIMENSION,
    context: Any = None,
) -> Iterator[CandidateModel]:
    if bound != 1:
        return
    if max_amalgam_prime < 2 or max_amalgam_order < 1 or max_amalgam_bridge_generators < 1:
        return
    if max_amalgam_scalars < 0:
        raise NotApplicable("the amalgam scalar bound must be non-negative")
    require_endomorphic_theory(theory, deadline, context)
    arity = target.arity
    if target.lhs.type != (arity, arity):
        return
    size = bounded_power(theory.wire_dimension, arity, max_amalgam_matrix_dimension)
    if size is None:
        raise NotApplicable(f"amalgam matrix dimension exceeds {max_amalgam_matrix_dimension}")
    presentation = fixed_arity.compile_presentation(
        theory,
        target,
        deadline,
        context,
    )
    templates = finite_field.templates(
        theory,
        presentation.signature,
        presentation.active_names,
        max_amalgam_scalars,
        deadline,
    )
    ordered = sorted(
        (fixed_arity.generator(name) for name in presentation.active_names),
        key=lambda token: (
            token not in presentation.target_generators,
            presentation.signature[str(token[1])] != arity,
            str(token[1]),
        ),
    )
    for prime in range(2, max_amalgam_prime + 1):
        deadline.check()
        if not is_prime(prime, deadline):
            continue
        for assignment in product(range(prime), repeat=len(templates.variables)):
            deadline.check()
            try:
                atoms, native = finite_field.instantiate(
                    templates,
                    presentation.signature,
                    assignment,
                    arity,
                    theory.wire_dimension,
                    prime,
                )
                if any(
                    _word_matrix(relation.left, atoms, size, prime, deadline)
                    != _word_matrix(relation.right, atoms, size, prime, deadline)
                    for relation in presentation.required_constraints
                ):
                    continue
                for bridges in _bridge_subsets(ordered, max_amalgam_bridge_generators, deadline):
                    deadline.check()
                    try:
                        model = _candidate(
                            theory,
                            presentation,
                            atoms,
                            native,
                            bridges,
                            prime,
                            assignment,
                            templates.variables,
                            max_amalgam_order,
                            deadline,
                        )
                    except finite_field.Reject:
                        continue
                    if model is not None:
                        yield model
            except finite_field.Reject:
                continue
