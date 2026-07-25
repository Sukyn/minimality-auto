"""Search finite factors and separate equations in their free amalgam.

Generator matrices supply only a sparsity/scalar template.  Small prime-field
assignments, factor partitions, shared subgroups, and conjugation actions are
all discovered from the input theory and then checked exactly.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations, product
from math import isqrt
from typing import Any

from ..core import Circuit
from ..search import CandidateModel, Deadline, NotApplicable
from . import finite_field, fixed_arity
from .common import require_endomorphic_theory
from .fixed_arity import Token, Word

MAX_MATRIX_DIMENSION = 32


def _bounded_power(base: int, exponent: int, limit: int) -> int | None:
    if base < 1 or exponent < 0 or limit < 1:
        return None
    result = 1
    for _ in range(exponent):
        if result > limit // base:
            return None
        result *= base
    return result


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


def _subgroup(
    group: finite_field.MatrixGroup,
    generators: Sequence[int],
    limit: int,
    deadline: Deadline,
) -> frozenset[int]:
    seen = {0}
    pending = deque((0,))
    steps = 0
    while pending:
        value = pending.popleft()
        for generator in generators:
            candidate = group.multiply(value, generator)
            if candidate not in seen:
                if len(seen) >= limit:
                    raise finite_field.Reject(
                        f"amalgamated subgroup order exceeds {limit}"
                    )
                seen.add(candidate)
                pending.append(candidate)
        steps += 1
        if steps % 128 == 0:
            deadline.check()
    return frozenset(seen)


def _cancel(
    left: Word, right: Word, deadline: Deadline | None = None
) -> tuple[Word, Word]:
    start = 0
    while start < min(len(left), len(right)) and left[start] == right[start]:
        start += 1
        if deadline is not None and start % 1024 == 0:
            deadline.check()
    left_end, right_end = len(left), len(right)
    while (
        left_end > start
        and right_end > start
        and left[left_end - 1] == right[right_end - 1]
    ):
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


@dataclass(frozen=True)
class _DElement:
    shared: int
    quotient: int


@dataclass
class _DGroup:
    base: finite_field.MatrixGroup
    quotient: finite_field.MatrixGroup
    action: Mapping[tuple[int, int], int]
    products: dict[tuple[_DElement, _DElement], _DElement] = field(default_factory=dict)

    def multiply(self, left: _DElement, right: _DElement) -> _DElement:
        key = left, right
        if key not in self.products:
            transported = self.action[left.quotient, right.shared]
            self.products[key] = _DElement(
                self.base.multiply(left.shared, transported),
                self.quotient.multiply(left.quotient, right.quotient),
            )
        return self.products[key]


def _close_shared_subgroup(
    base: finite_field.MatrixGroup,
    quotient: finite_field.MatrixGroup,
    seeds: Sequence[int],
    limit: int,
    deadline: Deadline,
) -> tuple[frozenset[int], dict[tuple[int, int], int]]:
    generators = set(seeds)
    while True:
        shared = _subgroup(base, tuple(sorted(generators)), limit, deadline)
        enlarged = set(generators)
        action: dict[tuple[int, int], int] = {}
        for quotient_value in range(quotient.order):
            q = quotient.elements[quotient_value]
            q_inverse = quotient.elements[quotient.inverse(quotient_value)]
            for index, shared_value in enumerate(shared):
                matrix = finite_field.multiply(
                    q_inverse,
                    finite_field.multiply(
                        base.elements[shared_value],
                        q,
                        base.size,
                        base.prime,
                    ),
                    base.size,
                    base.prime,
                )
                conjugate = base.element(matrix)
                enlarged.add(conjugate)
                action[quotient_value, shared_value] = conjugate
                if index % 128 == 0:
                    deadline.check()
        if enlarged <= shared:
            if len(shared) * quotient.order > limit:
                raise finite_field.Reject(
                    f"second factor order exceeds {limit}"
                )
            return shared, action
        generators = enlarged


def _cosets(
    group: finite_field.MatrixGroup,
    shared: frozenset[int],
    deadline: Deadline,
) -> tuple[dict[int, tuple[int, int]], tuple[int, ...]]:
    decomposition: dict[int, tuple[int, int]] = {}
    representatives: list[int] = []
    for representative in range(group.order):
        if representative in decomposition:
            continue
        representatives.append(representative)
        for head in sorted(shared):
            value = group.multiply(head, representative)
            if value in decomposition:
                raise finite_field.Reject("overlapping amalgam cosets")
            decomposition[value] = head, representative
        deadline.check()
    if len(decomposition) != group.order:
        raise finite_field.Reject("incomplete amalgam cosets")
    return decomposition, tuple(representatives)


@dataclass(frozen=True)
class _NormalForm:
    head: int
    syllables: tuple[tuple[str, int | _DElement], ...] = ()


@dataclass
class _Amalgam:
    base: finite_field.MatrixGroup
    second: _DGroup
    base_decomposition: Mapping[int, tuple[int, int]]
    base_representative_ids: Mapping[int, int]
    shared_ids: Mapping[int, int]
    tokens: Mapping[Token, tuple[str, int | _DElement]]

    def _embed(self, factor: str, value: int) -> int | _DElement:
        return value if factor == "C" else _DElement(value, 0)

    def _multiply(
        self, factor: str, left: int | _DElement, right: int | _DElement
    ) -> int | _DElement:
        if factor == "C" and isinstance(left, int) and isinstance(right, int):
            return self.base.multiply(left, right)
        if factor == "D" and isinstance(left, _DElement) and isinstance(right, _DElement):
            return self.second.multiply(left, right)
        raise RuntimeError("mixed amalgam factor values")

    def _decompose(
        self, factor: str, value: int | _DElement
    ) -> tuple[int, int | _DElement]:
        if factor == "C" and isinstance(value, int):
            return self.base_decomposition[value]
        if factor == "D" and isinstance(value, _DElement):
            return value.shared, _DElement(0, value.quotient)
        raise RuntimeError("invalid amalgam factor value")

    @staticmethod
    def _representative_is_identity(factor: str, value: int | _DElement) -> bool:
        return value == 0 if factor == "C" else value == _DElement(0, 0)

    def prepend(
        self, factor: str, value: int | _DElement, normal: _NormalForm
    ) -> _NormalForm:
        combined = self._multiply(factor, value, self._embed(factor, normal.head))
        head, representative = self._decompose(factor, combined)
        if self._representative_is_identity(factor, representative):
            return _NormalForm(head, normal.syllables)
        if not normal.syllables or normal.syllables[0][0] != factor:
            return _NormalForm(head, ((factor, representative), *normal.syllables))
        _, first = normal.syllables[0]
        transported, merged = self._decompose(
            factor, self._multiply(factor, representative, first)
        )
        new_head = self.base.multiply(head, transported)
        tail = normal.syllables[1:]
        if self._representative_is_identity(factor, merged):
            return _NormalForm(new_head, tail)
        return _NormalForm(new_head, ((factor, merged), *tail))

    def evaluate_word(
        self, word: Word, deadline: Deadline | None = None
    ) -> tuple[Any, ...]:
        normal = _NormalForm(0)
        for index, token in enumerate(reversed(word)):
            factor, value = self.tokens[token]
            normal = self.prepend(factor, value, normal)
            if deadline is not None and index % 128 == 0:
                deadline.check()
        syllables: list[tuple[str, int]] = []
        for factor, representative in normal.syllables:
            if factor == "C" and isinstance(representative, int):
                representative_id = self.base_representative_ids[representative]
            elif factor == "D" and isinstance(representative, _DElement):
                representative_id = representative.quotient
            else:  # pragma: no cover - guarded by the normalizer
                raise RuntimeError("invalid normal-form syllable")
            syllables.append((factor, representative_id))
            if deadline is not None and len(syllables) % 128 == 0:
                deadline.check()
        return self.shared_ids[normal.head], tuple(syllables)


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

    shared, action = _close_shared_subgroup(
        base, quotient, seeds, max_order, deadline
    )
    second = _DGroup(base, quotient, action)
    decomposition, representatives = _cosets(base, shared, deadline)
    tokens: dict[Token, tuple[str, int | _DElement]] = {
        token: ("C", value) for token, value in base_tokens.items()
    }
    tokens.update(
        {
            token: ("D", _DElement(0, value))
            for token, value in quotient_tokens.items()
        }
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
        raise finite_field.Reject(
            "a retained equation fails after amalgam normalization"
        )
    if amalgam.evaluate_word(
        target_words[0], deadline
    ) == amalgam.evaluate_word(target_words[1], deadline):
        return None

    def evaluate(term: Circuit) -> tuple[Any, ...]:
        if term.inputs != term.outputs or term.inputs > arity:
            raise NotApplicable(
                f"amalgam interpretation supports endomorphisms through arity {arity}"
            )
        return amalgam.evaluate_word(
            fixed_arity.compile_word(
                term,
                signature,
                arity,
                deadline=deadline,
            ),
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
        structural[
            fixed_arity.display_name(fixed_arity.swap(index))
        ] = order
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
                variable_label(index): value
                for index, value in enumerate(assignment)
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
        evaluator=evaluate,
    )


def _is_prime(value: int, deadline: Deadline | None = None) -> bool:
    if value < 2:
        return False
    for divisor in range(2, isqrt(value) + 1):
        if deadline is not None and divisor % 1024 == 0:
            deadline.check()
        if value % divisor == 0:
            return False
    return True


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
    if (
        max_amalgam_prime < 2
        or max_amalgam_order < 1
        or max_amalgam_bridge_generators < 1
    ):
        return
    if max_amalgam_scalars < 0:
        raise NotApplicable("the amalgam scalar bound must be non-negative")
    require_endomorphic_theory(theory, deadline, context)
    arity = target.arity
    if target.lhs.type != (arity, arity):
        return
    size = _bounded_power(
        theory.wire_dimension, arity, max_amalgam_matrix_dimension
    )
    if size is None:
        raise NotApplicable(
            f"amalgam matrix dimension exceeds {max_amalgam_matrix_dimension}"
        )
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
        if not _is_prime(prime, deadline):
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
                for bridges in _bridge_subsets(
                    ordered, max_amalgam_bridge_generators, deadline
                ):
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
