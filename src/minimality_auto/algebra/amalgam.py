"""Finite factors, shared subgroups, and free-amalgam normal forms."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..presentation import Token, Word
from ..search_types import Deadline
from . import finite_field


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
                    raise finite_field.Reject(f"amalgamated subgroup order exceeds {limit}")
                seen.add(candidate)
                pending.append(candidate)
        steps += 1
        if steps % 128 == 0:
            deadline.check()
    return frozenset(seen)


@dataclass
class _DGroup:
    """Semidirect factor with ids ``shared_base_id * |quotient| + quotient_id``."""

    base: finite_field.MatrixGroup
    quotient: finite_field.MatrixGroup
    action: Mapping[tuple[int, int], int]
    products: dict[tuple[int, int], int] = field(default_factory=dict)

    def element(self, shared: int, quotient: int) -> int:
        return shared * self.quotient.order + quotient

    def components(self, value: int) -> tuple[int, int]:
        return divmod(value, self.quotient.order)

    def multiply(self, left: int, right: int) -> int:
        key = left, right
        if key not in self.products:
            left_head, left_tail = self.components(left)
            right_head, right_tail = self.components(right)
            transported = self.action[left_tail, right_head]
            self.products[key] = self.element(
                self.base.multiply(left_head, transported),
                self.quotient.multiply(left_tail, right_tail),
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
                raise finite_field.Reject(f"second factor order exceeds {limit}")
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
            decomposition[value] = head, representative
        deadline.check()
    return decomposition, tuple(representatives)


@dataclass(frozen=True)
class _NormalForm:
    head: int
    syllables: tuple[tuple[str, int], ...] = ()


@dataclass
class _Amalgam:
    base: finite_field.MatrixGroup
    second: _DGroup
    base_decomposition: Mapping[int, tuple[int, int]]
    base_representative_ids: Mapping[int, int]
    shared_ids: Mapping[int, int]
    tokens: Mapping[Token, tuple[str, int]]

    def _embed(self, factor: str, value: int) -> int:
        return value if factor == "C" else self.second.element(value, 0)

    def _multiply(self, factor: str, left: int, right: int) -> int:
        group = self.base if factor == "C" else self.second
        return group.multiply(left, right)

    def _decompose(self, factor: str, value: int) -> tuple[int, int]:
        # A D representative has shared component zero, so its id is the quotient id.
        return self.base_decomposition[value] if factor == "C" else self.second.components(value)

    def prepend(self, factor: str, value: int, normal: _NormalForm) -> _NormalForm:
        combined = self._multiply(factor, value, self._embed(factor, normal.head))
        head, representative = self._decompose(factor, combined)
        if representative == 0:
            return _NormalForm(head, normal.syllables)
        if not normal.syllables or normal.syllables[0][0] != factor:
            return _NormalForm(head, ((factor, representative), *normal.syllables))
        _, first = normal.syllables[0]
        transported, merged = self._decompose(factor, self._multiply(factor, representative, first))
        new_head = self.base.multiply(head, transported)
        tail = normal.syllables[1:]
        if merged == 0:
            return _NormalForm(new_head, tail)
        return _NormalForm(new_head, ((factor, merged), *tail))

    def evaluate_word(
        self, word: Word, deadline: Deadline | None = None
    ) -> tuple[int, tuple[tuple[str, int], ...]]:
        normal = _NormalForm(0)
        for index, token in enumerate(reversed(word)):
            factor, value = self.tokens[token]
            normal = self.prepend(factor, value, normal)
            if deadline is not None and index % 128 == 0:
                deadline.check()
        syllables: list[tuple[str, int]] = []
        for factor, representative in normal.syllables:
            representative_id = (
                self.base_representative_ids[representative] if factor == "C" else representative
            )
            syllables.append((factor, representative_id))
            if deadline is not None and len(syllables) % 128 == 0:
                deadline.check()
        return self.shared_ids[normal.head], tuple(syllables)
