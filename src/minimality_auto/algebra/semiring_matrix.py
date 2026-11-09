"""Exact and symbolic rectangular matrices over three finite semirings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import z3

from ..matrix import basis_wire_permutation
from ..search_types import Deadline, NotApplicable
from ..syntax import Circuit

Matrix = tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class FiniteSemiring:
    kind: str
    prime: int = 2
    symbolic: bool = False

    @property
    def zero(self) -> Any:
        if self.kind == "relations":
            return False
        return (0, 0) if self.kind == "dual_numbers" else 0

    @property
    def one(self) -> Any:
        if self.kind == "relations":
            return True
        return (1, 0) if self.kind == "dual_numbers" else 1

    def add(self, values: Iterable[Any]) -> Any:
        values = tuple(values)
        if self.kind == "relations":
            return z3.Or(*values) if self.symbolic else any(values)
        if self.kind == "dual_numbers":
            return tuple(sum(value[i] for value in values) % self.prime for i in (0, 1))
        return sum(values) % self.prime

    def multiply(self, left: Any, right: Any) -> Any:
        if self.kind == "relations":
            return z3.And(left, right) if self.symbolic else left and right
        if self.kind == "dual_numbers":
            a, b = left
            c, d = right
            return a * c % self.prime, (a * d + b * c) % self.prime
        return left * right % self.prime

    def unknown(self, solver: z3.Solver, name: str) -> Any:
        if self.kind == "relations":
            return z3.Bool(name)
        if self.kind == "dual_numbers":
            return tuple(self._coefficient(solver, f"{name}_{i}") for i in (0, 1))
        return self._coefficient(solver, name)

    def _coefficient(self, solver: z3.Solver, name: str) -> z3.ArithRef:
        value = z3.Int(name)
        solver.add(value >= 0, value < self.prime)
        return value

    def decode(self, model: z3.ModelRef, value: Any) -> Any:
        if self.kind == "dual_numbers":
            return tuple(model.eval(item, model_completion=True).as_long() for item in value)
        result = model.eval(value, model_completion=True)
        return z3.is_true(result) if self.kind == "relations" else result.as_long()


class SemiringMatrixSemantics:
    """A fixed interpretation; the subterm cache is shared by all equation sides."""

    def __init__(
        self,
        dimension: int,
        ring: FiniteSemiring,
        interpretation: Mapping[str, Matrix],
        deadline: Deadline,
        max_entries: int = 256,
    ) -> None:
        self.dimension = dimension
        self.ring = ring
        self.interpretation = interpretation
        self.deadline = deadline
        self.max_entries = max_entries
        self._cache: dict[Circuit, Matrix] = {}

    def shape(self, term: Circuit) -> tuple[int, int]:
        def power(wires: int) -> int:
            size = 1
            if self.dimension == 1:
                return size
            for _ in range(wires):
                if size > self.max_entries // self.dimension:
                    raise NotApplicable("finite matrix exceeds the entry limit")
                size *= self.dimension
            return size

        rows, columns = power(term.outputs), power(term.inputs)
        if rows * columns > self.max_entries:
            raise NotApplicable("finite matrix exceeds the entry limit")
        return rows, columns

    def multiply(self, left: Matrix, right: Matrix) -> Matrix:
        if len(left) * len(right[0]) > self.max_entries:
            raise NotApplicable("finite matrix exceeds the entry limit")
        rows = []
        for row in left:
            self.deadline.check()
            rows.append(
                tuple(
                    self.ring.add(self.ring.multiply(a, right[k][j]) for k, a in enumerate(row))
                    for j in range(len(right[0]))
                )
            )
        return tuple(rows)

    def tensor(self, left: Matrix, right: Matrix) -> Matrix:
        if len(left) * len(left[0]) * len(right) * len(right[0]) > self.max_entries:
            raise NotApplicable("finite matrix exceeds the entry limit")
        rows = []
        for a in left:
            self.deadline.check()
            rows.extend(tuple(self.ring.multiply(x, y) for x in a for y in b) for b in right)
        return tuple(rows)

    def evaluate(self, term: Circuit) -> Matrix:
        self.deadline.check()
        if term in self._cache:
            return self._cache[term]
        rows, columns = self.shape(term)
        ring = self.ring
        if term.kind == "gen":
            result = self.interpretation[term.name]
            if len(result) != rows or any(len(row) != columns for row in result):
                raise ValueError("generator interpretation has the wrong matrix shape")
        elif term.kind in {"id", "perm"}:
            order = (
                basis_wire_permutation(term.order, self.dimension, self.deadline.check)
                if term.kind == "perm" and self.dimension != 1
                else tuple(range(columns))
            )
            result = tuple(
                tuple(ring.one if i == order[j] else ring.zero for j in range(columns))
                for i in range(rows)
            )
        elif term.kind == "compose":
            result = self.evaluate(term.parts[0])
            for part in term.parts[1:]:
                result = self.multiply(self.evaluate(part), result)
        elif term.kind == "tensor":
            result = ((ring.one,),)
            for part in term.parts:
                result = self.tensor(result, self.evaluate(part))
        else:
            raise ValueError(f"unsupported circuit node {term.kind!r}")
        self._cache[term] = result
        return result


def symbolic_equal(left: Matrix, right: Matrix, *, dual: bool = False) -> z3.BoolRef:
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        return z3.BoolVal(False)
    return z3.And(
        *(
            z3.And(*(a[i] == b[i] for i in (0, 1))) if dual else a == b
            for row, other in zip(left, right, strict=True)
            for a, b in zip(row, other, strict=True)
        )
    )
