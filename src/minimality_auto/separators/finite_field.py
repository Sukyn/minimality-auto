"""Small exact finite-field matrix groups used by separation searches."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Any, Mapping, Sequence

from ..search import Deadline, NotApplicable
from .finite_model import _generator, _swap


Mat = tuple[int, ...]
Token = tuple[str, str | int]
ENTRY_TOLERANCE = 1e-10


class Reject(RuntimeError):
    pass


def identity(size: int) -> Mat:
    return tuple(
        1 if row == column else 0
        for row in range(size)
        for column in range(size)
    )


def multiply(left: Mat, right: Mat, size: int, prime: int) -> Mat:
    result = [0] * (size * size)
    for row in range(size):
        for middle in range(size):
            value = left[row * size + middle]
            if value:
                for column in range(size):
                    index = row * size + column
                    result[index] = (
                        result[index] + value * right[middle * size + column]
                    ) % prime
    return tuple(result)


def inverse(value: Mat, size: int, prime: int) -> Mat:
    unit = identity(size)
    rows = [
        [
            *value[row * size : (row + 1) * size],
            *unit[row * size : (row + 1) * size],
        ]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if rows[row][column] % prime),
            None,
        )
        if pivot is None:
            raise Reject("singular finite-field matrix")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = pow(rows[column][column] % prime, -1, prime)
        rows[column] = [(entry * scale) % prime for entry in rows[column]]
        for row in range(size):
            if row == column:
                continue
            scale = rows[row][column] % prime
            if scale:
                rows[row] = [
                    (entry - scale * pivot_entry) % prime
                    for entry, pivot_entry in zip(
                        rows[row], rows[column], strict=True
                    )
                ]
    return tuple(entry for row in rows for entry in row[size:])


def kronecker(
    left: Mat,
    left_size: int,
    right: Mat,
    right_size: int,
    prime: int,
) -> Mat:
    return tuple(
        left[left_row * left_size + left_column]
        * right[right_row * right_size + right_column]
        % prime
        for left_row in range(left_size)
        for right_row in range(right_size)
        for left_column in range(left_size)
        for right_column in range(right_size)
    )


def permutation_matrix(
    order: Sequence[int], wire_dimension: int, prime: int
) -> Mat:
    wires = len(order)
    size = wire_dimension**wires
    result = [0] * (size * size)
    for column in range(size):
        digits = [0] * wires
        value = column
        for index in range(wires - 1, -1, -1):
            digits[index] = value % wire_dimension
            value //= wire_dimension
        row = 0
        for source in order:
            row = row * wire_dimension + digits[source]
        result[row * size + column] = 1 % prime
    return tuple(result)


@dataclass(frozen=True)
class _Scalar:
    kind: str
    first: int
    second: int


@dataclass(frozen=True)
class Templates:
    matrices: Mapping[str, tuple[_Scalar, ...]]
    sizes: Mapping[str, int]
    variables: tuple[complex, ...]


def templates(
    theory: Any,
    signature: Mapping[str, int],
    active: frozenset[str],
    max_variables: int,
    deadline: Deadline | None = None,
) -> Templates:
    variables: list[complex] = []

    def scalar(value: complex) -> _Scalar:
        if not isfinite(value.real) or not isfinite(value.imag):
            raise NotApplicable("finite-field templates require finite entries")
        if abs(value.imag) <= ENTRY_TOLERANCE:
            real = float(value.real)
            rational = Fraction(real).limit_denominator(64)
            if abs(real - float(rational)) <= ENTRY_TOLERANCE:
                return _Scalar("rational", rational.numerator, rational.denominator)

        # Non-rational entries are pattern variables.  Values that differ only
        # by a sign share a variable; no embedding of C into F_p is assumed.
        # Every candidate is subsequently checked by exact F_p arithmetic.
        sign = -1 if (
            value.real < -ENTRY_TOLERANCE
            or (
                abs(value.real) <= ENTRY_TOLERANCE
                and value.imag < -ENTRY_TOLERANCE
            )
        ) else 1
        representative = complex(sign * value)
        index = next(
            (
                position
                for position, known in enumerate(variables)
                if abs(representative - known) <= ENTRY_TOLERANCE
            ),
            None,
        )
        if index is None:
            variables.append(representative)
            index = len(variables) - 1
        return _Scalar("variable", index, sign)

    matrices: dict[str, tuple[_Scalar, ...]] = {}
    sizes: dict[str, int] = {}
    dimension = theory.wire_dimension
    for index, (name, width) in enumerate(signature.items()):
        if deadline is not None and index % 128 == 0:
            deadline.check()
        size = dimension**width
        generator = theory.signature[name]
        if name not in active or generator.matrix is None:
            raw = identity(size)
        else:
            try:
                if len(generator.matrix) != size or any(
                    len(row) != size for row in generator.matrix
                ):
                    raise NotApplicable(
                        f"matrix template for {name!r} has the wrong shape"
                    )
                raw = tuple(entry for row in generator.matrix for entry in row)
            except (TypeError, ValueError, OverflowError) as error:
                raise NotApplicable(
                    f"matrix template for {name!r} has invalid rows"
                ) from error
        try:
            matrices[name] = tuple(scalar(complex(entry)) for entry in raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise NotApplicable(
                f"matrix template for {name!r} has invalid entries"
            ) from error
        sizes[name] = size
    if len(variables) > max_variables:
        raise NotApplicable(
            f"finite-field templates contain more than {max_variables} free scalars"
        )
    return Templates(matrices, sizes, tuple(variables))


def instantiate(
    source: Templates,
    signature: Mapping[str, int],
    assignment: Sequence[int],
    arity: int,
    wire_dimension: int,
    prime: int,
) -> tuple[dict[Token, Mat], dict[str, Mat]]:
    native: dict[str, Mat] = {}
    atoms: dict[Token, Mat] = {}
    total_size = wire_dimension**arity
    for name, width in signature.items():
        entries: list[int] = []
        for scalar in source.matrices[name]:
            if scalar.kind == "rational":
                if scalar.second % prime == 0:
                    raise Reject("a template denominator vanishes")
                value = scalar.first * pow(scalar.second, -1, prime)
            else:
                value = scalar.second * assignment[scalar.first]
            entries.append(value % prime)
        matrix = tuple(entries)
        native[name] = matrix
        padding = wire_dimension ** (arity - width)
        atoms[_generator(name)] = kronecker(
            matrix,
            source.sizes[name],
            identity(padding),
            padding,
            prime,
        )
    for index in range(arity - 1):
        order = list(range(arity))
        order[index], order[index + 1] = order[index + 1], order[index]
        atoms[_swap(index)] = permutation_matrix(order, wire_dimension, prime)
    for matrix in atoms.values():
        inverse(matrix, total_size, prime)
    return atoms, native


class MatrixGroup:
    def __init__(
        self,
        generators: Sequence[Mat],
        size: int,
        prime: int,
        limit: int,
        deadline: Deadline,
    ) -> None:
        if size < 1 or prime < 2 or limit < 1:
            raise Reject("invalid finite matrix-group bound")
        if any(len(generator) != size * size for generator in generators):
            raise Reject("finite matrix generator has the wrong shape")
        self.size = size
        self.prime = prime
        unit = identity(size)
        self.elements: list[Mat] = [unit]
        self.indices: dict[Mat, int] = {unit: 0}
        self._products: dict[tuple[int, int], int] = {}
        self._inverses: dict[int, int] = {0: 0}
        pending = deque((0,))
        steps = 0
        while pending:
            left = self.elements[pending.popleft()]
            for right in generators:
                candidate = multiply(right, left, size, prime)
                if candidate not in self.indices:
                    if len(self.elements) >= limit:
                        raise Reject(f"factor order exceeds {limit}")
                    self.indices[candidate] = len(self.elements)
                    self.elements.append(candidate)
                    pending.append(len(self.elements) - 1)
            steps += 1
            if steps % 128 == 0:
                deadline.check()

    @property
    def order(self) -> int:
        return len(self.elements)

    def element(self, matrix: Mat) -> int:
        try:
            return self.indices[matrix]
        except KeyError as error:
            raise Reject("matrix is outside the generated factor") from error

    def multiply(self, left: int, right: int) -> int:
        key = left, right
        if key not in self._products:
            matrix = multiply(
                self.elements[right], self.elements[left], self.size, self.prime
            )
            self._products[key] = self.element(matrix)
        return self._products[key]

    def inverse(self, value: int) -> int:
        if value not in self._inverses:
            matrix = inverse(self.elements[value], self.size, self.prime)
            self._inverses[value] = self.element(matrix)
        return self._inverses[value]
