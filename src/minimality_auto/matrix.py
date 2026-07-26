"""Complex matrix semantics for PROP circuits."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache
from typing import Any

import numpy as np

from .core import (
    Circuit,
    Signature,
    ValidationError,
    _matrix_shape_matches,
    _natural,
)


def basis_digits(index: int, wires: int, dimension: int) -> tuple[int, ...]:
    """Decode a computational-basis index, most-significant wire first."""

    result = [0] * wires
    for position in range(wires - 1, -1, -1):
        index, result[position] = divmod(index, dimension)
    return tuple(result)


def basis_index(values: Iterable[int], dimension: int) -> int:
    """Encode computational-basis digits, most-significant wire first."""

    result = 0
    for value in values:
        result = result * dimension + value
    return result


def basis_permutation(
    value: Any,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[int, ...] | None:
    """Extract an exact column-to-row permutation from a 0/1 matrix."""

    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        return None

    result: list[int] = []
    seen: set[int] = set()
    for column in range(matrix.shape[1]):
        if checkpoint is not None and column % 128 == 0:
            checkpoint()
        output = -1
        for row in range(matrix.shape[0]):
            entry = matrix[row, column]
            if entry == 0:
                continue
            if entry != 1 or output >= 0:
                return None
            output = row
        if output < 0 or output in seen:
            return None
        seen.add(output)
        result.append(output)
    return tuple(result)


def basis_wire_permutation(
    order: Sequence[int],
    wire_dimension: int,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[int, ...]:
    """Return the column-to-row routing induced by a wire permutation."""

    order = tuple(order)
    if sorted(order) != list(range(len(order))):
        raise ValidationError(f"not a wire permutation: {list(order)}")
    result = []
    for column in range(wire_dimension ** len(order)):
        if checkpoint is not None and column % 256 == 0:
            checkpoint()
        digits = basis_digits(column, len(order), wire_dimension)
        result.append(
            basis_index((digits[source] for source in order), wire_dimension)
        )
    return tuple(result)


def permutation_matrix(
    order: Sequence[int],
    wire_dimension: int,
) -> np.ndarray:
    """Return the basis permutation whose output axes follow ``order``."""

    routing = basis_wire_permutation(order, wire_dimension)
    size = len(routing)
    result = np.zeros((size, size), dtype=np.complex128)
    result[np.asarray(routing), np.arange(size)] = 1
    return result


class MatrixSemantics:
    """Evaluate circuits as complex matrices over a fixed wire dimension."""

    def __init__(
        self,
        signature: Signature,
        wire_dimension: int = 2,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.signature = signature
        self.wire_dimension = _natural(wire_dimension, "wire dimension")
        if self.wire_dimension == 0:
            raise ValidationError("wire dimension must be positive")
        self.overrides = dict(overrides or {})

    def evaluate(self, circuit: Circuit) -> Any:
        matrices = self.overrides
        d = self.wire_dimension

        def generator_matrix(node: Circuit) -> Any:
            assert node.name is not None
            raw = matrices.get(node.name)
            if raw is None:
                raw = self.signature[node.name].matrix
            if raw is None:
                raise ValidationError(f"no matrix supplied for generator {node.name!r}")
            matrix = np.asarray(raw, dtype=np.complex128)
            if not _matrix_shape_matches(
                matrix.shape,
                d,
                node.inputs,
                node.outputs,
            ):
                raise ValidationError(
                    f"matrix for {node.name!r} has shape {matrix.shape}, expected "
                    f"({d}^{node.outputs}, {d}^{node.inputs})"
                )
            return matrix

        @cache
        def run(node: Circuit) -> Any:
            if node.kind == "gen":
                result = generator_matrix(node)
            elif node.kind == "id":
                result = np.eye(d**node.inputs, dtype=np.complex128)
            elif node.kind == "perm":
                result = permutation_matrix(node.order, d)
            elif node.kind == "tensor":
                result = np.ones((1, 1), dtype=np.complex128)
                for part in node.parts:
                    result = np.kron(result, run(part))
            elif node.kind == "compose":
                result = run(node.parts[0])
                for part in node.parts[1:]:
                    result = run(part) @ result
            else:
                raise AssertionError(f"unknown circuit kind {node.kind!r}")
            return result

        return run(circuit)


def evaluate_matrix(
    circuit: Circuit,
    signature: Signature,
    wire_dimension: int = 2,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Convenience wrapper around :class:`MatrixSemantics`."""

    return MatrixSemantics(signature, wire_dimension, overrides).evaluate(circuit)
