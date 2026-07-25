"""Complex matrix semantics for PROP circuits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .core import (
    Circuit,
    Signature,
    ValidationError,
    _matrix_shape_matches,
    _matrix_tuple,
    _natural,
)


def parse_complex_matrix(value: Any) -> np.ndarray:
    """Parse a JSON matrix as a NumPy complex array."""

    return np.asarray(_matrix_tuple(value), dtype=np.complex128)


def permutation_matrix(
    order: Sequence[int],
    wire_dimension: int,
) -> np.ndarray:
    """Return the basis permutation whose output axes follow ``order``."""

    size = wire_dimension ** len(order)
    columns = (
        np.arange(size)
        .reshape((wire_dimension,) * len(order))
        .transpose(order)
        .ravel()
    )
    result = np.zeros((size, size), dtype=np.complex128)
    result[np.arange(size), columns] = 1
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

    def evaluate(
        self,
        circuit: Circuit,
        overrides: Mapping[str, Any] | None = None,
    ) -> Any:
        matrices = dict(self.overrides)
        matrices.update(overrides or {})
        cache: dict[Circuit, Any] = {}
        d = self.wire_dimension

        def generator_matrix(node: Circuit) -> Any:
            assert node.name is not None
            raw = matrices.get(node.name)
            if raw is None:
                raw = self.signature[node.name].matrix
            if raw is None:
                raise ValidationError(f"no matrix supplied for generator {node.name!r}")
            matrix = np.asarray(
                raw if hasattr(raw, "shape") else _matrix_tuple(raw),
                dtype=np.complex128,
            )
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

        def run(node: Circuit) -> Any:
            if node in cache:
                return cache[node]
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
            cache[node] = result
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
