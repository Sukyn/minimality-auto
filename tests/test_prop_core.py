from __future__ import annotations

import numpy as np

from minimality_auto.core import Circuit, MatrixSemantics, Signature


def _block_swap(left: int, right: int) -> Circuit:
    return Circuit.perm((*range(left, left + right), *range(left)))


def test_matrix_semantics_satisfies_naturality_for_arity_changing_boxes():
    f_matrix = np.arange(8, dtype=float).reshape(4, 2) / 7
    g_matrix = np.arange(8, 16, dtype=float).reshape(2, 4) / 11
    signature = Signature.from_json(
        [
            {"id": "f", "inputs": 1, "outputs": 2, "matrix": f_matrix.tolist()},
            {"id": "g", "inputs": 2, "outputs": 1, "matrix": g_matrix.tolist()},
        ]
    )
    f = Circuit.generator(signature["f"])
    g = Circuit.generator(signature["g"])
    semantics = MatrixSemantics(signature)

    left = Circuit.compose((Circuit.tensor((f, g)), _block_swap(2, 1)))
    right = Circuit.compose((_block_swap(1, 2), Circuit.tensor((g, f))))

    np.testing.assert_allclose(semantics.evaluate(left), semantics.evaluate(right))


def test_matrix_semantics_satisfies_tensor_interchange_for_rectangular_maps():
    matrices = {
        "f1": np.arange(8, dtype=float).reshape(4, 2) / 5,
        "f2": np.arange(8, dtype=float).reshape(2, 4) / 7,
        "g1": np.arange(8, 16, dtype=float).reshape(2, 4) / 11,
        "g2": np.arange(8, dtype=float).reshape(4, 2) / 13,
    }
    signature = Signature.from_json(
        [
            {"id": "f1", "inputs": 1, "outputs": 2, "matrix": matrices["f1"].tolist()},
            {"id": "f2", "inputs": 2, "outputs": 1, "matrix": matrices["f2"].tolist()},
            {"id": "g1", "inputs": 2, "outputs": 1, "matrix": matrices["g1"].tolist()},
            {"id": "g2", "inputs": 1, "outputs": 2, "matrix": matrices["g2"].tolist()},
        ]
    )
    f1, f2, g1, g2 = (Circuit.generator(signature[name]) for name in matrices)
    semantics = MatrixSemantics(signature)

    left = Circuit.tensor((Circuit.compose((f1, f2)), Circuit.compose((g1, g2))))
    right = Circuit.compose((Circuit.tensor((f1, g1)), Circuit.tensor((f2, g2))))

    np.testing.assert_allclose(semantics.evaluate(left), semantics.evaluate(right))


def test_compact_arity_changing_placement_preserves_untouched_wire_order():
    grow = np.asarray(
        [
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1],
        ]
    )
    signature = Signature.from_json(
        [{"id": "grow", "inputs": 1, "outputs": 2, "matrix": grow.tolist()}]
    )
    circuit = Circuit.from_json(
        {"wires": 2, "ops": [{"gen": "grow", "on": [1]}]}, signature
    )
    value = MatrixSemantics(signature).evaluate(circuit)
    input_01 = np.zeros(4)
    input_01[1] = 1
    expected_011 = np.zeros(8)
    expected_011[3] = 1

    assert circuit.type == (2, 3)
    np.testing.assert_allclose(value @ input_01, expected_011)
