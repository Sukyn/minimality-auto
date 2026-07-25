from __future__ import annotations

from functools import cache
from itertools import pairwise, product
from math import comb, pi
from pathlib import Path

import numpy as np

from minimality_auto.core import Circuit, iter_circuit_nodes, iter_theory_terms, load_theory

DIMENSION = 5
THEORY_DIR = Path(__file__).parents[1] / "theories"
THEORY_PATHS = [
    THEORY_DIR / "affine_d5.json",
    THEORY_DIR / "linear_phase_d5.json",
    THEORY_DIR / "quadratic_phase_d5.json",
    THEORY_DIR / "cubic_phase_d5.json",
]
SIMPLIFIED_THEORY_PATHS = [
    THEORY_DIR / "affine_simplified_d5.json",
    THEORY_DIR / "linear_phase_simplified_d5.json",
    THEORY_DIR / "quadratic_phase_simplified_d5.json",
    THEORY_DIR / "cubic_phase_simplified_d5.json",
]
EXPECTED_GENERATORS = [
    {"X", "CX", "M_1", "M_2", "M_3", "M_4"},
    {"X", "CX", "M_1", "M_2", "M_3", "M_4", "omega", "Z"},
    {"X", "CX", "M_1", "M_2", "M_3", "M_4", "omega", "Z", "S"},
    {"X", "CX", "M_1", "M_2", "M_3", "M_4", "omega", "Z", "S", "T"},
]
EXPECTED_MACROS = [set(), set(), {"CZ"}, {"CZ", "CS", "SC", "CCZ"}]
EXPECTED_COUNTS = [32, 40, 50, 74]
EXPECTED_FIGURES = [[1], [1, 3], [1, 3, 4], [1, 3, 4, 5]]
SIMPLIFIED_GENERATORS = [
    {"X", "CX", "M_g"},
    {"X", "CX", "M_g", "omega", "Z"},
    {"X", "CX", "M_g", "omega", "Z", "S"},
    {"X", "CX", "M_g", "omega", "Z", "S", "T"},
]
SIMPLIFIED_COUNTS = [9, 14, 21, 33]
SIMPLIFIED_AFFINE_IDS = [
    "A1",
    "A3_order",
    "A3_g",
    "A4_g",
    "A5_order",
    "A5_g",
    "A6",
    "A7",
    "A8",
]


def test_theories_are_standalone_nested_presentations():
    theories = [load_theory(path) for path in THEORY_PATHS]

    for theory, generators, macros, count, figures in zip(
        theories,
        EXPECTED_GENERATORS,
        EXPECTED_MACROS,
        EXPECTED_COUNTS,
        EXPECTED_FIGURES,
        strict=True,
    ):
        assert theory.wire_dimension == DIMENSION
        assert set(theory.signature) == generators
        assert set(theory.macros) == macros
        assert len(theory.equations) == count
        assert theory.source["paper"] == "https://arxiv.org/abs/2603.06466"
        assert theory.source["version"] == 3
        assert theory.source["figures"] == figures

    for parent, child in pairwise(theories):
        assert child.equations[: len(parent.equations)] == parent.equations
        for name, generator in parent.signature.items():
            assert child.signature[name] == generator
        for name, body in parent.macros.items():
            assert child.macros[name] == body

    assert all(
        node.kind != "macro"
        for theory in theories
        for term in iter_theory_terms(theory)
        for node in iter_circuit_nodes(term)
    )


def test_single_multiplier_theories_are_standalone_and_nested():
    theories = [load_theory(path) for path in SIMPLIFIED_THEORY_PATHS]

    for theory, generators, macros, count, figures in zip(
        theories,
        SIMPLIFIED_GENERATORS,
        EXPECTED_MACROS,
        SIMPLIFIED_COUNTS,
        EXPECTED_FIGURES,
        strict=True,
    ):
        assert theory.wire_dimension == DIMENSION
        assert set(theory.signature) == generators
        assert set(theory.macros) == macros
        assert len(theory.equations) == count
        assert theory.source["figures"] == figures
        assert theory.source["multiplier"] == "g=2; M_(2^k)=M_g^k"

    assert [equation.id for equation in theories[0].equations] == SIMPLIFIED_AFFINE_IDS
    for parent, child in pairwise(theories):
        assert child.equations[: len(parent.equations)] == parent.equations
        for name, generator in parent.signature.items():
            assert child.signature[name] == generator
        for name, body in parent.macros.items():
            assert child.macros[name] == body

    assert all(
        node.kind != "macro"
        for theory in theories
        for term in iter_theory_terms(theory)
        for node in iter_circuit_nodes(term)
    )


def _permutation_matrix(arity, transform):
    size = DIMENSION**arity
    result = np.zeros((size, size), dtype=np.complex128)
    for values in product(range(DIMENSION), repeat=arity):
        column = sum(value * DIMENSION ** (arity - index - 1) for index, value in enumerate(values))
        output = transform(values)
        row = sum(value * DIMENSION ** (arity - index - 1) for index, value in enumerate(output))
        result[row, column] = 1
    return result


def test_primitive_matrices_use_the_papers_basis_actions():
    signature = load_theory(THEORY_PATHS[-1]).signature

    np.testing.assert_array_equal(
        signature["X"].matrix,
        _permutation_matrix(1, lambda values: ((values[0] + 1) % DIMENSION,)),
    )
    np.testing.assert_array_equal(
        signature["CX"].matrix,
        _permutation_matrix(
            2,
            lambda values: (values[0], (values[0] + values[1]) % DIMENSION),
        ),
    )
    for multiplier in range(1, DIMENSION):
        np.testing.assert_array_equal(
            signature[f"M_{multiplier}"].matrix,
            _permutation_matrix(
                1,
                lambda values, multiplier=multiplier: (multiplier * values[0] % DIMENSION,),
            ),
        )

    root = np.exp(2j * pi / DIMENSION)
    np.testing.assert_allclose(signature["omega"].matrix, [[root]])
    exponents = {
        "Z": lambda value: value,
        "S": lambda value: comb(value, 2),
        "T": lambda value: comb(value, 3),
    }
    for name, exponent in exponents.items():
        expected = np.diag([root ** (exponent(value) % DIMENSION) for value in range(DIMENSION)])
        np.testing.assert_allclose(signature[name].matrix, expected)


def test_single_multiplier_powers_recover_every_multiplier():
    full = load_theory(THEORY_PATHS[-1]).signature
    simplified = load_theory(SIMPLIFIED_THEORY_PATHS[-1]).signature
    multiplier = np.asarray(simplified["M_g"].matrix)
    powers = {1: 0, 2: 1, 4: 2, 3: 3}

    for value, exponent in powers.items():
        np.testing.assert_array_equal(
            np.linalg.matrix_power(multiplier, exponent),
            full[f"M_{value}"].matrix,
        )


@cache
def _action(circuit: Circuit, basis: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    if circuit.kind == "id":
        return basis, 0
    if circuit.kind == "perm":
        return tuple(basis[index] for index in circuit.order), 0
    if circuit.kind == "gen":
        assert circuit.name is not None
        name = circuit.name
        if name == "omega":
            return basis, 1
        if name == "X":
            return ((basis[0] + 1) % DIMENSION,), 0
        if name == "CX":
            return (basis[0], (basis[0] + basis[1]) % DIMENSION), 0
        if name == "M_g":
            return (2 * basis[0] % DIMENSION,), 0
        if name.startswith("M_"):
            return (int(name[2:]) * basis[0] % DIMENSION,), 0
        if name == "Z":
            return basis, basis[0]
        if name == "S":
            return basis, comb(basis[0], 2) % DIMENSION
        if name == "T":
            return basis, comb(basis[0], 3) % DIMENSION
        raise AssertionError(f"unexpected generator {name}")
    if circuit.kind == "compose":
        output, phase = basis, 0
        for part in circuit.parts:
            output, added = _action(part, output)
            phase = (phase + added) % DIMENSION
        return output, phase
    if circuit.kind == "tensor":
        output = []
        phase = offset = 0
        for part in circuit.parts:
            end = offset + part.inputs
            values, added = _action(part, basis[offset:end])
            output.extend(values)
            phase = (phase + added) % DIMENSION
            offset = end
        return tuple(output), phase
    raise AssertionError(f"unexpected circuit kind {circuit.kind}")


def test_derived_shortcuts_have_the_declared_phase_polynomials():
    theory = load_theory(THEORY_PATHS[-1])
    expected = {
        "CZ": (2, lambda values: values[0] * values[1]),
        "CS": (2, lambda values: values[0] * comb(values[1], 2)),
        "SC": (2, lambda values: values[1] * comb(values[0], 2)),
        "CCZ": (3, lambda values: values[0] * values[1] * values[2]),
    }

    for name, (arity, phase) in expected.items():
        for basis in product(range(DIMENSION), repeat=arity):
            assert _action(theory.macros[name], basis) == (basis, phase(basis) % DIMENSION)


def test_every_equation_is_exact_on_the_computational_basis():
    for path in (THEORY_PATHS[-1], SIMPLIFIED_THEORY_PATHS[-1]):
        theory = load_theory(path)
        for equation in theory.equations:
            for basis in product(range(DIMENSION), repeat=equation.lhs.inputs):
                assert _action(equation.lhs, basis) == _action(equation.rhs, basis), equation.id
