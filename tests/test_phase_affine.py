from __future__ import annotations

from itertools import pairwise, product
from math import comb, pi
from pathlib import Path

import numpy as np

from minimality_auto.core import Circuit, iter_circuit_nodes, iter_theory_terms, load_theory
from minimality_auto.search import search_theory

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
SIMPLIFIED_D7_THEORY_PATHS = [
    THEORY_DIR / "affine_simplified_d7.json",
    THEORY_DIR / "linear_phase_simplified_d7.json",
    THEORY_DIR / "quadratic_phase_simplified_d7.json",
    THEORY_DIR / "cubic_phase_simplified_d7.json",
]
SIMPLIFIED_D11_AFFINE_PATH = THEORY_DIR / "affine_simplified_d11.json"
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
SIMPLIFIED_COUNTS = [7, 12, 19, 31]
SIMPLIFIED_AFFINE_IDS = [
    "A1",
    "A3_order",
    "A3_g",
    "A4_A6_merged",
    "A5_merged",
    "A7",
    "A8",
]
D11_AFFINE_IDS = [
    "A1",
    "A3_order",
    "A3_g",
    "A4_A6_merged",
    "A5_order",
    "A5_g",
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
    instances = [
        (SIMPLIFIED_THEORY_PATHS, 5, "g=2; M_(2^k)=M_g^k"),
        (SIMPLIFIED_D7_THEORY_PATHS, 7, "g=3; M_(3^k)=M_g^k"),
    ]
    for paths, dimension, multiplier in instances:
        theories = [load_theory(path) for path in paths]
        for theory, generators, macros, count, figures in zip(
            theories,
            SIMPLIFIED_GENERATORS,
            EXPECTED_MACROS,
            SIMPLIFIED_COUNTS,
            EXPECTED_FIGURES,
            strict=True,
        ):
            assert theory.wire_dimension == dimension
            assert set(theory.signature) == generators
            assert set(theory.macros) == macros
            assert len(theory.equations) == count
            assert theory.source["figures"] == figures
            assert theory.source["multiplier"] == multiplier

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


def test_d11_single_multiplier_affine_theory_is_standalone():
    theory = load_theory(SIMPLIFIED_D11_AFFINE_PATH)

    assert theory.wire_dimension == 11
    assert set(theory.signature) == SIMPLIFIED_GENERATORS[0]
    assert not theory.macros
    assert [equation.id for equation in theory.equations] == D11_AFFINE_IDS
    assert theory.source["figures"] == [1]
    assert theory.source["multiplier"] == "g=2; M_(2^k)=M_g^k"


def _permutation_matrix(dimension, arity, transform):
    size = dimension**arity
    result = np.zeros((size, size), dtype=np.complex128)
    for values in product(range(dimension), repeat=arity):
        column = sum(
            value * dimension ** (arity - index - 1)
            for index, value in enumerate(values)
        )
        output = transform(values)
        row = sum(
            value * dimension ** (arity - index - 1)
            for index, value in enumerate(output)
        )
        result[row, column] = 1
    return result


def test_primitive_matrices_use_the_papers_basis_actions():
    signature = load_theory(THEORY_PATHS[-1]).signature

    np.testing.assert_array_equal(
        signature["X"].matrix,
        _permutation_matrix(DIMENSION, 1, lambda values: ((values[0] + 1) % DIMENSION,)),
    )
    np.testing.assert_array_equal(
        signature["CX"].matrix,
        _permutation_matrix(
            DIMENSION,
            2,
            lambda values: (values[0], (values[0] + values[1]) % DIMENSION),
        ),
    )
    for multiplier in range(1, DIMENSION):
        np.testing.assert_array_equal(
            signature[f"M_{multiplier}"].matrix,
            _permutation_matrix(
                DIMENSION,
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

    d7 = load_theory(SIMPLIFIED_D7_THEORY_PATHS[-1]).signature
    np.testing.assert_array_equal(
        d7["X"].matrix,
        _permutation_matrix(7, 1, lambda values: ((values[0] + 1) % 7,)),
    )
    np.testing.assert_array_equal(
        d7["CX"].matrix,
        _permutation_matrix(7, 2, lambda values: (values[0], sum(values) % 7)),
    )
    np.testing.assert_array_equal(
        d7["M_g"].matrix,
        _permutation_matrix(7, 1, lambda values: (3 * values[0] % 7,)),
    )
    root = np.exp(2j * pi / 7)
    np.testing.assert_allclose(d7["omega"].matrix, [[root]])
    for name, exponent in exponents.items():
        expected = np.diag([root ** (exponent(value) % 7) for value in range(7)])
        np.testing.assert_allclose(d7[name].matrix, expected)

    d11 = load_theory(SIMPLIFIED_D11_AFFINE_PATH).signature
    np.testing.assert_array_equal(
        d11["X"].matrix,
        _permutation_matrix(11, 1, lambda values: ((values[0] + 1) % 11,)),
    )
    np.testing.assert_array_equal(
        d11["CX"].matrix,
        _permutation_matrix(11, 2, lambda values: (values[0], sum(values) % 11)),
    )
    np.testing.assert_array_equal(
        d11["M_g"].matrix,
        _permutation_matrix(11, 1, lambda values: (2 * values[0] % 11,)),
    )


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

    d7 = load_theory(SIMPLIFIED_D7_THEORY_PATHS[-1]).signature
    multiplier = np.asarray(d7["M_g"].matrix)
    for value, exponent in {1: 0, 3: 1, 2: 2, 6: 3, 4: 4, 5: 5}.items():
        np.testing.assert_array_equal(
            np.linalg.matrix_power(multiplier, exponent),
            _permutation_matrix(7, 1, lambda basis, value=value: (value * basis[0] % 7,)),
        )

    d11 = load_theory(SIMPLIFIED_D11_AFFINE_PATH).signature
    multiplier = np.asarray(d11["M_g"].matrix)
    for exponent, value in enumerate((1, 2, 4, 8, 5, 10, 9, 7, 3, 6)):
        np.testing.assert_array_equal(
            np.linalg.matrix_power(multiplier, exponent),
            _permutation_matrix(
                11,
                1,
                lambda basis, value=value: (value * basis[0] % 11,),
            ),
        )


def test_merged_a5_has_a_small_certified_separator():
    theory = load_theory(SIMPLIFIED_THEORY_PATHS[0])
    report = search_theory(
        theory,
        strategies=("finite_model",),
        equation_ids={"A5_merged"},
        max_permutation_degree=3,
        timeout=2,
    )

    witness = report.witnesses["A5_merged"]
    assert witness.certified
    assert witness.parameters["degree"] == 3


def test_merged_a4_a6_has_exact_affine_lift_separators():
    instances = (
        (SIMPLIFIED_THEORY_PATHS[0], (2, 3)),
        (SIMPLIFIED_D7_THEORY_PATHS[0], (3, 4)),
        (SIMPLIFIED_D11_AFFINE_PATH, (5, 6)),
    )
    for path, (positive, negative) in instances:
        report = search_theory(
            load_theory(path),
            strategies=("affine_lift",),
            equation_ids={"A4_A6_merged"},
            max_affine_lift_prime=2,
            timeout=10,
        )

        witness = report.witnesses["A4_A6_merged"]
        assert witness.certified
        assert witness.parameters["prime"] == 2
        assert witness.parameters["translations"] == {
            "generators": {
                "CX": [
                    [[positive, 0], 1],
                    [[positive, positive], 1],
                    [[negative, 0], 1],
                    [[negative, negative], 1],
                ]
            },
            "structural_swaps": {},
        }


def test_d11_a8_has_an_exact_block_normal_form_separator():
    report = search_theory(
        load_theory(SIMPLIFIED_D11_AFFINE_PATH),
        strategies=("block_normal_form",),
        equation_ids={"A8"},
        timeout=2,
    )

    witness = report.witnesses["A8"]
    assert witness.certified
    assert witness.description.startswith("exact ")


_ACTION_CACHE: dict[
    tuple[int, tuple[int, ...], int, int],
    tuple[Circuit, tuple[tuple[int, ...], int]],
] = {}


def _action(
    circuit: Circuit,
    basis: tuple[int, ...],
    dimension: int = 5,
    multiplier: int = 2,
) -> tuple[tuple[int, ...], int]:
    key = id(circuit), basis, dimension, multiplier
    cached = _ACTION_CACHE.get(key)
    if cached is None or cached[0] is not circuit:
        result = _uncached_action(circuit, basis, dimension, multiplier)
        _ACTION_CACHE[key] = circuit, result
        return result
    return cached[1]


def _uncached_action(
    circuit: Circuit,
    basis: tuple[int, ...],
    dimension: int,
    multiplier: int,
) -> tuple[tuple[int, ...], int]:
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
            return ((basis[0] + 1) % dimension,), 0
        if name == "CX":
            return (basis[0], (basis[0] + basis[1]) % dimension), 0
        if name == "M_g":
            return (multiplier * basis[0] % dimension,), 0
        if name.startswith("M_"):
            return (int(name[2:]) * basis[0] % dimension,), 0
        if name == "Z":
            return basis, basis[0]
        if name == "S":
            return basis, comb(basis[0], 2) % dimension
        if name == "T":
            return basis, comb(basis[0], 3) % dimension
        raise AssertionError(f"unexpected generator {name}")
    if circuit.kind == "compose":
        output, phase = basis, 0
        for part in circuit.parts:
            output, added = _action(part, output, dimension, multiplier)
            phase = (phase + added) % dimension
        return output, phase
    if circuit.kind == "tensor":
        output = []
        phase = offset = 0
        for part in circuit.parts:
            end = offset + part.inputs
            values, added = _action(
                part,
                basis[offset:end],
                dimension,
                multiplier,
            )
            output.extend(values)
            phase = (phase + added) % dimension
            offset = end
        return tuple(output), phase
    raise AssertionError(f"unexpected circuit kind {circuit.kind}")


def test_derived_shortcuts_have_the_declared_phase_polynomials():
    expected = {
        "CZ": (2, lambda values: values[0] * values[1]),
        "CS": (2, lambda values: values[0] * comb(values[1], 2)),
        "SC": (2, lambda values: values[1] * comb(values[0], 2)),
        "CCZ": (3, lambda values: values[0] * values[1] * values[2]),
    }

    for path, dimension, multiplier in (
        (THEORY_PATHS[-1], 5, 2),
        (SIMPLIFIED_D7_THEORY_PATHS[-1], 7, 3),
    ):
        theory = load_theory(path)
        for name, (arity, phase) in expected.items():
            for basis in product(range(dimension), repeat=arity):
                assert _action(
                    theory.macros[name],
                    basis,
                    dimension,
                    multiplier,
                ) == (basis, phase(basis) % dimension)


def test_every_equation_is_exact_on_the_computational_basis():
    instances = (
        (THEORY_PATHS[-1], 5, 2),
        (SIMPLIFIED_THEORY_PATHS[-1], 5, 2),
        (SIMPLIFIED_D7_THEORY_PATHS[-1], 7, 3),
        (SIMPLIFIED_D11_AFFINE_PATH, 11, 2),
    )
    for path, dimension, multiplier in instances:
        theory = load_theory(path)
        for equation in theory.equations:
            for basis in product(range(dimension), repeat=equation.lhs.inputs):
                assert _action(
                    equation.lhs,
                    basis,
                    dimension,
                    multiplier,
                ) == _action(
                    equation.rhs,
                    basis,
                    dimension,
                    multiplier,
                ), equation.id
