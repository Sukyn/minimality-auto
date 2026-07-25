from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minimality_auto.core import (
    Circuit,
    iter_circuit_nodes,
    iter_theory_terms,
    load_theory,
    primitive_occurrences,
)
from minimality_auto.matrix import evaluate_matrix
from minimality_auto.search import search_theory

THEORY_DIR = Path(__file__).parents[1] / "theories"
THEORY_PATH = THEORY_DIR / "multi_qudit_clifford_d3.json"
HIGHER_DIMENSION_PATHS = [
    THEORY_DIR / "multi_qudit_clifford_d5.json",
    THEORY_DIR / "multi_qudit_clifford_d7.json",
]
ALL_THEORY_PATHS = [THEORY_PATH, *HIGHER_DIMENSION_PATHS]
RETAINED = [
    "C0",
    "C1",
    "C2",
    "C3_k2",
    "C4",
    "C5",
    "C6",
    "C8",
    "C9",
    "C18",
    "CZ-decomp",
    "SWAP-decomp",
]
EXTRA_C3 = {
    3: [],
    5: ["C3_k3", "C3_k4"],
    7: ["C3_k3", "C3_k4", "C3_k5", "C3_k6"],
}
MULTIPLIERS = {
    3: {"M_g": 2},
    5: {"M_g": 2, "M_3": 3, "M_4": 4},
    7: {"M_g": 3, "M_2": 2, "M_4": 4, "M_5": 5, "M_6": 6},
}


def test_rule_inventory_and_shortcuts_are_explicit():
    theory = load_theory(THEORY_PATH)
    assert [equation.id for equation in theory.equations] == RETAINED
    assert set(theory.signature) == {"omega", "minus_one", "H", "S", "CZ"}
    assert set(theory.macros) == {
        "H2",
        "H3",
        "S_inv",
        "Z",
        "X",
        "M_g",
        "CZ_inv",
        "CX",
        "CX_inv",
    }
    for equation in theory.equations:
        assert set(primitive_occurrences(equation.lhs)) <= set(theory.signature)
        assert set(primitive_occurrences(equation.rhs)) <= set(theory.signature)
    assert all(
        node.kind != "macro"
        for term in iter_theory_terms(theory)
        for node in iter_circuit_nodes(term)
    )


@pytest.mark.parametrize("theory_path", ALL_THEORY_PATHS, ids=lambda path: path.stem)
def test_every_retained_equation_is_sound(theory_path: Path):
    theory = load_theory(theory_path)
    failures = []
    for equation in theory.equations:
        lhs = evaluate_matrix(
            equation.lhs, theory.signature, theory.wire_dimension
        )
        rhs = evaluate_matrix(
            equation.rhs, theory.signature, theory.wire_dimension
        )
        if not np.allclose(lhs, rhs, atol=1e-8, rtol=1e-8):
            failures.append(
                f"{equation.id}: {float(np.max(np.abs(lhs - rhs))):.6g}"
            )
    assert not failures, ", ".join(failures)


@pytest.mark.parametrize("theory_path", ALL_THEORY_PATHS, ids=lambda path: path.stem)
def test_derived_gate_orientations_match_the_diagrams(theory_path: Path):
    theory = load_theory(theory_path)
    d = theory.wire_dimension
    for name, factor in MULTIPLIERS[d].items():
        multiplier = evaluate_matrix(
            theory.macros[name], theory.signature, d
        )
        expected_multiplier = np.zeros((d, d), dtype=np.complex128)
        for value in range(d):
            expected_multiplier[factor * value % d, value] = 1
        np.testing.assert_allclose(
            multiplier, expected_multiplier, atol=1e-8, rtol=1e-8
        )

    cx = evaluate_matrix(theory.macros["CX"], theory.signature, d)
    expected_cx = np.zeros((d * d, d * d), dtype=np.complex128)
    for target in range(d):
        for control in range(d):
            column = target * d + control
            row = (target + control) % d * d + control
            expected_cx[row, column] = 1
    np.testing.assert_allclose(cx, expected_cx, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("theory_path", ALL_THEORY_PATHS, ids=lambda path: path.stem)
def test_swap_decomposition_has_the_lambda_squared_phase(theory_path: Path):
    theory = load_theory(theory_path)
    equation = theory.equation("SWAP-decomp")
    counts = primitive_occurrences(equation.rhs)

    assert equation.lhs == Circuit.perm((1, 0))
    assert counts["H"] == 6
    assert counts["CZ"] == 3
    assert counts["minus_one"] == (theory.wire_dimension % 4 == 3)


@pytest.mark.parametrize("theory_path", ALL_THEORY_PATHS, ids=lambda path: path.stem)
def test_swap_decomposition_is_rediscovered_by_counting(theory_path: Path):
    theory = load_theory(theory_path)
    report = search_theory(
        theory,
        strategies=("counting",),
        equation_ids={"SWAP-decomp"},
        max_modulus=2,
        timeout=5,
    )

    assert report.witnesses["SWAP-decomp"].parameters == {
        "modulus": 2,
        "weights": {},
        "swap": 1,
    }


@pytest.mark.parametrize("theory_path", ALL_THEORY_PATHS, ids=lambda path: path.stem)
def test_c18_is_rediscovered_by_block_normal_form(theory_path: Path):
    theory = load_theory(theory_path)
    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"C18"},
        timeout=10,
    )

    witness = report.witnesses["C18"]
    assert report.unresolved == ("C18",)
    assert not witness.certified
    assert witness.lhs_value["block_count"] == 3
    assert witness.rhs_value["block_count"] == 5
    assert witness.parameters["wire_dimension"] == theory.wire_dimension
    assert "one-wire unitaries" in witness.parameters["local_equivalence"]


def test_c3_keeps_its_only_nontrivial_d3_instance():
    theory = load_theory(THEORY_PATH)
    equation = theory.equation("C3_k2")
    expected = Circuit.from_json(
        {
            "wires": 1,
            "ops": [
                {"macro": "M_g", "on": [0]},
                {"macro": "M_g", "on": [0]},
            ],
        },
        theory.signature,
        theory.macros,
    )
    assert equation.lhs == expected
    assert equation.rhs == Circuit.identity(1)


@pytest.mark.parametrize("theory_path", HIGHER_DIMENSION_PATHS, ids=lambda path: path.stem)
def test_higher_dimension_rule_inventory(theory_path: Path):
    theory = load_theory(theory_path)
    dimension = theory.wire_dimension
    expected = RETAINED[:4] + EXTRA_C3[dimension] + RETAINED[4:]
    assert [equation.id for equation in theory.equations] == expected
    assert theory.equation(f"C3_k{dimension - 1}").rhs.kind == "id"
