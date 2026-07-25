from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minimality_auto.core import Circuit, load_theory
from minimality_auto.matrix import evaluate_matrix
from minimality_auto.search import search_theory

THEORY_DIR = Path(__file__).parents[1] / "theories"
PAPER_THEORIES = [
    THEORY_DIR / name
    for name in (
        "qubit_clifford.json",
        "real_clifford.json",
        "qutrit_clifford.json",
        "clifford_t.json",
        "clifford_cs.json",
        "cnot_dihedral.json",
    )
]
SEPARATOR_CASES = [
    ("qubit_clifford", "presence", {"omega8", "H2", "S4", "CPh"}),
    ("qubit_clifford", "counting", {"E", "B", "CZ"}),
    ("qubit_clifford", "determinant", {"I"}),
    ("real_clifford", "presence", {"minus2", "H2", "Z2", "CX2"}),
    ("real_clifford", "counting", {"F", "B", "ZC"}),
    ("real_clifford", "substitution", {"CZr", "CF"}),
    ("real_clifford", "determinant", {"I"}),
    ("qutrit_clifford", "presence", {"omega12", "H4", "S3", "CPh"}),
    ("qutrit_clifford", "counting", {"E", "KC", "CZ", "B"}),
    ("qutrit_clifford", "substitution", {"SSprime"}),
    ("clifford_t", "presence", {"omega8", "T8", "H2", "CPh"}),
    ("clifford_t", "counting", {"E", "TX", "B", "CZ"}),
    ("clifford_cs", "presence", {"omega8", "H2", "S4", "CPh"}),
    ("clifford_cs", "counting", {"E", "B", "XCS"}),
    ("clifford_cs", "substitution", {"CSr", "CE"}),
    ("clifford_cs", "determinant", {"I"}),
    ("cnot_dihedral", "presence", {"omega8", "X2", "T8", "CPh"}),
    ("cnot_dihedral", "counting", {"TX", "XC", "B", "ZC", "I", "C2T"}),
    ("cnot_dihedral", "determinant", {"C3T"}),
]
CASE_LIMITS = {
    ("qubit_clifford", "counting"): {"max_modulus": 2},
    ("real_clifford", "counting"): {"max_modulus": 2},
    ("qutrit_clifford", "counting"): {"max_modulus": 3},
    ("clifford_t", "counting"): {"max_modulus": 2},
    ("clifford_cs", "counting"): {"max_modulus": 2},
    ("real_clifford", "substitution"): {"max_depth": 0},
    ("qutrit_clifford", "substitution"): {"max_depth": 2},
}


@pytest.mark.parametrize("theory_path", PAPER_THEORIES, ids=lambda path: path.stem)
def test_imported_paper_equations_are_sound_in_the_supplied_matrices(theory_path: Path):
    theory = load_theory(theory_path)
    failures: list[str] = []
    for equation in theory.equations:
        left = evaluate_matrix(
            equation.lhs, theory.signature, theory.macros, theory.wire_dimension
        )
        right = evaluate_matrix(
            equation.rhs, theory.signature, theory.macros, theory.wire_dimension
        )
        if not np.allclose(left, right, atol=1e-8, rtol=1e-8):
            failures.append(
                f"{equation.id} (max error {float(np.max(np.abs(left - right))):.6g})"
            )
    assert not failures, f"{theory_path.name}: " + ", ".join(failures)


@pytest.mark.parametrize(
    ("theory_name", "strategy", "equation_ids"),
    SEPARATOR_CASES,
    ids=[f"{theory}-{strategy}" for theory, strategy, _ in SEPARATOR_CASES],
)
@pytest.mark.integration
def test_published_separator_families_are_found(
    theory_name: str, strategy: str, equation_ids: set[str]
):
    theory_path = THEORY_DIR / f"{theory_name}.json"
    theory = load_theory(theory_path)
    report = search_theory(
        theory,
        strategies=(strategy,),
        equation_ids=equation_ids,
        timeout=20,
        **CASE_LIMITS.get((theory_name, strategy), {}),
    )
    assert not report.timed_out, f"{theory_path.name}: {strategy} timed out"
    if strategy in {"determinant", "substitution"}:
        assert set(report.witnesses) == equation_ids
        assert set(report.unresolved) == equation_ids
        assert all(not witness.certified for witness in report.witnesses.values())
    else:
        assert not report.unresolved, (
            f"{theory_path.name}: {strategy} did not separate "
            + ", ".join(report.unresolved)
        )


def test_real_cf_printed_rhs_is_unsound():
    theory = load_theory(THEORY_DIR / "real_clifford.json")
    equation = theory.equation("CF")
    printed_rhs = Circuit.from_json(
        {
            "wires": 2,
            "ops": [
                {"gen": "H", "on": [1]},
                {"gen": "Z", "on": [1]},
                {"gen": "H", "on": [1]},
                {"gen": "CNOT", "on": [0, 1]},
            ],
        },
        theory.signature,
        theory.macros,
    )
    left = evaluate_matrix(
        equation.lhs, theory.signature, theory.macros, theory.wire_dimension
    )
    printed = evaluate_matrix(
        printed_rhs, theory.signature, theory.macros, theory.wire_dimension
    )
    assert not np.allclose(left, printed, atol=1e-8, rtol=1e-8)


def test_qutrit_e_records_the_global_phase_conflict_in_the_prose_matrix():
    theory = load_theory(THEORY_DIR / "qutrit_clifford.json")
    equation = theory.equation("E")
    positive_fourier = -np.asarray(theory.signature["H"].matrix)
    override = {"H": positive_fourier}
    left = evaluate_matrix(
        equation.lhs,
        theory.signature,
        theory.macros,
        theory.wire_dimension,
        overrides=override,
    )
    right = evaluate_matrix(
        equation.rhs,
        theory.signature,
        theory.macros,
        theory.wire_dimension,
        overrides=override,
    )
    np.testing.assert_allclose(left, -right, atol=1e-8, rtol=1e-8)
