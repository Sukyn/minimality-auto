from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minimality_auto.core import (
    Circuit,
    evaluate_matrix,
    load_theory,
    primitive_occurrences,
    structural_permutation_parity,
)
from minimality_auto.search import CandidateModel, Deadline, search_theory, verify
from minimality_auto.separators import determinant
from minimality_auto.separators.substitution import _projectively_equal


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


def _separator_cases():
    cases = []
    for path in PAPER_THEORIES:
        theory = load_theory(path)
        groups: dict[str, set[str]] = {}
        for equation in theory.equations:
            expected = equation.metadata.get("expected_separator")
            if expected:
                groups.setdefault(str(expected["strategy"]), set()).add(equation.id)
        cases.extend((path, strategy, ids) for strategy, ids in groups.items())
    return cases


def _equation_cases():
    cases = []
    for path in PAPER_THEORIES:
        for equation in load_theory(path).equations:
            if equation.metadata.get("expected_separator"):
                cases.append((path, equation.id))
    return cases


def _published_model(theory, equation, expected):
    strategy = expected["strategy"]
    if strategy == "presence":
        selected = frozenset(expected["generators"])
        return CandidateModel(
            kind="presence",
            description="published occurrence detector",
            parameters=expected,
            evaluator=lambda term: any(
                primitive_occurrences(term, theory.macros).get(name, 0)
                for name in selected
            ),
        )
    if strategy == "counting":
        modulus = int(expected["modulus"])
        weights = dict(expected["weights"])
        swap = int(expected.get("swap", 0))

        def evaluate(term):
            counts = primitive_occurrences(term, theory.macros)
            value = sum(counts.get(name, 0) * weight for name, weight in weights.items())
            value += structural_permutation_parity(term, theory.macros) * swap
            return value % modulus

        return CandidateModel(
            kind="counting",
            description="published modular count",
            parameters=expected,
            evaluator=evaluate,
        )
    if strategy == "substitution":
        generator = theory.signature[expected["generator"]]
        replacement = Circuit.from_json(
            expected["replacement"], theory.signature, theory.macros
        )
        replacement_matrix = evaluate_matrix(
            replacement, theory.signature, theory.macros, theory.wire_dimension
        )
        override = {generator.name: replacement_matrix}
        return CandidateModel(
            kind="substitution",
            description="published projective substitution",
            parameters=expected,
            evaluator=lambda term: evaluate_matrix(
                term,
                theory.signature,
                theory.macros,
                theory.wire_dimension,
                overrides=override,
            ),
            equality=_projectively_equal,
        )
    if strategy == "determinant":
        return next(
            determinant.candidates(
                theory,
                equation,
                bound=int(expected["k"]),
                deadline=Deadline.after(10),
            )
        )
    raise AssertionError(f"unknown published strategy {strategy!r}")


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
    ("theory_path", "strategy", "equation_ids"),
    _separator_cases(),
    ids=lambda value: value.stem if isinstance(value, Path) else None,
)
def test_published_separator_families_are_found(
    theory_path: Path, strategy: str, equation_ids: set[str]
):
    theory = load_theory(theory_path)
    report = search_theory(
        theory,
        strategies=(strategy,),
        equation_ids=equation_ids,
        timeout=120,
    )
    assert not report.timed_out, f"{theory_path.name}: {strategy} timed out"
    assert not report.unresolved, (
        f"{theory_path.name}: {strategy} did not separate "
        + ", ".join(report.unresolved)
    )


@pytest.mark.parametrize(
    ("theory_path", "equation_id"),
    _equation_cases(),
    ids=lambda value: value.stem if isinstance(value, Path) else value,
)
def test_exact_published_separator_verifies(theory_path: Path, equation_id: str):
    theory = load_theory(theory_path)
    equation = theory.equation(equation_id)
    expected = equation.metadata["expected_separator"]
    model = _published_model(theory, equation, expected)
    assert verify(model, theory, equation, Deadline.after(30)) is not None


def test_real_cf_metadata_preserves_the_unsound_printed_rhs():
    theory = load_theory(THEORY_DIR / "real_clifford.json")
    equation = theory.equation("CF")
    printed_rhs = Circuit.from_json(
        equation.metadata["printed_rhs"], theory.signature, theory.macros
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
