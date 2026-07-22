from __future__ import annotations

import numpy as np
import pytest

from minimality_auto.core import Circuit, Equation, MacroDef, Signature, Theory
from minimality_auto.search import (
    CandidateModel,
    Deadline,
    relevant_equations,
    SearchTimeout,
    search_theory,
    verify,
)


def _presence_theory() -> Theory:
    return Theory.from_json(
        {
            "name": "presence_example",
            "generators": {"a": [1, 1], "b": [1, 1]},
            "equations": [
                {"id": "drop_a", "lhs": {"gen": "a"}, "rhs": {"id": 1}},
                {
                    "id": "commute",
                    "lhs": {"compose": [{"gen": "a"}, {"gen": "b"}]},
                    "rhs": {"compose": [{"gen": "b"}, {"gen": "a"}]},
                },
            ],
        }
    )


def test_presence_finds_exact_boolean_max_model():
    theory = _presence_theory()
    report = search_theory(
        theory, strategies=("presence",), equation_ids={"drop_a"}, timeout=2
    )
    witness = report.witnesses["drop_a"]
    assert witness.strategy == "presence"
    assert witness.parameters == {"generators": ["a"]}
    assert witness.checked_equations == ("commute",)


def test_auto_prefers_a_small_presence_witness_to_identity_substitution():
    theory = Theory.from_json(
        {
            "name": "auto_order",
            "generators": [
                {"id": "a", "inputs": 1, "outputs": 1, "matrix": [[0, 1], [1, 0]]}
            ],
            "equations": [{"id": "drop_a", "lhs": {"gen": "a"}, "rhs": {"id": 1}}],
        }
    )
    report = search_theory(theory, equation_ids={"drop_a"}, timeout=2)
    assert report.witnesses["drop_a"].strategy == "presence"


def test_counting_can_assign_structural_swap_parity():
    theory = Theory.from_json(
        {
            "name": "swap_example",
            "generators": {"u": [1, 1]},
            "equations": [
                {"id": "swap", "lhs": {"perm": [1, 0]}, "rhs": {"id": 2}},
                {
                    "id": "u2",
                    "lhs": {"compose": [{"gen": "u"}, {"gen": "u"}]},
                    "rhs": {"id": 1},
                },
            ],
        }
    )
    report = search_theory(
        theory, strategies=("counting",), equation_ids={"swap"}, max_modulus=2, timeout=2
    )
    witness = report.witnesses["swap"]
    assert witness.parameters["modulus"] == 2
    assert witness.parameters["swap"] == 1


def test_central_verifier_rejects_model_breaking_another_axiom():
    theory = _presence_theory()
    target = theory.equation("drop_a")
    commute_lhs = theory.equation("commute").lhs
    model = CandidateModel(
        kind="bad",
        description="deliberately bad",
        parameters={},
        evaluator=lambda term: term in (target.lhs, commute_lhs),
    )
    assert verify(model, theory, target, Deadline.after(1)) is None


def test_arity_changing_signature_disables_the_arity_shortcut():
    theory = Theory.from_json(
        {
            "generators": {"a": [1, 1], "grow": [1, 2], "shrink": [2, 1]},
            "equations": [
                {"id": "target", "lhs": {"gen": "a"}, "rhs": {"id": 1}},
                {
                    "id": "higher",
                    "lhs": {"tensor": [{"gen": "a"}, {"id": 1}]},
                    "rhs": {"id": 2},
                },
            ],
        }
    )
    target = theory.equation("target")
    model = CandidateModel(
        kind="bad",
        description="breaks the higher-arity axiom",
        parameters={},
        evaluator=lambda term: term.kind != "id",
    )
    assert verify(model, theory, target, Deadline.after(1)) is None


def test_undeclared_arity_changing_nodes_disable_the_arity_shortcut():
    grow = Circuit.generator("grow", 1, 2)
    shrink = Circuit.generator("shrink", 2, 1)
    wide_left = Circuit.generator("wide_left", 2, 2)
    wide_right = Circuit.generator("wide_right", 2, 2)
    higher = Equation("higher", wide_left, wide_right)
    target = Equation(
        "target",
        Circuit.compose((grow, wide_left, shrink)),
        Circuit.compose((grow, wide_right, shrink)),
    )
    theory = Theory(Signature(), (higher, target))

    assert relevant_equations(theory, target) == [higher]


def test_arity_changing_nodes_hidden_in_macros_disable_the_arity_shortcut():
    grow = Circuit.generator("grow", 1, 2)
    shrink = Circuit.generator("shrink", 2, 1)
    macro = MacroDef("round_trip", Circuit.compose((grow, shrink)))
    higher = Equation(
        "higher",
        Circuit.generator("wide_left", 2, 2),
        Circuit.generator("wide_right", 2, 2),
    )
    target = Equation(
        "target", Circuit.macro("round_trip", 1, 1), Circuit.identity(1)
    )
    theory = Theory(Signature(), (higher, target), {macro.name: macro})

    assert relevant_equations(theory, target) == [higher]


def test_external_arity_changing_target_checks_all_higher_arity_rules():
    retained = Equation(
        "retained",
        Circuit.generator("wide_left", 2, 2),
        Circuit.generator("wide_right", 2, 2),
    )
    theory = Theory(Signature(), (retained,))
    external = Equation(
        "external",
        Circuit.generator("effect_left", 1, 0),
        Circuit.generator("effect_right", 1, 0),
    )

    assert relevant_equations(theory, external) == [retained]


def test_verifier_checks_the_deadline_after_each_model_operation():
    theory = _presence_theory()
    evaluated: list[Circuit] = []

    class TwoChecks:
        calls = 0

        def check(self):
            self.calls += 1
            if self.calls == 2:
                raise SearchTimeout

    model = CandidateModel(
        kind="probe",
        description="deadline probe",
        parameters={},
        evaluator=lambda term: evaluated.append(term) or 0,
    )

    with pytest.raises(SearchTimeout):
        verify(model, theory, theory.equation("drop_a"), TwoChecks())
    assert len(evaluated) == 1


def test_projective_substitution_search():
    x = [[0, 1], [1, 0]]
    z = [[1, 0], [0, -1]]
    h = (np.asarray([[1, 1], [1, -1]]) / np.sqrt(2)).tolist()
    theory = Theory.from_json(
        {
            "name": "substitution_example",
            "generators": [
                {"id": "X", "inputs": 1, "outputs": 1, "matrix": x},
                {"id": "Z", "inputs": 1, "outputs": 1, "matrix": z},
                {"id": "H", "inputs": 1, "outputs": 1, "matrix": h},
            ],
            "equations": [
                {
                    "id": "projective_commutation",
                    "lhs": {"compose": [{"gen": "X"}, {"gen": "Z"}]},
                    "rhs": {"compose": [{"gen": "Z"}, {"gen": "X"}]},
                }
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("substitution",),
        equation_ids={"projective_commutation"},
        max_depth=1,
        timeout=5,
    )
    assert report.witnesses["projective_commutation"].strategy == "substitution"


def test_scaled_determinant_separates_three_wire_interaction():
    cnot = np.zeros((4, 4)).tolist()
    for x in (0, 1):
        for y in (0, 1):
            cnot[2 * x + (x ^ y)][2 * x + y] = 1
    theory = Theory.from_json(
        {
            "name": "determinant_example",
            "generators": [
                {"id": "CNOT", "inputs": 2, "outputs": 2, "matrix": cnot}
            ],
            "equations": [
                {
                    "id": "I",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "CNOT", "on": [0, 1]},
                            {"gen": "CNOT", "on": [0, 1]},
                            {"gen": "CNOT", "on": [0, 1]},
                        ],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "CNOT", "on": [0, 1]},
                            {"gen": "CNOT", "on": [0, 1]},
                        ],
                    },
                }
            ],
        }
    )
    report = search_theory(
        theory, strategies=("determinant",), equation_ids={"I"}, timeout=2
    )
    witness = report.witnesses["I"]
    assert witness.parameters["k"] == 2


def test_zero_timeout_is_cooperative():
    report = search_theory(_presence_theory(), strategies=("presence",), timeout=0)
    assert report.timed_out
    assert set(report.unresolved) == {"drop_a", "commute"}

