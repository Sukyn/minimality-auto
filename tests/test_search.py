from __future__ import annotations

import numpy as np
import pytest

import minimality_auto.search as search_module
from minimality_auto.core import Circuit, Equation, Signature, Theory
from minimality_auto.search import (
    CandidateModel,
    Deadline,
    DirectCertificate,
    SearchTimeout,
    relevant_equations,
    search_theory,
    verify,
)
from minimality_auto.separators import amalgam, presence


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


def test_direct_certificates_are_uncertified_unless_explicitly_promoted():
    certificate = DirectCertificate("numerical", {}, 0, 1)

    assert not certificate.certified


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
    macro = Circuit.compose((grow, shrink))
    higher = Equation(
        "higher",
        Circuit.generator("wide_left", 2, 2),
        Circuit.generator("wide_right", 2, 2),
    )
    target = Equation(
        "target", Circuit.macro("round_trip", 1, 1), Circuit.identity(1)
    )
    theory = Theory(Signature(), (higher, target), {"round_trip": macro})

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
    witness = report.witnesses["projective_commutation"]
    assert witness.strategy == "substitution"
    assert not witness.certified
    assert witness.description.startswith("numerical ")


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
    assert not witness.certified
    assert witness.description.startswith("numerical ")


def test_zero_timeout_is_cooperative():
    report = search_theory(_presence_theory(), strategies=("presence",), timeout=0)
    assert report.timed_out
    assert set(report.unresolved) == {"drop_a", "commute"}


def test_empty_search_does_not_time_out():
    report = search_theory(
        _presence_theory(),
        strategies=("presence",),
        equation_ids=set(),
        timeout=0,
    )
    assert not report.timed_out
    assert report.unresolved == ()


def test_empty_strategy_set_returns_without_a_timeout():
    report = search_theory(
        _presence_theory(),
        strategies=(),
        timeout=0,
    )

    assert not report.timed_out
    assert set(report.unresolved) == {"drop_a", "commute"}


def test_max_arity_includes_outputs():
    theory = Theory.from_json(
        {
            "generators": {"left": [1, 3], "right": [1, 3]},
            "equations": [
                {
                    "id": "grow",
                    "lhs": {"gen": "left"},
                    "rhs": {"gen": "right"},
                }
            ],
        }
    )
    report = search_theory(
        theory, strategies=("presence",), max_arity=2, timeout=0
    )
    assert report.unresolved == ()
    assert not report.timed_out


def test_not_applicable_candidate_does_not_discard_later_candidates(monkeypatch):
    def candidates(*_, **__):
        def reject(_term):
            raise search_module.NotApplicable("candidate-specific")

        yield CandidateModel("presence", "bad", {}, reject, key="shared")
        yield CandidateModel(
            "presence",
            "good",
            {},
            lambda term: term.kind == "gen",
            key="shared",
        )

    monkeypatch.setattr(presence, "candidates", candidates)
    report = search_theory(
        _presence_theory(),
        strategies=("presence",),
        equation_ids={"drop_a"},
        timeout=2,
    )
    assert report.witnesses["drop_a"].description == "good"


def test_models_without_keys_are_not_deduplicated_by_description(monkeypatch):
    def candidates(*_, **__):
        yield CandidateModel("presence", "same", {}, lambda _term: 0)
        yield CandidateModel(
            "presence", "same", {}, lambda term: term.kind == "gen"
        )

    monkeypatch.setattr(presence, "candidates", candidates)
    report = search_theory(
        _presence_theory(),
        strategies=("presence",),
        equation_ids={"drop_a"},
        timeout=2,
    )
    assert "drop_a" in report.witnesses


def test_search_context_computes_endomorphic_status_once(monkeypatch):
    calls = 0
    original = search_module._theory_is_endomorphic

    def counted(theory, deadline=None):
        nonlocal calls
        calls += 1
        return original(theory, deadline)

    monkeypatch.setattr(search_module, "_theory_is_endomorphic", counted)
    search_theory(_presence_theory(), strategies=("presence",), timeout=2)
    assert calls == 1


def test_search_context_reuses_term_features_across_candidates(monkeypatch):
    occurrence_calls = 0
    parity_calls = 0
    original_occurrences = search_module.primitive_occurrences
    original_parity = search_module.structural_permutation_parity

    def counted_occurrences(*args, **kwargs):
        nonlocal occurrence_calls
        occurrence_calls += 1
        return original_occurrences(*args, **kwargs)

    def counted_parity(*args, **kwargs):
        nonlocal parity_calls
        parity_calls += 1
        return original_parity(*args, **kwargs)

    monkeypatch.setattr(search_module, "primitive_occurrences", counted_occurrences)
    monkeypatch.setattr(search_module, "structural_permutation_parity", counted_parity)
    report = search_theory(
        _presence_theory(),
        strategies=("counting",),
        equation_ids={"drop_a"},
        max_modulus=2,
        timeout=2,
    )

    assert "drop_a" in report.witnesses
    assert occurrence_calls == 4
    assert parity_calls == 4


def test_cheap_strategy_reaches_every_target_before_later_strategy(monkeypatch):
    visited: list[str] = []

    def cheap(_theory, target, **_kwargs):
        visited.append(target.id)
        return iter(())

    def expensive(*_args, **_kwargs):
        raise SearchTimeout

    monkeypatch.setattr(presence, "candidates", cheap)
    monkeypatch.setattr(amalgam, "candidates", expensive)
    report = search_theory(
        _presence_theory(),
        strategies=("presence", "amalgam"),
        timeout=2,
    )

    assert report.timed_out
    assert visited[:2] == ["drop_a", "commute"]


def test_exact_candidate_replaces_numerical_candidate_with_the_same_key(monkeypatch):
    def candidates(*_, **__):
        def evaluate(term):
            return term.kind == "gen"

        yield CandidateModel(
            "presence", "numerical", {}, evaluate, key="same", certified=False
        )
        yield CandidateModel(
            "presence", "exact", {}, evaluate, key="same", certified=True
        )

    monkeypatch.setattr(presence, "candidates", candidates)
    report = search_theory(
        _presence_theory(),
        strategies=("presence",),
        equation_ids={"drop_a"},
        timeout=2,
    )
    assert report.witnesses["drop_a"].description == "exact"
    assert report.witnesses["drop_a"].certified
    assert report.unresolved == ()


def test_numerical_candidate_is_not_a_proof(monkeypatch):
    def candidates(*_, **__):
        yield CandidateModel(
            "presence",
            "numerical",
            {},
            lambda term: term.kind == "gen",
            certified=False,
        )

    monkeypatch.setattr(presence, "candidates", candidates)
    report = search_theory(
        _presence_theory(),
        strategies=("presence",),
        equation_ids={"drop_a"},
        timeout=2,
    )
    assert not report.witnesses["drop_a"].certified
    assert report.unresolved == ("drop_a",)
