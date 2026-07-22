from __future__ import annotations

from itertools import permutations
from pathlib import Path

import pytest

from minimality_auto.core import Circuit, Theory, load_theory
from minimality_auto.search import Deadline, NotApplicable, search_theory
from minimality_auto.separators import finite_model


THEORY_DIR = Path(__file__).parents[1] / "theories"
REAL_CH = THEORY_DIR / "realclifford-ch-simplified.json"


def _inverse(order: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(order)
    for output, source in enumerate(order):
        result[source] = output
    return tuple(result)


def _place(term: Circuit, selected: tuple[int, ...], total: int) -> Circuit:
    rest = tuple(wire for wire in range(total) if wire not in selected)
    order = selected + rest
    return Circuit.compose(
        (
            Circuit.perm(order),
            Circuit.tensor((term, Circuit.identity(total - term.inputs))),
            Circuit.perm(_inverse(order)),
        )
    )


def _candidate(theory: Theory, target: str, degree: int):
    return next(
        finite_model.candidates(
            theory,
            theory.equation(target),
            bound=degree,
            deadline=Deadline.after(5),
        )
    )


def test_generic_search_rediscovers_the_real_ch_witnesses():
    theory = load_theory(REAL_CH)
    expected = {"1", "2", "3", "5", "6", "7", "8", "10", "15", "18"}
    report = search_theory(
        theory,
        strategies=("finite_model",),
        equation_ids=expected,
        max_permutation_degree=4,
        timeout=10,
    )

    assert set(report.witnesses) == expected
    assert not report.unresolved
    assert {name: witness.parameters["degree"] for name, witness in report.witnesses.items()} == {
        "1": 3,
        "2": 3,
        "3": 3,
        "5": 4,
        "6": 2,
        "7": 2,
        "8": 2,
        "10": 3,
        "15": 2,
        "18": 2,
    }
    for witness in report.witnesses.values():
        assert set(witness.parameters) == {"arity", "degree", "interpretation"}


def test_report_contains_the_actual_named_interpretation():
    theory = load_theory(REAL_CH)
    report = search_theory(
        theory,
        strategies=("finite_model",),
        equation_ids={"15", "18"},
        max_permutation_degree=2,
        timeout=5,
    )

    assert report.witnesses["15"].parameters["interpretation"] == {
        "generators": {
            "H": [0, 1],
            "Z": [0, 1],
            "CZ": [1, 0],
            "CH": [1, 0],
        },
        "structural_swaps": {
            "swap[0,1]": [1, 0],
            "swap[1,2]": [1, 0],
        },
    }
    assert report.witnesses["18"].parameters["interpretation"] == {
        "generators": {
            "H": [0, 1],
            "Z": [0, 1],
            "CZ": [0, 1],
            "CH": [1, 0],
        },
        "structural_swaps": {
            "swap[0,1]": [0, 1],
            "swap[1,2]": [0, 1],
        },
    }


def test_rules_12_through_18_have_only_two_witnesses_through_degree_five():
    theory = load_theory(REAL_CH)
    targets = {str(number) for number in range(12, 19)}
    report = search_theory(
        theory,
        strategies=("finite_model",),
        equation_ids=targets,
        max_permutation_degree=5,
        timeout=10,
    )

    assert set(report.witnesses) == {"15", "18"}
    assert set(report.unresolved) == {"12", "13", "14", "16", "17"}
    assert not report.timed_out


def test_search_is_independent_of_generator_names_and_equation_ids():
    pair = lambda left, right: {
        "compose": [{"gen": left}, {"gen": right}] * 8
    }
    theory = Theory.from_json(
        {
            "name": "renamed_one_wire_fragment",
            "generators": {"alpha": [1, 1], "beta": [1, 1]},
            "equations": [
                {
                    "id": "alpha_square",
                    "lhs": {"compose": [{"gen": "alpha"}, {"gen": "alpha"}]},
                    "rhs": {"id": 1},
                },
                {
                    "id": "beta_square",
                    "lhs": {"compose": [{"gen": "beta"}, {"gen": "beta"}]},
                    "rhs": {"id": 1},
                },
                {"id": "octagon", "lhs": pair("alpha", "beta"), "rhs": {"id": 1}},
            ],
        }
    )

    report = search_theory(
        theory,
        strategies=("finite_model",),
        max_permutation_degree=3,
        timeout=5,
    )
    assert set(report.witnesses) == {"alpha_square", "beta_square", "octagon"}
    for witness in report.witnesses.values():
        assert set(witness.parameters["interpretation"]["generators"]) == {
            "alpha",
            "beta",
        }


def test_generator_names_cannot_collide_with_structural_output_names():
    name = "swap[0,1]"
    theory = Theory.from_json(
        {
            "generators": {name: [2, 2]},
            "equations": [
                {
                    "id": "square",
                    "lhs": {"compose": [{"gen": name}, {"gen": name}]},
                    "rhs": {"id": 2},
                }
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("finite_model",),
        max_permutation_degree=3,
        timeout=2,
    )
    interpretation = report.witnesses["square"].parameters["interpretation"]

    assert name in interpretation["generators"]
    assert name in interpretation["structural_swaps"]


def test_generic_search_supports_scalars_in_an_unrelated_fragment():
    theory = load_theory(THEORY_DIR / "qubit_clifford.json")
    report = search_theory(
        theory,
        strategies=("finite_model",),
        equation_ids={"omega8"},
        max_permutation_degree=3,
        timeout=5,
    )
    witness = report.witnesses["omega8"]
    assert witness.parameters == {
        "arity": 0,
        "degree": 3,
        "interpretation": {
            "generators": {"omega": [1, 2, 0]},
            "structural_swaps": {},
        },
    }


def test_permutation_degree_is_increased_step_by_step():
    theory = Theory.from_json(
        {
            "generators": {"turn": [1, 1]},
            "equations": [
                {
                    "id": "turn2",
                    "lhs": {"compose": [{"gen": "turn"}, {"gen": "turn"}]},
                    "rhs": {"id": 1},
                }
            ],
        }
    )
    too_small = search_theory(
        theory,
        strategies=("finite_model",),
        max_permutation_degree=2,
        timeout=2,
    )
    enough = search_theory(
        theory,
        strategies=("finite_model",),
        max_permutation_degree=3,
        timeout=2,
    )

    assert too_small.unresolved == ("turn2",)
    assert enough.witnesses["turn2"].parameters["degree"] == 3


def test_structural_permutation_compiler_uses_the_core_convention():
    degree = 4
    assignment = {}
    for index in range(degree - 1):
        value = list(range(degree))
        value[index], value[index + 1] = value[index + 1], value[index]
        assignment[finite_model._swap(index)] = tuple(value)

    for order in permutations(range(degree)):
        word = finite_model._adjacent_word(order)
        assert finite_model._evaluate(word, assignment, degree) == _inverse(order)


def test_compact_and_explicit_routing_agree_on_every_three_wire_placement():
    theory = load_theory(REAL_CH)
    model = _candidate(theory, "15", 2)
    for name, generator in theory.signature.items():
        base = Circuit.generator(generator)
        for selected in permutations(range(3), generator.inputs):
            compact = Circuit.from_json(
                {"wires": 3, "ops": [{"gen": name, "on": list(selected)}]},
                theory.signature,
            )
            assert model.evaluate(compact) == model.evaluate(_place(base, selected, 3))


@pytest.mark.parametrize("target", ["15", "18"])
def test_three_wire_models_validate_every_ordered_axiom_embedding(target: str):
    theory = load_theory(REAL_CH)
    model = _candidate(theory, target, 2)
    for equation in theory.equations:
        if equation.lhs.inputs > 3:
            continue
        equalities = [
            model.equal(
                model.evaluate(_place(equation.lhs, selected, 3)),
                model.evaluate(_place(equation.rhs, selected, 3)),
            )
            for selected in permutations(range(3), equation.lhs.inputs)
        ]
        assert all(equalities) == (equation.id != target), equation.id


def test_prop_validator_rejects_a_broken_braid_relation():
    identity = (0, 1, 2)
    assignment = {
        finite_model._swap(0): (1, 0, 2),
        finite_model._swap(1): identity,
    }
    with pytest.raises(ValueError, match="required relation"):
        finite_model._validate_prop_assignment(3, (), assignment, 3)


def test_prop_validator_rejects_noncentral_scalars():
    assignment = {
        finite_model._generator("phase"): (1, 2, 0),
        finite_model._generator("gate"): (1, 0, 2),
    }
    with pytest.raises(ValueError, match="required relation"):
        finite_model._validate_prop_assignment(
            1, (("phase", 0), ("gate", 1)), assignment, 3
        )


def test_prop_validator_rejects_broken_disjoint_interchange():
    assignment = {
        finite_model._generator("left"): (1, 2, 0),
        finite_model._generator("right"): (1, 0, 2),
        finite_model._swap(0): (0, 1, 2),
    }
    with pytest.raises(ValueError, match="required relation"):
        finite_model._validate_prop_assignment(
            2, (("left", 1), ("right", 1)), assignment, 3
        )


def test_models_reject_terms_above_their_fixed_arity():
    theory = load_theory(REAL_CH)
    model = _candidate(theory, "5", 4)
    with pytest.raises(NotApplicable, match="through arity 2"):
        model.evaluate(Circuit.identity(3))


def test_arity_changing_signatures_are_rejected():
    theory = Theory.from_json(
        {
            "generators": {"gate": [1, 1], "grow": [1, 2]},
            "equations": [
                {"id": "drop", "lhs": {"gen": "gate"}, "rhs": {"id": 1}}
            ],
        }
    )
    with pytest.raises(NotApplicable, match="endomorphic signature"):
        next(
            finite_model.candidates(
                theory,
                theory.equation("drop"),
                bound=2,
                deadline=Deadline.after(1),
            )
        )
