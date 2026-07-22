import cmath
import math
from itertools import permutations

import numpy as np

from minimality_auto.core import Circuit, Equation, Signature, Theory
from minimality_auto.search import Deadline, search_theory
from minimality_auto.separators import counting, determinant, substitution


def test_presence_handles_arity_changes_and_inline_primitive_names():
    theory = Theory.from_json(
        {
            "generators": {"unused": [4, 0], "unit": [0, 1], "counit": [1, 0]},
            "equations": [
                {
                    "id": "loop",
                    "lhs": {"compose": [{"gen": "unit"}, {"gen": "counit"}]},
                    "rhs": {"id": 0},
                }
            ],
        }
    )
    inline_theory = Theory(
        Signature.from_json({"unused": [4, 0]}),
        (
            Equation(
                "inline",
                Circuit.generator("ghost", 1, 1),
                Circuit.identity(1),
            ),
        ),
    )
    loop = search_theory(
        theory, strategies=("presence",), equation_ids={"loop"}, timeout=2
    )
    inline = search_theory(
        inline_theory,
        strategies=("presence",),
        equation_ids={"inline"},
        timeout=2,
    )

    assert loop.witnesses["loop"].parameters["generators"] in (["unit"], ["counit"])
    assert inline.witnesses["inline"].parameters == {"generators": ["ghost"]}


def test_counting_disables_swap_sign_when_a_generator_changes_wire_parity():
    theory = Theory.from_json(
        {
            "generators": {"unit": [0, 1]},
            "equations": [
                {"id": "swap", "lhs": {"perm": [1, 0]}, "rhs": {"id": 2}}
            ],
        }
    )
    models = list(
        counting.candidates(
            theory,
            theory.equation("swap"),
            bound=2,
            deadline=Deadline.after(2),
            max_modulus=2,
        )
    )
    report = search_theory(
        theory,
        strategies=("counting",),
        equation_ids={"swap"},
        max_modulus=2,
        timeout=2,
    )

    assert all(model.parameters["swap"] == 0 for model in models)
    assert report.unresolved == ("swap",)


def test_counting_swap_sign_supports_parity_preserving_arity_changes():
    theory = Theory.from_json(
        {
            "generators": {"pair": [0, 2]},
            "equations": [
                {"id": "swap", "lhs": {"perm": [1, 0]}, "rhs": {"id": 2}}
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("counting",),
        equation_ids={"swap"},
        max_modulus=2,
        timeout=2,
    )

    assert report.witnesses["swap"].parameters["swap"] == 1


def test_wire_parity_check_is_iterative_for_deep_programmatic_terms():
    theory = Theory.from_json({"generators": {}, "equations": []})

    def nest(term, depth):
        for index in range(depth):
            term = (
                Circuit.tensor((term, Circuit.identity(0)))
                if index % 2 == 0
                else Circuit.compose((term, Circuit.identity(term.outputs)))
            )
        return term

    preserving = nest(Circuit.generator("inline", 1, 1), 1_500)
    changing = nest(Circuit.generator("inline grow", 1, 2), 1_500)

    assert counting._wire_parity_is_preserved(theory, preserving)
    assert not counting._wire_parity_is_preserved(theory, changing)


def test_swap_phase_checks_an_external_target_for_inline_arity_changes():
    theory = Theory.from_json(
        {
            "generators": {"gate": [1, 1]},
            "equations": [
                {"id": "gate", "lhs": {"gen": "gate"}, "rhs": {"gen": "gate"}}
            ],
        }
    )
    grow = Circuit.generator("inline grow", 1, 2)
    target = Equation(
        "external target",
        Circuit.compose((grow, Circuit.perm((1, 0)))),
        grow,
    )

    counting_models = list(
        counting.candidates(
            theory,
            target,
            bound=2,
            deadline=Deadline.after(2),
            max_modulus=2,
        )
    )
    determinant_models = list(
        determinant.candidates(
            theory,
            target,
            bound=2,
            deadline=Deadline.after(2),
        )
    )

    assert all(model.parameters["swap"] == 0 for model in counting_models)
    assert determinant_models == []


def test_determinant_generalizes_to_qutrits_and_scalars():
    phase = cmath.exp(1j * math.pi / 6)
    phase_theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {
                "qutrit_phase": {
                    "type": [1, 1],
                    "matrix": [[phase, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            },
            "equations": [
                {"id": "phase", "lhs": {"gen": "qutrit_phase"}, "rhs": {"id": 1}}
            ],
        }
    )
    scalar_theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {"scalar": {"type": [0, 0], "matrix": [[phase]]}},
            "equations": [
                {"id": "scalar", "lhs": {"gen": "scalar"}, "rhs": {"id": 0}}
            ],
        }
    )
    phase_report = search_theory(
        phase_theory, strategies=("determinant",), equation_ids={"phase"}, timeout=2
    )
    scalar_report = search_theory(
        scalar_theory, strategies=("determinant",), equation_ids={"scalar"}, timeout=2
    )

    assert phase_report.witnesses["phase"].parameters["wire_dimension"] == 3
    assert scalar_report.witnesses["scalar"].strategy == "determinant"


def test_determinant_defaults_unsupported_generators_and_preserves_prop_naturality():
    phase = cmath.exp(1j * math.pi / 4)
    theory = Theory.from_json(
        {
            "wire_dimension": 2,
            "generators": {
                "phase": {"type": [1, 1], "matrix": [[phase, 0], [0, 1]]},
                "singular": {"type": [1, 1], "matrix": [[0, 0], [0, 0]]},
                "unit": {"type": [0, 1], "matrix": [[1], [0]]},
            },
            "equations": [
                {"id": "target", "lhs": {"gen": "phase"}, "rhs": {"id": 1}},
                {
                    "id": "singular tautology",
                    "lhs": {"gen": "singular"},
                    "rhs": {"gen": "singular"},
                },
                {
                    "id": "unit tautology",
                    "lhs": {"gen": "unit"},
                    "rhs": {"gen": "unit"},
                },
            ],
        }
    )
    model = next(
        determinant.candidates(
            theory,
            theory.equation("target"),
            bound=2,
            deadline=Deadline.after(2),
        )
    )

    assert set(model.parameters["zero_weight_generators"]) == {"singular", "unit"}
    assert model.parameters["swap_phase_over_pi"] == 0
    assert not model.equal(
        model.evaluate(theory.equation("target").lhs),
        model.evaluate(theory.equation("target").rhs),
    )


def test_substitution_searches_typed_arity_changing_replacements():
    ket_zero = [[1], [0]]
    ket_one = [[0], [1]]
    theory = Theory.from_json(
        {
            "generators": {
                "make": [0, 1],
                "other": {"type": [0, 1], "matrix": ket_zero},
                "third": {"type": [0, 1], "matrix": ket_one},
            },
            "equations": [
                {"id": "target", "lhs": {"gen": "make"}, "rhs": {"gen": "other"}}
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("substitution",),
        equation_ids={"target"},
        max_depth=1,
        timeout=2,
    )

    witness = report.witnesses["target"]
    assert witness.parameters["generator"] == "make"
    assert "third@[]" in witness.parameters["replacement"]


def test_substitution_searches_nullary_scalars():
    theory = Theory.from_json(
        {
            "generators": {
                "s": {"type": [0, 0], "matrix": [[1]]},
                "t": {"type": [0, 0], "matrix": [[2]]},
            },
            "equations": [
                {"id": "target", "lhs": {"gen": "s"}, "rhs": {"id": 0}}
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("substitution",),
        equation_ids={"target"},
        max_depth=1,
        timeout=2,
    )

    assert report.witnesses["target"].parameters["replacement"] == "t@[]"


def test_substitution_ignores_irrelevant_higher_arity_missing_matrices():
    identity = np.eye(2).tolist()
    z = [[1, 0], [0, -1]]
    theory = Theory.from_json(
        {
            "generators": {
                "X": {"type": [1, 1], "matrix": identity},
                "Z": {"type": [1, 1], "matrix": z},
                "large_missing": [3, 3],
            },
            "equations": [
                {
                    "id": "large",
                    "lhs": {"gen": "large_missing"},
                    "rhs": {"id": 3},
                },
                {"id": "target", "lhs": {"gen": "X"}, "rhs": {"id": 1}},
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("substitution",),
        equation_ids={"target"},
        max_depth=1,
        timeout=2,
    )

    assert report.witnesses["target"].parameters["replacement"] == "Z@[0]"
    assert report.witnesses["target"].checked_equations == ()


def test_substitution_can_override_an_inline_typed_generator():
    theory = Theory(
        Signature.from_json(
            {"Z": {"type": [1, 1], "matrix": [[1, 0], [0, -1]]}}
        ),
        (
            Equation(
                "target",
                Circuit.generator("inline", 1, 1),
                Circuit.identity(1),
            ),
        ),
    )
    report = search_theory(
        theory,
        strategies=("substitution",),
        equation_ids={"target"},
        max_depth=1,
        timeout=2,
    )

    witness = report.witnesses["target"]
    assert witness.parameters["generator"] == "inline"
    assert witness.parameters["replacement"] == "Z@[0]"


def test_substitution_word_search_does_not_recurse_with_depth():
    theory = Theory.from_json(
        {
            "generators": {
                "tick": {"type": [0, 0], "matrix": [[1]]},
            },
            "equations": [
                {"id": "target", "lhs": {"gen": "tick"}, "rhs": {"id": 0}}
            ],
        }
    )
    term, label = next(
        substitution._replacement_terms(
            theory,
            0,
            0,
            1_100,
            Deadline.after(5),
            frozenset({"tick"}),
            {},
            1,
        )
    )

    assert term.type == (0, 0)
    assert label.count("tick@[]") == 1_100


def test_substitution_placements_are_lazy_in_the_ambient_wire_count():
    assert list(substitution._placements(4, 2, Deadline.after(2))) == list(
        permutations(range(4), 2)
    )
    theory = Theory.from_json(
        {
            "wire_dimension": 1,
            "generators": {"tick": {"type": [1, 1], "matrix": [[-1]]}},
            "equations": [
                {"id": "target", "lhs": {"gen": "tick"}, "rhs": {"id": 1}}
            ],
        }
    )
    operations = list(
        substitution._operation_alphabet(
            theory,
            1_000_000,
            frozenset({"tick"}),
            {},
            1,
            Deadline.after(2),
        )
    )

    assert operations == [({"gen": "tick", "on": [0]}, 1_000_000)]


def test_substitution_matrix_budget_is_configurable_and_nonfinite_is_rejected():
    theory = Theory.from_json(
        {
            "generators": {
                "X": {"type": [1, 1], "matrix": [[1, 0], [0, 1]]},
                "Z": {"type": [1, 1], "matrix": [[1, 0], [0, -1]]},
            },
            "equations": [
                {"id": "target", "lhs": {"gen": "X"}, "rhs": {"id": 1}}
            ],
        }
    )
    blocked = list(
        substitution.candidates(
            theory,
            theory.equation("target"),
            bound=1,
            deadline=Deadline.after(2),
            max_substitution_matrix_entries=1,
        )
    )
    allowed = list(
        substitution.candidates(
            theory,
            theory.equation("target"),
            bound=1,
            deadline=Deadline.after(2),
            max_substitution_matrix_entries=4,
        )
    )

    assert blocked == []
    assert allowed
    assert not substitution._projectively_equal(
        np.asarray([[np.nan]]), np.asarray([[np.nan]])
    )


def test_projective_matrix_equality_does_not_collapse_small_matrices_to_zero():
    tiny_but_not_zero = np.full((2, 2), 0.75e-8, dtype=complex)
    zero = np.zeros((2, 2), dtype=complex)

    assert not substitution._projectively_equal(tiny_but_not_zero, zero)
    assert not substitution._projectively_equal(zero, tiny_but_not_zero)
    assert substitution._projectively_equal(tiny_but_not_zero, 1j * tiny_but_not_zero)

    visible = tiny_but_not_zero.copy()
    visible[0, 0] = 2e-8
    assert not substitution._projectively_equal(tiny_but_not_zero, visible)


def test_substitution_budget_accounts_for_wires_hidden_inside_macros():
    split = np.zeros((8, 2), dtype=int)
    split[0, 0] = split[1, 1] = 1
    merge = np.zeros((2, 8), dtype=int)
    merge[0, 0], merge[1, 1] = 1, -1
    theory = Theory.from_json(
        {
            "generators": {
                "X": {"type": [1, 1], "matrix": np.eye(2).tolist()},
                "split": {"type": [1, 3], "matrix": split.tolist()},
                "merge": {"type": [3, 1], "matrix": merge.tolist()},
            },
            "macros": {
                "burst": {
                    "body": {
                        "compose": [{"gen": "split"}, {"gen": "merge"}]
                    }
                }
            },
            "equations": [
                {"id": "target", "lhs": {"gen": "X"}, "rhs": {"id": 1}}
            ],
        }
    )

    def replacements(budget: int) -> set[str]:
        return {
            model.parameters["replacement"]
            for model in substitution.candidates(
                theory,
                theory.equation("target"),
                bound=1,
                deadline=Deadline.after(2),
                max_substitution_matrix_entries=budget,
            )
        }

    assert "burst@[0]" not in replacements(4)
    assert "burst@[0]" in replacements(64)
