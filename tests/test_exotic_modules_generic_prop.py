from pathlib import Path

import numpy as np
import pytest

from minimality_auto.core import (
    Circuit,
    Equation,
    Generator,
    MacroDef,
    Signature,
    Theory,
    load_theory,
)
from minimality_auto.search import Deadline, NotApplicable, search_theory
from minimality_auto.separators import amalgam, spin
from minimality_auto.separators.finite_field import instantiate, templates


THEORY_PATH = (
    Path(__file__).parents[1] / "theories" / "realclifford-ch-simplified.json"
)


def test_finite_field_templates_accept_complex_patterns():
    theory = Theory.from_json(
        {
            "wire_dimension": 2,
            "generators": [
                {
                    "name": "phase gate",
                    "source": 1,
                    "target": 1,
                    "matrix": [[0, "i"], ["-i", 0]],
                }
            ],
            "equations": [
                {
                    "id": "arbitrary target id",
                    "lhs": {"gen": "phase gate"},
                    "rhs": {"id": 1},
                }
            ],
        }
    )

    source = templates(theory, {"phase gate": 1}, frozenset({"phase gate"}), 1)
    atoms, native = instantiate(source, {"phase gate": 1}, (2,), 1, 2, 5)

    assert source.variables == (1j,)
    assert native["phase gate"] == (0, 2, 3, 0)
    assert atoms[("generator", "phase gate")] == native["phase gate"]


def test_amalgam_supports_zero_wire_scalars_and_scalar_naturality():
    theory = Theory.from_json(
        {
            "wire_dimension": 1,
            "generators": [
                {"name": "left scalar", "source": 0, "target": 0, "matrix": [[-1]]},
                {"name": "right scalar", "source": 0, "target": 0, "matrix": [[-1]]},
            ],
            "equations": [
                {
                    "id": "left square",
                    "lhs": {"compose": [{"gen": "left scalar"}] * 2},
                    "rhs": {"id": 0},
                },
                {
                    "id": "right square",
                    "lhs": {"compose": [{"gen": "right scalar"}] * 2},
                    "rhs": {"id": 0},
                },
                {
                    "id": "independent scalar equality",
                    "lhs": {"gen": "left scalar"},
                    "rhs": {"gen": "right scalar"},
                },
            ],
        }
    )

    report = search_theory(
        theory,
        strategies=("amalgam",),
        equation_ids={"independent scalar equality"},
        max_amalgam_prime=3,
        max_amalgam_order=32,
        timeout=5,
    )

    witness = report.witnesses["independent scalar equality"]
    assert witness.parameters["arity"] == 0
    assert witness.parameters["structural_swaps"] == {}


def test_amalgam_supports_explicitly_typed_inline_generators():
    theory = Theory(
        Signature.from_json(
            {
                "declared": {
                    "type": [1, 1],
                    "matrix": [[0, 1], [1, 0]],
                }
            }
        ),
        (
            Equation(
                "inline target",
                Circuit.generator("inline", 1, 1),
                Circuit.generator("declared", 1, 1),
            ),
        ),
    )

    model = next(
        amalgam.candidates(
            theory,
            theory.equation("inline target"),
            bound=1,
            deadline=Deadline.after(2),
            max_amalgam_prime=2,
            max_amalgam_order=16,
        )
    )

    target = theory.equation("inline target")
    assert "inline" in model.parameters["interpretation"]
    assert model.evaluate(target.lhs) != model.evaluate(target.rhs)


def test_amalgam_matrix_bound_does_not_impose_an_unrelated_arity_bound():
    theory = Theory.from_json(
        {
            "wire_dimension": 1,
            "generators": [
                {"name": "left", "source": 3, "target": 3, "matrix": [[-1]]},
                {"name": "right", "source": 3, "target": 3, "matrix": [[-1]]},
            ],
            "equations": [
                {
                    "id": "left square",
                    "lhs": {"compose": [{"gen": "left"}, {"gen": "left"}]},
                    "rhs": {"id": 3},
                },
                {
                    "id": "right square",
                    "lhs": {"compose": [{"gen": "right"}, {"gen": "right"}]},
                    "rhs": {"id": 3},
                },
                {
                    "id": "target",
                    "lhs": {"gen": "left"},
                    "rhs": {"gen": "right"},
                },
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("amalgam",),
        equation_ids={"target"},
        max_amalgam_prime=3,
        max_amalgam_order=32,
        max_amalgam_matrix_dimension=1,
        timeout=5,
    )

    assert set(report.witnesses) == {"target"}


def test_amalgam_cannot_separate_prop_interchange():
    theory = Theory.from_json(
        {
            "wire_dimension": 2,
            "generators": [
                {
                    "name": "first",
                    "source": 1,
                    "target": 1,
                    "matrix": [[0, 1], [1, 0]],
                },
                {
                    "name": "second",
                    "source": 1,
                    "target": 1,
                    "matrix": [[1, 0], [0, -1]],
                },
            ],
            "equations": [
                {
                    "id": "PROP interchange",
                    "lhs": {
                        "compose": [
                            {"tensor": [{"gen": "first"}, {"id": 1}]},
                            {"tensor": [{"id": 1}, {"gen": "second"}]},
                        ]
                    },
                    "rhs": {
                        "compose": [
                            {"tensor": [{"id": 1}, {"gen": "second"}]},
                            {"tensor": [{"gen": "first"}, {"id": 1}]},
                        ]
                    },
                }
            ],
        }
    )

    models = list(
        amalgam.candidates(
            theory,
            theory.equation("PROP interchange"),
            bound=1,
            deadline=Deadline.after(5),
            max_amalgam_prime=2,
            max_amalgam_order=64,
        )
    )

    assert models == []


@pytest.mark.parametrize("separator", [amalgam.candidates, spin.candidates])
def test_fixed_arity_exotic_searches_reject_hidden_arity_changes(separator):
    theory = Theory.from_json(
        {
            "wire_dimension": 2,
            "generators": [
                {
                    "name": "turn",
                    "source": 1,
                    "target": 1,
                    "matrix": [[0, 1], [1, 0]],
                }
            ],
            "equations": [
                {
                    "id": "turn square",
                    "lhs": {"compose": [{"gen": "turn"}, {"gen": "turn"}]},
                    "rhs": {"id": 1},
                }
            ],
        }
    )
    theory.macros["hidden grow"] = MacroDef(
        "hidden grow", Circuit.generator("fresh", 0, 1)
    )

    with pytest.raises(NotApplicable, match="terms and macros"):
        list(
            separator(
                theory,
                theory.equation("turn square"),
                bound=1 if separator is amalgam.candidates else 0,
                deadline=Deadline.after(2),
            )
        )


def _macro_wrapped_spin_theory() -> Theory:
    original = load_theory(THEORY_PATH)
    target = original.equation("19_n5")
    macros = dict(original.macros)
    macros["arbitrary left wrapper"] = MacroDef("arbitrary left wrapper", target.lhs)
    right_parts = list(target.rhs.parts)
    right_parts.insert(1, Circuit.identity(5))
    macros["arbitrary right wrapper"] = MacroDef(
        "arbitrary right wrapper", Circuit.compose(right_parts)
    )
    wrapped = Equation(
        "unrelated target name",
        Circuit.macro("arbitrary left wrapper", 5, 5),
        Circuit.macro("arbitrary right wrapper", 5, 5),
    )
    unused_matrix = tuple(
        tuple(1 if row == column else 0 for column in range(8))
        for row in range(8)
    )
    signature = Signature(
        (
            *original.signature.values(),
            Generator("unused non-reflection", 3, 3, unused_matrix),
        )
    )
    equations = tuple(
        equation for equation in original.equations if equation.id != "19_n5"
    )
    return Theory(
        signature,
        (*equations, wrapped),
        macros,
        "generic spin input",
        original.wire_dimension,
    )


def test_spin_expands_target_macros_and_ignores_unused_matrix_templates():
    theory = _macro_wrapped_spin_theory()
    report = search_theory(
        theory,
        strategies=("spin",),
        equation_ids={"unrelated target name"},
        timeout=10,
    )

    assert set(report.witnesses) == {"unrelated target name"}
    assert "unused non-reflection" not in report.witnesses[
        "unrelated target name"
    ].parameters["primitive_reflections"]


@pytest.mark.parametrize("width", [0, 3])
def test_spin_accepts_active_reflections_of_arbitrary_width(width: int):
    size = 1 << width
    diagonal = [[0] * size for _ in range(size)]
    for index in range(size):
        diagonal[index][index] = -1 if index == 0 else 1
    theory = Theory.from_json(
        {
            "wire_dimension": 2,
            "generators": [
                {
                    "name": "reflection",
                    "source": width,
                    "target": width,
                    "matrix": diagonal,
                }
            ],
            "equations": [],
        }
    )

    assert spin._reflection_generators(
        theory, frozenset({"reflection"}), Deadline.after(2)
    ) == (("reflection", width),)


def test_spin_rejects_a_near_reflection():
    near_reflection = np.diag([-1.0, 1.0 - 1e-10]).astype(complex)

    with pytest.raises(NotApplicable, match="not a real orthogonal reflection"):
        spin._check_reflection("near reflection", near_reflection, 1)


def test_spin_dense_dimension_guard_is_configurable_and_exact():
    theory = _macro_wrapped_spin_theory()
    target = theory.equation("unrelated target name")

    with pytest.raises(NotApplicable, match="exceeds 31"):
        list(
            spin.candidates(
                theory,
                target,
                bound=0,
                deadline=Deadline.after(2),
                max_spin_matrix_dimension=31,
            )
        )

    models = list(
        spin.candidates(
            theory,
            target,
            bound=0,
            deadline=Deadline.after(5),
            max_spin_matrix_dimension=32,
        )
    )
    assert len(models) == 1


def test_spin_rejects_matrix_free_inline_primitives_cleanly():
    inline = Circuit.generator("inline reflection", 5, 5)
    theory = Theory(
        Signature(),
        (Equation("inline target", inline, Circuit.identity(5)),),
        wire_dimension=2,
    )

    with pytest.raises(NotApplicable, match="valid matrices"):
        list(
            spin.candidates(
                theory,
                theory.equation("inline target"),
                bound=0,
                deadline=Deadline.after(2),
            )
        )
