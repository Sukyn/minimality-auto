from __future__ import annotations

from itertools import permutations, product
import sys

import pytest

from minimality_auto.core import Circuit, Equation, MacroDef, Signature, Theory
from minimality_auto.search import Deadline, NotApplicable, search_theory
from minimality_auto.separators import finite_model


def _compiled(term: Circuit, theory: Theory, arity: int):
    signature = {
        name: generator.inputs for name, generator in theory.signature.items()
    }
    return finite_model._word(term, signature, arity)


def test_every_small_prop_assignment_respects_naturality_and_interchange():
    theory = Theory.from_json(
        {
            "generators": {"left box": [1, 1], "right/box": [1, 1]},
            "equations": [
                {
                    "id": "dummy",
                    "lhs": {"gen": "left box"},
                    "rhs": {"gen": "right/box"},
                }
            ],
        }
    )
    left = Circuit.generator(theory.signature["left box"])
    right = Circuit.generator(theory.signature["right/box"])
    swap = Circuit.perm((1, 0))
    naturality = (
        Circuit.compose((Circuit.tensor((left, right)), swap)),
        Circuit.compose((swap, Circuit.tensor((right, left)))),
    )
    interchange = (
        Circuit.compose(
            (Circuit.tensor((left, right)), Circuit.tensor((right, left)))
        ),
        Circuit.tensor(
            (Circuit.compose((left, right)), Circuit.compose((right, left)))
        ),
    )

    degree = 3
    values = tuple(permutations(range(degree)))
    generators = (("left box", 1), ("right/box", 1))
    constraints = finite_model._prop_constraints(2, generators)
    variables = (
        finite_model._generator("left box"),
        finite_model._generator("right/box"),
        finite_model._swap(0),
    )
    valid = 0
    for assigned in product(values, repeat=len(variables)):
        assignment = dict(zip(variables, assigned, strict=True))
        try:
            finite_model._validate_assignment(
                variables, constraints, assignment, degree
            )
        except ValueError:
            continue
        valid += 1
        for lhs, rhs in (naturality, interchange):
            assert finite_model._evaluate(
                _compiled(lhs, theory, 2), assignment, degree
            ) == finite_model._evaluate(
                _compiled(rhs, theory, 2), assignment, degree
            )
    assert valid > 1


def test_scalar_centrality_holds_for_every_small_prop_assignment():
    theory = Theory.from_json(
        {
            "generators": {"scalar": [0, 0], "gate": [1, 1]},
            "equations": [
                {
                    "id": "dummy",
                    "lhs": {"tensor": [{"gen": "scalar"}, {"gen": "gate"}]},
                    "rhs": {"tensor": [{"gen": "gate"}, {"gen": "scalar"}]},
                }
            ],
        }
    )
    equation = theory.equation("dummy")
    degree = 3
    values = tuple(permutations(range(degree)))
    generators = (("scalar", 0), ("gate", 1))
    constraints = finite_model._prop_constraints(1, generators)
    variables = tuple(finite_model._generator(name) for name, _ in generators)
    valid = 0
    for assigned in product(values, repeat=2):
        assignment = dict(zip(variables, assigned, strict=True))
        try:
            finite_model._validate_assignment(
                variables, constraints, assignment, degree
            )
        except ValueError:
            continue
        valid += 1
        assert finite_model._evaluate(
            _compiled(equation.lhs, theory, 1), assignment, degree
        ) == finite_model._evaluate(
            _compiled(equation.rhs, theory, 1), assignment, degree
        )
    assert valid > 1


def test_every_small_mixed_width_assignment_respects_prop_laws():
    """Exercise the complete fixed-arity presentation, not one paper fragment."""

    theory = Theory.from_json(
        {
            "generators": {
                "scalar": [0, 0],
                "unary": [1, 1],
                "binary": [2, 2],
            },
            "equations": [
                {"id": "dummy", "lhs": {"gen": "unary"}, "rhs": {"gen": "unary"}}
            ],
        }
    )
    scalar = Circuit.generator(theory.signature["scalar"])
    unary = Circuit.generator(theory.signature["unary"])
    binary = Circuit.generator(theory.signature["binary"])

    def block_swap(left: int, right: int) -> Circuit:
        return Circuit.perm((*range(left, left + right), *range(left)))

    laws = (
        # Naturality for mixed generator widths and for identity blocks.
        (
            Circuit.compose(
                (Circuit.tensor((unary, binary)), block_swap(1, 2))
            ),
            Circuit.compose(
                (block_swap(1, 2), Circuit.tensor((binary, unary)))
            ),
        ),
        (
            Circuit.compose(
                (Circuit.tensor((unary, Circuit.identity(2))), block_swap(1, 2))
            ),
            Circuit.compose(
                (block_swap(1, 2), Circuit.tensor((Circuit.identity(2), unary)))
            ),
        ),
        (
            Circuit.compose(
                (Circuit.tensor((binary, Circuit.identity(1))), block_swap(2, 1))
            ),
            Circuit.compose(
                (block_swap(2, 1), Circuit.tensor((Circuit.identity(1), binary)))
            ),
        ),
        # Interchange on a 1+2 split.
        (
            Circuit.compose(
                (Circuit.tensor((unary, binary)), Circuit.tensor((unary, binary)))
            ),
            Circuit.tensor(
                (Circuit.compose((unary, unary)), Circuit.compose((binary, binary)))
            ),
        ),
        # A nullary generator is central with boxes and structural maps.
        (Circuit.tensor((scalar, unary)), Circuit.tensor((unary, scalar))),
        (Circuit.tensor((scalar, binary)), Circuit.tensor((binary, scalar))),
        (
            Circuit.tensor((scalar, Circuit.perm((1, 0)))),
            Circuit.tensor((Circuit.perm((1, 0)), scalar)),
        ),
    )

    degree = 3
    values = tuple(permutations(range(degree)))
    generators = (("scalar", 0), ("unary", 1), ("binary", 2))
    variables = tuple(finite_model._generator(name) for name, _ in generators) + (
        finite_model._swap(0),
        finite_model._swap(1),
    )
    constraints = finite_model._prop_constraints(3, generators)
    valid = 0
    for assigned in product(values, repeat=len(variables)):
        assignment = dict(zip(variables, assigned, strict=True))
        try:
            finite_model._validate_assignment(
                variables, constraints, assignment, degree
            )
        except ValueError:
            continue
        valid += 1
        for lhs, rhs in laws:
            assert finite_model._evaluate(
                _compiled(lhs, theory, 3), assignment, degree
            ) == finite_model._evaluate(
                _compiled(rhs, theory, 3), assignment, degree
            )
        # Interchange must also hold after every possible wire routing, not
        # only for the canonical left/right placement used in the relation.
        for binary_wires in permutations(range(3), 2):
            unary_wire = tuple(
                wire for wire in range(3) if wire not in binary_wires
            )
            binary_word = finite_model._placed(
                finite_model._generator("binary"), binary_wires, 3
            )
            unary_word = finite_model._placed(
                finite_model._generator("unary"), unary_wire, 3
            )
            assert finite_model._evaluate(
                binary_word + unary_word, assignment, degree
            ) == finite_model._evaluate(
                unary_word + binary_word, assignment, degree
            )
    assert valid > 1


def test_structural_compilation_respects_composition_in_every_small_model():
    degree = 3
    values = tuple(permutations(range(degree)))
    variables = (finite_model._swap(0), finite_model._swap(1))
    constraints = finite_model._prop_constraints(3, ())
    orders = tuple(permutations(range(3)))
    valid = 0
    for assigned in product(values, repeat=2):
        assignment = dict(zip(variables, assigned, strict=True))
        try:
            finite_model._validate_assignment(
                variables, constraints, assignment, degree
            )
        except ValueError:
            continue
        valid += 1
        for first, second in product(orders, repeat=2):
            composite = tuple(first[second[index]] for index in range(3))
            word = finite_model._adjacent_word(first) + finite_model._adjacent_word(
                second
            )
            assert finite_model._evaluate(
                word, assignment, degree
            ) == finite_model._evaluate(
                finite_model._adjacent_word(composite), assignment, degree
            )
    assert valid > 1


def test_pure_structural_theories_need_no_named_generators():
    theory = Theory.from_json(
        {
            "generators": {},
            "equations": [
                {
                    "id": "erase_swap",
                    "lhs": {"perm": [1, 0]},
                    "rhs": {"id": 2},
                }
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("finite_model",),
        max_permutation_degree=2,
        timeout=2,
    )

    witness = report.witnesses["erase_swap"]
    assert witness.parameters["interpretation"] == {
        "generators": {},
        "structural_swaps": {"swap[0,1]": [1, 0]},
    }


def test_finite_model_interprets_inline_typed_generators():
    theory = Theory(
        Signature(()),
        (
            Equation(
                "inline target",
                Circuit.generator("fresh", 1, 1),
                Circuit.identity(1),
            ),
        ),
    )
    report = search_theory(
        theory,
        strategies=("finite_model",),
        max_permutation_degree=2,
        timeout=2,
    )

    interpretation = report.witnesses["inline target"].parameters[
        "interpretation"
    ]
    assert interpretation["generators"] == {"fresh": [1, 0]}


def test_hidden_arity_changing_macro_rejects_fixed_wire_search():
    theory = Theory.from_json(
        {
            "generators": {"turn": [1, 1]},
            "equations": [
                {
                    "id": "turn2",
                    "lhs": {
                        "compose": [{"gen": "turn"}, {"gen": "turn"}]
                    },
                    "rhs": {"id": 1},
                }
            ],
        }
    )
    theory.macros["hidden grow"] = MacroDef(
        "hidden grow", Circuit.generator("fresh", 0, 1)
    )

    with pytest.raises(NotApplicable, match="terms and macros"):
        next(
            finite_model.candidates(
                theory,
                theory.equation("turn2"),
                bound=3,
                deadline=Deadline.after(1),
            )
        )


def test_prop_helpers_reject_ill_scoped_inputs():
    with pytest.raises(ValueError, match="fixed-arity fragment"):
        finite_model._prop_constraints(1, (("wide", 2),))
    with pytest.raises(ValueError, match="distinct positions"):
        finite_model._placed(finite_model._generator("g"), (0, 0), 2)

    constraint = finite_model._Constraint(
        (finite_model._generator("missing"),), ()
    )
    with pytest.raises(ValueError, match="undeclared interpretation"):
        finite_model._validate_assignment((), (constraint,), {}, 2)


def test_solver_does_not_depend_on_python_recursion_depth():
    variables = tuple(
        finite_model._generator(f"generic {index}") for index in range(220)
    )
    constraint = finite_model._Constraint(variables, (), equal=False)
    previous_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(150)
        solution = finite_model._find_assignment(
            variables,
            (constraint,),
            2,
            Deadline.after(5),
        )
    finally:
        sys.setrecursionlimit(previous_limit)

    assert solution is not None
    finite_model._validate_assignment(variables, (constraint,), solution, 2)
