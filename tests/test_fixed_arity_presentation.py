from minimality_auto.core import Circuit, Equation, Generator, Signature, Theory
from minimality_auto.search import Deadline
from minimality_auto.separators import finite_model, fixed_arity


def test_presentation_compiles_shared_fixed_arity_inputs_once():
    generators = tuple(Generator(name, 1, 1) for name in ("left", "right", "retained"))
    signature = Signature(generators)
    left, right, retained = map(Circuit.generator, generators)
    theory = Theory(
        signature,
        (
            Equation(
                "retained involution",
                Circuit.compose((retained, retained)),
                Circuit.identity(1),
            ),
            Equation("target", left, right),
        ),
    )

    presentation = fixed_arity.compile_presentation(
        theory,
        theory.equation("target"),
        Deadline.after(1),
    )

    assert presentation.active_names == {"left", "right", "retained"}
    assert presentation.target_generators == {
        fixed_arity.generator("left"),
        fixed_arity.generator("right"),
    }
    assert presentation.required_constraints == (
        *presentation.retained,
        *presentation.prop,
    )
    assert presentation.separation_constraints[-1] == fixed_arity.Constraint(
        *presentation.target_pair,
        equal=False,
    )

    model = next(
        finite_model.candidates(
            theory,
            theory.equation("target"),
            bound=2,
            deadline=Deadline.after(1),
        )
    )
    assert model.evaluate(left) != model.evaluate(right)
