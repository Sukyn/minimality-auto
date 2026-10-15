"""Adversarial checks for the fixed-ambient Spin certificate.

The ambient-wire guards ensure that Spin lifts preserve these PROP equalities.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from minimality_auto.core import (
    Circuit,
    Equation,
    Generator,
    Signature,
    Theory,
    load_theory,
)
from minimality_auto.search import Deadline, NotApplicable, search_theory
from minimality_auto.separators import spin

THEORY_PATH = (
    Path(__file__).parents[1] / "theories" / "realclifford-ch-simplified.json"
)


def _rank_one_reflection(width: int) -> tuple[tuple[int, ...], ...]:
    size = 1 << width
    return tuple(
        tuple(-1 if row == column == 0 else int(row == column) for column in range(size))
        for row in range(size)
    )


def _interchange_theory(
    *, ambient_width: int, gate_width: int
) -> tuple[Theory, Equation, Circuit, Circuit]:
    first_generator = Generator(
        "first reflection",
        gate_width,
        gate_width,
        _rank_one_reflection(gate_width),
    )
    second_generator = Generator(
        "second reflection",
        gate_width,
        gate_width,
        _rank_one_reflection(gate_width),
    )
    spare = ambient_width - 2 * gate_width
    assert spare >= 0

    first = Circuit.tensor(
        (
            Circuit.generator(first_generator),
            Circuit.identity(ambient_width - gate_width),
        )
    )
    second = Circuit.tensor(
        (
            Circuit.identity(gate_width),
            Circuit.generator(second_generator),
            Circuit.identity(spare),
        )
    )
    equation = Equation(
        "PROP interchange",
        Circuit.compose((first, second)),
        Circuit.compose((second, first)),
    )
    theory = Theory(
        Signature((first_generator, second_generator)),
        (equation,),
        wire_dimension=2,
    )
    return theory, equation, first, second


def _commutator_data(theory: Theory, first: Circuit, second: Circuit):
    return spin._commutator_data(
        spin._matrix(theory, first),
        spin._matrix(theory, second),
    )


def _with_extra_rules(
    theory: Theory,
    generators: tuple[Generator, ...],
    equations: tuple[Equation, ...],
) -> Theory:
    return replace(
        theory,
        signature=Signature((*theory.signature.values(), *generators)),
        equations=(*theory.equations, *equations),
    )


def test_low_arity_spin_sign_cannot_separate_prop_interchange():
    theory, target, first, second = _interchange_theory(
        ambient_width=4,
        gate_width=2,
    )

    # The commuting matrices have anticommuting Spin lifts:
    # p=q=4 and their minus spaces meet in one direction, so pq-t=15 is odd.
    data = _commutator_data(theory, first, second)
    assert data == (4, 4, 1)
    assert spin._commutator_sign(data) == -1

    # PROP interchange forces this equality, so the separator rejects the
    # arity-four lift that gives it a negative sign.
    assert list(
        spin.candidates(
            theory,
            target,
            bound=0,
            deadline=Deadline.after(2),
        )
    ) == []


def test_one_spare_wire_removes_the_false_interchange_sign():
    theory, target, first, second = _interchange_theory(
        ambient_width=5,
        gate_width=2,
    )

    # One untouched wire doubles all three dimensions.  The even exponent
    # 8*8-2=62 makes the lifts commute, as PROP interchange requires.
    data = _commutator_data(theory, first, second)
    assert data == (8, 8, 2)
    assert spin._commutator_sign(data) == 1
    assert data == tuple(2 * value for value in (4, 4, 1))

    assert list(
        spin.candidates(
            theory,
            target,
            bound=0,
            deadline=Deadline.after(2),
        )
    ) == []


def test_one_spare_wire_accepts_composition_tensor_and_routing_equalities():
    original = load_theory(THEORY_PATH)
    target = original.equation("19_n5")
    local, interchange, first, second = _interchange_theory(
        ambient_width=4,
        gate_width=2,
    )
    first_generator, second_generator = tuple(local.signature.values())

    block_swap = Circuit.perm((2, 3, 0, 1))
    routed = Equation(
        "routed reflection",
        Circuit.compose((block_swap, first, block_swap)),
        second,
    )
    theory = _with_extra_rules(
        original,
        (first_generator, second_generator),
        (interchange, routed),
    )

    # The four-wire interchange equality has negative local Spin sign.
    # The routed equality checks composition with structural permutations.
    assert spin._commutator_sign(_commutator_data(theory, first, second)) == -1
    for equation in (interchange, routed):
        assert spin._close(
            spin._matrix(theory, equation.lhs),
            spin._matrix(theory, equation.rhs),
        )

    report = search_theory(
        theory,
        strategies=("spin",),
        equation_ids={target.id},
        timeout=10,
    )
    witness = report.witnesses[target.id]
    assert {interchange.id, routed.id}.issubset(witness.checked_equations)
    assert not witness.certified
    assert witness.description.startswith("numerical ")


def test_full_arity_positive_reversal_is_recorded_explicitly():
    original = load_theory(THEORY_PATH)
    target = original.equation("19_n5")
    local, interchange, first, second = _interchange_theory(
        ambient_width=5,
        gate_width=2,
    )
    theory = _with_extra_rules(
        original,
        tuple(local.signature.values()),
        (interchange,),
    )

    assert _commutator_data(theory, first, second) == (8, 8, 2)
    witnesses = list(
        spin.candidates(
            theory,
            target,
            bound=0,
            deadline=Deadline.after(10),
        )
    )

    assert len(witnesses) == 1
    assert witnesses[0].parameters["full_arity_retained_reversals"] == [
        {
            "equation": interchange.id,
            "minus_dimensions": [8, 8],
            "intersection_dimension": 2,
            "commutator_sign": 1,
        }
    ]


def test_one_spare_wire_still_rejects_a_matrix_unsound_equation():
    original = load_theory(THEORY_PATH)
    target = original.equation("19_n5")
    reflection = Generator(
        "extra reflection",
        2,
        2,
        _rank_one_reflection(2),
    )
    placed = Circuit.tensor((Circuit.generator(reflection), Circuit.identity(2)))
    false_equation = Equation("false four-wire rule", placed, Circuit.identity(4))
    theory = _with_extra_rules(original, (reflection,), (false_equation,))

    with pytest.raises(NotApplicable, match="not matrix-sound"):
        list(
            spin.candidates(
                theory,
                target,
                bound=0,
                deadline=Deadline.after(10),
            )
        )


def test_spin_rejects_wide_primitives_that_fill_the_ambient_tensor():
    theory, target, first, second = _interchange_theory(
        ambient_width=6,
        gate_width=3,
    )

    data = _commutator_data(theory, first, second)
    assert data == (8, 8, 1)
    assert spin._commutator_sign(data) == -1

    # Each three-wire primitive has three complement wires in this tensor.
    # Together they fill disjoint halves and have negative commutator sign,
    # so the pairwise width guard rejects this interchange violation.
    with pytest.raises(NotApplicable, match="too few wires for PROP interchange"):
        list(
            spin.candidates(
                theory,
                target,
                bound=0,
                deadline=Deadline.after(2),
            )
        )


def test_spin_rejects_a_primitive_with_only_two_complement_wires():
    reflection = Generator(
        "wide reflection",
        3,
        3,
        _rank_one_reflection(3),
    )
    placed = Circuit.tensor((Circuit.generator(reflection), Circuit.identity(2)))
    target = Equation("wide target", placed, Circuit.identity(5))
    theory = Theory(Signature((reflection,)), (target,), wire_dimension=2)

    with pytest.raises(NotApplicable, match="too few complement wires"):
        list(
            spin.candidates(
                theory,
                target,
                bound=0,
                deadline=Deadline.after(2),
            )
        )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # An odd-dimensional minus space gives determinant -1.
        (np.diag((-1, 1)), np.diag((-1, 1))),
        # These reflections anticommute.
        (np.diag((-1, 1)), np.asarray(((0, 1), (1, 0)))),
    ],
)
def test_commutator_data_rejects_non_spin_or_noncommuting_inputs(first, second):
    assert spin._commutator_data(
        np.asarray(first, dtype=np.complex128),
        np.asarray(second, dtype=np.complex128),
    ) is None
