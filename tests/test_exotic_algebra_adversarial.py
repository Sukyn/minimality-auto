from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from minimality_auto.core import Circuit, Equation, Generator, Signature, Theory
from minimality_auto.search import Deadline, NotApplicable, SearchTimeout
from minimality_auto.separators import amalgam, spin
from minimality_auto.separators.finite_field import (
    MatrixGroup,
    identity,
    inverse,
    multiply,
    templates,
)


def _symmetric_group_factors():
    # The two matrices generate S_3 over F_2.  Its normal C_3 subgroup is the
    # shared factor, while the reflection generates the quotient C_2.
    cycle = (0, 0, 1, 1, 0, 0, 0, 1, 0)
    reflection = (1, 0, 0, 0, 0, 1, 0, 1, 0)
    deadline = Deadline.after(2)
    base = MatrixGroup((cycle, reflection), 3, 2, 16, deadline)
    quotient = MatrixGroup((reflection,), 3, 2, 16, deadline)
    shared, action = amalgam._close_shared_subgroup(
        base,
        quotient,
        (base.element(cycle),),
        16,
        deadline,
    )
    second = amalgam._DGroup(base, quotient, action)
    decomposition, representatives = amalgam._cosets(base, shared, deadline)
    return base, quotient, shared, second, decomposition, representatives


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_finite_field_matrix_operations_are_exact_over_multiple_fields(prime):
    matrices = (
        identity(2),
        (0, 1, 1, 0),
        (1, 1, 0, 1),
    )
    invertible = []
    for value in matrices:
        try:
            value_inverse = inverse(value, 2, prime)
        except RuntimeError:
            continue
        invertible.append(value)
        assert multiply(value, value_inverse, 2, prime) == identity(2)
        assert multiply(value_inverse, value, 2, prime) == identity(2)

    for first, second, third in product(invertible, repeat=3):
        assert multiply(
            multiply(first, second, 2, prime), third, 2, prime
        ) == multiply(first, multiply(second, third, 2, prime), 2, prime)


def test_amalgam_normalizer_is_a_homomorphism_on_nonabelian_factors():
    (
        base,
        quotient,
        shared,
        second,
        decomposition,
        representatives,
    ) = _symmetric_group_factors()
    c_tokens = {value: ("c", value) for value in range(base.order)}
    d_values = tuple(
        amalgam._DElement(head, tail)
        for head in sorted(shared)
        for tail in range(quotient.order)
    )
    d_tokens = {value: ("d", index) for index, value in enumerate(d_values)}
    tokens = {
        **{token: ("C", value) for value, token in c_tokens.items()},
        **{token: ("D", value) for value, token in d_tokens.items()},
    }
    model = amalgam._Amalgam(
        base,
        shared,
        second,
        decomposition,
        {value: index for index, value in enumerate(representatives)},
        {value: index for index, value in enumerate(sorted(shared))},
        tokens,
    )

    # Both factor embeddings preserve every multiplication, not just the
    # handful of relations used by a particular input theory.
    for left, right in product(range(base.order), repeat=2):
        assert model.evaluate_word((c_tokens[left], c_tokens[right])) == (
            model.evaluate_word((c_tokens[base.multiply(left, right)],))
        )
    for left, right in product(d_values, repeat=2):
        assert model.evaluate_word((d_tokens[left], d_tokens[right])) == (
            model.evaluate_word((d_tokens[second.multiply(left, right)],))
        )

    # The two copies of the shared C_3 subgroup are identified exactly.
    for value in shared:
        assert model.evaluate_word((c_tokens[value],)) == model.evaluate_word(
            (d_tokens[amalgam._DElement(value, 0)],)
        )

    # Moving the shared head out of a D syllable may trigger reductions with
    # both neighbours.  Both spellings must therefore have one normal form.
    for left, middle, right in product(range(base.order), d_values, range(base.order)):
        word = (c_tokens[left], d_tokens[middle], c_tokens[right])
        expanded = (
            c_tokens[left],
            c_tokens[middle.shared],
            d_tokens[amalgam._DElement(0, middle.quotient)],
            c_tokens[right],
        )
        assert model.evaluate_word(word) == model.evaluate_word(expanded)


def test_amalgam_semidirect_factor_is_associative():
    _, quotient, shared, second, _, _ = _symmetric_group_factors()
    values = tuple(
        amalgam._DElement(head, tail)
        for head in sorted(shared)
        for tail in range(quotient.order)
    )
    for first, second_value, third in product(values, repeat=3):
        assert second.multiply(second.multiply(first, second_value), third) == (
            second.multiply(first, second.multiply(second_value, third))
        )


def test_amalgam_normalizer_honours_the_shared_search_deadline():
    (
        base,
        quotient,
        shared,
        second,
        decomposition,
        representatives,
    ) = _symmetric_group_factors()
    token = ("c", 1)
    model = amalgam._Amalgam(
        base,
        shared,
        second,
        decomposition,
        {value: index for index, value in enumerate(representatives)},
        {value: index for index, value in enumerate(sorted(shared))},
        {token: ("C", 1)},
    )

    with pytest.raises(SearchTimeout):
        model.evaluate_word((token,) * 10_000, Deadline.after(0))


def test_amalgam_bridge_search_includes_retained_only_generators():
    flip = [[0, 1], [1, 0]]
    theory = Theory.from_json(
        {
            "wire_dimension": 2,
            "generators": [
                {"name": "target left", "source": 1, "target": 1, "matrix": flip},
                {"name": "target right", "source": 1, "target": 1, "matrix": flip},
                {"name": "retained only", "source": 1, "target": 1, "matrix": flip},
            ],
            "equations": [
                {
                    "id": "retained involution",
                    "lhs": {
                        "compose": [
                            {"gen": "retained only"},
                            {"gen": "retained only"},
                        ]
                    },
                    "rhs": {"id": 1},
                },
                {
                    "id": "arbitrary target",
                    "lhs": {"gen": "target left"},
                    "rhs": {"gen": "target right"},
                },
            ],
        }
    )
    relations, _, target_words = amalgam._relations(
        theory,
        theory.equation("arbitrary target"),
        1,
        {name: 1 for name in theory.signature},
        Deadline.after(2),
    )
    active = {
        token[1]
        for relation in relations
        for word in (relation.left, relation.right)
        for token in word
        if token[0] == "generator"
    }
    active.update(
        token[1]
        for word in target_words
        for token in word
        if token[0] == "generator"
    )

    assert active == {"target left", "target right", "retained only"}


def test_amalgam_candidate_evaluator_respects_prop_contexts():
    theory = Theory.from_json(
        {
            "wire_dimension": 2,
            "generators": [
                {
                    "name": "horizontal flip",
                    "source": 1,
                    "target": 1,
                    "matrix": [[0, 1], [1, 0]],
                },
                {
                    "name": "vertical sign",
                    "source": 1,
                    "target": 1,
                    "matrix": [[1, 0], [0, -1]],
                },
            ],
            "equations": [
                {
                    "id": "flip square",
                    "lhs": {
                        "compose": [
                            {"gen": "horizontal flip"},
                            {"gen": "horizontal flip"},
                        ]
                    },
                    "rhs": {"id": 1},
                },
                {
                    "id": "sign square",
                    "lhs": {
                        "compose": [
                            {"gen": "vertical sign"},
                            {"gen": "vertical sign"},
                        ]
                    },
                    "rhs": {"id": 1},
                },
                {
                    "id": "generic two-wire target",
                    "lhs": {
                        "tensor": [{"gen": "horizontal flip"}, {"id": 1}]
                    },
                    "rhs": {
                        "tensor": [{"gen": "vertical sign"}, {"id": 1}]
                    },
                },
            ],
        }
    )
    model = next(
        amalgam.candidates(
            theory,
            theory.equation("generic two-wire target"),
            bound=1,
            deadline=Deadline.after(5),
            max_amalgam_prime=3,
            max_amalgam_order=128,
        )
    )
    flip = Circuit.generator(theory.signature["horizontal flip"])
    sign = Circuit.generator(theory.signature["vertical sign"])
    unit = Circuit.identity(1)
    swap = Circuit.perm((1, 0))
    flip_left = Circuit.tensor((flip, unit))
    flip_right = Circuit.tensor((unit, flip))
    sign_right = Circuit.tensor((unit, sign))

    assert model.evaluate(
        Circuit.compose((flip_left, sign_right))
    ) == model.evaluate(Circuit.compose((sign_right, flip_left)))
    assert model.evaluate(
        Circuit.compose((swap, flip_left, swap))
    ) == model.evaluate(flip_right)


def test_spin_rejects_invalid_retained_only_primitive():
    reflection = ((-1, 0), (0, 1))
    nonreflection = ((1, 0), (0, 2))
    target_left = Circuit.tensor(
        (
            Circuit.generator("left", 1, 1),
            Circuit.identity(4),
        )
    )
    target_right = Circuit.identity(5)
    retained = Equation(
        "retained invalid primitive",
        Circuit.generator("bad", 1, 1),
        Circuit.generator("bad", 1, 1),
    )
    target = Equation("arbitrary spin target", target_left, target_right)
    theory = Theory(
        Signature(
            (
                Generator("left", 1, 1, reflection),
                Generator("bad", 1, 1, nonreflection),
            )
        ),
        (retained, target),
        wire_dimension=2,
    )

    with pytest.raises(NotApplicable, match="bad.*not a real orthogonal reflection"):
        list(
            spin.candidates(
                theory,
                target,
                bound=0,
                deadline=Deadline.after(2),
            )
        )


def test_spin_reflection_check_rejects_complex_and_nearly_rank_one_values():
    complex_value = np.asarray(((1j, 0), (0, 1)), dtype=np.complex128)
    almost_reflection = np.diag((-1.0, 1.0 - 1e-5)).astype(np.complex128)

    with pytest.raises(NotApplicable, match="not a real orthogonal reflection"):
        spin._check_reflection("complex", complex_value, 1)
    with pytest.raises(NotApplicable, match="not a real orthogonal reflection"):
        spin._check_reflection("perturbed", almost_reflection, 1)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (((1, 0), (0,)), "wrong shape"),
        (((1, 0), (0, object())), "invalid entries"),
    ],
)
def test_finite_field_templates_reject_malformed_programmatic_matrices_cleanly(
    matrix, message
):
    theory = Theory(
        Signature((Generator("malformed", 1, 1, matrix),)),
        (
            Equation(
                "target",
                Circuit.generator("malformed", 1, 1),
                Circuit.identity(1),
            ),
        ),
        wire_dimension=2,
    )

    with pytest.raises(NotApplicable, match=message):
        templates(theory, {"malformed": 1}, frozenset({"malformed"}), 1)


def test_spin_rejects_malformed_programmatic_matrix_cleanly():
    malformed = ((1, 0), (0, object()))
    generator = Generator("malformed", 1, 1, malformed)
    term = Circuit.tensor((Circuit.generator(generator), Circuit.identity(4)))
    theory = Theory(
        Signature((generator,)),
        (Equation("target", term, Circuit.identity(5)),),
        wire_dimension=2,
    )

    with pytest.raises(NotApplicable, match="valid matrices"):
        list(
            spin.candidates(
                theory,
                theory.equation("target"),
                bound=0,
                deadline=Deadline.after(2),
            )
        )
