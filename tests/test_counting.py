from itertools import product
from random import Random

import pytest

from minimality_auto.algebra.abelian import separating_weights
from minimality_auto.core import Theory
from minimality_auto.search import Deadline, SearchTimeout, search_theory


@pytest.mark.parametrize("modulus", range(2, 13))
def test_modular_solver_matches_exhaustive_search(modulus):
    random = Random(modulus)
    deadline = Deadline.after(5)
    for columns in range(4):
        for _ in range(12):
            source = [
                tuple(random.randrange(-modulus, modulus + 1) for _ in range(columns))
                for _ in range(random.randrange(4))
            ]
            target = tuple(random.randrange(-modulus, modulus + 1) for _ in range(columns))

            def separates(weights, source=source, target=target):
                return (
                    all(
                        sum(a * b for a, b in zip(row, weights, strict=True)) % modulus == 0
                        for row in source
                    )
                    and sum(a * b for a, b in zip(target, weights, strict=True)) % modulus != 0
                )

            expected = any(
                separates(weights) for weights in product(range(modulus), repeat=columns)
            )
            witness = separating_weights(source, target, modulus, deadline)
            assert (witness is not None) == expected
            if witness is not None:
                assert separates(witness)


@pytest.mark.parametrize("modulus", [4, 8])
def test_counting_finds_witnesses_requiring_composite_moduli(modulus):
    theory = Theory.from_json(
        {
            "generators": {"g": 1},
            "equations": [
                {
                    "id": "retained",
                    "lhs": {"gen": "g", "repeat": modulus},
                    "rhs": {"id": 1},
                },
                {
                    "id": "target",
                    "lhs": {"gen": "g", "repeat": modulus // 2},
                    "rhs": {"id": 1},
                },
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("counting",),
        equation_ids={"target"},
        max_modulus=modulus,
        timeout=2,
    )
    witness = report.witnesses["target"]
    assert witness.certified
    assert witness.parameters["modulus"] == modulus
    assert witness.checked_equations == ("retained",)
    assert not report.unresolved


def test_modular_solver_handles_noninvertible_pivots():
    witness = separating_weights(((2,),), (1,), 4, Deadline.after(2))
    assert witness == (2,)


def test_modular_solver_honours_deadline():
    with pytest.raises(SearchTimeout):
        separating_weights(((1, 1),), (1, 0), 4, Deadline.after(0))
