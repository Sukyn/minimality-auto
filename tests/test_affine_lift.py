from __future__ import annotations

from itertools import product
from random import Random

from minimality_auto.core import Theory
from minimality_auto.search import Deadline, search_theory
from minimality_auto.separators.affine_lift import (
    _dot,
    _evaluate_word,
    _separating_vector,
    _word_coefficients,
)


def _search(theory: Theory, target: str):
    return search_theory(
        theory,
        strategies=("affine_lift",),
        equation_ids={target},
        max_affine_lift_prime=2,
        timeout=2,
    )


def test_dependent_retained_rows_do_not_hide_or_invent_a_witness():
    square = {
        "compose": [{"gen": "g"}, {"gen": "g"}],
    }
    theory = Theory.from_json(
        {
            "generators": {
                "g": {"type": 1, "matrix": {"permutation": [1, 0]}},
            },
            "equations": [
                {"id": "first", "lhs": square, "rhs": {"id": 1}},
                {"id": "duplicate", "lhs": square, "rhs": {"id": 1}},
                {"id": "target", "lhs": square, "rhs": {"id": 1}},
            ],
        }
    )

    report = _search(theory, "target")

    assert not report.witnesses
    assert report.unresolved == ("target",)


def test_nonpermutation_templates_are_not_treated_as_exact_affine_lifts():
    theory = Theory.from_json(
        {
            "generators": {
                "z": {"type": 1, "matrix": {"diagonal": [1, -1]}},
            },
            "equations": [
                {"id": "target", "lhs": {"gen": "z"}, "rhs": {"id": 1}},
            ],
        }
    )

    assert not _search(theory, "target").witnesses


def test_a_false_retained_base_equation_prevents_a_certificate():
    theory = Theory.from_json(
        {
            "generators": {
                "a": {"type": 1, "matrix": {"permutation": [1, 0]}},
                "b": {"type": 1, "matrix": {"permutation": [0, 1]}},
            },
            "equations": [
                {"id": "retained", "lhs": {"gen": "a"}, "rhs": {"gen": "b"}},
                {"id": "target", "lhs": {"gen": "a"}, "rhs": {"id": 1}},
            ],
        }
    )

    assert not _search(theory, "target").witnesses


def test_nonbinary_affine_lift_is_found_exactly():
    cube = {"compose": [{"gen": "g"}] * 3}
    theory = Theory.from_json(
        {
            "generators": {
                "g": {"type": 1, "matrix": {"permutation": [0, 1]}},
            },
            "equations": [
                {"id": "cube", "lhs": cube, "rhs": {"id": 1}},
                {"id": "target", "lhs": {"gen": "g"}, "rhs": {"id": 1}},
            ],
        }
    )

    report = search_theory(
        theory,
        strategies=("affine_lift",),
        equation_ids={"target"},
        max_affine_lift_prime=3,
        timeout=2,
    )

    witness = report.witnesses["target"]
    assert witness.certified
    assert witness.parameters["prime"] == 3


def test_base_permutations_can_separate_with_zero_translation():
    theory = Theory.from_json(
        {
            "generators": {
                "g": {"type": 1, "matrix": {"permutation": [1, 0]}},
            },
            "equations": [
                {"id": "target", "lhs": {"gen": "g"}, "rhs": {"id": 1}},
            ],
        }
    )

    witness = _search(theory, "target").witnesses["target"]

    assert witness.parameters["translations"] == {
        "generators": {},
        "structural_swaps": {},
    }
    assert witness.lhs_value != witness.rhs_value


def test_generator_and_structural_swap_names_do_not_collide():
    theory = Theory.from_json(
        {
            "generators": {
                "swap[0,1]": {
                    "type": 2,
                    "matrix": {"permutation": [0, 2, 1, 3]},
                },
            },
            "equations": [
                {
                    "id": "target",
                    "lhs": {
                        "compose": [
                            {"gen": "swap[0,1]"},
                            {"perm": [1, 0]},
                        ]
                    },
                    "rhs": {"id": 2},
                },
            ],
        }
    )

    witness = _search(theory, "target").witnesses["target"]
    parameters = witness.parameters

    assert "swap[0,1]" in parameters["translations"]["generators"]
    assert "swap[0,1]" in parameters["base_interpretation"]["generators"]
    assert "swap[0,1]" in parameters["base_interpretation"]["structural_swaps"]


def test_affine_lifts_do_not_separate_prop_coherence():
    theory = Theory.from_json(
        {
            "generators": {
                "a": {"type": 1, "matrix": {"permutation": [1, 0]}},
                "b": {"type": 1, "matrix": {"permutation": [0, 1]}},
                "binary": {
                    "type": 2,
                    "matrix": {"permutation": [0, 2, 1, 3]},
                },
                "scalar": {"type": 0, "matrix": {"permutation": [0]}},
            },
            "equations": [
                {
                    "id": "naturality",
                    "lhs": {
                        "compose": [
                            {"tensor": [{"gen": "a"}, {"gen": "b"}]},
                            {"perm": [1, 0]},
                        ]
                    },
                    "rhs": {
                        "compose": [
                            {"perm": [1, 0]},
                            {"tensor": [{"gen": "b"}, {"gen": "a"}]},
                        ]
                    },
                },
                {
                    "id": "braid",
                    "lhs": {
                        "compose": [
                            {"perm": [1, 0, 2]},
                            {"perm": [0, 2, 1]},
                            {"perm": [1, 0, 2]},
                        ]
                    },
                    "rhs": {
                        "compose": [
                            {"perm": [0, 2, 1]},
                            {"perm": [1, 0, 2]},
                            {"perm": [0, 2, 1]},
                        ]
                    },
                },
                {
                    "id": "mixed_interchange",
                    "lhs": {
                        "compose": [
                            {"tensor": [{"gen": "binary"}, {"id": 1}]},
                            {"tensor": [{"id": 2}, {"gen": "a"}]},
                        ]
                    },
                    "rhs": {
                        "compose": [
                            {"tensor": [{"id": 2}, {"gen": "a"}]},
                            {"tensor": [{"gen": "binary"}, {"id": 1}]},
                        ]
                    },
                },
                {
                    "id": "scalar_centrality",
                    "lhs": {"tensor": [{"gen": "scalar"}, {"gen": "a"}]},
                    "rhs": {"tensor": [{"gen": "a"}, {"gen": "scalar"}]},
                },
            ],
        }
    )

    report = search_theory(
        theory,
        strategies=("affine_lift",),
        max_affine_lift_prime=3,
        timeout=2,
    )

    assert not report.witnesses
    assert report.unresolved == (
        "naturality",
        "braid",
        "mixed_interchange",
        "scalar_centrality",
    )


def test_symbolic_coefficients_match_concrete_affine_composition():
    first = ("generator", "first")
    second = ("generator", "second")
    assignment = {
        first: (1, 2, 0),
        second: (0, 2, 1),
    }
    offsets = {first: 0, second: 3}
    word = (first, second, first, first, second)
    deadline = Deadline.after(2)

    for prime in (2, 3, 5):
        base, rows = _word_coefficients(
            word,
            assignment,
            offsets,
            6,
            prime,
            deadline,
        )
        witness = tuple(index % prime for index in range(6))
        translations = {
            first: witness[:3],
            second: witness[3:],
        }

        concrete = _evaluate_word(
            word,
            assignment,
            translations,
            prime,
            deadline,
        )
        symbolic = base, tuple(_dot(row, witness, prime) for row in rows)

        assert concrete == symbolic


def test_separating_vector_matches_brute_force_kernels():
    random = Random(0)
    deadline = Deadline.after(5)
    for prime in (2, 3, 5):
        for columns in range(1, 5):
            for _ in range(4):
                source = [
                    tuple(random.randrange(prime) for _ in range(columns))
                    for _ in range(random.randrange(5))
                ]
                targets = [
                    tuple(random.randrange(prime) for _ in range(columns))
                    for _ in range(2)
                ]
                expected = any(
                    all(_dot(row, value, prime) == 0 for row in source)
                    and any(_dot(row, value, prime) != 0 for row in targets)
                    for value in product(range(prime), repeat=columns)
                )

                witness = _separating_vector(
                    source,
                    targets,
                    columns,
                    prime,
                    deadline,
                )

                assert (witness is not None) == expected
                if witness is not None:
                    assert all(_dot(row, witness, prime) == 0 for row in source)
                    assert any(_dot(row, witness, prime) != 0 for row in targets)
