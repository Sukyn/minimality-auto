from __future__ import annotations

import json

import numpy as np
import pytest

from minimality_auto.core import MatrixSemantics, Theory
from minimality_auto.main import main
from minimality_auto.search import search_theory


def _assert_matrix_sound(theory: Theory) -> None:
    semantics = MatrixSemantics(theory.signature, theory.wire_dimension)
    for equation in theory.equations:
        np.testing.assert_allclose(
            semantics.evaluate(equation.lhs, theory.macros),
            semantics.evaluate(equation.rhs, theory.macros),
            atol=1e-10,
        )


def _rectangular_prop_laws() -> Theory:
    copy = [[1, 0], [0, 0], [0, 0], [0, 1]]
    merge = [[1, 0, 0, 0], [0, 0, 0, 1]]
    return Theory.from_json(
        {
            "generators": {
                "copy": {"type": [1, 2], "matrix": copy},
                "merge": {"type": [2, 1], "matrix": merge},
                "x": {"type": [1, 1], "matrix": [[0, 1], [1, 0]]},
                "z": {"type": [1, 1], "matrix": [[1, 0], [0, -1]]},
            },
            "equations": [
                {
                    "id": "naturality",
                    "lhs": {
                        "compose": [
                            {"tensor": [{"gen": "copy"}, {"id": 1}]},
                            {"perm": [2, 0, 1]},
                        ]
                    },
                    "rhs": {
                        "compose": [
                            {"perm": [1, 0]},
                            {"tensor": [{"id": 1}, {"gen": "copy"}]},
                        ]
                    },
                },
                {
                    "id": "interchange",
                    "lhs": {
                        "compose": [
                            {"tensor": [{"gen": "copy"}, {"gen": "x"}]},
                            {"tensor": [{"gen": "merge"}, {"gen": "z"}]},
                        ]
                    },
                    "rhs": {
                        "tensor": [
                            {
                                "compose": [
                                    {"gen": "copy"},
                                    {"gen": "merge"},
                                ]
                            },
                            {"compose": [{"gen": "x"}, {"gen": "z"}]},
                        ]
                    },
                },
            ],
        }
    )


def test_structural_searches_do_not_separate_rectangular_prop_laws():
    theory = _rectangular_prop_laws()
    _assert_matrix_sound(theory)
    report = search_theory(
        theory,
        strategies=("presence", "counting", "determinant", "substitution"),
        max_modulus=3,
        max_depth=1,
        timeout=10,
    )

    assert report.witnesses == {}
    assert report.unresolved == ("naturality", "interchange")
    assert not report.timed_out


def _contextual_consequence() -> Theory:
    return Theory.from_json(
        {
            "generators": {
                "a": {"type": [1, 1], "matrix": [[0, 1], [1, 0]]},
                "b": {"type": [1, 1], "matrix": [[1, 0], [0, -1]]},
            },
            "equations": [
                {
                    "id": "a_involution",
                    "lhs": {"compose": [{"gen": "a"}, {"gen": "a"}]},
                    "rhs": {"id": 1},
                },
                {
                    "id": "contextual_consequence",
                    "lhs": {
                        "compose": [
                            {"tensor": [{"gen": "a"}, {"gen": "b"}]},
                            {"tensor": [{"gen": "a"}, {"id": 1}]},
                        ]
                    },
                    "rhs": {"tensor": [{"id": 1}, {"gen": "b"}]},
                },
            ],
        }
    )


def _endomorphic_prop_laws() -> Theory:
    return Theory.from_json(
        {
            "generators": {
                "a": {"type": [1, 1], "matrix": [[0, 1], [1, 0]]},
                "b": {"type": [1, 1], "matrix": [[1, 0], [0, -1]]},
            },
            "equations": [
                {
                    "id": "swap_naturality",
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
                    "id": "swap_braid",
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
            ],
        }
    )


@pytest.mark.parametrize(
    ("strategy", "options"),
    [
        ("finite_model", {"max_permutation_degree": 3}),
        (
            "amalgam",
            {"max_amalgam_prime": 3, "max_amalgam_order": 64},
        ),
    ],
)
def test_fixed_arity_searches_do_not_separate_symmetric_prop_laws(
    strategy, options
):
    theory = _endomorphic_prop_laws()
    _assert_matrix_sound(theory)
    report = search_theory(
        theory,
        strategies=(strategy,),
        timeout=10,
        **options,
    )

    assert report.witnesses == {}
    assert report.unresolved == ("swap_naturality", "swap_braid")
    assert not report.timed_out


@pytest.mark.parametrize(
    ("strategies", "options"),
    [
        (
            ("presence", "counting", "determinant", "substitution"),
            {"max_modulus": 3, "max_depth": 1},
        ),
        (("finite_model",), {"max_permutation_degree": 3}),
        (
            ("amalgam",),
            {"max_amalgam_prime": 3, "max_amalgam_order": 64},
        ),
    ],
)
def test_searches_do_not_separate_a_retained_axiom_in_prop_context(
    strategies, options
):
    theory = _contextual_consequence()
    _assert_matrix_sound(theory)
    report = search_theory(
        theory,
        strategies=strategies,
        equation_ids={"contextual_consequence"},
        timeout=10,
        **options,
    )

    assert report.witnesses == {}
    assert report.unresolved == ("contextual_consequence",)
    assert not report.timed_out


def test_fixed_arity_searches_reject_non_endomorphic_input_without_a_witness():
    theory = _rectangular_prop_laws()
    report = search_theory(
        theory,
        strategies=("finite_model", "amalgam"),
        equation_ids={"naturality"},
        max_permutation_degree=3,
        max_amalgam_prime=3,
        max_amalgam_order=64,
        timeout=5,
    )

    assert report.witnesses == {}
    assert report.unresolved == ("naturality",)
    assert not report.timed_out


def test_cli_rejects_an_ill_typed_equation_before_search(tmp_path, capsys):
    path = tmp_path / "ill_typed.json"
    path.write_text(
        json.dumps(
            {
                "generators": {"gate": [1, 1]},
                "equations": [
                    {
                        "id": "bad",
                        "lhs": {"gen": "gate"},
                        "rhs": {"id": 2},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        main([str(path), "--auto", "--timeout", "1"])

    assert error.value.code == 2
    assert "equation 'bad' is ill-typed" in capsys.readouterr().err
