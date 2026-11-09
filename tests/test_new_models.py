from __future__ import annotations

import json
from itertools import combinations, pairwise
from math import gcd
from random import Random

import pytest
import z3

from minimality_auto.algebra.abelian import smith_basis
from minimality_auto.algebra.semiring_matrix import FiniteSemiring, SemiringMatrixSemantics
from minimality_auto.core import Circuit, Theory
from minimality_auto.main import main
from minimality_auto.search import Deadline, NotApplicable, SearchTimeout, search_theory
from minimality_auto.separators import actions, matrices

MATRIX_STRATEGIES = ("relations", "finite_field_matrices", "dual_numbers")


def _determinant(matrix):
    if not matrix:
        return 1
    return sum(
        (-1) ** j
        * a
        * _determinant(
            [row[:j] + row[j + 1 :] for row in matrix[1:]],
        )
        for j, a in enumerate(matrix[0])
    )


def test_smith_basis_preserves_the_integer_lattice_and_invariant_factors():
    random = Random(12)
    for columns in range(5):
        for _ in range(30):
            rows = [
                [random.randrange(-12, 13) for _ in range(columns)]
                for _ in range(random.randrange(5))
            ]
            diagonal, basis = smith_basis(rows, columns, Deadline.after(2))
            assert abs(_determinant([list(row) for row in basis])) == 1
            nonzero = [value for value in diagonal if value]
            assert all(value > 0 for value in nonzero)
            assert all(b % a == 0 for a, b in pairwise(nonzero))
            assert diagonal == tuple(nonzero + [0] * (columns - len(nonzero)))
            for row in rows:
                for i, modulus in enumerate(diagonal):
                    value = sum(a * basis[j][i] for j, a in enumerate(row))
                    assert (value % modulus if modulus else value) == 0
            product = 1
            for size in range(1, min(len(rows), columns) + 1):
                minors = (
                    abs(
                        _determinant(
                            [[rows[i][j] for j in selected_columns] for i in selected_rows]
                        )
                    )
                    for selected_rows in combinations(range(len(rows)), size)
                    for selected_columns in combinations(range(columns), size)
                )
                expected = gcd(*minors)
                product *= diagonal[size - 1]
                assert product == expected


def test_smith_reduction_honours_deadlines():
    with pytest.raises(SearchTimeout):
        smith_basis(((1,),), 1, Deadline.after(0))


def test_abelianisation_detects_torsion_beyond_the_counting_modulus_ceiling():
    theory = Theory.from_json(
        {
            "generators": {"g": 1},
            "equations": [
                {"id": "retained", "lhs": {"gen": "g", "repeat": 32}, "rhs": {"id": 1}},
                {"id": "target", "lhs": {"gen": "g", "repeat": 16}, "rhs": {"id": 1}},
            ],
        }
    )
    bounded = search_theory(
        theory, strategies=("counting",), equation_ids={"target"}, max_modulus=8, timeout=2
    )
    assert bounded.unresolved == ("target",)
    full = search_theory(
        theory, strategies=("abelianisation",), equation_ids={"target"}, max_modulus=2, timeout=2
    )
    witness = full.witnesses["target"]
    assert witness.parameters["free_rank"] == 0
    assert 32 in witness.parameters["torsion"]
    assert witness.certified
    assert witness.checked_equations == ("retained",)


def test_abelianisation_keeps_free_and_torsion_components_together():
    theory = Theory.from_json(
        {
            "generators": {"a": 1, "b": 1},
            "equations": [
                {"id": "a12", "lhs": {"gen": "a", "repeat": 12}, "rhs": {"id": 1}},
                {
                    "id": "target",
                    "lhs": {"compose": [{"gen": "a"}, {"gen": "b"}]},
                    "rhs": {"id": 1},
                },
            ],
        }
    )
    witness = search_theory(
        theory, strategies=("abelianisation",), equation_ids={"target"}, timeout=2
    ).witnesses["target"]
    assert witness.parameters["free_rank"] == 1
    assert witness.parameters["torsion"] == (2, 12)
    assert 0 in witness.parameters["coordinate_moduli"]
    assert witness.lhs_value != witness.rhs_value


def _nilpotent_theory():
    def product(*names):
        return {"compose": [{"gen": name} for name in names]}

    return Theory.from_json(
        {
            "generators": {"e": 0, "zero": 0},
            "equations": [
                {"id": "idempotent_zero", "lhs": product("zero", "zero"), "rhs": {"gen": "zero"}},
                {"id": "absorbing", "lhs": product("zero", "e"), "rhs": {"gen": "zero"}},
                {"id": "nilpotent", "lhs": product("e", "e"), "rhs": {"gen": "zero"}},
                {"id": "target", "lhs": {"gen": "e"}, "rhs": {"gen": "zero"}},
            ],
        }
    )


def test_dual_numbers_find_a_genuine_nilpotent_unavailable_over_fields():
    theory = _nilpotent_theory()
    report = search_theory(
        theory,
        strategies=("dual_numbers",),
        equation_ids={"target"},
        max_matrix_dimension=1,
        max_matrix_prime=2,
        timeout=5,
    )
    witness = report.witnesses["target"]
    assert witness.parameters["interpretation"]["e"] == (((0, 1),),)
    assert witness.parameters["interpretation"]["zero"] == (((0, 0),),)
    assert witness.parameters["ring"] == "F_2[epsilon]/(epsilon^2)"
    assert witness.certified
    for strategy in ("finite_field_matrices", "relations", "abelianisation"):
        report = search_theory(
            theory,
            strategies=(strategy,),
            equation_ids={"target"},
            max_matrix_dimension=1,
            max_matrix_prime=5,
            timeout=5,
        )
        assert not report.witnesses
        assert not report.timed_out


@pytest.mark.parametrize("strategy", MATRIX_STRATEGIES)
def test_matrix_models_synthesize_rectangular_maps_without_supplied_matrices(strategy):
    theory = Theory.from_json(
        {
            "wire_dimension": 17,
            "generators": {"state": [0, 1], "effect": [1, 0]},
            "equations": [
                {
                    "id": "retraction",
                    "lhs": {"compose": [{"gen": "state"}, {"gen": "effect"}]},
                    "rhs": {"id": 0},
                },
                {
                    "id": "target",
                    "lhs": {"compose": [{"gen": "effect"}, {"gen": "state"}]},
                    "rhs": {"id": 1},
                },
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=(strategy,),
        equation_ids={"target"},
        max_matrix_prime=2,
        max_matrix_dimension=2,
        timeout=5,
    )
    witness = report.witnesses["target"]
    assert witness.certified
    assert witness.parameters["dimension"] == 2
    assert witness.checked_equations == ("retraction",)
    matrices = witness.parameters["interpretation"]
    assert (len(matrices["state"]), len(matrices["state"][0])) == (2, 1)
    assert (len(matrices["effect"]), len(matrices["effect"][0])) == (1, 2)
    json.dumps(report.as_dict(), allow_nan=False)


def test_general_field_matrices_can_find_noncommuting_interpretations():
    theory = Theory.from_json(
        {
            "generators": {"a": 1, "b": 1},
            "equations": [
                {
                    "id": "target",
                    "lhs": {"compose": [{"gen": "a"}, {"gen": "b"}]},
                    "rhs": {"compose": [{"gen": "b"}, {"gen": "a"}]},
                }
            ],
        }
    )
    report = search_theory(
        theory, strategies=("finite_field_matrices",), max_matrix_prime=2, timeout=5
    )
    witness = report.witnesses["target"]
    assert witness.certified
    assert witness.parameters["dimension"] == 2
    assert witness.lhs_value != witness.rhs_value


def test_field_synthesis_advances_to_a_larger_characteristic():
    theory = Theory.from_json(
        {
            "generators": {"g": 1},
            "equations": [
                {"id": "order4", "lhs": {"gen": "g", "repeat": 4}, "rhs": {"id": 1}},
                {"id": "target", "lhs": {"gen": "g", "repeat": 2}, "rhs": {"id": 1}},
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("finite_field_matrices",),
        equation_ids={"target"},
        max_matrix_dimension=1,
        timeout=5,
    )
    assert report.witnesses["target"].parameters["characteristic"] == 5


def test_matrix_synthesis_visits_each_prime_dimension_pair_once(monkeypatch):
    visited = []

    def synthesize(_theory, _target, _kind, dimension, prime, *_):
        visited.append((prime, dimension))

    monkeypatch.setattr(matrices, "_synthesize", synthesize)
    search_theory(
        _nilpotent_theory(),
        strategies=("dual_numbers",),
        equation_ids={"target"},
        max_matrix_prime=3,
        max_matrix_dimension=4,
        timeout=2,
    )
    assert sorted(visited) == [(p, d) for p in (2, 3) for d in range(1, 5)]


def test_transformation_monoid_finds_nonbijective_actions():
    theory = Theory.from_json(
        {
            "generators": {"e": 1},
            "equations": [
                {"id": "idempotent", "lhs": {"gen": "e", "repeat": 2}, "rhs": {"gen": "e"}},
                {"id": "target", "lhs": {"gen": "e"}, "rhs": {"id": 1}},
            ],
        }
    )
    permutations = search_theory(
        theory,
        strategies=("finite_model",),
        equation_ids={"target"},
        max_permutation_degree=3,
        timeout=2,
    )
    assert permutations.unresolved == ("target",)
    report = search_theory(
        theory, strategies=("transformation_monoid",), equation_ids={"target"}, timeout=5
    )
    witness = report.witnesses["target"]
    assert len(set(witness.parameters["interpretation"]["generators"]["e"])) == 1
    assert witness.certified
    assert witness.checked_equations == ("idempotent",)


def test_transformation_monoid_rejects_arity_changing_signatures():
    theory = Theory.from_json(
        {
            "generators": {"grow": [1, 2]},
            "equations": [{"id": "target", "lhs": {"gen": "grow"}, "rhs": {"gen": "grow"}}],
        }
    )
    with pytest.raises(NotApplicable, match="endomorphic"):
        next(
            actions.transformations(
                theory, theory.equation("target"), bound=2, deadline=Deadline.after(2)
            )
        )


@pytest.mark.parametrize("kind", MATRIX_STRATEGIES)
def test_exact_semiring_matrix_semantics_and_candidate_cache(kind):
    ring = FiniteSemiring(kind, 5)
    theory = Theory.from_json({"generators": {"g": 1}, "equations": []})
    value = ((ring.one, ring.one), (ring.zero, ring.one))
    semantics = SemiringMatrixSemantics(2, ring, {"g": value}, Deadline.after(2))
    g = Circuit.generator(theory.signature["g"])
    square = semantics.evaluate(Circuit.compose((g, g)))
    expected = ring.one if kind == "relations" else (2, 0) if kind == "dual_numbers" else 2
    assert square == ((ring.one, expected), (ring.zero, ring.one))
    assert semantics.evaluate(g) is semantics.evaluate(g)
    assert g in semantics._cache
    other = SemiringMatrixSemantics(2, ring, {"g": ((ring.zero,) * 2,) * 2}, Deadline.after(2))
    assert other.evaluate(g) != semantics.evaluate(g)


def test_epsilon_squares_to_zero_and_has_the_correct_product_rule():
    ring = FiniteSemiring("dual_numbers", 5)
    assert ring.multiply((0, 1), (0, 1)) == (0, 0)
    assert ring.multiply((2, 3), (4, 2)) == (3, 1)
    assert ring.add(((1, 4), (4, 3))) == (0, 2)


@pytest.mark.parametrize("kind", MATRIX_STRATEGIES)
def test_wire_permutations_use_output_axis_order_for_every_semiring(kind):
    ring = FiniteSemiring(kind, 5)
    semantics = SemiringMatrixSemantics(2, ring, {}, Deadline.after(2))
    value = semantics.evaluate(Circuit.perm((1, 2, 0)))
    expected = (0, 2, 4, 6, 1, 3, 5, 7)
    for j, output in enumerate(expected):
        assert tuple(value[i][j] for i in range(8)) == tuple(
            ring.one if i == output else ring.zero for i in range(8)
        )


def test_entry_cap_also_limits_partial_compositions():
    theory = Theory.from_json(
        {
            "generators": {"shrink": [4, 0], "grow": [0, 4]},
            "equations": [],
        }
    )
    shrink = Circuit.generator(theory.signature["shrink"])
    grow = Circuit.generator(theory.signature["grow"])
    semantics = SemiringMatrixSemantics(
        2,
        FiniteSemiring("finite_field_matrices", 2),
        {"shrink": ((1,) * 16,), "grow": ((1,),) * 16},
        Deadline.after(2),
        max_entries=16,
    )
    # Every individual box and the final result fit, but the intermediate is 16x16.
    with pytest.raises(NotApplicable, match="entry limit"):
        semantics.evaluate(Circuit.compose((shrink, grow, shrink)))


@pytest.mark.parametrize("strategy", MATRIX_STRATEGIES)
def test_matrix_entry_limit_leaves_large_interpretations_unresolved(strategy):
    theory = Theory.from_json(
        {
            "generators": {},
            "equations": [{"id": "target", "lhs": {"perm": [1, 0]}, "rhs": {"id": 2}}],
        }
    )
    report = search_theory(
        theory, strategies=(strategy,), max_matrix_entries=1, max_matrix_prime=2, timeout=3
    )
    assert report.unresolved == ("target",)
    assert not report.timed_out


def test_synthesis_unknown_never_becomes_a_certificate(monkeypatch):
    theory = _nilpotent_theory()
    monkeypatch.setattr(matrices, "solve", lambda *_: None)
    monkeypatch.setattr(actions, "solve", lambda *_: None)
    report = search_theory(
        theory,
        strategies=("dual_numbers", "transformation_monoid"),
        equation_ids={"target"},
        max_matrix_prime=2,
        timeout=3,
    )
    assert not report.witnesses
    assert not report.timed_out


@pytest.mark.parametrize("value", [0, 1], ids=["breaks-retained", "does-not-separate"])
def test_common_verifier_rejects_incorrect_synthesized_matrices(monkeypatch, value):
    theory = Theory.from_json(
        {
            "generators": {"g": 1},
            "equations": [
                {"id": "retained", "lhs": {"gen": "g", "repeat": 2}, "rhs": {"id": 1}},
                {"id": "target", "lhs": {"gen": "g"}, "rhs": {"id": 1}},
            ],
        }
    )
    solver = z3.Solver()
    solver.add(z3.Int("g0_0_0") == value)
    assert solver.check() == z3.sat
    monkeypatch.setattr(matrices, "solve", lambda *_: solver.model())
    report = search_theory(
        theory,
        strategies=("finite_field_matrices",),
        equation_ids={"target"},
        max_matrix_prime=2,
        max_matrix_dimension=1,
        timeout=3,
    )
    assert not report.witnesses
    assert report.unresolved == ("target",)


def test_transformation_replay_still_checks_structural_laws(monkeypatch):
    theory = Theory.from_json(
        {
            "generators": {},
            "equations": [{"id": "target", "lhs": {"perm": [1, 0]}, "rhs": {"id": 2}}],
        }
    )
    # A three-cycle separates the target but cannot interpret a structural swap.
    solver = z3.Solver()
    solver.add(*(z3.Int(f"t0_{i}") == value for i, value in enumerate((1, 2, 0))))
    assert solver.check() == z3.sat
    monkeypatch.setattr(actions, "solve", lambda *_: solver.model())
    with pytest.raises(ValueError, match="required relation"):
        next(
            actions.transformations(
                theory,
                theory.equation("target"),
                bound=3,
                deadline=Deadline.after(2),
            )
        )


@pytest.mark.parametrize(
    "flag",
    [
        "--dual-numbers",
        "--relations",
        "--finite-field-matrices",
        "--transformation-monoid",
        "--abelianisation",
        "--abelianization",
    ],
)
def test_new_strategy_cli_flags(tmp_path, capsys, flag):
    path = tmp_path / "example.json"
    path.write_text(
        json.dumps(
            {
                "generators": {"g": 1},
                "equations": [{"id": "target", "lhs": {"gen": "g"}, "rhs": {"id": 1}}],
            }
        ),
        encoding="utf-8",
    )
    assert main([str(path), flag, "--max-matrix-prime", "2", "--timeout", "5", "--json"]) == 0
    witness = json.loads(capsys.readouterr().out)["witnesses"]["target"]
    assert witness["certified"] is True
