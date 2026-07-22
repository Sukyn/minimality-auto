from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from minimality_auto.core import (
    Circuit,
    Equation,
    Generator,
    Signature,
    Theory,
    evaluate_matrix,
    expand_macros,
    load_theory,
)
from minimality_auto.main import main
from minimality_auto.search import (
    Deadline,
    Separation,
    _validate_direct_separation,
    relevant_equations,
    search_theory,
)
from minimality_auto.separators import amalgam


THEORY_PATH = (
    Path(__file__).parents[1] / "theories" / "realclifford-ch-simplified.json"
)


@pytest.fixture(scope="module")
def theory() -> Theory:
    return load_theory(THEORY_PATH)


def _with_equations(theory: Theory, equations: tuple[Equation, ...]) -> Theory:
    return Theory(
        theory.signature,
        equations,
        dict(theory.macros),
        theory.name,
        theory.wire_dimension,
    )


def _rename_generators(raw: dict, renaming: dict[str, str]) -> dict:
    def rename(value):
        if isinstance(value, list):
            return [rename(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: rename(item) for key, item in value.items()}
        if isinstance(result.get("gen"), str):
            result["gen"] = renaming[result["gen"]]
        return result

    renamed = rename(deepcopy(raw))
    for generator in renamed["generators"]:
        generator["name"] = renaming[generator["name"]]
    return renamed


def _replace_first_generator(
    term: Circuit, name: str, replacement: Circuit
) -> tuple[Circuit, bool]:
    replaced = False

    def visit(node: Circuit) -> Circuit:
        nonlocal replaced
        if node.kind == "gen" and node.name == name and not replaced:
            replaced = True
            return replacement
        if node.kind == "compose":
            return Circuit.compose(visit(part) for part in node.parts)
        if node.kind == "tensor":
            return Circuit.tensor(visit(part) for part in node.parts)
        return node

    return visit(term), replaced


def _assert_matrix_sound(theory: Theory, equation: Equation) -> None:
    np.testing.assert_allclose(
        evaluate_matrix(
            equation.lhs,
            theory.signature,
            theory.macros,
            theory.wire_dimension,
        ),
        evaluate_matrix(
            equation.rhs,
            theory.signature,
            theory.macros,
            theory.wire_dimension,
        ),
        atol=1e-8,
        rtol=1e-8,
    )


def test_amalgam_cli_separates_rule_11(capsys: pytest.CaptureFixture[str]):
    assert main(
        [
            str(THEORY_PATH),
            "--amalgam",
            "--equation",
            "11",
            "--timeout",
            "30",
            "--json",
        ]
    ) == 0
    witness = json.loads(capsys.readouterr().out)["witnesses"]["11"]

    assert witness["strategy"] == "amalgam"
    assert witness["checked_equations"] == ["1", "2", "3", "5", "6", "7", "8", "10"]
    assert set(witness["parameters"]["interpretation"]) == {"H", "Z", "CZ", "CH"}
    assert witness["parameters"]["structural_swaps"] == {"swap[0,1]": [1, 0]}
    assert "quotient_seed_matrix" in witness["parameters"]["interpretation"]["CH"]
    assert witness["lhs_value"] != witness["rhs_value"]


def test_amalgam_does_not_depend_on_the_target_id(theory: Theory):
    target = theory.equation("11")
    renamed = Equation("renamed_rule_11", target.lhs, target.rhs, target.metadata)
    renamed_theory = _with_equations(
        theory,
        tuple(renamed if equation.id == "11" else equation for equation in theory.equations),
    )

    report = search_theory(
        renamed_theory,
        strategies=("amalgam",),
        equation_ids={renamed.id},
        timeout=30,
    )

    assert set(report.witnesses) == {renamed.id}
    assert report.witnesses[renamed.id].equation == renamed.id
    assert not report.unresolved


def test_amalgam_normal_form_preserves_composition_order(theory: Theory):
    deadline = Deadline.after(30)
    model = next(
        amalgam.candidates(
            theory,
            theory.equation("11"),
            bound=1,
            deadline=deadline,
            max_amalgam_prime=7,
        )
    )

    h = Circuit.generator(theory.signature["H"])
    z = Circuit.generator(theory.signature["Z"])
    hz = model.evaluate(Circuit.compose((h, z)))
    zh = model.evaluate(Circuit.compose((z, h)))

    assert hz != zh


def test_amalgam_discovers_roles_after_every_generator_is_renamed():
    raw = json.loads(THEORY_PATH.read_text(encoding="utf-8"))
    renaming = {"H": "p", "Z": "q", "CZ": "r", "CH": "b"}
    raw = _rename_generators(raw, renaming)
    for equation in raw["equations"]:
        equation["id"] = f"rule_{equation['id']}"
        equation["lhs"], equation["rhs"] = equation["rhs"], equation["lhs"]
    raw["equations"].reverse()
    renamed = Theory.from_json(raw)

    report = search_theory(
        renamed,
        strategies=("amalgam",),
        equation_ids={"rule_11"},
        max_amalgam_prime=7,
        timeout=30,
    )

    witness = report.witnesses["rule_11"]
    assert witness.parameters["bridge_generators"] == ["b"]
    assert witness.parameters["prime"] == 7
    assert witness.parameters["factor_orders"] == {
        "C": 2304,
        "A": 256,
        "Q": 2,
        "D": 512,
    }


def test_amalgam_increases_the_prime_bound(theory: Theory):
    too_small = search_theory(
        theory,
        strategies=("amalgam",),
        equation_ids={"11"},
        max_amalgam_prime=5,
        timeout=10,
    )
    enough = search_theory(
        theory,
        strategies=("amalgam",),
        equation_ids={"11"},
        max_amalgam_prime=7,
        timeout=30,
    )

    assert too_small.unresolved == ("11",)
    assert enough.witnesses["11"].parameters["prime"] == 7


def test_unused_complex_generator_does_not_block_amalgam_search():
    raw = json.loads(THEORY_PATH.read_text(encoding="utf-8"))
    raw["generators"].append(
        {
            "name": "unused",
            "source": 1,
            "target": 1,
            "matrix": [[1, "i"], [0, 1]],
        }
    )
    extended = Theory.from_json(raw)

    report = search_theory(
        extended,
        strategies=("amalgam",),
        equation_ids={"11"},
        max_amalgam_prime=7,
        timeout=30,
    )

    unused = report.witnesses["11"].parameters["interpretation"]["unused"]
    assert unused == {"factor": "C", "matrix": [[1, 0], [0, 1]]}


def test_amalgam_separates_an_unrelated_one_wire_theory():
    cycle = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    reflection = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    toy = Theory.from_json(
        {
            "name": "generic_amalgam_toy",
            "wire_dimension": 3,
            "generators": [
                {"name": "cycle", "source": 1, "target": 1, "matrix": cycle},
                {"name": "left_flip", "source": 1, "target": 1, "matrix": reflection},
                {"name": "bridge", "source": 1, "target": 1, "matrix": reflection},
            ],
            "equations": [
                {
                    "id": "cycle3",
                    "lhs": {"compose": [{"gen": "cycle"}] * 3},
                    "rhs": {"id": 1},
                },
                {
                    "id": "left_flip2",
                    "lhs": {"compose": [{"gen": "left_flip"}] * 2},
                    "rhs": {"id": 1},
                },
                {
                    "id": "bridge2",
                    "lhs": {"compose": [{"gen": "bridge"}] * 2},
                    "rhs": {"id": 1},
                },
                {
                    "id": "left_action",
                    "lhs": {
                        "compose": [
                            {"gen": "left_flip"},
                            {"gen": "cycle"},
                            {"gen": "left_flip"},
                        ]
                    },
                    "rhs": {"compose": [{"gen": "cycle"}] * 2},
                },
                {
                    "id": "bridge_action",
                    "lhs": {
                        "compose": [
                            {"gen": "bridge"},
                            {"gen": "cycle"},
                            {"gen": "bridge"},
                        ]
                    },
                    "rhs": {"compose": [{"gen": "cycle"}] * 2},
                },
                {
                    "id": "target",
                    "lhs": {
                        "compose": [{"gen": "bridge"}, {"gen": "left_flip"}]
                    },
                    "rhs": {
                        "compose": [{"gen": "left_flip"}, {"gen": "bridge"}]
                    },
                },
            ],
        }
    )

    report = search_theory(
        toy,
        strategies=("amalgam",),
        equation_ids={"target"},
        max_amalgam_prime=2,
        timeout=10,
    )

    witness = report.witnesses["target"]
    assert witness.parameters["bridge_generators"] == ["bridge"]
    assert witness.parameters["prime"] == 2
    assert witness.parameters["factor_orders"] == {
        "C": 6,
        "A": 3,
        "Q": 2,
        "D": 6,
    }


def test_amalgam_can_search_two_bridge_generators():
    cycle = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    flip_12 = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    flip_01 = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]

    def composed(*names: str) -> dict[str, list[dict[str, str]]]:
        return {"compose": [{"gen": name} for name in names]}

    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": [
                {"name": "c", "source": 1, "target": 1, "matrix": cycle},
                {"name": "d", "source": 1, "target": 1, "matrix": flip_12},
                {"name": "b1", "source": 1, "target": 1, "matrix": flip_12},
                {"name": "b2", "source": 1, "target": 1, "matrix": flip_01},
            ],
            "equations": [
                {"id": "c3", "lhs": composed("c", "c", "c"), "rhs": {"id": 1}},
                {"id": "d2", "lhs": composed("d", "d"), "rhs": {"id": 1}},
                {"id": "b1_2", "lhs": composed("b1", "b1"), "rhs": {"id": 1}},
                {"id": "b2_2", "lhs": composed("b2", "b2"), "rhs": {"id": 1}},
                {"id": "d_action", "lhs": composed("d", "c", "d"), "rhs": composed("c", "c")},
                {"id": "b1_action", "lhs": composed("b1", "c", "b1"), "rhs": composed("c", "c")},
                {"id": "b2_action", "lhs": composed("b2", "c", "b2"), "rhs": composed("c", "c")},
                {
                    "id": "target",
                    "lhs": composed("b1", "b2", "d"),
                    "rhs": composed("d", "b1", "b2"),
                },
            ],
        }
    )

    models = list(
        amalgam.candidates(
            theory,
            theory.equation("target"),
            bound=1,
            deadline=Deadline.after(10),
            max_amalgam_prime=2,
            max_amalgam_order=100,
            max_amalgam_bridge_generators=2,
        )
    )

    assert any(len(model.parameters["bridge_generators"]) == 2 for model in models)


@pytest.mark.parametrize(
    ("strategy", "target"),
    [("amalgam", "11"), ("spin", "19_n5")],
)
def test_false_retained_equation_blocks_direct_certificates(
    theory: Theory, strategy: str, target: str
):
    false_axiom = Equation(
        "false_H_identity",
        Circuit.generator(theory.signature["H"]),
        Circuit.identity(1),
    )
    extended = _with_equations(theory, (*theory.equations, false_axiom))

    report = search_theory(
        extended,
        strategies=(strategy,),
        equation_ids={target},
        timeout=30,
    )

    assert not report.witnesses
    assert report.unresolved == (target,)
    assert not report.timed_out


def test_spin_separates_every_materialized_high_arity_instance(theory: Theory):
    targets = {f"19_n{arity}" for arity in range(5, 11)}
    report = search_theory(
        theory,
        strategies=("spin",),
        equation_ids=targets,
        timeout=60,
    )

    assert set(report.witnesses) == targets
    assert not report.unresolved
    assert not report.timed_out

    for arity in range(5, 11):
        equation_id = f"19_n{arity}"
        witness = report.witnesses[equation_id]
        expected_checked = tuple(
            equation.id
            for equation in relevant_equations(theory, theory.equation(equation_id))
        )
        assert witness.checked_equations == expected_checked
        assert witness.parameters["minus_dimensions"] == [1 << (arity - 1), 2]
        assert witness.parameters["intersection_dimension"] == 1
        assert witness.parameters["commutator_sign"] == -1
        assert witness.lhs_value["spin_lift"] == "u"
        assert witness.rhs_value["spin_lift"] == "-u"

        assert witness.parameters["full_arity_retained_reversals"] == []


def test_spin_one_spare_wire_uses_matrix_equality_not_schema_metadata(theory: Theory):
    lower = theory.equation("19_n5")
    upper = theory.equation("19_n6")
    renamed_lower = Equation("unrelated retained name", lower.lhs, lower.rhs)
    renamed_upper = Equation("unrelated target name", upper.lhs, upper.rhs)
    ordinary = tuple(
        equation
        for equation in theory.equations
        if not equation.id.startswith("19_n")
    )
    renamed = _with_equations(
        theory,
        (*ordinary, renamed_upper, renamed_lower),
    )

    report = search_theory(
        renamed,
        strategies=("spin",),
        equation_ids={renamed_upper.id},
        timeout=20,
    )

    witness = report.witnesses[renamed_upper.id]
    assert renamed_lower.id in witness.checked_equations
    assert witness.parameters["full_arity_retained_reversals"] == []


def test_spin_does_not_depend_on_active_generator_names():
    raw = json.loads(THEORY_PATH.read_text(encoding="utf-8"))
    renaming = {"H": "p", "Z": "q", "CZ": "r", "CH": "b"}
    renamed = Theory.from_json(_rename_generators(raw, renaming))

    report = search_theory(
        renamed,
        strategies=("spin",),
        equation_ids={"19_n6"},
        timeout=20,
    )

    witness = report.witnesses["19_n6"]
    assert set(witness.parameters["primitive_reflections"]) == set(renaming.values())


def test_spin_rejects_a_retained_reversal_with_negative_lift_sign(theory: Theory):
    original = theory.equation("19_n6")
    retained = Equation("retained negative reversal", original.lhs, original.rhs)
    target = Equation("duplicate target reversal", original.lhs, original.rhs)
    ordinary = tuple(
        equation
        for equation in theory.equations
        if not equation.id.startswith("19_n")
    )
    duplicated = _with_equations(theory, (*ordinary, retained, target))

    report = search_theory(
        duplicated,
        strategies=("spin",),
        equation_ids={target.id},
        timeout=20,
    )

    assert not report.witnesses
    assert report.unresolved == (target.id,)
    assert not report.timed_out


def test_spin_one_spare_wire_accepts_matrix_equal_aliases(theory: Theory):
    lower = theory.equation("19_n5")
    target = theory.equation("19_n6")
    alias = Generator("matrix-only alias", 2, 2, theory.signature["CH"].matrix)
    signature = Signature((*theory.signature.values(), alias))
    right, replaced = _replace_first_generator(
        expand_macros(lower.rhs, theory.macros),
        "CH",
        Circuit.generator(alias),
    )

    retained = Equation(
        "matrix-equal retained row",
        expand_macros(lower.lhs, theory.macros),
        right,
    )
    assert replaced
    ordinary = tuple(
        equation
        for equation in theory.equations
        if not equation.id.startswith("19_n")
    )
    adversarial = Theory(
        signature,
        (*ordinary, retained, target),
        theory.macros,
        theory.name,
        theory.wire_dimension,
    )

    _assert_matrix_sound(adversarial, retained)
    report = search_theory(
        adversarial,
        strategies=("spin",),
        equation_ids={target.id},
        timeout=20,
    )

    assert set(report.witnesses) == {target.id}
    assert retained.id in report.witnesses[target.id].checked_equations
    assert not report.unresolved
    assert not report.timed_out


def test_spin_normalizes_identity_factors_in_an_exact_reversal(theory: Theory):
    original = theory.equation("19_n5")
    right_parts = list(original.rhs.parts)
    first_nonidentity = next(
        index for index, part in enumerate(right_parts) if part.kind != "id"
    )
    right_parts.insert(first_nonidentity + 1, Circuit.identity(original.rhs.inputs))
    padded = Equation(
        "identity_padded_factor_reversal",
        original.lhs,
        Circuit.compose(right_parts),
    )
    reduced = tuple(
        equation
        for equation in theory.equations
        if equation.id not in {"19_n5", "19_n6"}
    )
    padded_theory = _with_equations(theory, (*reduced, padded))

    _assert_matrix_sound(theory, padded)
    left = tuple(part for part in padded.lhs.parts if part.kind != "id")
    right = tuple(part for part in padded.rhs.parts if part.kind != "id")
    assert any(right == left[index:] + left[:index] for index in range(1, len(left)))

    report = search_theory(
        padded_theory,
        strategies=("spin",),
        equation_ids={padded.id},
        timeout=30,
    )

    assert set(report.witnesses) == {padded.id}
    assert not report.unresolved
    assert not report.timed_out


def test_spin_rejects_only_matrix_equal_nonidentical_factors(theory: Theory):
    original = theory.equation("19_n5")
    alias = Generator("CH alias", 2, 2, theory.signature["CH"].matrix)
    signature = Signature((*theory.signature.values(), alias))
    right, replaced = _replace_first_generator(
        expand_macros(original.rhs, theory.macros),
        "CH",
        Circuit.generator(alias),
    )

    target = Equation(
        "matrix_only_factor_reversal",
        expand_macros(original.lhs, theory.macros),
        right,
    )
    assert replaced
    reduced = tuple(
        equation
        for equation in theory.equations
        if equation.id not in {"19_n5", "19_n6"}
    )
    matrix_only_theory = Theory(
        signature,
        (*reduced, target),
        theory.macros,
        theory.name,
        theory.wire_dimension,
    )

    _assert_matrix_sound(matrix_only_theory, target)
    report = search_theory(
        matrix_only_theory,
        strategies=("spin",),
        equation_ids={target.id},
        timeout=30,
    )

    assert not report.witnesses
    assert report.unresolved == (target.id,)
    assert not report.timed_out


def test_direct_certificate_must_list_every_retained_equation(theory: Theory):
    target = theory.equation("19_n5")
    incomplete = Separation(
        equation=target.id,
        strategy="spin",
        description="incomplete",
        parameters={},
        checked_equations=(),
        lhs_value="u",
        rhs_value="-u",
    )

    with pytest.raises(ValueError, match="did not check every retained equation"):
        _validate_direct_separation(incomplete, theory, target, "spin")
