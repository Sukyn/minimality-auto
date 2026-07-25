from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minimality_auto.core import Circuit, load_theory
from minimality_auto.matrix import evaluate_matrix
from minimality_auto.search import search_theory

THEORY_DIR = Path(__file__).parents[1] / "theories"
FULL_PATH = THEORY_DIR / "real_clifford_ch.json"
SIMPLIFIED_PATH = THEORY_DIR / "realclifford-ch-simplified.json"


def test_full_real_clifford_ch_theory_is_sound():
    theory = load_theory(FULL_PATH)
    for equation in theory.equations:
        if equation.id.startswith("19_") and equation.lhs.inputs > 6:
            continue
        left = evaluate_matrix(equation.lhs, theory.signature, theory.wire_dimension)
        right = evaluate_matrix(equation.rhs, theory.signature, theory.wire_dimension)
        np.testing.assert_allclose(left, right, atol=1e-8, rtol=1e-8)


def test_simplified_theory_differs_only_by_rules_4_9_and_16():
    full = json.loads(FULL_PATH.read_text(encoding="utf-8"))
    simplified = json.loads(SIMPLIFIED_PATH.read_text(encoding="utf-8"))
    full_equations = full.pop("equations")
    simplified_equations = simplified.pop("equations")

    assert simplified == full
    assert simplified_equations == [
        equation
        for equation in full_equations
        if equation["id"] not in {"4", "9", "16"}
    ]


def test_shortcut_matrices_match_their_published_semantics():
    theory = load_theory(FULL_PATH)
    cosine, sine = np.cos(np.pi / 8), np.sin(np.pi / 8)
    p = np.array([[cosine, sine], [sine, -cosine]])
    p_pair = evaluate_matrix(
        theory.macros["P_pair"], theory.signature, theory.wire_dimension
    )
    np.testing.assert_allclose(p_pair, np.kron(p, p), atol=1e-8, rtol=1e-8)

    for wires in range(5, 9):
        actual = evaluate_matrix(
            theory.macros[f"MCZ_pad_{wires}"],
            theory.signature,
            theory.wire_dimension,
        )
        expected = np.eye(2**wires)
        expected[-2, -2] = expected[-1, -1] = -1
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)


def test_rule_19_targets_are_materialized_through_n10():
    theory = load_theory(FULL_PATH)

    for wires in range(5, 11):
        tail = list(range(wires - 3, wires))
        previous_wires = [*range(wires - 3), wires - 1, wires - 2]
        name = f"MCZ_pad_{wires}"
        expected_body = Circuit.from_json(
            {
                "wires": wires,
                "ops": [
                    {"macro": "MCZX_2", "on": tail},
                    {"macro": f"MCZ_pad_{wires - 1}", "on": previous_wires},
                    {"macro": "MCXZ_2", "on": tail},
                    {"macro": f"MCZ_pad_{wires - 1}", "on": previous_wires},
                ],
            },
            theory.signature,
            theory.macros,
        )
        assert theory.macros[name] == expected_body

        equation = theory.equation(f"19_n{wires}")
        macro_op = {"macro": name, "on": list(range(wires))}
        z_op = {"gen": "Z", "on": [wires - 1]}
        assert equation.lhs == Circuit.from_json(
            {"wires": wires, "ops": [z_op, macro_op]},
            theory.signature,
            theory.macros,
        )
        assert equation.rhs == Circuit.from_json(
            {"wires": wires, "ops": [macro_op, z_op]},
            theory.signature,
            theory.macros,
        )


def test_counting_search_finds_new_witnesses_for_rules_15_and_18():
    theory = load_theory(SIMPLIFIED_PATH)
    report = search_theory(
        theory,
        strategies=("counting",),
        equation_ids={"15", "18"},
        max_modulus=2,
        timeout=5,
    )
    assert set(report.witnesses) == {"15", "18"}
    assert report.witnesses["15"].parameters == {
        "modulus": 2,
        "weights": {"CH": 1, "CZ": 1},
        "swap": 1,
    }
    assert report.witnesses["18"].parameters == {
        "modulus": 2,
        "weights": {"CH": 1},
        "swap": 0,
    }
