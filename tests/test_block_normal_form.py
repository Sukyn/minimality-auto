from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from minimality_auto.core import (
    Circuit,
    Equation,
    Theory,
    load_theory,
)
from minimality_auto.main import main
from minimality_auto.matrix import evaluate_matrix, permutation_matrix
from minimality_auto.search import Deadline, search_theory
from minimality_auto.separators.block_normal_form import (
    _block_kind,
    _logical_trace,
    _normal_form,
)

THEORY_DIR = Path(__file__).parents[1] / "theories"


def _sum_matrix(dimension: int) -> list[list[int]]:
    matrix = np.zeros((dimension**2, dimension**2), dtype=int)
    for control in range(dimension):
        for target in range(dimension):
            column = control * dimension + target
            row = control * dimension + (target + control) % dimension
            matrix[row, column] = 1
    return matrix.tolist()


def _i_equation(generator: str = "SUM", equation_id: str = "I") -> dict:
    return {
        "id": equation_id,
        "lhs": {
            "wires": 3,
            "ops": [
                {"gen": generator, "on": [1, 2]},
                {"gen": generator, "on": [0, 1]},
                {"gen": generator, "on": [0, 2]},
            ],
        },
        "rhs": {
            "wires": 3,
            "ops": [
                {"gen": generator, "on": [0, 1]},
                {"gen": generator, "on": [1, 2]},
            ],
        },
    }


def _sum_theory(
    dimension: int,
    *,
    matrix: list[list[int]] | None = None,
    retained: list[dict] | None = None,
    extra_generators: dict[str, object] | None = None,
) -> Theory:
    return Theory.from_json(
        {
            "name": f"SUM dimension {dimension}",
            "wire_dimension": dimension,
            "generators": {
                "SUM": {
                    "type": 2,
                    "matrix": matrix if matrix is not None else _sum_matrix(dimension),
                },
                **(extra_generators or {}),
            },
            "equations": [*(retained or []), _i_equation()],
        }
    )


@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_generalized_sum_i_is_separated_in_higher_dimensions(dimension):
    report = search_theory(
        _sum_theory(dimension),
        strategies=("block_normal_form",),
        equation_ids={"I"},
        max_block_matrix_dimension=dimension**3,
        timeout=10,
    )

    assert not report.timed_out
    witness = report.witnesses["I"]
    assert witness.strategy == "block_normal_form"
    assert not witness.certified
    assert witness.description.startswith("numerical ")
    assert witness.lhs_value["block_count"] == 5
    assert witness.rhs_value["block_count"] == 3
    assert [block["support"] for block in witness.lhs_value["blocks"]] == [
        "12",
        "01",
        "12",
        "01",
        "12",
    ]
    assert witness.lhs_value["signature"][-2:] == [
        "routing_swap",
        "routing_identity",
    ]
    assert witness.rhs_value["signature"][0] == "routing_identity"


def test_qutrit_clifford_open_i_case_is_now_separated():
    theory = load_theory(THEORY_DIR / "qutrit_clifford.json")
    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"I"},
        timeout=10,
    )

    witness = report.witnesses["I"]
    assert witness.checked_equations == tuple(
        equation.id for equation in theory.equations if equation.id != "I"
    )
    assert "5 versus 3 blocks" in witness.description


def test_reversing_the_words_is_routed_dynamically():
    theory = Theory.from_json(
        {
            "wire_dimension": 2,
            "generators": {"G": {"type": 2, "matrix": _sum_matrix(2)}},
            "equations": [
                {
                    "id": "reversed",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "G", "on": [0, 2]},
                            {"gen": "G", "on": [0, 1]},
                            {"gen": "G", "on": [1, 2]},
                        ],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "G", "on": [1, 2]},
                            {"gen": "G", "on": [0, 1]},
                        ],
                    },
                }
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"reversed"},
        timeout=10,
    )

    blocks = report.witnesses["reversed"].lhs_value["blocks"]
    assert blocks[0]["kind"] == "routing_swap"
    assert sum(block["kind"] == "entangling" for block in blocks) == 3


def test_the_note_example_is_found_from_boundary_types_not_counts():
    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {"G": {"type": 2, "matrix": _sum_matrix(3)}},
            "equations": [
                {
                    "id": "same_control_commutation",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "G", "on": [0, 2]},
                            {"gen": "G", "on": [0, 1]},
                        ],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "G", "on": [0, 1]},
                            {"gen": "G", "on": [0, 2]},
                        ],
                    },
                }
            ],
        }
    )

    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        timeout=10,
    )

    witness = report.witnesses["same_control_commutation"]
    assert witness.lhs_value["block_count"] == 5
    assert witness.rhs_value["block_count"] == 5
    assert witness.lhs_value["signature"] == [
        "routing_swap",
        "entangling",
        "entangling",
        "routing_swap",
        "routing_swap",
    ]
    assert witness.rhs_value["signature"] == [
        "routing_identity",
        "entangling",
        "entangling",
        "routing_swap",
        "routing_identity",
    ]


@pytest.mark.parametrize("target_position", [0, 1, 2])
@pytest.mark.parametrize("retained_order", [("alpha", "omega"), ("omega", "alpha")])
def test_gate_names_rule_names_and_equation_order_are_irrelevant(
    target_position, retained_order
):
    target = _i_equation("zebra_gate", "target_927")
    retained = {
        "alpha": {"id": "alpha", "lhs": {"id": 1}, "rhs": {"id": 1}},
        "omega": {
            "id": "omega",
            "lhs": {"gen": "zebra_gate"},
            "rhs": {"gen": "zebra_gate"},
        },
    }
    equations = [retained[name] for name in retained_order]
    equations.insert(target_position, target)
    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {
                "zebra_gate": {
                    "type": 2,
                    "matrix": _sum_matrix(3),
                }
            },
            "equations": equations,
        }
    )

    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"target_927"},
        timeout=10,
    )

    witness = report.witnesses["target_927"]
    assert witness.lhs_value["block_count"] == 5
    assert witness.rhs_value["block_count"] == 3
    assert {
        factor["generator"]
        for block in witness.lhs_value["blocks"]
        for factor in block["factors"]
        if "generator" in factor
    } == {"zebra_gate"}


def test_nested_macro_boxes_still_produce_a_positive_witness():
    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {"G": {"type": 2, "matrix": _sum_matrix(3)}},
            "macros": {
                "inner": {"gen": "G"},
                "outer": {"macro": "inner"},
            },
            "equations": [
                {
                    "id": "macro_target",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"macro": "outer", "on": [1, 2]},
                            {"macro": "outer", "on": [0, 1]},
                            {"macro": "outer", "on": [0, 2]},
                        ],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [
                            {"macro": "outer", "on": [0, 1]},
                            {"macro": "outer", "on": [1, 2]},
                        ],
                    },
                }
            ],
        }
    )

    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        timeout=10,
    )

    witness = report.witnesses["macro_target"]
    assert witness.lhs_value["block_count"] == 5
    assert witness.rhs_value["block_count"] == 3


def test_distinct_binary_generator_names_are_supported():
    names = ("ch", "cz", "h", "z", "phase")
    matrix = _sum_matrix(3)
    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {
                name: {"type": 2, "matrix": matrix}
                for name in names
            },
            "equations": [
                {
                    "id": "mixed",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "ch", "on": [1, 2]},
                            {"gen": "cz", "on": [0, 1]},
                            {"gen": "h", "on": [0, 2]},
                        ],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "z", "on": [0, 1]},
                            {"gen": "phase", "on": [1, 2]},
                        ],
                    },
                }
            ],
        }
    )

    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        timeout=10,
    )

    assert report.witnesses["mixed"].lhs_value["block_count"] == 5
    assert report.witnesses["mixed"].rhs_value["block_count"] == 3


def test_generator_names_do_not_make_equal_normal_forms_different():
    matrix = _sum_matrix(3)
    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {
                name: {"type": 2, "matrix": matrix}
                for name in ("first_name", "second_name")
            },
            "equations": [
                {
                    "id": "renaming",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "first_name", "on": [0, 1]},
                            {"gen": "first_name", "on": [1, 2]},
                        ],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "second_name", "on": [0, 1]},
                            {"gen": "second_name", "on": [1, 2]},
                        ],
                    },
                }
            ],
        }
    )

    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        timeout=10,
    )

    assert report.unresolved == ("renaming",)


@pytest.mark.parametrize("wire_map", tuple(permutations(range(3))))
def test_every_global_wire_relabeling_is_discovered(wire_map):
    def op(left, right):
        return {"gen": "G", "on": [wire_map[left], wire_map[right]]}

    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {"G": {"type": 2, "matrix": _sum_matrix(3)}},
            "equations": [
                {
                    "id": "relabelled",
                    "lhs": {
                        "wires": 3,
                        "ops": [op(1, 2), op(0, 1), op(0, 2)],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [op(0, 1), op(1, 2)],
                    },
                }
            ],
        }
    )

    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        timeout=10,
    )

    assert "relabelled" in report.witnesses


def test_operator_schmidt_test_distinguishes_all_three_brylinski_cases():
    dimension = 3
    identity = np.eye(dimension, dtype=complex)
    phase = np.diag([1, -1, 1]).astype(complex)
    local = np.kron(phase, identity)
    swap = permutation_matrix((1, 0), dimension)
    generalized_sum = np.asarray(_sum_matrix(dimension), dtype=complex)

    assert _block_kind(local, dimension)[0] == "local"
    assert _block_kind(local @ swap, dimension)[0] == "swap_local"
    assert _block_kind(generalized_sum, dimension)[0] == "entangling"


def test_compact_and_explicit_prop_routing_have_the_same_blocks():
    compact = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {
                "SUM": {
                    "type": 2,
                    "matrix": _sum_matrix(3),
                }
            },
            "equations": [
                {
                    "id": "routing",
                    "lhs": {
                        "wires": 3,
                        "ops": [{"gen": "SUM", "on": [0, 2]}],
                    },
                    "rhs": {
                        "compose": [
                            {"perm": [1, 0, 2]},
                            {"tensor": [{"id": 1}, {"gen": "SUM"}]},
                            {"perm": [1, 0, 2]},
                        ]
                    },
                }
            ],
        }
    )
    report = search_theory(
        compact,
        strategies=("block_normal_form",),
        equation_ids={"routing"},
        timeout=10,
    )

    equation = compact.equation("routing")
    assert _logical_trace(equation.lhs) == _logical_trace(equation.rhs)
    deadline = Deadline.after(10)
    assert _normal_form(
        _logical_trace(equation.lhs), {"SUM"}, deadline
    ) == _normal_form(
        _logical_trace(equation.rhs), {"SUM"}, deadline
    )
    assert report.unresolved == ("routing",)


def test_matrix_unsound_retained_equation_prevents_a_certificate():
    bad = {
        "id": "bad",
        "lhs": {"gen": "SUM"},
        "rhs": {"id": 2},
    }
    report = search_theory(
        _sum_theory(3, retained=[bad]),
        strategies=("block_normal_form",),
        equation_ids={"I"},
        timeout=10,
    )

    assert report.unresolved == ("I",)
    assert not report.witnesses


def test_retained_three_wire_equation_prevents_a_certificate():
    retained = {"id": "three", "lhs": {"id": 3}, "rhs": {"id": 3}}
    report = search_theory(
        _sum_theory(3, retained=[retained]),
        strategies=("block_normal_form",),
        equation_ids={"I"},
        timeout=10,
    )

    assert report.unresolved == ("I",)


def test_local_two_wire_generator_cannot_fake_an_irreducible_block():
    identity = np.eye(9, dtype=int).tolist()
    report = search_theory(
        _sum_theory(3, matrix=identity),
        strategies=("block_normal_form",),
        equation_ids={"I"},
        timeout=10,
    )

    assert report.unresolved == ("I",)


def test_multiple_boxes_in_one_block_are_conservatively_rejected():
    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {
                "SUM": {
                    "type": 2,
                    "matrix": _sum_matrix(3),
                }
            },
            "equations": [
                {
                    "id": "SUM3",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "SUM", "on": [0, 1]},
                            {"gen": "SUM", "on": [0, 1]},
                            {"gen": "SUM", "on": [0, 1]},
                        ],
                    },
                    "rhs": {"id": 3},
                }
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"SUM3"},
        timeout=10,
    )

    assert report.unresolved == ("SUM3",)


def test_a_lifted_retained_rule_on_nonadjacent_wires_cannot_be_separated():
    dimension = 3
    gate = np.asarray(_sum_matrix(dimension), dtype=int)
    squared = (gate @ gate).tolist()
    theory = Theory.from_json(
        {
            "wire_dimension": dimension,
            "generators": {
                "G": {"type": 2, "matrix": gate.tolist()},
                "K": {"type": 2, "matrix": squared},
            },
            "equations": [
                {
                    "id": "retained",
                    "lhs": {"compose": [{"gen": "G"}, {"gen": "G"}]},
                    "rhs": {"gen": "K"},
                },
                {
                    "id": "lifted",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "G", "on": [0, 2]},
                            {"gen": "G", "on": [0, 2]},
                        ],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [{"gen": "K", "on": [0, 2]}],
                    },
                },
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"lifted"},
        timeout=10,
    )

    assert report.unresolved == ("lifted",)


def test_a_contextual_use_of_qutrit_b_cannot_be_separated():
    original = load_theory(THEORY_DIR / "qutrit_clifford.json")
    braid = original.equation("B")
    cnot = Circuit.generator(original.signature["CNOT"])
    middle = Circuit.tensor((Circuit.identity(1), cnot))
    left_block = Circuit.tensor((braid.lhs, Circuit.identity(1)))
    right_block = Circuit.tensor((braid.rhs, Circuit.identity(1)))
    derived = Equation(
        "derived_from_B",
        Circuit.compose((left_block, middle, left_block)),
        Circuit.compose((right_block, middle, right_block)),
    )
    theory = Theory(
        original.signature,
        tuple(
            equation for equation in original.equations if equation.id != "I"
        )
        + (derived,),
        original.macros,
        "qutrit B contextual regression",
        original.wire_dimension,
    )

    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"derived_from_B"},
        timeout=10,
    )

    assert report.unresolved == ("derived_from_B",)


def test_explicit_and_compact_contextual_routing_cannot_be_separated():
    matrix = _sum_matrix(3)
    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {
                "G": {"type": 2, "matrix": matrix},
                "K": {"type": 2, "matrix": matrix},
            },
            "equations": [
                {"id": "retained", "lhs": {"gen": "G"}, "rhs": {"gen": "K"}},
                {
                    "id": "contextual_lift",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "G", "on": [0, 1]},
                            {"gen": "G", "on": [0, 2]},
                            {"gen": "G", "on": [0, 1]},
                        ],
                    },
                    "rhs": {
                        "compose": [
                            {"tensor": [{"gen": "G"}, {"id": 1}]},
                            {"tensor": [{"id": 1}, {"perm": [1, 0]}]},
                            {"tensor": [{"gen": "K"}, {"id": 1}]},
                            {"tensor": [{"id": 1}, {"perm": [1, 0]}]},
                            {"tensor": [{"gen": "G"}, {"id": 1}]},
                        ]
                    },
                },
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"contextual_lift"},
        timeout=10,
    )

    assert report.unresolved == ("contextual_lift",)


def test_structural_swap_hidden_in_a_macro_cannot_fake_a_separator():
    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {"G": {"type": 2, "matrix": _sum_matrix(3)}},
            "macros": {"P": {"perm": [1, 0]}},
            "equations": [
                {
                    "id": "naturality",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"macro": "P", "on": [0, 2]},
                            {"gen": "G", "on": [0, 1]},
                            {"macro": "P", "on": [0, 2]},
                        ],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [{"gen": "G", "on": [2, 1]}],
                    },
                }
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"naturality"},
        timeout=10,
    )

    assert report.unresolved == ("naturality",)


def test_entangling_macros_cannot_hide_structural_routing():
    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {"G": {"type": 2, "matrix": _sum_matrix(3)}},
            "macros": {
                "T": {
                    "compose": [{"perm": [1, 0]}, {"gen": "G"}]
                },
                "R": {
                    "compose": [
                        {"perm": [1, 0]},
                        {"gen": "G"},
                        {"perm": [1, 0]},
                    ]
                },
            },
            "equations": [
                {
                    "id": "coherence",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"macro": "T", "on": [0, 1]},
                            {"gen": "G", "on": [1, 2]},
                            {"gen": "G", "on": [0, 2]},
                            {"gen": "G", "on": [1, 2]},
                            {"gen": "G", "on": [0, 1]},
                        ],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [
                            {"macro": "R", "on": [0, 1]},
                            {"gen": "G", "on": [0, 2]},
                            {"gen": "G", "on": [1, 2]},
                            {"gen": "G", "on": [0, 2]},
                            {"macro": "T", "on": [0, 1]},
                        ],
                    },
                }
            ],
        }
    )
    equation = theory.equation("coherence")
    left = evaluate_matrix(
        equation.lhs, theory.signature, theory.wire_dimension
    )
    right = evaluate_matrix(
        equation.rhs, theory.signature, theory.wire_dimension
    )
    np.testing.assert_allclose(left, right, atol=1e-8, rtol=1e-8)

    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"coherence"},
        timeout=10,
    )

    assert report.unresolved == ("coherence",)


def test_swap_local_primitive_cannot_hide_routing_inside_entangling_blocks():
    theory = Theory.from_json(
        {
            "wire_dimension": 3,
            "generators": {
                "G": {"type": 2, "matrix": _sum_matrix(3)},
                "P": {
                    "type": 2,
                    "matrix": permutation_matrix((1, 0), 3)
                    .real.astype(int)
                    .tolist(),
                },
            },
            "equations": [
                {"id": "Pswap", "lhs": {"gen": "P"}, "rhs": {"perm": [1, 0]}},
                {
                    "id": "derived",
                    "lhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "P", "on": [0, 1]},
                            {"gen": "G", "on": [0, 1]},
                            {"gen": "G", "on": [1, 2]},
                            {"gen": "G", "on": [0, 2]},
                            {"gen": "G", "on": [1, 2]},
                            {"gen": "G", "on": [0, 1]},
                        ],
                    },
                    "rhs": {
                        "wires": 3,
                        "ops": [
                            {"gen": "P", "on": [0, 1]},
                            {"gen": "G", "on": [0, 1]},
                            {"gen": "P", "on": [0, 1]},
                            {"gen": "G", "on": [0, 2]},
                            {"gen": "G", "on": [1, 2]},
                            {"gen": "G", "on": [0, 2]},
                            {"gen": "P", "on": [0, 1]},
                            {"gen": "G", "on": [0, 1]},
                        ],
                    },
                },
            ],
        }
    )
    equation = theory.equation("derived")
    left = evaluate_matrix(
        equation.lhs, theory.signature, theory.wire_dimension
    )
    right = evaluate_matrix(
        equation.rhs, theory.signature, theory.wire_dimension
    )
    np.testing.assert_allclose(left, right, atol=1e-8, rtol=1e-8)

    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"derived"},
        timeout=10,
    )

    assert report.unresolved == ("derived",)


def test_nonunitary_generator_is_rejected():
    nonunitary = (2 * np.eye(9, dtype=int)).tolist()
    report = search_theory(
        _sum_theory(3, matrix=nonunitary),
        strategies=("block_normal_form",),
        equation_ids={"I"},
        timeout=10,
    )

    assert report.unresolved == ("I",)


def test_nonidentity_net_routing_is_not_guessed_about():
    theory = Theory.from_json(
        {
            "generators": {},
            "equations": [
                {
                    "id": "routing",
                    "lhs": {"perm": [1, 0, 2]},
                    "rhs": {"perm": [1, 0, 2]},
                }
            ],
        }
    )
    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"routing"},
        timeout=10,
    )

    assert report.unresolved == ("routing",)


def test_arity_changing_signature_is_rejected_even_when_unused():
    theory = _sum_theory(
        3,
        extra_generators={
            "state": {
                "type": [0, 1],
                "matrix": [[1], [0], [0]],
            }
        },
    )
    report = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"I"},
        timeout=10,
    )

    assert report.unresolved == ("I",)


def test_block_matrix_dimension_bound_is_enforced():
    theory = _sum_theory(5)
    too_small = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"I"},
        max_block_matrix_dimension=124,
        timeout=10,
    )
    enough = search_theory(
        theory,
        strategies=("block_normal_form",),
        equation_ids={"I"},
        max_block_matrix_dimension=125,
        timeout=10,
    )

    assert too_small.unresolved == ("I",)
    assert "I" in enough.witnesses


@pytest.mark.parametrize("flag", ["--block-normal-form", "--brylinski"])
def test_block_normal_form_cli_aliases(capsys, flag):
    result = main(
        [
            str(THEORY_DIR / "qutrit_clifford.json"),
            flag,
            "--equation",
            "I",
            "--timeout",
            "10",
            "--json",
        ]
    )

    assert result == 1
    output = capsys.readouterr().out
    assert '"strategy": "block_normal_form"' in output
    assert '"certified": false' in output
