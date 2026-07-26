import json
from collections import Counter

import numpy as np
import pytest

from minimality_auto.core import (
    Circuit,
    Generator,
    Signature,
    Theory,
    ValidationError,
    analyze_circuit,
    contains_arity_change,
    iter_circuit_nodes,
    load_json,
    parse_complex,
)
from minimality_auto.matrix import (
    MatrixSemantics,
    basis_wire_permutation,
    permutation_matrix,
)


@pytest.fixture
def signature():
    return Signature.from_json({"u": 1, "mul": [2, 1], "unit": [0, 1]})


def test_ast_is_typed_and_composition_runs_left_to_right(signature):
    circuit = Circuit.from_json(
        {
            "compose": [
                {"tensor": [{"gen": "u"}, {"gen": "u"}]},
                {"gen": "mul"},
                {"gen": "u"},
            ]
        },
        signature,
    )
    assert circuit.type == (2, 1)
    assert analyze_circuit(circuit).occurrence_counter() == Counter(u=3, mul=1)

    with pytest.raises(ValidationError, match="composition type mismatch"):
        Circuit.from_json({"compose": [{"gen": "mul"}, {"gen": "mul"}]}, signature)


def test_circuit_constructors_apply_prop_unit_laws(signature):
    u = Circuit.generator(signature["u"])

    assert Circuit.perm([0, 1]) == Circuit.identity(2)
    assert Circuit.compose((Circuit.identity(1), u, Circuit.identity(1))) == u
    assert Circuit.compose((Circuit.identity(1), Circuit.identity(1))) == Circuit.identity(1)
    assert Circuit.tensor((Circuit.identity(0), u, Circuit.identity(0))) == u
    assert Circuit.tensor((Circuit.identity(0), Circuit.identity(0))) == Circuit.identity(0)
    assert Circuit.tensor((Circuit.identity(1), Circuit.identity(2))) == Circuit.identity(3)
    assert Circuit.compose((Circuit.compose((u, u)), u)).parts == (u, u, u)
    assert Circuit.tensor((Circuit.tensor((u, u)), u)).parts == (u, u, u)

    with pytest.raises(ValidationError, match="composition type mismatch"):
        Circuit.compose((Circuit.identity(1), Circuit.identity(2)))


def test_scalar_type_is_endomorphism_shorthand_without_losing_prop_types():
    signature = Signature.from_json(
        {
            "scalar": 0,
            "H": 1,
            "copy": [1, 2],
        }
    )

    assert signature["scalar"].type == (0, 0)
    assert signature["H"].type == (1, 1)
    assert signature["copy"].type == (1, 2)

    theory = Theory.from_json(
        {
            "generators": {"H": 1},
            "macros": {"H2": {"compose": [{"gen": "H"}, {"gen": "H"}]}},
            "equations": [],
        }
    )
    assert theory.macros["H2"].type == (1, 1)


@pytest.mark.parametrize("bad_type", [True, -1, [1], [1, -1], [1, 1]])
def test_scalar_and_pair_types_remain_strict(bad_type):
    with pytest.raises(ValidationError):
        Signature.from_json({"bad": bad_type})


def test_generator_specs_reject_noncanonical_fields():
    with pytest.raises(ValidationError, match="unknown generator"):
        Signature.from_json({"bad": {"type": 1, "inputs": 1, "outputs": 1}})
    with pytest.raises(ValidationError, match="bare type"):
        Signature.from_json({"bad": {"type": 1}})


def test_generators_is_not_a_reserved_generator_name():
    signature = Signature.from_json({"generators": 1})

    assert signature["generators"].type == (1, 1)


def test_permutations_and_compact_wiring(signature):
    swap_then_mul = Circuit.from_json(
        {"wires": 2, "ops": [{"perm": [1, 0]}, {"gen": "mul"}]},
        signature,
    )
    assert swap_then_mul.type == (2, 1)
    assert analyze_circuit(swap_then_mul).occurrence_counter() == Counter(mul=1)
    assert analyze_circuit(swap_then_mul).permutation_parity == 1

    nonadjacent = Circuit.from_json({"wires": 3, "ops": [{"gen": "mul", "on": [2, 0]}]}, signature)
    assert nonadjacent.type == (3, 2)
    assert Circuit.from_json({"wires": 2, "ops": [{"gen": "u", "on": 1}]}, signature).type == (2, 2)

    with pytest.raises(ValidationError, match="more than once"):
        Circuit.from_json({"wires": 2, "ops": [{"gen": "mul", "on": [0, 0]}]}, signature)
    with pytest.raises(ValidationError, match="integer on"):
        Circuit.from_json({"wires": 2, "ops": [{"gen": "mul", "on": 0}]}, signature)
    with pytest.raises(ValidationError, match="needs an on array"):
        Circuit.from_json({"wires": 3, "ops": [{"gen": "mul"}]}, signature)


def test_compact_parser_canonicalizes_units_and_validates_permutations(signature):
    assert Circuit.from_json({"wires": 1, "ops": [{"gen": "u"}]}, signature) == Circuit.generator(
        signature["u"]
    )

    with pytest.raises(ValidationError, match="permutation must be an array"):
        Circuit.from_json(
            {"wires": 2, "ops": [{"perm": 1}]},
            signature,
        )


def test_compact_operations_support_bounded_repetition(signature):
    circuit = Circuit.from_json(
        {"wires": 1, "ops": [{"gen": "u", "repeat": 3}]},
        signature,
    )
    assert analyze_circuit(circuit).occurrence_counter() == Counter(u=3)
    assert Circuit.from_json({"gen": "u", "repeat": 3}, signature) == circuit

    with pytest.raises(ValidationError, match="repeat must be positive"):
        Circuit.from_json({"wires": 1, "ops": [{"gen": "u", "repeat": 0}]}, signature)
    with pytest.raises(ValidationError, match="repeat needs an endomorphism"):
        Circuit.from_json({"wires": 2, "ops": [{"gen": "mul", "repeat": 2}]}, signature)
    with pytest.raises(ValidationError, match="exceeds 10000"):
        Circuit.from_json({"wires": 1, "ops": [{"gen": "u", "repeat": 10_001}]}, signature)


def test_macros_are_expanded_during_loading_and_cycles_fail():
    theory = Theory.from_json(
        {
            "generators": {"u": 1},
            "macros": {
                "twice": {"gen": "u", "repeat": 2},
                "four": {"macro": "twice", "repeat": 2},
            },
            "equations": [{"id": "e", "lhs": {"macro": "four"}, "rhs": {"gen": "u"}}],
        }
    )
    lhs = theory.equations[0].lhs
    assert analyze_circuit(lhs).occurrence_counter() == Counter(u=4)
    assert lhs is theory.macros["four"]
    assert lhs.parts == 2 * theory.macros["twice"].parts
    assert all(
        node.kind != "macro"
        for term in (*theory.macros.values(), lhs)
        for node in iter_circuit_nodes(term)
    )
    assert "macro" not in json.dumps(lhs.to_json())

    with pytest.raises(ValidationError, match=r"macro cycle: a -> b -> a"):
        Theory.from_json(
            {
                "generators": {"u": 1},
                "macros": {
                    "a": {"macro": "b"},
                    "b": {"macro": "a"},
                },
                "equations": [],
            }
        )


def test_duplicate_ids_and_keys_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="duplicate equation id"):
        Theory.from_json(
            {
                "generators": {"u": 1},
                "equations": [
                    {"id": "same", "lhs": {"gen": "u"}, "rhs": {"gen": "u"}},
                    {"id": "same", "lhs": {"gen": "u"}, "rhs": {"gen": "u"}},
                ],
            }
        )
    path = tmp_path / "duplicate.json"
    path.write_text('{"equations": [], "equations": []}', encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate JSON key"):
        load_json(path)


def test_matrix_semantics_and_overrides():
    signature = Signature.from_json({"x": {"type": 1, "matrix": [[0, 1], [1, 0]]}})
    circuit = Circuit.from_json({"compose": [{"gen": "x"}, {"gen": "x"}]}, signature)
    np.testing.assert_allclose(MatrixSemantics(signature).evaluate(circuit), np.eye(2))

    identity = [[1, 0], [0, 1]]
    np.testing.assert_allclose(
        MatrixSemantics(signature, overrides={"x": identity}).evaluate(
            Circuit.from_json({"gen": "x"}, signature)
        ),
        np.eye(2),
    )

    def parsed_matrix(value):
        parsed = Signature.from_json({"m": {"type": 0, "matrix": value}})["m"].matrix
        return np.asarray(parsed)

    parsed = parsed_matrix([["1+i", "-2i"]])
    np.testing.assert_allclose(parsed, [[1 + 1j, -2j]])

    diagonal = parsed_matrix({"diagonal": [1, "i", -1]})
    np.testing.assert_allclose(diagonal, np.diag([1, 1j, -1]))

    basis_cycle = parsed_matrix({"permutation": [1, 2, 0]})
    np.testing.assert_array_equal(
        basis_cycle,
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    )

    cycle = permutation_matrix((1, 2, 0), 2)
    assert basis_wire_permutation((1, 2, 0), 2)[5] == 3
    assert cycle[3, 5] == 1  # |101> maps to |011>.
    np.testing.assert_array_equal(cycle.sum(axis=0), np.ones(8))


def test_matrix_evaluation_caches_repeated_subterms(monkeypatch):
    import minimality_auto.matrix as matrix_module

    conversions = 0
    original = matrix_module.np.asarray

    def counted(value, *args, **kwargs):
        nonlocal conversions
        conversions += 1
        return original(value, *args, **kwargs)

    monkeypatch.setattr(matrix_module.np, "asarray", counted)
    signature = Signature.from_json({"x": {"type": 1, "matrix": [[0, 1], [1, 0]]}})
    gate = Circuit.generator(signature["x"])
    MatrixSemantics(signature).evaluate(Circuit.compose((gate, gate)))

    assert conversions == 1


@pytest.mark.parametrize(
    "value",
    [
        {"real": [[1]]},
        {"imag": [[1]]},
        {"diagonal": []},
        {"diagonal": "1"},
        {"diagonal": [1], "unexpected": []},
        {"permutation": []},
        {"permutation": [0, 0]},
        {"permutation": [1], "unexpected": []},
    ],
)
def test_malformed_matrix_encodings_raise_validation_error(value):
    with pytest.raises(ValidationError):
        Signature.from_json({"m": {"type": 0, "matrix": value}})


def test_matrix_permutation_convention():
    signature = Signature([])
    swap = Circuit.from_json({"perm": [1, 0]}, signature)
    matrix = MatrixSemantics(signature).evaluate(swap)
    # |01> is routed to |10>.
    basis_01 = np.array([0, 1, 0, 0], dtype=complex)
    np.testing.assert_allclose(matrix @ basis_01, [0, 0, 1, 0])


def test_compact_endomorphism_returns_nonadjacent_wires_to_their_slots():
    cnot = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    signature = Signature.from_json({"CNOT": {"type": 2, "matrix": cnot}})
    circuit = Circuit.from_json({"wires": 3, "ops": [{"gen": "CNOT", "on": [0, 2]}]}, signature)
    matrix = MatrixSemantics(signature).evaluate(circuit)
    basis_101 = np.zeros(8, dtype=complex)
    basis_101[5] = 1
    expected_100 = np.zeros(8, dtype=complex)
    expected_100[4] = 1
    np.testing.assert_allclose(matrix @ basis_101, expected_100)
    assert analyze_circuit(circuit).permutation_parity == 0


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        "nan",
        {"re": 0},
        {"real": 0},
        True,
    ],
)
def test_complex_scalars_must_be_finite_numbers(value):
    with pytest.raises(ValidationError):
        parse_complex(value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_rejects_nonfinite_constants(tmp_path, constant):
    path = tmp_path / "invalid.json"
    path.write_text(constant, encoding="utf-8")
    with pytest.raises(ValidationError, match="non-finite"):
        load_json(path)


def test_theory_rejects_unknown_circuit_fields_and_wrong_matrix_shapes():
    with pytest.raises(ValidationError, match="unknown circuit fields"):
        Theory.from_json(
            {
                "generators": {"x": 1},
                "equations": [
                    {
                        "id": "bad",
                        "lhs": {"id": 1, "typo": True},
                        "rhs": {"id": 1},
                    }
                ],
            }
        )

    with pytest.raises(ValidationError, match="has shape"):
        Theory.from_json(
            {
                "wire_dimension": 2,
                "generators": {
                    "x": {
                        "type": 1,
                        "matrix": [[1]],
                    }
                },
                "equations": [],
            }
        )


def test_matrix_shape_validation_rejects_huge_arities_without_huge_integers():
    with pytest.raises(ValidationError, match="has shape"):
        Theory.from_json(
            {
                "wire_dimension": 2,
                "generators": {
                    "hostile": {
                        "type": 100_000_000,
                        "matrix": [[1]],
                    }
                },
                "equations": [],
            }
        )


def test_noncanonical_declaration_forms_are_rejected():
    with pytest.raises(ValidationError, match="generators must be an object"):
        Signature.from_json([{"id": "x", "type": 1}])
    with pytest.raises(ValidationError, match="unknown theory"):
        Theory.from_json(
            {
                "dimension": 2,
                "generators": {},
                "equations": [],
            }
        )
    with pytest.raises(ValidationError, match="unknown equation"):
        Theory.from_json(
            {
                "generators": {"x": 1},
                "equations": [
                    {
                        "id": "e",
                        "name": "e",
                        "lhs": {"gen": "x"},
                        "rhs": {"gen": "x"},
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):
        Theory.from_json(
            {
                "generators": {"x": 1},
                "macros": {"m": {"body": {"gen": "x"}}},
                "equations": [],
            }
        )
    with pytest.raises(ValidationError, match="source must be an object"):
        Theory.from_json({"source": "paper", "generators": {}, "equations": []})


def test_arity_change_detection_checks_nested_subterms():
    grow = Circuit.generator(Generator("grow", 1, 2))
    shrink = Circuit.generator(Generator("shrink", 2, 1))
    round_trip = Circuit.compose((grow, shrink))

    assert round_trip.type == (1, 1)
    assert contains_arity_change(round_trip)
    assert not contains_arity_change(Circuit.identity(1))


def test_declared_generator_round_trips_in_canonical_form():
    signature = Signature.from_json({"grow": [1, 2]})
    changing = Circuit.from_json({"gen": "grow"}, signature)

    assert changing.to_json() == {"gen": "grow"}
    assert Circuit.from_json(changing.to_json(), signature) == changing
