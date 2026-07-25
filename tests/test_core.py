import json
from collections import Counter

import pytest

from minimality_auto.core import (
    Circuit,
    Signature,
    Theory,
    ValidationError,
    contains_arity_change,
    expand_macros,
    load_json,
    parse_complex,
    primitive_occurrences,
    structural_permutation_parity,
)
from minimality_auto.matrix import (
    MatrixSemantics,
    parse_complex_matrix,
    permutation_matrix,
)


@pytest.fixture
def signature():
    return Signature.from_json(
        [
            {"id": "u", "inputs": 1, "outputs": 1},
            {"id": "mul", "inputs": 2, "outputs": 1},
            {"id": "unit", "inputs": 0, "outputs": 1},
        ]
    )


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
    assert primitive_occurrences(circuit) == Counter(u=3, mul=1)

    with pytest.raises(ValidationError, match="composition type mismatch"):
        Circuit.from_json({"compose": [{"gen": "mul"}, {"gen": "mul"}]}, signature)


def test_circuit_constructors_apply_prop_unit_laws(signature):
    u = Circuit.generator(signature["u"])

    assert Circuit.perm([0, 1]) == Circuit.identity(2)
    assert Circuit.compose((Circuit.identity(1), u, Circuit.identity(1))) == u
    assert Circuit.compose(
        (Circuit.identity(1), Circuit.identity(1))
    ) == Circuit.identity(1)
    assert Circuit.tensor((Circuit.identity(0), u, Circuit.identity(0))) == u
    assert Circuit.tensor(
        (Circuit.identity(0), Circuit.identity(0))
    ) == Circuit.identity(0)

    with pytest.raises(ValidationError, match="composition type mismatch"):
        Circuit.compose((Circuit.identity(1), Circuit.identity(2)))


def test_source_target_generator_aliases_match_the_documented_schema():
    signature = Signature.from_json(
        [{"name": "u", "source": 1, "target": 1, "matrix": [[1, 0], [0, 1]]}]
    )
    assert signature["u"].type == (1, 1)


def test_scalar_type_is_endomorphism_shorthand_without_losing_prop_types():
    signature = Signature.from_json(
        {
            "scalar": 0,
            "H": {"type": 1},
            "copy": [1, 2],
        }
    )

    assert signature["scalar"].type == (0, 0)
    assert signature["H"].type == (1, 1)
    assert signature["copy"].type == (1, 2)

    theory = Theory.from_json(
        {
            "generators": {"H": 1},
            "macros": {
                "H2": {
                    "type": 1,
                    "body": {
                        "compose": [{"gen": "H"}, {"gen": "H"}],
                    },
                }
            },
            "equations": [],
        }
    )
    assert theory.macros["H2"].type == (1, 1)


@pytest.mark.parametrize("bad_type", [True, -1, [1], [1, -1]])
def test_scalar_and_pair_types_remain_strict(bad_type):
    with pytest.raises(ValidationError):
        Signature.from_json({"bad": bad_type})


def test_type_cannot_be_mixed_with_endpoint_fields():
    with pytest.raises(ValidationError, match="cannot mix type"):
        Signature.from_json({"bad": {"type": 1, "inputs": 1, "outputs": 1}})


def test_generators_is_not_a_reserved_generator_name():
    signature = Signature.from_json({"generators": [1, 1]})

    assert signature["generators"].type == (1, 1)


def test_permutations_and_compact_wiring(signature):
    swap_then_mul = Circuit.from_json(
        {"wires": 2, "ops": [{"perm": [1, 0]}, {"gen": "mul"}]},
        signature,
    )
    assert swap_then_mul.type == (2, 1)
    assert primitive_occurrences(swap_then_mul) == Counter(mul=1)
    assert structural_permutation_parity(swap_then_mul) == 1

    nonadjacent = Circuit.from_json(
        {"wires": 3, "ops": [{"gen": "mul", "on": [2, 0]}]}, signature
    )
    assert nonadjacent.type == (3, 2)
    assert Circuit.from_json(
        {"wires": 2, "ops": [{"gen": "u", "on": 1}]}, signature
    ).type == (2, 2)

    with pytest.raises(ValidationError, match="more than once"):
        Circuit.from_json(
            {"wires": 2, "ops": [{"gen": "mul", "on": [0, 0]}]}, signature
        )
    with pytest.raises(ValidationError, match="integer on"):
        Circuit.from_json(
            {"wires": 2, "ops": [{"gen": "mul", "on": 0}]}, signature
        )
    with pytest.raises(ValidationError, match="needs an on array"):
        Circuit.from_json({"wires": 3, "ops": [{"gen": "mul"}]}, signature)


def test_compact_parser_canonicalizes_units_and_validates_permutations(signature):
    assert Circuit.from_json(
        {"wires": 1, "ops": [{"gen": "u"}]}, signature
    ) == Circuit.generator(signature["u"])

    with pytest.raises(ValidationError, match="permutation must be an array"):
        Circuit.from_json(
            {"wires": 2, "ops": [{"perm": 1}]},
            signature,
        )


def test_compact_wiring_accepts_typed_inline_generators_and_checks_types():
    signature = Signature.from_json({"known": [1, 1]})
    inline = Circuit.from_json(
        {
            "wires": 1,
            "ops": [
                {"gen": {"id": "external", "type": [1, 2]}, "on": 0}
            ],
        },
        signature,
    )

    assert inline.type == (1, 2)
    assert primitive_occurrences(inline) == Counter(external=1)
    with pytest.raises(ValidationError, match="inline type disagrees"):
        Circuit.from_json(
            {
                "wires": 1,
                "ops": [
                    {"gen": {"id": "known", "type": [1, 2]}, "on": [0]}
                ],
            },
            signature,
        )


def test_macros_expand_before_features_and_cycles_fail():
    theory = Theory.from_json(
        {
            "generators": {"u": [1, 1]},
            "macros": [
                {"id": "twice", "body": {"compose": [{"gen": "u"}, {"gen": "u"}]}},
                {"id": "four", "body": {"compose": [{"macro": "twice"}, {"macro": "twice"}]}},
            ],
            "equations": [{"id": "e", "lhs": {"macro": "four"}, "rhs": {"gen": "u"}}],
        }
    )
    lhs = theory.equations[0].lhs
    assert primitive_occurrences(lhs, theory.macros) == Counter(u=4)
    assert "macro" not in json.dumps(expand_macros(lhs, theory.macros).to_json())

    with pytest.raises(ValidationError, match=r"macro cycle: a -> b -> a"):
        Theory.from_json(
            {
                "generators": {"u": [1, 1]},
                "macros": {
                    "a": {"type": [1, 1], "body": {"macro": "b"}},
                    "b": {"type": [1, 1], "body": {"macro": "a"}},
                },
                "equations": [],
            }
        )


def test_mapping_macros_accept_direct_circuit_bodies():
    theory = Theory.from_json(
        {
            "generators": {"u": 1},
            "macros": {
                "twice": {"compose": [{"gen": "u"}, {"gen": "u"}]}
            },
            "equations": [],
        }
    )

    assert primitive_occurrences(theory.macros["twice"]) == Counter(u=2)


def test_cached_macro_uses_still_validate_each_occurrence_type():
    signature = Signature([])
    macros = {"wire": Circuit.identity(1)}
    malformed = Circuit.tensor(
        (Circuit.macro("wire", 1, 1), Circuit.macro("wire", 2, 2))
    )

    with pytest.raises(ValidationError, match="inconsistent type"):
        primitive_occurrences(malformed, macros)
    with pytest.raises(ValidationError, match="inconsistent type"):
        structural_permutation_parity(malformed, macros)
    with pytest.raises(ValidationError, match="inconsistent type"):
        MatrixSemantics(signature).evaluate(malformed, macros)


def test_duplicate_ids_and_keys_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="duplicate equation id"):
        Theory.from_json(
            {
                "generators": {"u": [1, 1]},
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


def test_matrix_semantics_and_call_overrides():
    np = pytest.importorskip("numpy")
    signature = Signature.from_json(
        [{"id": "x", "inputs": 1, "outputs": 1, "matrix": [[0, 1], [1, 0]]}]
    )
    circuit = Circuit.from_json(
        {"compose": [{"gen": "x"}, {"gen": "x"}]}, signature
    )
    np.testing.assert_allclose(MatrixSemantics(signature).evaluate(circuit), np.eye(2))

    identity = [[1, 0], [0, 1]]
    np.testing.assert_allclose(
        MatrixSemantics(signature).evaluate(
            Circuit.from_json({"gen": "x"}, signature),
            overrides={"x": identity},
        ),
        np.eye(2),
    )
    parsed = parse_complex_matrix([["1+i", {"re": 0, "im": -2}]])
    np.testing.assert_allclose(parsed, [[1 + 1j, -2j]])

    diagonal = parse_complex_matrix({"diagonal": [1, "i", -1]})
    np.testing.assert_allclose(diagonal, np.diag([1, 1j, -1]))

    cycle = permutation_matrix((1, 2, 0), 2)
    assert cycle[3, 5] == 1  # |101> maps to |011>.
    np.testing.assert_array_equal(cycle.sum(axis=0), np.ones(8))


def test_matrix_evaluation_caches_repeated_subterms(monkeypatch):
    import minimality_auto.matrix as matrix_module

    conversions = 0
    original = matrix_module._matrix_tuple

    def counted(value):
        nonlocal conversions
        conversions += 1
        return original(value)

    monkeypatch.setattr(matrix_module, "_matrix_tuple", counted)
    signature = Signature.from_json(
        {"x": {"type": 1, "matrix": [[0, 1], [1, 0]]}}
    )
    gate = Circuit.generator(signature["x"])
    MatrixSemantics(signature).evaluate(Circuit.compose((gate, gate)))

    assert conversions == 1


@pytest.mark.parametrize(
    "value",
    [
        {"real": [1]},
        {"real": [[1]], "imag": [1]},
        {"real": [[1]], "unexpected": []},
        {"diagonal": []},
        {"diagonal": "1"},
        {"diagonal": [1], "unexpected": []},
    ],
)
def test_malformed_matrix_encodings_raise_validation_error(value):
    with pytest.raises(ValidationError):
        parse_complex_matrix(value)


def test_matrix_permutation_convention():
    np = pytest.importorskip("numpy")
    signature = Signature([])
    swap = Circuit.from_json({"perm": [1, 0]}, signature)
    matrix = MatrixSemantics(signature).evaluate(swap)
    # |01> is routed to |10>.
    basis_01 = np.array([0, 1, 0, 0], dtype=complex)
    np.testing.assert_allclose(matrix @ basis_01, [0, 0, 1, 0])


def test_compact_endomorphism_returns_nonadjacent_wires_to_their_slots():
    np = pytest.importorskip("numpy")
    cnot = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    signature = Signature.from_json(
        [{"id": "CNOT", "inputs": 2, "outputs": 2, "matrix": cnot}]
    )
    circuit = Circuit.from_json(
        {"wires": 3, "ops": [{"gen": "CNOT", "on": [0, 2]}]}, signature
    )
    matrix = MatrixSemantics(signature).evaluate(circuit)
    basis_101 = np.zeros(8, dtype=complex)
    basis_101[5] = 1
    expected_100 = np.zeros(8, dtype=complex)
    expected_100[4] = 1
    np.testing.assert_allclose(matrix @ basis_101, expected_100)
    assert structural_permutation_parity(circuit) == 0


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), "nan", {"re": True}, {"im": False}],
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
                "generators": {"x": [1, 1]},
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
                        "inputs": 1,
                        "outputs": 1,
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


@pytest.mark.parametrize(
    "parse",
    [
        lambda: parse_complex({"re": 1, "real": 1}),
        lambda: Signature.from_json(
            [{"id": "x", "name": "x", "type": 1}]
        ),
        lambda: Circuit.from_json(
            {"gen": {"id": "x", "name": "x"}},
            Signature.from_json({"x": 1}),
        ),
        lambda: Theory.from_json(
            {
                "wire_dimension": 2,
                "dimension": 2,
                "generators": {},
                "equations": [],
            }
        ),
        lambda: Theory.from_json(
            {
                "generators": {},
                "signature": {},
                "equations": [],
            }
        ),
        lambda: Theory.from_json(
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
        ),
        lambda: Theory.from_json(
            {
                "generators": {"x": 1},
                "macros": {
                    "m": {
                        "body": {"gen": "x"},
                        "circuit": {"gen": "x"},
                    }
                },
                "equations": [],
            }
        ),
    ],
)
def test_json_aliases_are_mutually_exclusive(parse):
    with pytest.raises(ValidationError, match="cannot use aliases together"):
        parse()


def test_arity_change_detection_checks_nested_subterms():
    grow = Circuit.generator("grow", 1, 2)
    shrink = Circuit.generator("shrink", 2, 1)
    round_trip = Circuit.compose((grow, shrink))

    assert round_trip.type == (1, 1)
    assert contains_arity_change(round_trip)
    assert not contains_arity_change(Circuit.identity(1))


def test_typed_inline_generator_round_trips_without_a_signature():
    signature = Signature()
    changing = Circuit.from_json(
        {"gen": {"id": "external", "type": [1, 2]}},
        signature,
    )
    endomorphism = Circuit.from_json(
        {"gen": {"id": "endo", "type": 1}},
        signature,
    )

    assert changing.to_json()["gen"]["type"] == [1, 2]
    assert endomorphism.to_json()["gen"]["type"] == 1
    assert Circuit.from_json(changing.to_json(), signature) == changing
    assert Circuit.from_json(endomorphism.to_json(), signature) == endomorphism


def test_outer_type_annotation_is_checked_for_nested_inline_generator():
    signature = Signature.from_json({"known": 1})

    with pytest.raises(ValidationError, match="inline type disagrees"):
        Circuit.from_json(
            {
                "gen": {"id": "known"},
                "inputs": 2,
                "outputs": 2,
            },
            signature,
        )
    with pytest.raises(ValidationError, match="declared twice"):
        Circuit.from_json(
            {
                "gen": {"id": "external", "type": 1},
                "type": 1,
            },
            signature,
        )
