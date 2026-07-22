from collections import Counter
import json

import pytest

from minimality_auto.core import (
    Circuit,
    MacroDef,
    MatrixSemantics,
    Signature,
    Theory,
    ValidationError,
    expand_macros,
    load_json,
    parse_complex_matrix,
    primitive_occurrences,
    structural_permutation_parity,
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


def test_source_target_generator_aliases_match_the_documented_schema():
    signature = Signature.from_json(
        [{"name": "u", "source": 1, "target": 1, "matrix": [[1, 0], [0, 1]]}]
    )
    assert signature["u"].type == (1, 1)


def test_generators_is_not_a_reserved_generator_name():
    signature = Signature.from_json({"generators": [1, 1]})

    assert signature["generators"].type == (1, 1)


def test_permutations_and_compact_wiring(signature):
    swap_then_mul = Circuit.from_json(
        {"wires": 2, "ops": [{"perm": [1, 0]}, {"gen": "mul", "on": [0, 1]}]},
        signature,
    )
    assert swap_then_mul.type == (2, 1)
    assert primitive_occurrences(swap_then_mul) == Counter(mul=1)
    assert structural_permutation_parity(swap_then_mul) == 1

    nonadjacent = Circuit.from_json(
        {"wires": 3, "ops": [{"gen": "mul", "on": [2, 0]}]}, signature
    )
    assert nonadjacent.type == (3, 2)

    with pytest.raises(ValidationError, match="more than once"):
        Circuit.from_json(
            {"wires": 2, "ops": [{"gen": "mul", "on": [0, 0]}]}, signature
        )


def test_compact_wiring_accepts_typed_inline_generators_and_checks_types():
    signature = Signature.from_json({"known": [1, 1]})
    inline = Circuit.from_json(
        {
            "wires": 1,
            "ops": [
                {"gen": {"id": "external", "type": [1, 2]}, "on": [0]}
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


def test_cached_macro_uses_still_validate_each_occurrence_type():
    signature = Signature([])
    macros = {"wire": MacroDef("wire", Circuit.identity(1))}
    malformed = Circuit.tensor(
        (Circuit.macro("wire", 1, 1), Circuit.macro("wire", 2, 2))
    )

    with pytest.raises(ValidationError, match="inconsistent type"):
        primitive_occurrences(malformed, macros)
    with pytest.raises(ValidationError, match="inconsistent type"):
        structural_permutation_parity(malformed, macros)
    with pytest.raises(ValidationError, match="inconsistent type"):
        MatrixSemantics(signature).evaluate(malformed, macros)


def test_duplicate_ids_and_keys_are_rejected(tmp_path, signature):
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
        MatrixSemantics(signature).evaluate(Circuit.from_json({"gen": "x"}, signature), overrides={"x": identity}),
        np.eye(2),
    )
    parsed = parse_complex_matrix([["1+i", {"re": 0, "im": -2}]])
    np.testing.assert_allclose(parsed, [[1 + 1j, -2j]])


@pytest.mark.parametrize(
    "value",
    [
        {"real": [1]},
        {"real": [[1]], "imag": [1]},
        {"real": [[1]], "unexpected": []},
    ],
)
def test_malformed_split_matrix_encodings_raise_validation_error(value):
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
