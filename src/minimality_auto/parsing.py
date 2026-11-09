"""JSON loading, validation, macro expansion, and compact circuit syntax."""

from __future__ import annotations

import cmath
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .syntax import (
    Circuit,
    Equation,
    Generator,
    Signature,
    Theory,
    ValidationError,
    _identifier,
    _inverse_permutation,
    _is_sequence,
    _matrix_shape_matches,
    _natural,
)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValidationError(f"unknown {where} fields: {sorted(unknown)}")


def _type_pair(value: Any, where: str) -> tuple[int, int]:
    """Parse ``a`` as ``a -> a`` and ``[a, b]`` as ``a -> b``."""

    if isinstance(value, int) and not isinstance(value, bool):
        arity = _natural(value, where)
        return arity, arity
    if not _is_sequence(value) or len(value) != 2:
        raise ValidationError(f"{where} must be an arity or [inputs, outputs]")
    inputs = _natural(value[0], f"{where} inputs")
    outputs = _natural(value[1], f"{where} outputs")
    if inputs == outputs:
        raise ValidationError(f"{where} endomorphism must use the scalar arity {inputs}")
    return inputs, outputs


def parse_complex(value: Any) -> complex:
    """Parse a JSON complex scalar.

    Accepted forms are real JSON numbers and strings such as ``"1-2i"``.
    """

    if isinstance(value, bool):
        raise ValidationError("boolean is not a complex scalar")
    parsed: complex
    if isinstance(value, (int, float, complex)):
        parsed = complex(value)
    elif isinstance(value, str):
        try:
            parsed = complex(value.strip().replace("I", "j").replace("i", "j"))
        except ValueError as exc:
            raise ValidationError(f"invalid complex scalar {value!r}") from exc
    else:
        raise ValidationError(f"invalid complex scalar {value!r}")
    if not cmath.isfinite(parsed):
        raise ValidationError(f"complex scalar must be finite, got {value!r}")
    return parsed


def _matrix_tuple(value: Any) -> tuple[tuple[complex, ...], ...]:
    if isinstance(value, Mapping):
        if set(value) == {"diagonal"}:
            diagonal = value["diagonal"]
            if not _is_sequence(diagonal) or not diagonal:
                raise ValidationError("matrix diagonal must be a non-empty array")
            entries = tuple(parse_complex(item) for item in diagonal)
            return tuple(
                tuple(entry if row == column else 0j for column, entry in enumerate(entries))
                for row in range(len(entries))
            )
        if set(value) == {"permutation"}:
            raw = value["permutation"]
            if not _is_sequence(raw) or not raw:
                raise ValidationError("matrix permutation must be a non-empty array")
            permutation = tuple(_natural(item, "matrix permutation entry") for item in raw)
            if tuple(sorted(permutation)) != tuple(range(len(permutation))):
                raise ValidationError("matrix permutation must contain each index once")
            return tuple(
                tuple(
                    1 + 0j if permutation[column] == row else 0j
                    for column in range(len(permutation))
                )
                for row in range(len(permutation))
            )
        raise ValidationError("matrix object must contain only diagonal or permutation")
    if not _is_sequence(value) or not value:
        raise ValidationError("matrix must be a non-empty array of rows")
    rows: list[tuple[complex, ...]] = []
    width: int | None = None
    for row in value:
        if not _is_sequence(row) or not row:
            raise ValidationError("matrix rows must be non-empty arrays")
        parsed = tuple(parse_complex(item) for item in row)
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            raise ValidationError("matrix rows have different lengths")
        rows.append(parsed)
    return tuple(rows)


def parse_signature(value: Any, cls: type[Signature] = Signature) -> Signature:
    if not isinstance(value, Mapping):
        raise ValidationError("generators must be an object")
    generators: list[Generator] = []
    for name, spec in value.items():
        _identifier(name, "generator name")
        if not isinstance(spec, Mapping):
            matrix = None
            inputs, outputs = _type_pair(spec, f"generator {name!r} type")
        else:
            _reject_unknown(spec, {"type", "matrix"}, f"generator {name!r}")
            if "type" not in spec:
                raise ValidationError(f"generator {name!r} has no type")
            if "matrix" not in spec:
                raise ValidationError(f"generator {name!r} without a matrix must use a bare type")
            inputs, outputs = _type_pair(
                spec["type"],
                f"generator {name!r} type",
            )
            matrix = _matrix_tuple(spec["matrix"])
        generators.append(Generator(name, inputs, outputs, matrix))
    return cls(generators)


def _validate_generator_matrices(signature: Signature, dimension: int) -> None:
    for generator in signature.values():
        if generator.matrix is None:
            continue
        shape = (len(generator.matrix), len(generator.matrix[0]))
        if not _matrix_shape_matches(
            shape,
            dimension,
            generator.inputs,
            generator.outputs,
        ):
            raise ValidationError(
                f"matrix for {generator.name!r} has shape {shape}, expected "
                f"({dimension}^{generator.outputs}, {dimension}^{generator.inputs})"
            )


MacroResolver = Callable[[str], Circuit]


def parse_circuit(
    value: Any,
    signature: Signature,
    macros: Mapping[str, Circuit] | None = None,
) -> Circuit:
    def resolve_macro(name: str) -> Circuit:
        if macros is None or name not in macros:
            raise ValidationError(f"unknown macro {name!r}")
        return macros[name]

    return _parse_circuit(value, signature, resolve_macro)


def _parse_named_circuit(
    value: Mapping[str, Any],
    key: str,
    signature: Signature,
    macro: MacroResolver,
) -> Circuit:
    name = _identifier(value[key], f"{key} name")
    if key == "gen":
        return Circuit.generator(signature[name])
    return macro(name)


def _parse_circuit(value: Any, signature: Signature, macro: MacroResolver) -> Circuit:
    if not isinstance(value, Mapping):
        raise ValidationError("circuit must be an object")
    if "wires" in value or "ops" in value:
        if "wires" not in value or "ops" not in value:
            raise ValidationError("compact circuit needs both wires and ops")
        _reject_unknown(value, {"wires", "ops"}, "compact circuit")
        return _parse_compact(value, signature, macro)

    keys = [key for key in ("gen", "id", "perm", "compose", "tensor", "macro") if key in value]
    if len(keys) != 1:
        raise ValidationError(
            "circuit must contain exactly one of gen/id/perm/compose/tensor/macro"
        )
    key = keys[0]
    repeatable = key in {"gen", "macro", "perm"}
    _reject_unknown(value, {key, "repeat"} if repeatable else {key}, "circuit")
    if repeatable:
        base = (
            Circuit.perm(value[key])
            if key == "perm"
            else _parse_named_circuit(value, key, signature, macro)
        )
        return _repeat_circuit(base, value.get("repeat", 1), "circuit")
    if key == "id":
        return Circuit.identity(value["id"])
    if key in {"compose", "tensor"}:
        raw_parts = value[key]
        if not _is_sequence(raw_parts):
            raise ValidationError(f"{key} must be an array")
        parts = [_parse_circuit(part, signature, macro) for part in raw_parts]
        return Circuit.compose(parts) if key == "compose" else Circuit.tensor(parts)
    raise AssertionError(key)


_MAX_COMPACT_OPERATIONS = 10_000


def _repeat_circuit(circuit: Circuit, value: Any, where: str) -> Circuit:
    repeat = _natural(value, f"{where} repeat")
    if repeat == 0:
        raise ValidationError(f"{where} repeat must be positive")
    if repeat > _MAX_COMPACT_OPERATIONS:
        raise ValidationError(f"{where} repeat exceeds {_MAX_COMPACT_OPERATIONS}")
    if repeat > 1 and circuit.inputs != circuit.outputs:
        raise ValidationError(f"{where} repeat needs an endomorphism")
    return Circuit.compose([circuit] * repeat) if repeat > 1 else circuit


def _parse_compact(value: Mapping[str, Any], signature: Signature, macro: MacroResolver) -> Circuit:
    wires = _natural(value["wires"], "compact circuit wires")
    ops = value["ops"]
    if not _is_sequence(ops):
        raise ValidationError("compact circuit ops must be an array")
    steps: list[Circuit] = []
    current = wires
    expanded = 0
    for index, op in enumerate(ops):
        if not isinstance(op, Mapping):
            raise ValidationError(f"operation {index} must be an object")
        op_keys = [key for key in ("gen", "macro", "perm") if key in op]
        if len(op_keys) != 1:
            raise ValidationError(f"operation {index} needs exactly one of gen/macro/perm")
        key = op_keys[0]
        allowed = {key, "repeat"}
        if key in {"gen", "macro"}:
            allowed.add("on")
        _reject_unknown(op, allowed, f"operation {index}")
        repeat = op.get("repeat", 1)
        expanded += _natural(repeat, f"operation {index} repeat")
        if expanded > _MAX_COMPACT_OPERATIONS:
            raise ValidationError(f"compact circuit exceeds {_MAX_COMPACT_OPERATIONS} operations")
        if key == "perm":
            step = Circuit.perm(op["perm"])
            if step.inputs != current:
                raise ValidationError(
                    f"operation {index} permutation has {step.inputs} wires, expected {current}"
                )
        else:
            base = _parse_named_circuit(op, key, signature, macro)
            name = base.name or ""
            if "on" not in op:
                if base.inputs not in {0, current}:
                    raise ValidationError(f"operation {index} needs an on array")
                selected = tuple(range(base.inputs))
            else:
                on = op["on"]
                if isinstance(on, int) and not isinstance(on, bool):
                    if base.inputs != 1:
                        raise ValidationError(
                            f"operation {index} uses an integer on for a {base.inputs}-input box"
                        )
                    selected = (on,)
                else:
                    if not _is_sequence(on):
                        raise ValidationError(f"operation {index} needs an on array or unary index")
                    selected = tuple(on)
            if len(selected) != base.inputs:
                raise ValidationError(
                    f"operation {index} selects {len(selected)} wires for "
                    f"{base.inputs}-input {name!r}"
                )
            if any(isinstance(i, bool) or not isinstance(i, int) for i in selected):
                raise ValidationError(f"operation {index} wire positions must be integers")
            if len(set(selected)) != len(selected):
                raise ValidationError(f"operation {index} selects a wire more than once")
            if any(i < 0 or i >= current for i in selected):
                raise ValidationError(f"operation {index} selects a wire outside 0..{current - 1}")
            step = _place_on_wires(base, current, selected)
        steps.append(_repeat_circuit(step, repeat, f"operation {index}"))
        current = step.outputs
    return Circuit.compose(steps) if steps else Circuit.identity(wires)


def _place_on_wires(base: Circuit, wires: int, selected: tuple[int, ...]) -> Circuit:
    """Place a box using the compact circuit syntax's wire-routing rules.

    Selected wires are fed to the box in ``on`` order. Endomorphism outputs
    return to the corresponding selected positions. For arity-changing boxes,
    the outputs replace the selected wires at their leftmost position; untouched
    wires keep their order. Nullary outputs are appended.
    """

    chosen = set(selected)
    rest = tuple(i for i in range(wires) if i not in chosen)
    order = selected + rest
    pre = Circuit.perm(order)
    middle = Circuit.tensor((base, Circuit.identity(len(rest))))
    if base.inputs == base.outputs:
        output_order = _inverse_permutation(order)
    else:
        anchor = min(selected, default=wires)
        before = tuple(base.outputs + i for i, wire in enumerate(rest) if wire < anchor)
        after = tuple(base.outputs + i for i, wire in enumerate(rest) if wire >= anchor)
        output_order = before + tuple(range(base.outputs)) + after
    return Circuit.compose((pre, middle, Circuit.perm(output_order)))


def _parse_macro_definitions(value: Any, signature: Signature) -> dict[str, Circuit]:
    if not isinstance(value, Mapping):
        raise ValidationError("macros must be an object")
    specs = {_identifier(name, "macro id"): body for name, body in value.items()}

    resolved: dict[str, Circuit] = {}
    visiting: list[str] = []

    def resolve(name: str) -> Circuit:
        if name in resolved:
            return resolved[name]
        if name not in specs:
            raise ValidationError(f"unknown macro {name!r}")
        if name in visiting:
            start = visiting.index(name)
            cycle = visiting[start:] + [name]
            raise ValidationError(f"macro cycle: {' -> '.join(cycle)}")
        visiting.append(name)
        body = _parse_circuit(specs[name], signature, resolve)
        visiting.pop()
        resolved[name] = body
        return body

    for name in specs:
        resolve(name)
    return resolved


def parse_theory(value: Any, cls: type[Theory] = Theory) -> Theory:
    if not isinstance(value, Mapping):
        raise ValidationError("theory must be an object")
    data = value
    _reject_unknown(
        data,
        {"name", "wire_dimension", "source", "generators", "macros", "equations"},
        "theory",
    )
    dimension = _natural(data.get("wire_dimension", 2), "wire dimension")
    if dimension == 0:
        raise ValidationError("wire dimension must be positive")
    if "generators" not in data:
        raise ValidationError("theory has no generators")
    signature = Signature.from_json(data["generators"])
    _validate_generator_matrices(signature, dimension)

    macros = _parse_macro_definitions(data.get("macros", {}), signature)
    raw_equations = data.get("equations")
    if not _is_sequence(raw_equations):
        raise ValidationError("equations must be an array")
    equations: list[Equation] = []
    ids: set[str] = set()
    for raw in raw_equations:
        if not isinstance(raw, Mapping):
            raise ValidationError("each equation must be an object")
        _reject_unknown(raw, {"id", "lhs", "rhs"}, "equation")
        equation_id = _identifier(raw.get("id"), "equation id")
        if equation_id in ids:
            raise ValidationError(f"duplicate equation id {equation_id!r}")
        ids.add(equation_id)
        if "lhs" not in raw or "rhs" not in raw:
            raise ValidationError(f"equation {equation_id!r} needs lhs and rhs")
        lhs = Circuit.from_json(raw["lhs"], signature, macros)
        rhs = Circuit.from_json(raw["rhs"], signature, macros)
        equations.append(Equation(equation_id, lhs, rhs))
    name = data.get("name")
    if name is not None:
        name = _identifier(name, "theory name")
    raw_source = data.get("source")
    if raw_source is not None and not isinstance(raw_source, Mapping):
        raise ValidationError("theory source must be an object")
    source = dict(raw_source) if raw_source is not None else None
    return cls(signature, tuple(equations), macros, name, dimension, source)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON number {value!r}")


def load_json(path: str | Path) -> Any:
    """Load JSON, enforcing unique keys and rejecting NaN/Infinity literals."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_nonfinite_constant,
            )
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def load_theory(path: str | Path) -> Theory:
    return Theory.from_json(load_json(path))
