"""A compact, typed JSON representation of circuits in a PROP.

Composition is written in execution order: ``compose: [f, g]`` means first
``f``, then ``g``.  A permutation list gives, for each output position, the
input position routed there.
"""

from __future__ import annotations

import cmath
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain, pairwise
from pathlib import Path
from typing import Any


class PropError(ValueError):
    """Base class for malformed PROP data."""


class ValidationError(PropError):
    """Raised when JSON is well formed but not a typed PROP expression."""


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValidationError(f"unknown {where} fields: {sorted(unknown)}")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


_MISSING = object()


def _one_of(
    value: Mapping[str, Any],
    names: tuple[str, ...],
    where: str,
    *,
    default: Any = _MISSING,
) -> Any:
    present = [name for name in names if name in value]
    if len(present) > 1:
        raise ValidationError(f"{where} cannot use aliases together: {present}")
    return value[present[0]] if present else default


def _natural(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{where} must be a non-negative integer")
    return value


def _wire_size_matches(size: int, dimension: int, wires: int) -> bool:
    """Check ``size == dimension**wires`` without constructing a hostile integer."""

    if size < 1 or dimension < 1 or wires < 0:
        return False
    if dimension == 1:
        return size == 1
    return wires <= size.bit_length() and dimension**wires == size


def _matrix_shape_matches(
    shape: tuple[int, ...],
    dimension: int,
    inputs: int,
    outputs: int,
) -> bool:
    return (
        len(shape) == 2
        and _wire_size_matches(shape[0], dimension, outputs)
        and _wire_size_matches(shape[1], dimension, inputs)
    )


def _type_pair(value: Any, where: str) -> tuple[int, int]:
    """Parse ``a`` as ``a -> a`` and ``[a, b]`` as ``a -> b``."""

    if isinstance(value, int) and not isinstance(value, bool):
        arity = _natural(value, where)
        return arity, arity
    if not _is_sequence(value) or len(value) != 2:
        raise ValidationError(f"{where} must be an arity or [inputs, outputs]")
    return (
        _natural(value[0], f"{where} inputs"),
        _natural(value[1], f"{where} outputs"),
    )


def _optional_type(
    value: Mapping[str, Any],
    where: str,
    input_names: tuple[str, ...] = ("inputs",),
    output_names: tuple[str, ...] = ("outputs",),
) -> tuple[int | None, int | None]:
    has_type = "type" in value
    endpoint_names = input_names + output_names
    if has_type and any(name in value for name in endpoint_names):
        raise ValidationError(f"{where} cannot mix type with input/output fields")
    if has_type:
        return _type_pair(value["type"], f"{where} type")

    def endpoint(names: tuple[str, ...], label: str) -> int | None:
        raw = _one_of(value, names, f"{where} {label}")
        if raw is _MISSING:
            return None
        return _natural(raw, f"{where} {label}")

    return endpoint(input_names, "inputs"), endpoint(output_names, "outputs")


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where} must be a non-empty string")
    return value


def parse_complex(value: Any) -> complex:
    """Parse a JSON complex scalar.

    Accepted forms are real JSON numbers, strings such as ``"1-2i"``, and
    objects ``{"re": 1, "im": -2}`` (``real``/``imag`` are aliases).
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
    elif isinstance(value, Mapping):
        allowed = {"re", "im", "real", "imag"}
        _reject_unknown(value, allowed, "complex scalar")
        re = _one_of(value, ("re", "real"), "complex scalar real part", default=0)
        im = _one_of(value, ("im", "imag"), "complex scalar imaginary part", default=0)
        if isinstance(re, bool) or isinstance(im, bool):
            raise ValidationError("boolean is not a complex scalar component")
        try:
            parsed = complex(float(re), float(im))
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"invalid complex scalar {value!r}") from exc
    else:
        raise ValidationError(f"invalid complex scalar {value!r}")
    if not cmath.isfinite(parsed):
        raise ValidationError(f"complex scalar must be finite, got {value!r}")
    return parsed


def _matrix_tuple(value: Any) -> tuple[tuple[complex, ...], ...]:
    if isinstance(value, Mapping) and "diagonal" in value:
        unknown = set(value) - {"diagonal"}
        if unknown:
            raise ValidationError(f"unknown diagonal matrix fields: {sorted(unknown)}")
        diagonal = value["diagonal"]
        if not _is_sequence(diagonal) or not diagonal:
            raise ValidationError("matrix diagonal must be a non-empty array")
        entries = tuple(parse_complex(item) for item in diagonal)
        return tuple(
            tuple(entry if row == column else 0j for column, entry in enumerate(entries))
            for row in range(len(entries))
        )
    if isinstance(value, Mapping) and "real" in value:
        unknown = set(value) - {"real", "imag"}
        if unknown:
            raise ValidationError(f"unknown matrix fields: {sorted(unknown)}")
        real = value["real"]
        imag = value.get("imag")
        if not _is_sequence(real):
            raise ValidationError("matrix real part must be an array of rows")
        if imag is None:
            imag = []
            for row in real:
                if not _is_sequence(row):
                    raise ValidationError("matrix must be an array of rows")
                imag.append([0] * len(row))
        if not _is_sequence(imag):
            raise ValidationError("matrix imaginary part must be an array of rows")
        if len(real) != len(imag):
            raise ValidationError("matrix real and imaginary parts have different shapes")
        value = []
        for real_row, imag_row in zip(real, imag, strict=True):
            if not _is_sequence(real_row):
                raise ValidationError("matrix must be an array of rows")
            if not _is_sequence(imag_row):
                raise ValidationError("matrix must be an array of rows")
            if len(real_row) != len(imag_row):
                raise ValidationError("matrix real and imaginary parts have different shapes")
            value.append(
                [{"re": re, "im": im} for re, im in zip(real_row, imag_row, strict=True)]
            )
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


@dataclass(frozen=True, slots=True)
class Generator:
    name: str
    inputs: int
    outputs: int
    matrix: tuple[tuple[complex, ...], ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _identifier(self.name, "generator name")
        _natural(self.inputs, f"generator {self.name!r} inputs")
        _natural(self.outputs, f"generator {self.name!r} outputs")

    @property
    def type(self) -> tuple[int, int]:
        return self.inputs, self.outputs


class Signature(Mapping[str, Generator]):
    """Finite typed generator signature."""

    def __init__(self, generators: Iterable[Generator] = ()) -> None:
        by_name: dict[str, Generator] = {}
        for generator in generators:
            if generator.name in by_name:
                raise ValidationError(f"duplicate generator id {generator.name!r}")
            by_name[generator.name] = generator
        self._generators = by_name

    def __getitem__(self, name: str) -> Generator:
        try:
            return self._generators[name]
        except KeyError as exc:
            raise ValidationError(f"unknown generator {name!r}") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._generators)

    def __len__(self) -> int:
        return len(self._generators)

    def __contains__(self, name: object) -> bool:
        # Mapping.__contains__ probes through __getitem__, but this class turns
        # missing lookups into the user-facing ValidationError.  Membership
        # must remain a non-raising predicate so inline typed generators can be
        # distinguished from misspelled untyped ones.
        return name in self._generators

    def get(self, name: str, default: Any = None) -> Generator | Any:
        return self._generators.get(name, default)

    @classmethod
    def from_json(cls, value: Any) -> Signature:
        entries: list[tuple[str | None, Any]]
        if isinstance(value, Mapping):
            entries = [(name, spec) for name, spec in value.items()]
        elif _is_sequence(value):
            entries = [(None, spec) for spec in value]
        else:
            raise ValidationError("generators must be an object or array")

        generators: list[Generator] = []
        for key_name, spec in entries:
            if not isinstance(spec, Mapping) and key_name is not None:
                name, matrix = key_name, None
                inputs, outputs = _type_pair(spec, f"generator {name!r} type")
            elif isinstance(spec, Mapping):
                _reject_unknown(
                    spec,
                    {
                        "id",
                        "name",
                        "type",
                        "inputs",
                        "in",
                        "source",
                        "outputs",
                        "out",
                        "target",
                        "matrix",
                    },
                    "generator",
                )
                name = _one_of(
                    spec,
                    ("id", "name"),
                    "generator identity",
                    default=key_name,
                )
                if key_name is not None and name is not None and name != key_name:
                    raise ValidationError(
                        f"generator key {key_name!r} disagrees with id {name!r}"
                    )
                inputs, outputs = _optional_type(
                    spec,
                    f"generator {name!r}",
                    ("inputs", "in", "source"),
                    ("outputs", "out", "target"),
                )
                raw_matrix = spec.get("matrix")
                matrix = _matrix_tuple(raw_matrix) if raw_matrix is not None else None
            else:
                raise ValidationError(
                    "array generators must be objects; object entries use an arity "
                    "or [inputs, outputs]"
                )
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


@dataclass(frozen=True, slots=True)
class Circuit:
    """An immutable typed circuit expression."""

    kind: str
    inputs: int
    outputs: int
    name: str | None = None
    order: tuple[int, ...] = ()
    parts: tuple[Circuit, ...] = ()

    @property
    def type(self) -> tuple[int, int]:
        return self.inputs, self.outputs

    @classmethod
    def generator(
        cls,
        generator: Generator | str,
        inputs: int | None = None,
        outputs: int | None = None,
    ) -> Circuit:
        if isinstance(generator, Generator):
            return cls("gen", generator.inputs, generator.outputs, name=generator.name)
        name = _identifier(generator, "generator name")
        if inputs is None or outputs is None:
            raise ValidationError(f"type required for generator {name!r}")
        return cls(
            "gen",
            _natural(inputs, "generator inputs"),
            _natural(outputs, "generator outputs"),
            name=name,
        )

    @classmethod
    def identity(cls, wires: int) -> Circuit:
        wires = _natural(wires, "identity size")
        return cls("id", wires, wires)

    @classmethod
    def perm(cls, order: Sequence[int]) -> Circuit:
        if not _is_sequence(order):
            raise ValidationError("permutation must be an array")
        parsed = tuple(order)
        if any(isinstance(i, bool) or not isinstance(i, int) for i in parsed):
            raise ValidationError("permutation entries must be integers")
        if sorted(parsed) != list(range(len(parsed))):
            raise ValidationError(f"not a permutation of 0..{len(parsed) - 1}: {list(parsed)}")
        if parsed == tuple(range(len(parsed))):
            return cls.identity(len(parsed))
        return cls("perm", len(parsed), len(parsed), order=parsed)

    @classmethod
    def macro(cls, name: str, inputs: int, outputs: int) -> Circuit:
        return cls(
            "macro",
            _natural(inputs, "macro inputs"),
            _natural(outputs, "macro outputs"),
            name=_identifier(name, "macro name"),
        )

    @classmethod
    def compose(cls, parts: Iterable[Circuit]) -> Circuit:
        flat: list[Circuit] = []
        for part in parts:
            flat.extend(part.parts if part.kind == "compose" else (part,))
        if not flat:
            raise ValidationError("empty composition has no type")
        for left, right in pairwise(flat):
            if left.outputs != right.inputs:
                raise ValidationError(
                    "composition type mismatch: "
                    f"{left.inputs}->{left.outputs} then {right.inputs}->{right.outputs}"
                )
        inputs, outputs = flat[0].inputs, flat[-1].outputs
        nonidentities = [part for part in flat if part.kind != "id"]
        if not nonidentities:
            return cls.identity(inputs)
        if len(nonidentities) == 1:
            return nonidentities[0]
        return cls("compose", inputs, outputs, parts=tuple(nonidentities))

    @classmethod
    def tensor(cls, parts: Iterable[Circuit]) -> Circuit:
        flat: list[Circuit] = []
        for part in parts:
            flat.extend(part.parts if part.kind == "tensor" else (part,))
        flat = [
            part
            for part in flat
            if not (part.kind == "id" and part.inputs == 0)
        ]
        if not flat:
            return cls.identity(0)
        if len(flat) == 1:
            return flat[0]
        return cls(
            "tensor",
            sum(part.inputs for part in flat),
            sum(part.outputs for part in flat),
            parts=tuple(flat),
        )

    def to_json(self) -> dict[str, Any]:
        if self.kind == "gen":
            return {
                "gen": {
                    "id": self.name,
                    "type": (
                        self.inputs
                        if self.inputs == self.outputs
                        else [self.inputs, self.outputs]
                    ),
                }
            }
        if self.kind == "macro":
            return {"macro": self.name}
        if self.kind == "id":
            return {"id": self.inputs}
        if self.kind == "perm":
            return {"perm": list(self.order)}
        if self.kind in {"compose", "tensor"}:
            return {self.kind: [part.to_json() for part in self.parts]}
        raise AssertionError(f"unknown circuit kind {self.kind!r}")

    @classmethod
    def from_json(
        cls,
        value: Any,
        signature: Signature | Mapping[str, Generator],
        macros: Mapping[str, Circuit] | None = None,
    ) -> Circuit:
        signature = signature if isinstance(signature, Signature) else Signature(signature.values())

        def resolve_macro(name: str) -> Circuit:
            if macros is None or name not in macros:
                raise ValidationError(f"unknown macro {name!r}")
            return macros[name]

        return _parse_circuit(value, signature, resolve_macro)


def _name_from_term(value: Any, key: str) -> tuple[str, int | None, int | None]:
    raw = value[key]
    outer_type = _optional_type(value, f"inline {key}")
    if isinstance(raw, str):
        return raw, *outer_type
    if isinstance(raw, Mapping):
        _reject_unknown(
            raw,
            {"id", "name", "type", "inputs", "outputs"},
            f"inline {key}",
        )
        name = _one_of(raw, ("id", "name"), f"inline {key} identity")
        nested_type = _optional_type(raw, f"inline {key}")
        if outer_type != (None, None) and nested_type != (None, None):
            raise ValidationError(f"inline {key} type is declared twice")
        inputs, outputs = (
            nested_type if nested_type != (None, None) else outer_type
        )
        return _identifier(name, f"{key} name"), inputs, outputs
    raise ValidationError(f"{key} must name a {key}")


MacroResolver = Callable[[str], Circuit]


def _parse_named_circuit(
    value: Mapping[str, Any],
    key: str,
    signature: Signature,
    macro: MacroResolver,
) -> Circuit:
    name, inline_inputs, inline_outputs = _name_from_term(value, key)
    noun = "generator" if key == "gen" else "macro"
    if key == "gen":
        if name not in signature:
            if inline_inputs is None or inline_outputs is None:
                raise ValidationError(f"unknown generator {name!r}")
            return Circuit.generator(name, inline_inputs, inline_outputs)
        circuit = Circuit.generator(signature[name])
    else:
        definition = macro(name)
        circuit = Circuit.macro(name, definition.inputs, definition.outputs)
    if inline_inputs is not None and inline_inputs != circuit.inputs:
        raise ValidationError(f"inline type disagrees with {noun} {name!r}")
    if inline_outputs is not None and inline_outputs != circuit.outputs:
        raise ValidationError(f"inline type disagrees with {noun} {name!r}")
    return circuit


def _parse_circuit(value: Any, signature: Signature, macro: MacroResolver) -> Circuit:
    if not isinstance(value, Mapping):
        raise ValidationError("circuit must be an object")
    if "wires" in value or "ops" in value:
        if "wires" not in value or "ops" not in value:
            raise ValidationError("compact circuit needs both wires and ops")
        _reject_unknown(value, {"wires", "ops"}, "compact circuit")
        return _parse_compact(value, signature, macro)

    keys = [
        key
        for key in ("gen", "id", "perm", "compose", "tensor", "macro")
        if key in value
    ]
    if len(keys) != 1:
        raise ValidationError(
            "circuit must contain exactly one of gen/id/perm/compose/tensor/macro"
        )
    key = keys[0]
    allowed = (
        {key, "type", "inputs", "outputs"}
        if key in {"gen", "macro"}
        else {key}
    )
    _reject_unknown(value, allowed, "circuit")
    if key == "gen":
        return _parse_named_circuit(value, key, signature, macro)
    if key == "id":
        return Circuit.identity(value["id"])
    if key == "perm":
        return Circuit.perm(value["perm"])
    if key in {"compose", "tensor"}:
        raw_parts = value[key]
        if not _is_sequence(raw_parts):
            raise ValidationError(f"{key} must be an array")
        parts = [_parse_circuit(part, signature, macro) for part in raw_parts]
        return Circuit.compose(parts) if key == "compose" else Circuit.tensor(parts)
    return _parse_named_circuit(value, key, signature, macro)


def _parse_compact(value: Mapping[str, Any], signature: Signature, macro: MacroResolver) -> Circuit:
    wires = _natural(value["wires"], "compact circuit wires")
    ops = value["ops"]
    if not _is_sequence(ops):
        raise ValidationError("compact circuit ops must be an array")
    steps: list[Circuit] = []
    current = wires
    for index, op in enumerate(ops):
        if not isinstance(op, Mapping):
            raise ValidationError(f"operation {index} must be an object")
        op_keys = [key for key in ("gen", "macro", "perm") if key in op]
        if len(op_keys) != 1:
            raise ValidationError(f"operation {index} needs exactly one of gen/macro/perm")
        key = op_keys[0]
        allowed = (
            {key, "on", "type", "inputs", "outputs"}
            if key in {"gen", "macro"}
            else {key}
        )
        _reject_unknown(op, allowed, f"operation {index}")
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
                            f"operation {index} uses an integer on for a "
                            f"{base.inputs}-input box"
                        )
                    selected = (on,)
                else:
                    if not _is_sequence(on):
                        raise ValidationError(
                            f"operation {index} needs an on array or unary index"
                        )
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
        steps.append(step)
        current = step.outputs
    return Circuit.compose(steps) if steps else Circuit.identity(wires)


def _place_on_wires(base: Circuit, wires: int, selected: tuple[int, ...]) -> Circuit:
    """Lift a generator to selected positions in a deterministic compact syntax.

    Selected wires are fed to the box in ``on`` order. Endomorphism outputs
    return to the corresponding selected positions. For arity-changing boxes,
    the outputs replace the selected wires at their leftmost position; untouched
    wires keep their order. Nullary outputs are appended.
    """

    chosen = set(selected)
    rest = tuple(i for i in range(wires) if i not in chosen)
    pre = Circuit.perm(selected + rest)
    middle = Circuit.tensor((base, Circuit.identity(len(rest))))
    generated = tuple(("g", i) for i in range(base.outputs))
    source = generated + tuple(("w", i) for i in rest)
    positions = {token: i for i, token in enumerate(source)}
    if base.inputs == base.outputs:
        output_for_wire = {wire: ("g", index) for index, wire in enumerate(selected)}
        desired = tuple(output_for_wire.get(wire, ("w", wire)) for wire in range(wires))
        post = Circuit.perm(tuple(positions[token] for token in desired))
        return Circuit.compose((pre, middle, post))

    anchor = min(selected) if selected else wires
    before = tuple(("w", i) for i in rest if i < anchor)
    after = tuple(("w", i) for i in rest if i >= anchor)
    post = Circuit.perm(tuple(positions[token] for token in before + generated + after))
    return Circuit.compose((pre, middle, post))


@dataclass(frozen=True, slots=True)
class Equation:
    id: str
    lhs: Circuit
    rhs: Circuit

    def __post_init__(self) -> None:
        _identifier(self.id, "equation id")
        if self.lhs.type != self.rhs.type:
            raise ValidationError(
                f"equation {self.id!r} is ill-typed: {self.lhs.type} != {self.rhs.type}"
            )

    @property
    def arity(self) -> int:
        return max(self.lhs.inputs, self.lhs.outputs)


@dataclass(slots=True)
class Theory:
    signature: Signature
    equations: tuple[Equation, ...]
    macros: dict[str, Circuit] = field(default_factory=dict)
    name: str | None = None
    wire_dimension: int = 2

    @classmethod
    def from_json(cls, value: Any, signature: Signature | None = None) -> Theory:
        if _is_sequence(value):
            if signature is None:
                raise ValidationError(
                    "a signature is required when the top level is an equation array"
                )
            data: Mapping[str, Any] = {"equations": value}
        elif isinstance(value, Mapping):
            data = value
            _reject_unknown(
                data,
                {
                    "name",
                    "wire_dimension",
                    "dimension",
                    "source",
                    "generators",
                    "signature",
                    "macros",
                    "equations",
                },
                "theory",
            )
        else:
            raise ValidationError("theory must be an object or equation array")
        dimension = _natural(
            _one_of(
                data,
                ("wire_dimension", "dimension"),
                "theory wire dimension",
                default=2,
            ),
            "wire dimension",
        )
        if dimension == 0:
            raise ValidationError("wire dimension must be positive")
        raw_signature = _one_of(
            data,
            ("generators", "signature"),
            "theory signature",
            default=None,
        )
        if signature is None:
            if raw_signature is None:
                raise ValidationError("theory has no generators/signature")
            signature = Signature.from_json(raw_signature)
        _validate_generator_matrices(signature, dimension)

        macros = _parse_macro_definitions(data.get("macros", []), signature)
        raw_equations = data.get("equations")
        if not _is_sequence(raw_equations):
            raise ValidationError("equations must be an array")
        equations: list[Equation] = []
        ids: set[str] = set()
        for raw in raw_equations:
            if not isinstance(raw, Mapping):
                raise ValidationError("each equation must be an object")
            _reject_unknown(raw, {"id", "name", "lhs", "rhs"}, "equation")
            equation_id = _identifier(
                _one_of(raw, ("id", "name"), "equation identity", default=None),
                "equation id",
            )
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
        return cls(signature, tuple(equations), macros, name, dimension)

    def equation(self, equation_id: str) -> Equation:
        for equation in self.equations:
            if equation.id == equation_id:
                return equation
        raise KeyError(equation_id)


def iter_theory_terms(theory: Theory, *extra: Circuit) -> Iterator[Circuit]:
    """Iterate over macro bodies, equation sides, then any extra terms."""

    return chain(
        theory.macros.values(),
        (
            side
            for equation in theory.equations
            for side in (equation.lhs, equation.rhs)
        ),
        extra,
    )


def iter_circuit_nodes(
    circuit: Circuit,
    checkpoint: Callable[[], None] | None = None,
) -> Iterator[Circuit]:
    """Iterate over a circuit tree without recursion."""

    pending = [circuit]
    visited = 0
    while pending:
        if checkpoint is not None and visited % 1024 == 0:
            checkpoint()
        node = pending.pop()
        yield node
        pending.extend(node.parts)
        visited += 1


def _looks_like_circuit(value: Mapping[str, Any]) -> bool:
    return (
        "wires" in value
        or "ops" in value
        or any(
            key in value
            for key in ("gen", "perm", "compose", "tensor", "macro")
        )
        or ("id" in value and isinstance(value["id"], int))
    )


def _parse_macro_definitions(value: Any, signature: Signature) -> dict[str, Circuit]:
    entries: list[tuple[str | None, Any]]
    if isinstance(value, Mapping):
        entries = list(value.items())
    elif _is_sequence(value):
        entries = [(None, item) for item in value]
    else:
        raise ValidationError("macros must be an object or array")

    specs: dict[str, tuple[Any, int | None, int | None]] = {}
    for key_name, raw in entries:
        if (
            key_name is not None
            and isinstance(raw, Mapping)
            and _looks_like_circuit(raw)
        ):
            body, inputs, outputs = raw, None, None
        elif not isinstance(raw, Mapping):
            if key_name is None:
                raise ValidationError("each macro must be an object")
            body, inputs, outputs = raw, None, None
        else:
            _reject_unknown(
                raw,
                {
                    "id",
                    "name",
                    "body",
                    "circuit",
                    "type",
                    "inputs",
                    "source",
                    "outputs",
                    "target",
                },
                "macro",
            )
            name_in_body = _one_of(
                raw,
                ("id", "name"),
                "macro identity",
                default=key_name,
            )
            if key_name is not None and name_in_body is not None and name_in_body != key_name:
                raise ValidationError(f"macro key {key_name!r} disagrees with id {name_in_body!r}")
            key_name = name_in_body
            body = _one_of(
                raw,
                ("body", "circuit"),
                f"macro {key_name!r} body",
                default=None,
            )
            inputs, outputs = _optional_type(
                raw,
                f"macro {key_name!r}",
                ("inputs", "source"),
                ("outputs", "target"),
            )
        name = _identifier(key_name, "macro id")
        if name in specs:
            raise ValidationError(f"duplicate macro id {name!r}")
        if body is None:
            raise ValidationError(f"macro {name!r} has no body")
        specs[name] = (body, inputs, outputs)

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
        raw_body, declared_inputs, declared_outputs = specs[name]
        body = _parse_circuit(raw_body, signature, resolve)
        if declared_inputs is not None and declared_inputs != body.inputs:
            raise ValidationError(
                f"declared input type of macro {name!r} disagrees with its body"
            )
        if declared_outputs is not None and declared_outputs != body.outputs:
            raise ValidationError(
                f"declared output type of macro {name!r} disagrees with its body"
            )
        visiting.pop()
        resolved[name] = body
        return body

    for name in specs:
        resolve(name)
    return resolved


def _checked_macro_body(
    node: Circuit,
    registry: Mapping[str, Circuit],
) -> tuple[str, Circuit]:
    name = node.name or ""
    if name not in registry:
        raise ValidationError(f"unknown macro {name!r}")
    body = registry[name]
    if body.type != node.type:
        raise ValidationError(f"macro {name!r} has inconsistent type")
    return name, body


def contains_arity_change(
    circuit: Circuit,
    checkpoint: Callable[[], None] | None = None,
) -> bool:
    """Whether a term contains any non-endomorphic subterm."""

    return any(
        node.inputs != node.outputs
        for node in iter_circuit_nodes(circuit, checkpoint)
    )


def expand_macros(
    circuit: Circuit, macros: Mapping[str, Circuit] | None = None
) -> Circuit:
    """Return a macro-free circuit, detecting cycles in hand-built registries."""

    macros = macros or {}
    active: list[str] = []

    def expand(node: Circuit) -> Circuit:
        if node.kind == "macro":
            name, body = _checked_macro_body(node, macros)
            if name in active:
                start = active.index(name)
                raise ValidationError(f"macro cycle: {' -> '.join(active[start:] + [name])}")
            active.append(name)
            result = expand(body)
            active.pop()
            return result
        if node.kind == "compose":
            return Circuit.compose(expand(part) for part in node.parts)
        if node.kind == "tensor":
            return Circuit.tensor(expand(part) for part in node.parts)
        return node

    return expand(circuit)


def primitive_occurrences(
    circuit: Circuit, macros: Mapping[str, Circuit] | None = None
) -> Counter[str]:
    """Count primitive generators after expanding every macro."""

    registry = macros or {}
    cache: dict[str, Counter[str]] = {}
    active: list[str] = []

    def visit(node: Circuit) -> Counter[str]:
        if node.kind == "gen":
            return Counter((node.name or "",))
        if node.kind == "macro":
            name, body = _checked_macro_body(node, registry)
            if name in cache:
                return cache[name]
            if name in active:
                start = active.index(name)
                raise ValidationError(
                    f"macro cycle: {' -> '.join(active[start:] + [name])}"
                )
            active.append(name)
            cache[name] = visit(body)
            active.pop()
            return cache[name]
        result: Counter[str] = Counter()
        for part in node.parts:
            result.update(visit(part))
        return result

    return visit(circuit)


def inversion_count(order: Sequence[int]) -> int:
    """Number of inversions in a permutation."""

    parsed = Circuit.perm(order).order
    return sum(left > right for i, left in enumerate(parsed) for right in parsed[i + 1 :])


def permutation_parity(order: Sequence[int]) -> int:
    """Return 0 for an even permutation and 1 for an odd one."""

    return inversion_count(order) & 1


def structural_permutation_parity(
    circuit: Circuit, macros: Mapping[str, Circuit] | None = None
) -> int:
    """XOR the inversion parity of all explicit structural permutations."""

    registry = macros or {}
    cache: dict[str, int] = {}
    active: list[str] = []

    def parity(node: Circuit) -> int:
        if node.kind == "macro":
            name, body = _checked_macro_body(node, registry)
            if name in cache:
                return cache[name]
            if name in active:
                start = active.index(name)
                raise ValidationError(
                    f"macro cycle: {' -> '.join(active[start:] + [name])}"
                )
            active.append(name)
            cache[name] = parity(body)
            active.pop()
            return cache[name]
        own = permutation_parity(node.order) if node.kind == "perm" else 0
        return own ^ (sum(parity(part) for part in node.parts) & 1)

    return parity(circuit)


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
    """Load JSON while rejecting duplicate keys and non-finite numbers."""

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
