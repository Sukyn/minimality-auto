"""A compact, typed JSON representation of circuits in a PROP.

Composition is written in execution order: ``compose: [f, g]`` means first
``f``, then ``g``.  A permutation list gives, for each output position, the
input position routed there.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


class PropError(ValueError):
    """Base class for malformed PROP data."""


class ValidationError(PropError):
    """Raised when JSON is well formed but not a typed PROP expression."""


def _natural(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{where} must be a non-negative integer")
    return value


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
    if isinstance(value, (int, float, complex)):
        return complex(value)
    if isinstance(value, str):
        try:
            return complex(value.strip().replace("I", "j").replace("i", "j"))
        except ValueError as exc:
            raise ValidationError(f"invalid complex scalar {value!r}") from exc
    if isinstance(value, Mapping):
        allowed = {"re", "im", "real", "imag"}
        unknown = set(value) - allowed
        if unknown:
            raise ValidationError(f"unknown complex scalar fields: {sorted(unknown)}")
        re = value.get("re", value.get("real", 0))
        im = value.get("im", value.get("imag", 0))
        try:
            return complex(float(re), float(im))
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"invalid complex scalar {value!r}") from exc
    raise ValidationError(f"invalid complex scalar {value!r}")


def _matrix_tuple(value: Any) -> tuple[tuple[complex, ...], ...]:
    if isinstance(value, Mapping) and "real" in value:
        unknown = set(value) - {"real", "imag"}
        if unknown:
            raise ValidationError(f"unknown matrix fields: {sorted(unknown)}")
        real = value["real"]
        imag = value.get("imag")
        if not isinstance(real, Sequence) or isinstance(real, (str, bytes)):
            raise ValidationError("matrix real part must be an array of rows")
        if imag is None:
            imag = []
            for row in real:
                if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                    raise ValidationError("matrix must be an array of rows")
                imag.append([0] * len(row))
        if not isinstance(imag, Sequence) or isinstance(imag, (str, bytes)):
            raise ValidationError("matrix imaginary part must be an array of rows")
        if len(real) != len(imag):
            raise ValidationError("matrix real and imaginary parts have different shapes")
        value = []
        for real_row, imag_row in zip(real, imag, strict=True):
            if not isinstance(real_row, Sequence) or isinstance(real_row, (str, bytes)):
                raise ValidationError("matrix must be an array of rows")
            if not isinstance(imag_row, Sequence) or isinstance(imag_row, (str, bytes)):
                raise ValidationError("matrix must be an array of rows")
            if len(real_row) != len(imag_row):
                raise ValidationError("matrix real and imaginary parts have different shapes")
            value.append(
                [{"re": re, "im": im} for re, im in zip(real_row, imag_row, strict=True)]
            )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValidationError("matrix must be a non-empty array of rows")
    rows: list[tuple[complex, ...]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not row:
            raise ValidationError("matrix rows must be non-empty arrays")
        parsed = tuple(parse_complex(item) for item in row)
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            raise ValidationError("matrix rows have different lengths")
        rows.append(parsed)
    return tuple(rows)


def parse_complex_matrix(value: Any) -> Any:
    """Parse a JSON matrix and return a NumPy complex array.

    NumPy is imported lazily so structural searches stay lightweight.
    """

    np = _numpy()
    return np.asarray(_matrix_tuple(value), dtype=np.complex128)


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
    def dom(self) -> int:
        return self.inputs

    @property
    def cod(self) -> int:
        return self.outputs

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

    @classmethod
    def from_json(cls, value: Any) -> Signature:
        entries: list[tuple[str | None, Any]]
        if isinstance(value, Mapping):
            entries = [(name, spec) for name, spec in value.items()]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            entries = [(None, spec) for spec in value]
        else:
            raise ValidationError("generators must be an object or array")

        generators: list[Generator] = []
        seen: set[str] = set()
        for key_name, spec in entries:
            if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, Mapping)):
                if key_name is None or len(spec) != 2:
                    raise ValidationError("generator pair form is name: [inputs, outputs]")
                name, inputs, outputs, matrix = key_name, spec[0], spec[1], None
            elif isinstance(spec, Mapping):
                name = spec.get("id", spec.get("name", key_name))
                if key_name is not None and name is not None and name != key_name:
                    raise ValidationError(
                        f"generator key {key_name!r} disagrees with id {name!r}"
                    )
                typ = spec.get("type")
                if typ is not None:
                    if not isinstance(typ, Sequence) or len(typ) != 2:
                        raise ValidationError(f"generator {name!r} type must be [inputs, outputs]")
                    inputs, outputs = typ
                else:
                    inputs = spec.get("inputs", spec.get("in", spec.get("source")))
                    outputs = spec.get("outputs", spec.get("out", spec.get("target")))
                raw_matrix = spec.get("matrix")
                matrix = _matrix_tuple(raw_matrix) if raw_matrix is not None else None
            else:
                raise ValidationError("each generator must be an object or [inputs, outputs]")
            name = _identifier(name, "generator id")
            if name in seen:
                raise ValidationError(f"duplicate generator id {name!r}")
            seen.add(name)
            generators.append(
                Generator(name, _natural(inputs, f"{name} inputs"), _natural(outputs, f"{name} outputs"), matrix)
            )
        return cls(generators)


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
    def dom(self) -> int:
        return self.inputs

    @property
    def cod(self) -> int:
        return self.outputs

    @property
    def type(self) -> tuple[int, int]:
        return self.inputs, self.outputs

    @classmethod
    def generator(cls, generator: Generator | str, inputs: int | None = None, outputs: int | None = None) -> Circuit:
        if isinstance(generator, Generator):
            return cls("gen", generator.inputs, generator.outputs, name=generator.name)
        name = _identifier(generator, "generator name")
        if inputs is None or outputs is None:
            raise ValidationError(f"type required for generator {name!r}")
        return cls("gen", _natural(inputs, "generator inputs"), _natural(outputs, "generator outputs"), name=name)

    @classmethod
    def identity(cls, wires: int) -> Circuit:
        wires = _natural(wires, "identity size")
        return cls("id", wires, wires)

    @classmethod
    def perm(cls, order: Sequence[int]) -> Circuit:
        if isinstance(order, (str, bytes)):
            raise ValidationError("permutation must be an array")
        parsed = tuple(order)
        if any(isinstance(i, bool) or not isinstance(i, int) for i in parsed):
            raise ValidationError("permutation entries must be integers")
        if sorted(parsed) != list(range(len(parsed))):
            raise ValidationError(f"not a permutation of 0..{len(parsed) - 1}: {list(parsed)}")
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
        for left, right in zip(flat, flat[1:]):
            if left.outputs != right.inputs:
                raise ValidationError(
                    "composition type mismatch: "
                    f"{left.inputs}->{left.outputs} then {right.inputs}->{right.outputs}"
                )
        if len(flat) == 1:
            return flat[0]
        return cls("compose", flat[0].inputs, flat[-1].outputs, parts=tuple(flat))

    @classmethod
    def tensor(cls, parts: Iterable[Circuit]) -> Circuit:
        flat: list[Circuit] = []
        for part in parts:
            flat.extend(part.parts if part.kind == "tensor" else (part,))
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

    def then(self, other: Circuit) -> Circuit:
        return Circuit.compose((self, other))

    def tensor_with(self, other: Circuit) -> Circuit:
        return Circuit.tensor((self, other))

    def to_json(self) -> dict[str, Any]:
        if self.kind == "gen":
            return {"gen": self.name}
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
        macros: Mapping[str, MacroDef | Circuit] | None = None,
    ) -> Circuit:
        signature = signature if isinstance(signature, Signature) else Signature(signature.values())

        def resolve_macro(name: str) -> MacroDef:
            if macros is None or name not in macros:
                raise ValidationError(f"unknown macro {name!r}")
            macro = macros[name]
            return macro if isinstance(macro, MacroDef) else MacroDef(name, macro)

        return _parse_circuit(value, signature, resolve_macro)


@dataclass(frozen=True, slots=True)
class MacroDef:
    name: str
    body: Circuit

    def __post_init__(self) -> None:
        _identifier(self.name, "macro id")

    @property
    def inputs(self) -> int:
        return self.body.inputs

    @property
    def outputs(self) -> int:
        return self.body.outputs

    @property
    def type(self) -> tuple[int, int]:
        return self.body.type


def _name_from_term(value: Any, key: str) -> tuple[str, int | None, int | None]:
    raw = value[key]
    if isinstance(raw, str):
        return raw, value.get("inputs"), value.get("outputs")
    if isinstance(raw, Mapping):
        name = raw.get("id", raw.get("name"))
        typ = raw.get("type")
        if typ is not None:
            if not isinstance(typ, Sequence) or len(typ) != 2:
                raise ValidationError(f"inline {key} type must be [inputs, outputs]")
            return _identifier(name, f"{key} name"), typ[0], typ[1]
        return _identifier(name, f"{key} name"), raw.get("inputs"), raw.get("outputs")
    raise ValidationError(f"{key} must name a {key}")


MacroResolver = Callable[[str], MacroDef]


def _parse_circuit(value: Any, signature: Signature, macro: MacroResolver) -> Circuit:
    if not isinstance(value, Mapping):
        raise ValidationError("circuit must be an object")
    if "wires" in value or "ops" in value:
        if "wires" not in value or "ops" not in value:
            raise ValidationError("compact circuit needs both wires and ops")
        return _parse_compact(value, signature, macro)

    keys = [key for key in ("gen", "id", "perm", "compose", "tensor", "macro") if key in value]
    if len(keys) != 1:
        raise ValidationError("circuit must contain exactly one of gen/id/perm/compose/tensor/macro")
    key = keys[0]
    if key == "gen":
        name, inline_inputs, inline_outputs = _name_from_term(value, "gen")
        if name in signature:
            generator = signature[name]
            if inline_inputs is not None and inline_inputs != generator.inputs:
                raise ValidationError(f"inline type disagrees with generator {name!r}")
            if inline_outputs is not None and inline_outputs != generator.outputs:
                raise ValidationError(f"inline type disagrees with generator {name!r}")
            return Circuit.generator(generator)
        if inline_inputs is None or inline_outputs is None:
            raise ValidationError(f"unknown generator {name!r}")
        return Circuit.generator(name, inline_inputs, inline_outputs)
    if key == "id":
        return Circuit.identity(value["id"])
    if key == "perm":
        order = value["perm"]
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            raise ValidationError("permutation must be an array")
        return Circuit.perm(order)
    if key in {"compose", "tensor"}:
        raw_parts = value[key]
        if not isinstance(raw_parts, Sequence) or isinstance(raw_parts, (str, bytes)):
            raise ValidationError(f"{key} must be an array")
        parts = [_parse_circuit(part, signature, macro) for part in raw_parts]
        if key == "compose" and not parts and "wires" in value:
            return Circuit.identity(_natural(value["wires"], "empty composition wires"))
        return Circuit.compose(parts) if key == "compose" else Circuit.tensor(parts)
    name, inline_inputs, inline_outputs = _name_from_term(value, "macro")
    definition = macro(name)
    if inline_inputs is not None and inline_inputs != definition.inputs:
        raise ValidationError(f"inline type disagrees with macro {name!r}")
    if inline_outputs is not None and inline_outputs != definition.outputs:
        raise ValidationError(f"inline type disagrees with macro {name!r}")
    return Circuit.macro(name, definition.inputs, definition.outputs)


def _parse_compact(value: Mapping[str, Any], signature: Signature, macro: MacroResolver) -> Circuit:
    wires = _natural(value["wires"], "compact circuit wires")
    ops = value["ops"]
    if not isinstance(ops, Sequence) or isinstance(ops, (str, bytes)):
        raise ValidationError("compact circuit ops must be an array")
    circuit = Circuit.identity(wires)
    current = wires
    for index, op in enumerate(ops):
        if not isinstance(op, Mapping):
            raise ValidationError(f"operation {index} must be an object")
        op_keys = [key for key in ("gen", "macro", "perm") if key in op]
        if len(op_keys) != 1:
            raise ValidationError(f"operation {index} needs exactly one of gen/macro/perm")
        if op_keys[0] == "perm":
            step = Circuit.perm(op["perm"])
            if step.inputs != current:
                raise ValidationError(
                    f"operation {index} permutation has {step.inputs} wires, expected {current}"
                )
        else:
            key = op_keys[0]
            name, inline_inputs, inline_outputs = _name_from_term(op, key)
            if key == "gen":
                if name in signature:
                    generator = signature[name]
                    if inline_inputs is not None and inline_inputs != generator.inputs:
                        raise ValidationError(
                            f"inline type disagrees with generator {name!r}"
                        )
                    if inline_outputs is not None and inline_outputs != generator.outputs:
                        raise ValidationError(
                            f"inline type disagrees with generator {name!r}"
                        )
                    base = Circuit.generator(generator)
                else:
                    if inline_inputs is None or inline_outputs is None:
                        raise ValidationError(f"unknown generator {name!r}")
                    base = Circuit.generator(name, inline_inputs, inline_outputs)
            else:
                definition = macro(name)
                if inline_inputs is not None and inline_inputs != definition.inputs:
                    raise ValidationError(
                        f"inline type disagrees with macro {name!r}"
                    )
                if inline_outputs is not None and inline_outputs != definition.outputs:
                    raise ValidationError(
                        f"inline type disagrees with macro {name!r}"
                    )
                base = Circuit.macro(name, definition.inputs, definition.outputs)
            on = op.get("on")
            if not isinstance(on, Sequence) or isinstance(on, (str, bytes)):
                raise ValidationError(f"operation {index} needs an on array")
            selected = tuple(on)
            if len(selected) != base.inputs:
                raise ValidationError(
                    f"operation {index} selects {len(selected)} wires for {base.inputs}-input {name!r}"
                )
            if any(isinstance(i, bool) or not isinstance(i, int) for i in selected):
                raise ValidationError(f"operation {index} wire positions must be integers")
            if len(set(selected)) != len(selected):
                raise ValidationError(f"operation {index} selects a wire more than once")
            if any(i < 0 or i >= current for i in selected):
                raise ValidationError(f"operation {index} selects a wire outside 0..{current - 1}")
            step = _place_on_wires(base, current, selected)
        circuit = Circuit.compose((circuit, step))
        current = step.outputs
    return circuit


def _place_on_wires(base: Circuit, wires: int, selected: tuple[int, ...]) -> Circuit:
    """Lift a generator to selected positions in a deterministic compact syntax.

    Selected wires are fed to the box in ``on`` order. Endomorphism outputs
    return to the corresponding selected positions. For arity-changing boxes,
    the outputs replace the selected wires at their leftmost position; untouched
    wires keep their order. Nullary outputs are appended.
    """

    rest = tuple(i for i in range(wires) if i not in set(selected))
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
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        _identifier(self.id, "equation id")
        if self.lhs.type != self.rhs.type:
            raise ValidationError(
                f"equation {self.id!r} is ill-typed: {self.lhs.type} != {self.rhs.type}"
            )

    @property
    def type(self) -> tuple[int, int]:
        return self.lhs.type


@dataclass(slots=True)
class Theory:
    signature: Signature
    equations: tuple[Equation, ...]
    macros: dict[str, MacroDef] = field(default_factory=dict)
    name: str | None = None
    wire_dimension: int = 2

    @classmethod
    def from_json(cls, value: Any, signature: Signature | None = None) -> Theory:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if signature is None:
                raise ValidationError("a signature is required when the top level is an equation array")
            data: Mapping[str, Any] = {"equations": value}
        elif isinstance(value, Mapping):
            data = value
        else:
            raise ValidationError("theory must be an object or equation array")
        if signature is None:
            raw_signature = data.get("generators", data.get("signature"))
            if raw_signature is None:
                raise ValidationError("theory has no generators/signature")
            signature = Signature.from_json(raw_signature)

        macros = _parse_macro_definitions(data.get("macros", []), signature)
        raw_equations = data.get("equations")
        if not isinstance(raw_equations, Sequence) or isinstance(raw_equations, (str, bytes)):
            raise ValidationError("equations must be an array")
        equations: list[Equation] = []
        ids: set[str] = set()
        for raw in raw_equations:
            if not isinstance(raw, Mapping):
                raise ValidationError("each equation must be an object")
            equation_id = _identifier(raw.get("id", raw.get("name")), "equation id")
            if equation_id in ids:
                raise ValidationError(f"duplicate equation id {equation_id!r}")
            ids.add(equation_id)
            if "lhs" not in raw or "rhs" not in raw:
                raise ValidationError(f"equation {equation_id!r} needs lhs and rhs")
            lhs = Circuit.from_json(raw["lhs"], signature, macros)
            rhs = Circuit.from_json(raw["rhs"], signature, macros)
            metadata = {key: item for key, item in raw.items() if key not in {"id", "name", "lhs", "rhs"}}
            equations.append(Equation(equation_id, lhs, rhs, metadata))
        dimension = _natural(data.get("wire_dimension", data.get("dimension", 2)), "wire dimension")
        if dimension == 0:
            raise ValidationError("wire dimension must be positive")
        name = data.get("name")
        if name is not None:
            name = _identifier(name, "theory name")
        return cls(signature, tuple(equations), macros, name, dimension)

    def equation(self, equation_id: str) -> Equation:
        for equation in self.equations:
            if equation.id == equation_id:
                return equation
        raise KeyError(equation_id)


def _parse_macro_definitions(value: Any, signature: Signature) -> dict[str, MacroDef]:
    entries: list[tuple[str | None, Any]]
    if isinstance(value, Mapping):
        entries = list(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = [(None, item) for item in value]
    else:
        raise ValidationError("macros must be an object or array")

    specs: dict[str, tuple[Any, int | None, int | None]] = {}
    for key_name, raw in entries:
        if not isinstance(raw, Mapping):
            if key_name is None:
                raise ValidationError("each macro must be an object")
            body, inputs, outputs = raw, None, None
        else:
            name_in_body = raw.get("id", raw.get("name", key_name))
            if key_name is not None and name_in_body is not None and name_in_body != key_name:
                raise ValidationError(f"macro key {key_name!r} disagrees with id {name_in_body!r}")
            key_name = name_in_body
            body = raw.get("body", raw.get("circuit"))
            typ = raw.get("type")
            if typ is not None:
                if not isinstance(typ, Sequence) or len(typ) != 2:
                    raise ValidationError(f"macro {key_name!r} type must be [inputs, outputs]")
                inputs, outputs = typ
            else:
                inputs = raw.get("inputs", raw.get("source"))
                outputs = raw.get("outputs", raw.get("target"))
        name = _identifier(key_name, "macro id")
        if name in specs:
            raise ValidationError(f"duplicate macro id {name!r}")
        if body is None:
            raise ValidationError(f"macro {name!r} has no body")
        specs[name] = (body, inputs, outputs)

    resolved: dict[str, MacroDef] = {}
    visiting: list[str] = []

    def resolve(name: str) -> MacroDef:
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
        if declared_inputs is not None and _natural(declared_inputs, f"macro {name} inputs") != body.inputs:
            raise ValidationError(f"declared input type of macro {name!r} disagrees with its body")
        if declared_outputs is not None and _natural(declared_outputs, f"macro {name} outputs") != body.outputs:
            raise ValidationError(f"declared output type of macro {name!r} disagrees with its body")
        visiting.pop()
        result = MacroDef(name, body)
        resolved[name] = result
        return result

    for name in specs:
        resolve(name)
    return resolved


def expand_macros(circuit: Circuit, macros: Mapping[str, MacroDef | Circuit] | None = None) -> Circuit:
    """Return a macro-free circuit, detecting cycles in hand-built registries."""

    macros = macros or {}
    active: list[str] = []

    def expand(node: Circuit) -> Circuit:
        if node.kind == "macro":
            name = node.name or ""
            if name not in macros:
                raise ValidationError(f"unknown macro {name!r}")
            if name in active:
                start = active.index(name)
                raise ValidationError(f"macro cycle: {' -> '.join(active[start:] + [name])}")
            definition = macros[name]
            body = definition.body if isinstance(definition, MacroDef) else definition
            if body.type != node.type:
                raise ValidationError(f"macro {name!r} has inconsistent type")
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
    circuit: Circuit, macros: Mapping[str, MacroDef | Circuit] | None = None
) -> Counter[str]:
    """Count primitive generators after expanding every macro."""

    registry = macros or {}
    cache: dict[str, Counter[str]] = {}
    active: list[str] = []

    def visit(node: Circuit) -> Counter[str]:
        if node.kind == "gen":
            return Counter((node.name or "",))
        if node.kind == "macro":
            name = node.name or ""
            if name not in registry:
                raise ValidationError(f"unknown macro {name!r}")
            definition = registry[name]
            body = definition.body if isinstance(definition, MacroDef) else definition
            if body.type != node.type:
                raise ValidationError(f"macro {name!r} has inconsistent type")
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
    circuit: Circuit, macros: Mapping[str, MacroDef | Circuit] | None = None
) -> int:
    """XOR the inversion parity of all explicit structural permutations."""

    registry = macros or {}
    cache: dict[str, int] = {}
    active: list[str] = []

    def parity(node: Circuit) -> int:
        if node.kind == "macro":
            name = node.name or ""
            if name not in registry:
                raise ValidationError(f"unknown macro {name!r}")
            definition = registry[name]
            body = definition.body if isinstance(definition, MacroDef) else definition
            if body.type != node.type:
                raise ValidationError(f"macro {name!r} has inconsistent type")
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


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise PropError("matrix evaluation requires NumPy") from exc
    return np


class MatrixSemantics:
    """Evaluate circuits as complex matrices over a fixed wire dimension."""

    def __init__(
        self,
        signature: Signature,
        wire_dimension: int = 2,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.signature = signature
        self.wire_dimension = _natural(wire_dimension, "wire dimension")
        if self.wire_dimension == 0:
            raise ValidationError("wire dimension must be positive")
        self.overrides = dict(overrides or {})

    def evaluate(
        self,
        circuit: Circuit,
        macros: Mapping[str, MacroDef | Circuit] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Any:
        np = _numpy()
        matrices = dict(self.overrides)
        matrices.update(overrides or {})
        registry = macros or {}
        macro_matrices: dict[str, Any] = {}
        active_macros: list[str] = []
        d = self.wire_dimension

        def generator_matrix(node: Circuit) -> Any:
            assert node.name is not None
            raw = matrices.get(node.name)
            if raw is None:
                raw = self.signature[node.name].matrix
            if raw is None:
                raise ValidationError(f"no matrix supplied for generator {node.name!r}")
            matrix = np.asarray(
                raw if hasattr(raw, "shape") else _matrix_tuple(raw), dtype=np.complex128
            )
            expected = (d**node.outputs, d**node.inputs)
            if matrix.shape != expected:
                raise ValidationError(
                    f"matrix for {node.name!r} has shape {matrix.shape}, expected {expected}"
                )
            return matrix

        def permutation_matrix(order: tuple[int, ...]) -> Any:
            size = d ** len(order)
            matrix = np.zeros((size, size), dtype=np.complex128)
            for column in range(size):
                digits = [0] * len(order)
                remainder = column
                for index in range(len(order) - 1, -1, -1):
                    digits[index] = remainder % d
                    remainder //= d
                row = 0
                for source in order:
                    row = row * d + digits[source]
                matrix[row, column] = 1
            return matrix

        def run(node: Circuit) -> Any:
            if node.kind == "gen":
                return generator_matrix(node)
            if node.kind == "id":
                return np.eye(d**node.inputs, dtype=np.complex128)
            if node.kind == "perm":
                return permutation_matrix(node.order)
            if node.kind == "tensor":
                result = np.ones((1, 1), dtype=np.complex128)
                for part in node.parts:
                    result = np.kron(result, run(part))
                return result
            if node.kind == "compose":
                result = run(node.parts[0])
                for part in node.parts[1:]:
                    result = run(part) @ result
                return result
            if node.kind == "macro":
                name = node.name or ""
                if name not in registry:
                    raise ValidationError(f"unknown macro {name!r}")
                definition = registry[name]
                body = (
                    definition.body
                    if isinstance(definition, MacroDef)
                    else definition
                )
                if body.type != node.type:
                    raise ValidationError(f"macro {name!r} has inconsistent type")
                if name in macro_matrices:
                    return macro_matrices[name]
                if name in active_macros:
                    start = active_macros.index(name)
                    raise ValidationError(
                        "macro cycle: "
                        + " -> ".join(active_macros[start:] + [name])
                    )
                active_macros.append(name)
                macro_matrices[name] = run(body)
                active_macros.pop()
                return macro_matrices[name]
            raise AssertionError(f"unknown circuit kind {node.kind!r}")

        return run(circuit)


def evaluate_matrix(
    circuit: Circuit,
    signature: Signature,
    macros: Mapping[str, MacroDef | Circuit] | None = None,
    wire_dimension: int = 2,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Convenience wrapper around :class:`MatrixSemantics`."""

    return MatrixSemantics(signature, wire_dimension, overrides).evaluate(circuit, macros)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: str | Path) -> Any:
    """Load JSON while rejecting duplicate object keys."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def load_theory(path: str | Path) -> Theory:
    return Theory.from_json(load_json(path))
