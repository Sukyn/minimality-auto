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
    inputs = _natural(value[0], f"{where} inputs")
    outputs = _natural(value[1], f"{where} outputs")
    if inputs == outputs:
        raise ValidationError(f"{where} endomorphism must use the scalar arity {inputs}")
    return inputs, outputs


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where} must be a non-empty string")
    return value


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
        # Mapping.__contains__ probes through __getitem__, which deliberately
        # turns a missing lookup into a user-facing ValidationError.
        return name in self._generators

    def get(self, name: str, default: Any = None) -> Generator | Any:
        return self._generators.get(name, default)

    @classmethod
    def from_json(cls, value: Any) -> Signature:
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
                inputs, outputs = _type_pair(
                    spec["type"],
                    f"generator {name!r} type",
                )
                raw_matrix = spec.get("matrix")
                matrix = _matrix_tuple(raw_matrix) if raw_matrix is not None else None
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
    def compose(cls, parts: Iterable[Circuit]) -> Circuit:
        items = list(parts)
        if not items:
            raise ValidationError("empty composition has no type")
        for left, right in pairwise(items):
            if left.outputs != right.inputs:
                raise ValidationError(
                    "composition type mismatch: "
                    f"{left.inputs}->{left.outputs} then {right.inputs}->{right.outputs}"
                )
        inputs, outputs = items[0].inputs, items[-1].outputs
        nonidentities = [part for part in items if part.kind != "id"]
        if not nonidentities:
            return cls.identity(inputs)
        if len(nonidentities) == 1:
            return nonidentities[0]
        return cls("compose", inputs, outputs, parts=tuple(nonidentities))

    @classmethod
    def tensor(cls, parts: Iterable[Circuit]) -> Circuit:
        items = [
            part
            for part in parts
            if not (part.kind == "id" and part.inputs == 0)
        ]
        if not items:
            return cls.identity(0)
        if len(items) == 1:
            return items[0]
        return cls(
            "tensor",
            sum(part.inputs for part in items),
            sum(part.outputs for part in items),
            parts=tuple(items),
        )

    def to_json(self) -> dict[str, Any]:
        if self.kind == "gen":
            return {"gen": self.name}
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
        signature: Signature,
        macros: Mapping[str, Circuit] | None = None,
    ) -> Circuit:
        def resolve_macro(name: str) -> Circuit:
            if macros is None or name not in macros:
                raise ValidationError(f"unknown macro {name!r}")
            return macros[name]

        return _parse_circuit(value, signature, resolve_macro)


MacroResolver = Callable[[str], Circuit]


def _parse_named_circuit(
    value: Mapping[str, Any],
    key: str,
    signature: Signature,
    macro: MacroResolver,
) -> Circuit:
    name = _identifier(value[key], f"{key} name")
    if key == "gen":
        if name not in signature:
            raise ValidationError(f"unknown generator {name!r}")
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
    _reject_unknown(value, {key}, "circuit")
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
        allowed = {key, "on"} if key in {"gen", "macro"} else {key}
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
    source: Any = None

    @classmethod
    def from_json(cls, value: Any) -> Theory:
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
        if raw_source is not None and not isinstance(raw_source, (str, Mapping)):
            raise ValidationError("theory source must be a string or object")
        source = dict(raw_source) if isinstance(raw_source, Mapping) else raw_source
        return cls(signature, tuple(equations), macros, name, dimension, source)

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


def _parse_macro_definitions(value: Any, signature: Signature) -> dict[str, Circuit]:
    if not isinstance(value, Mapping):
        raise ValidationError("macros must be an object")
    specs = {
        _identifier(name, "macro id"): body
        for name, body in value.items()
    }

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


@dataclass(frozen=True, slots=True)
class CircuitAnalysis:
    """Structural facts collected in one pass over a macro-free circuit."""

    occurrences: tuple[tuple[str, int], ...]
    maximum_wires: int
    changes_arity: bool
    permutation_parity: int

    def occurrence_counter(self) -> Counter[str]:
        return Counter(dict(self.occurrences))


def analyze_circuit(
    circuit: Circuit,
    checkpoint: Callable[[], None] | None = None,
) -> CircuitAnalysis:
    counts: Counter[str] = Counter()
    maximum_wires = 0
    changes_arity = False
    parity = 0
    for node in iter_circuit_nodes(circuit, checkpoint):
        maximum_wires = max(maximum_wires, node.inputs, node.outputs)
        changes_arity |= node.inputs != node.outputs
        if node.kind == "gen":
            counts[node.name or ""] += 1
        elif node.kind == "perm":
            parity ^= permutation_parity(node.order)
    return CircuitAnalysis(
        tuple(sorted(counts.items())),
        maximum_wires,
        changes_arity,
        parity,
    )


def contains_arity_change(
    circuit: Circuit,
    checkpoint: Callable[[], None] | None = None,
) -> bool:
    """Whether a term contains any non-endomorphic subterm."""

    return analyze_circuit(circuit, checkpoint).changes_arity


def primitive_occurrences(circuit: Circuit) -> Counter[str]:
    """Count primitive generators in a macro-free circuit."""

    return analyze_circuit(circuit).occurrence_counter()


def inversion_count(order: Sequence[int]) -> int:
    """Number of inversions in a permutation."""

    parsed = Circuit.perm(order).order
    return sum(left > right for i, left in enumerate(parsed) for right in parsed[i + 1 :])


def permutation_parity(order: Sequence[int]) -> int:
    """Return 0 for an even permutation and 1 for an odd one."""

    return inversion_count(order) & 1


def structural_permutation_parity(circuit: Circuit) -> int:
    """XOR the inversion parity of all explicit structural permutations."""

    return analyze_circuit(circuit).permutation_parity


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
