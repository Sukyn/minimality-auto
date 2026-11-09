"""Typed generators, circuits, and equational presentations of a PROP.

Composition is stored in execution order. A permutation lists the input
position routed to each output position.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any


class PropError(ValueError):
    """Base class for malformed PROP data."""


class ValidationError(PropError):
    """Raised for JSON that violates PROP syntax or typing rules."""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _natural(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{where} must be a non-negative integer")
    return value


def _wire_size_matches(size: int, dimension: int, wires: int) -> bool:
    """Check ``size == dimension**wires``, bounding ``wires`` by ``size.bit_length()``."""

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


def _inverse_permutation(order: Sequence[int]) -> tuple[int, ...]:
    """Invert an already validated wire permutation."""
    result = [0] * len(order)
    for output, source in enumerate(order):
        result[source] = output
    return tuple(result)


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where} must be a non-empty string")
    return value


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
        # Use dictionary membership because __getitem__ raises ValidationError
        # for unknown generator names.
        return name in self._generators

    def get(self, name: str, default: Any = None) -> Generator | Any:
        return self._generators.get(name, default)

    @classmethod
    def from_json(cls, value: Any) -> Signature:
        from .parsing import parse_signature

        return parse_signature(value, cls)


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
    def generator(cls, generator: Generator) -> Circuit:
        return cls(
            "gen",
            generator.inputs,
            generator.outputs,
            name=generator.name,
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
        items = [
            item for part in parts for item in (part.parts if part.kind == "compose" else (part,))
        ]
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
        items: list[Circuit] = []
        identity_wires = 0
        for part in parts:
            for item in part.parts if part.kind == "tensor" else (part,):
                if item.kind == "id":
                    identity_wires += item.inputs
                    continue
                if identity_wires:
                    items.append(cls.identity(identity_wires))
                    identity_wires = 0
                items.append(item)
        if identity_wires:
            items.append(cls.identity(identity_wires))
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
        from .parsing import parse_circuit

        return parse_circuit(value, signature, macros)


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
    source: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, value: Any) -> Theory:
        from .parsing import parse_theory

        return parse_theory(value, cls)

    def equation(self, equation_id: str) -> Equation:
        for equation in self.equations:
            if equation.id == equation_id:
                return equation
        raise KeyError(equation_id)
