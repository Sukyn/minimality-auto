"""Compile an endomorphic PROP fragment to words in boxes and adjacent swaps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from ..core import Circuit, expand_macros, iter_circuit_nodes, iter_theory_terms
from ..search import Deadline, NotApplicable

Token = tuple[str, str | int]
Word = tuple[Token, ...]


def generator(name: str) -> Token:
    return "generator", name


def swap(index: int) -> Token:
    return "swap", index


def _inverse(order: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(order)
    for output, source in enumerate(order):
        result[source] = output
    return tuple(result)


def adjacent_word(
    order: tuple[int, ...],
    offset: int = 0,
    deadline: Deadline | None = None,
) -> Word:
    current = list(range(len(order)))
    result: list[Token] = []
    for position, wanted in enumerate(order):
        if deadline is not None and position % 256 == 0:
            deadline.check()
        found = current.index(wanted, position)
        while found > position:
            current[found - 1], current[found] = current[found], current[found - 1]
            result.append(swap(offset + found - 1))
            if deadline is not None and len(result) % 1024 == 0:
                deadline.check()
            found -= 1
    assert tuple(current) == order
    return tuple(result)


def place(
    token: Token,
    selected: tuple[int, ...],
    total: int,
    deadline: Deadline | None = None,
) -> Word:
    chosen = frozenset(selected)
    if (
        total < 0
        or len(chosen) != len(selected)
        or any(wire < 0 or wire >= total for wire in chosen)
    ):
        raise ValueError("selected wires must be distinct positions in the ambient arity")
    rest = tuple(wire for wire in range(total) if wire not in chosen)
    order = selected + rest
    return (
        adjacent_word(order, deadline=deadline)
        + (token,)
        + adjacent_word(_inverse(order), deadline=deadline)
    )


def compile_word(
    node: Circuit,
    signature: Mapping[str, int],
    total: int,
    *,
    offset: int = 0,
    deadline: Deadline | None = None,
) -> Word:
    if deadline is not None:
        deadline.check()
    if (
        total < 0
        or offset < 0
        or node.inputs != node.outputs
        or offset + node.inputs > total
    ):
        raise NotApplicable("fixed-arity models require endomorphisms")
    if node.kind == "id":
        return ()
    if node.kind == "gen":
        if node.name not in signature or signature[node.name] != node.inputs:
            raise NotApplicable(f"unsupported generator {node.name!r}")
        selected = tuple(range(offset, offset + node.inputs))
        return place(generator(node.name), selected, total, deadline)
    if node.kind == "perm":
        return adjacent_word(node.order, offset, deadline)
    if node.kind == "compose":
        return tuple(
            token
            for part in node.parts
            for token in compile_word(
                part,
                signature,
                total,
                offset=offset,
                deadline=deadline,
            )
        )
    if node.kind == "tensor":
        result: list[Token] = []
        position = offset
        for part in node.parts:
            result.extend(
                compile_word(
                    part,
                    signature,
                    total,
                    offset=position,
                    deadline=deadline,
                )
            )
            position += part.inputs
        return tuple(result)
    raise NotApplicable(f"unsupported circuit node {node.kind!r}")


def theory_word_compiler(
    macros: Mapping[str, Circuit],
    signature: Mapping[str, int],
    total: int,
    deadline: Deadline,
) -> Callable[[Circuit], Word]:
    """Bind the common theory data for compiling several terms."""

    def compile_term(node: Circuit) -> Word:
        return compile_word(
            expand_macros(node, macros),
            signature,
            total,
            deadline=deadline,
        )

    return compile_term


@dataclass(frozen=True)
class Constraint:
    left: Word
    right: Word
    equal: bool = True

    @cached_property
    def variables(self) -> frozenset[Token]:
        return frozenset((*self.left, *self.right))


def prop_constraints(
    arity: int,
    generators: tuple[tuple[str, int], ...],
    deadline: Deadline | None = None,
) -> tuple[Constraint, ...]:
    if deadline is not None:
        deadline.check()
    if arity < 0:
        raise ValueError("PROP arity must be non-negative")
    names = tuple(name for name, _ in generators)
    if len(set(names)) != len(names):
        raise ValueError("PROP generator names must be distinct")
    if any(width < 0 or width > arity for _, width in generators):
        raise ValueError("PROP generator arities must lie in the fixed-arity fragment")

    constraints: list[Constraint] = []
    for index in range(arity - 1):
        if deadline is not None and index % 256 == 0:
            deadline.check()
        token = swap(index)
        constraints.append(Constraint((token, token), ()))
    for index in range(arity - 2):
        if deadline is not None and index % 256 == 0:
            deadline.check()
        left, right = swap(index), swap(index + 1)
        constraints.append(Constraint((left, right, left), (right, left, right)))
    for left_index in range(arity - 1):
        if deadline is not None:
            deadline.check()
        for right_index in range(left_index + 2, arity - 1):
            left, right = swap(left_index), swap(right_index)
            constraints.append(Constraint((left, right), (right, left)))

    for name, width in generators:
        if deadline is not None:
            deadline.check()
        token = generator(name)
        for index in range(width, arity - 1):
            structural = swap(index)
            constraints.append(Constraint((token, structural), (structural, token)))

    checked_pairs = 0
    for index, (left_name, left_width) in enumerate(generators):
        if deadline is not None:
            deadline.check()
        for right_name, right_width in generators[index:]:
            checked_pairs += 1
            if deadline is not None and checked_pairs % 1024 == 0:
                deadline.check()
            if left_width + right_width > arity:
                continue
            left = (generator(left_name),)
            selected = tuple(range(left_width, left_width + right_width))
            right = place(generator(right_name), selected, arity, deadline)
            constraints.append(Constraint(left + right, right + left))

    return tuple(
        dict.fromkeys(
            constraint for constraint in constraints if constraint.left != constraint.right
        )
    )


def display_name(token: Token) -> str:
    if token[0] == "generator":
        return str(token[1])
    index = int(token[1])
    return f"swap[{index},{index + 1}]"


def _register_inline_generators(
    term: Circuit,
    signature: dict[str, int],
    deadline: Deadline,
) -> None:
    for node in iter_circuit_nodes(term, deadline.check):
        if node.kind == "gen":
            name = node.name or ""
            width = int(node.inputs)
            existing = signature.get(name)
            if existing is not None and existing != width:
                raise NotApplicable(
                    f"generator {name!r} is used at inconsistent arities"
                )
            signature[name] = width


def generator_arities(theory: Any, target: Any, deadline: Deadline) -> dict[str, int]:
    """Return declared and explicitly typed inline generator arities."""

    signature = {
        name: generator.inputs for name, generator in theory.signature.items()
    }
    for term in iter_theory_terms(theory, target.lhs, target.rhs):
        _register_inline_generators(term, signature, deadline)
    return signature
