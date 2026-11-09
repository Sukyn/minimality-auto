"""Traversal and structural invariants of typed circuits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain

from .syntax import Circuit, Theory


def iter_theory_terms(theory: Theory, *extra: Circuit) -> Iterator[Circuit]:
    """Iterate over macro bodies, equation sides, then any extra terms."""

    return chain(
        theory.macros.values(),
        (side for equation in theory.equations for side in (equation.lhs, equation.rhs)),
        extra,
    )


def iter_circuit_nodes(
    circuit: Circuit,
    checkpoint: Callable[[], None] | None = None,
) -> Iterator[Circuit]:
    """Traverse a circuit tree with an explicit stack."""

    pending = [circuit]
    visited = 0
    while pending:
        if checkpoint is not None and visited % 1024 == 0:
            checkpoint()
        node = pending.pop()
        yield node
        pending.extend(node.parts)
        visited += 1


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
    """Return whether any subterm changes the number of wires."""

    return any(node.inputs != node.outputs for node in iter_circuit_nodes(circuit, checkpoint))


def permutation_parity(order: Sequence[int]) -> int:
    """Return 0 for an even permutation and 1 for an odd one."""

    parsed = Circuit.perm(order).order
    return (
        sum(left > right for index, left in enumerate(parsed) for right in parsed[index + 1 :]) & 1
    )
