# Idea and original proof by Alexandre Clement.
#
# TODO: This argument is written to show the necessity of a three-wire rule
# when all generators have at most two wires.  It may generalize to show the
# necessity of a four-wire rule in theories with generators such as Toffoli.

"""Conservative numerical three-wire block-normal-form evidence.

PROP coherence supplies the routed word; the Brylinski--Brylinski criterion
classifies each two-wire block as local, SWAP-local, or entangling.  We compare
every minimum adjacent routing and retain only alternating forms whose
non-routing blocks each contain one entangling box.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np

from ..core import Circuit, ValidationError, expand_macros, primitive_occurrences
from ..matrix import permutation_matrix
from ..search import (
    Deadline,
    DirectCertificate,
    NotApplicable,
)
from .common import (
    numerical_matrix,
    projectively_equal,
    require_endomorphic_theory,
    retained_equations,
)

UNITARY_TOLERANCE = 1e-8
LOCAL_TOLERANCE = 1e-10
ENTANGLING_TOLERANCE = 1e-8
DEFAULT_MAX_MATRIX_DIMENSION = 512
MAX_ROUTING_PATHS = 256

Layout = tuple[int, int, int]


@dataclass(frozen=True)
class _Event:
    name: str
    wires: tuple[int, int]


@dataclass(frozen=True)
class _Step:
    pair: int
    name: str | None = None
    wires: tuple[int, int] = ()
    reversed_ports: bool = False


def _matrix(theory: Any, term: Circuit) -> np.ndarray:
    return numerical_matrix(theory, term, "the block argument")


def _active_generator_matrices(
    theory: Any, terms: tuple[Circuit, ...], deadline: Deadline
) -> dict[str, np.ndarray]:
    active: set[str] = set()
    for term in terms:
        active.update(primitive_occurrences(term, theory.macros))

    dimension = int(theory.wire_dimension)
    matrices: dict[str, np.ndarray] = {}
    for name in sorted(active):
        deadline.check()
        if name not in theory.signature:
            raise NotApplicable(f"active primitive {name!r} is not declared")
        generator = theory.signature[name]
        if (
            generator.inputs != generator.outputs
            or generator.inputs > 2
            or generator.matrix is None
        ):
            raise NotApplicable(
                "active primitives must be matrix-defined 0-, 1-, or 2-wire "
                "endomorphisms"
            )
        value = np.asarray(generator.matrix, dtype=np.complex128)
        size = dimension**generator.inputs
        identity = np.eye(size, dtype=np.complex128)
        if value.shape != (size, size) or not np.allclose(
            value.conj().T @ value,
            identity,
            atol=UNITARY_TOLERANCE,
            rtol=UNITARY_TOLERANCE,
        ):
            raise NotApplicable(f"active primitive {name!r} is not unitary")
        matrices[name] = value
    return matrices


def _trace_node(
    term: Circuit,
    layout: tuple[int, ...],
) -> tuple[list[_Event], tuple[int, ...]]:
    """Move structural permutations to the boundary and retain logical boxes."""

    if len(layout) != term.inputs or term.inputs != term.outputs:
        raise NotApplicable("the block argument needs endomorphisms")
    if term.kind == "id":
        return [], layout
    if term.kind == "perm":
        return [], tuple(layout[index] for index in term.order)
    if term.kind == "gen":
        if term.inputs == 0:
            return [], layout  # Work projectively, as in the proof.
        if term.inputs == 1:
            return [], layout  # Quotient by movable one-wire unitaries.
        if term.inputs != 2:
            raise NotApplicable("target boxes must act on at most two wires")
        return [_Event(term.name or "", (layout[0], layout[1]))], layout
    if term.kind == "compose":
        events: list[_Event] = []
        current = layout
        for part in term.parts:
            found, current = _trace_node(part, current)
            events.extend(found)
        return events, current
    if term.kind == "tensor":
        events = []
        output: list[int] = []
        offset = 0
        for part in term.parts:
            part_layout = layout[offset : offset + part.inputs]
            found, part_output = _trace_node(part, part_layout)
            events.extend(found)
            output.extend(part_output)
            offset += part.inputs
        return events, tuple(output)
    raise NotApplicable(f"unsupported target node {term.kind!r}")


def _logical_trace(
    theory: Any, term: Circuit
) -> tuple[tuple[_Event, ...], Layout]:
    if term.type != (3, 3):
        raise NotApplicable("the block argument is a three-wire invariant")
    try:
        expanded = expand_macros(term, theory.macros)
    except ValidationError as exc:
        raise NotApplicable(str(exc)) from exc
    events, output = _trace_node(expanded, (0, 1, 2))
    return tuple(events), (output[0], output[1], output[2])


def _append_cost(
    blocks: int,
    last_pair: int | None,
    pair: int,
) -> int:
    if last_pair is None:
        blocks += 1 + (pair == 0)  # Formal lower-pair identity prefix.
    elif last_pair != pair:
        blocks += 1
    return blocks


def _routing_paths(
    events: tuple[_Event, ...],
    output_layout: Layout,
    deadline: Deadline,
) -> tuple[tuple[_Step, ...], ...]:
    """Find every minimum-block adjacent routing, up to a safe cap."""

    State = tuple[int, Layout, int | None]
    start: State = (0, (0, 1, 2), None)
    best: dict[State, int] = {start: 0}
    paths: dict[State, set[tuple[_Step, ...]]] = {start: {()}}
    serial = count()
    queue: list[tuple[int, int, State, tuple[_Step, ...]]] = [
        (0, next(serial), start, ())
    ]

    def push(
        state: State, blocks: int, path: tuple[_Step, ...]
    ) -> None:
        previous = best.get(state)
        if previous is None or blocks < previous:
            best[state] = blocks
            paths[state] = {path}
        elif blocks == previous:
            bucket = paths[state]
            if path in bucket:
                return
            if len(bucket) >= MAX_ROUTING_PATHS:
                raise NotApplicable("too many shortest routing decompositions")
            bucket.add(path)
        else:
            return
        heapq.heappush(queue, (blocks, next(serial), state, path))

    while queue:
        deadline.check()
        blocks, _, state, path = heapq.heappop(queue)
        if best.get(state) != blocks or path not in paths[state]:
            continue
        index, layout, last_pair = state

        for pair in (0, 1):
            if (
                path
                and path[-1].name is None
                and path[-1].pair == pair
            ):
                continue  # Delete the structural relation SWAP ; SWAP = id.
            changed = list(layout)
            changed[pair], changed[pair + 1] = (
                changed[pair + 1],
                changed[pair],
            )
            push(
                (index, tuple(changed), pair),
                _append_cost(blocks, last_pair, pair),
                (*path, _Step(pair)),
            )

        if index == len(events):
            continue
        event = events[index]
        left = layout.index(event.wires[0])
        right = layout.index(event.wires[1])
        if abs(left - right) != 1:
            continue
        pair = min(left, right)
        physical_order = (layout[pair], layout[pair + 1])
        push(
            (index + 1, layout, pair),
            _append_cost(blocks, last_pair, pair),
            (
                *path,
                _Step(
                    pair,
                    event.name,
                    event.wires,
                    physical_order != event.wires,
                ),
            ),
        )

    goals: list[tuple[int, tuple[_Step, ...]]] = []
    for state, state_paths in paths.items():
        index, layout, last_pair = state
        if index != len(events) or layout != output_layout:
            continue
        cost = best[state]
        if last_pair == 0:
            cost += 1  # Formal lower identity suffix.
        goals.extend((cost, path) for path in state_paths)
    if not goals:
        raise NotApplicable("no adjacent routing realizes the target")
    optimum = min(cost for cost, _ in goals)
    result = sorted(
        {path for cost, path in goals if cost == optimum},
        key=repr,
    )
    if len(result) > MAX_ROUTING_PATHS:
        raise NotApplicable("too many shortest routing decompositions")
    return tuple(result)


def _path_summary(
    path: tuple[_Step, ...],
) -> tuple[dict[str, Any] | None, bool]:
    """Return ``(summary, canonical)``.

    A canonical route may still be unsupported when a block contains several
    boxes; that is represented by ``(None, True)``.  Routes with an internal
    generatorless block are not decompositions of the maximal-block form and
    are simply discarded.
    """

    groups: list[list[_Step]] = []
    for step in path:
        if not groups or groups[-1][0].pair != step.pair:
            groups.append([])
        groups[-1].append(step)

    blocks: list[dict[str, Any]] = []
    if groups and groups[0][0].pair == 0:
        blocks.append(
            {"support": "12", "kind": "routing_identity", "factors": []}
        )
    for group in groups:
        gates = [step for step in group if step.name is not None]
        if len(gates) > 1:
            return None, True
        if gates:
            kind = "entangling"
        elif len(group) % 2:
            kind = "routing_swap"
        else:
            return None, False
        factors = []
        for step in group:
            if step.name is None:
                factors.append({"structural": "swap"})
            else:
                factors.append(
                    {
                        "generator": step.name,
                        "wires": list(step.wires),
                        "reversed_ports": step.reversed_ports,
                    }
                )
        blocks.append(
            {
                "support": "01" if group[0].pair == 0 else "12",
                "kind": kind,
                "factors": factors,
            }
        )
    if groups and groups[-1][0].pair == 0:
        blocks.append(
            {"support": "12", "kind": "routing_identity", "factors": []}
        )

    allowed_routing = {0, len(blocks) - 2, len(blocks) - 1}
    for index, block in enumerate(blocks):
        if (
            block["kind"] == "routing_swap"
            and index not in allowed_routing
        ):
            return None, False
    return (
        {
            "block_count": len(blocks),
            "signature": [block["kind"] for block in blocks],
            "blocks": blocks,
            "blocks_are_representative": True,
            "routing_swaps": sum(step.name is None for step in path),
        },
        True,
    )


def _normal_form(
    trace: tuple[tuple[_Event, ...], Layout],
    entangling: set[str],
    deadline: Deadline,
) -> dict[str, Any] | None:
    events, output_layout = trace
    if any(event.name not in entangling for event in events):
        return None
    summaries: list[dict[str, Any]] = []
    for path in _routing_paths(events, output_layout, deadline):
        summary, canonical = _path_summary(path)
        if not canonical:
            continue
        if summary is None:
            return None
        summaries.append(summary)
    if not summaries:
        return None
    signatures = {
        (summary["block_count"], tuple(summary["signature"]))
        for summary in summaries
    }
    if len(signatures) != 1:
        return None
    return min(summaries, key=repr)


def _rank_status(
    matrix: np.ndarray, dimension: int
) -> tuple[str, float, dict[str, Any] | None]:
    realigned = (
        matrix.reshape(dimension, dimension, dimension, dimension)
        .transpose(0, 2, 1, 3)
        .reshape(dimension**2, dimension**2)
    )
    pivot = np.unravel_index(int(np.argmax(np.abs(realigned))), realigned.shape)
    pivot_value = realigned[pivot]
    if not np.isfinite(pivot_value) or pivot_value == 0:
        raise NotApplicable("could not test the operator-Schmidt rank")
    minors = (
        pivot_value * realigned
        - np.outer(realigned[:, pivot[1]], realigned[pivot[0], :])
    )
    witness_index = np.unravel_index(int(np.argmax(np.abs(minors))), minors.shape)
    witness_value = complex(minors[witness_index])
    ratio = float(abs(witness_value) / abs(pivot_value) ** 2)
    witness = {
        "rows": [pivot[0], witness_index[0]],
        "columns": [pivot[1], witness_index[1]],
        "determinant": witness_value,
    }
    if ratio <= LOCAL_TOLERANCE:
        return "rank_one", ratio, None
    if ratio >= ENTANGLING_TOLERANCE:
        return "higher_rank", ratio, witness
    return "ambiguous", ratio, witness


def _block_kind(
    matrix: np.ndarray, dimension: int
) -> tuple[str, dict[str, Any]]:
    local, local_minor, local_witness = _rank_status(matrix, dimension)
    swapped, swapped_minor, swapped_witness = _rank_status(
        matrix @ permutation_matrix((1, 0), dimension),
        dimension,
    )
    data = {
        "largest_normalized_minor": local_minor,
        "after_swap_largest_normalized_minor": swapped_minor,
        "nonzero_minor": local_witness,
        "after_swap_nonzero_minor": swapped_witness,
    }
    if local == "rank_one":
        return "local", data
    if swapped == "rank_one":
        return "swap_local", data
    if local == swapped == "higher_rank":
        return "entangling", data
    raise NotApplicable("operator-Schmidt classification is numerically ambiguous")


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_block_matrix_dimension: int = DEFAULT_MAX_MATRIX_DIMENSION,
    context: Any = None,
) -> Iterator[DirectCertificate]:
    if bound != 3:
        return
    require_endomorphic_theory(theory, deadline, context)
    if target.lhs.type != (3, 3) or target.rhs.type != (3, 3):
        return

    dimension = int(theory.wire_dimension)
    if dimension < 2:
        raise NotApplicable("the Brylinski classification requires dimension >= 2")
    if dimension**3 > max_block_matrix_dimension:
        raise NotApplicable(
            f"ambient matrix dimension exceeds {max_block_matrix_dimension}"
        )

    retained = retained_equations(theory, target, context)
    if any(equation.arity > 2 for equation in retained):
        raise NotApplicable("a retained relevant equation acts on three wires")

    terms = (
        target.lhs,
        target.rhs,
        *(side for equation in retained for side in (equation.lhs, equation.rhs)),
    )
    matrices = _active_generator_matrices(theory, terms, deadline)

    left_trace = _logical_trace(theory, target.lhs)
    right_trace = _logical_trace(theory, target.rhs)
    target_events = (*left_trace[0], *right_trace[0])
    certificates: dict[str, dict[str, Any]] = {}
    for name in sorted({event.name for event in target_events}):
        kind, data = _block_kind(matrices[name], dimension)
        if kind != "entangling":
            raise NotApplicable(f"target box {name!r} is degenerate")
        certificates[name] = data

    for equation in (*retained, target):
        deadline.check()
        if not projectively_equal(
            _matrix(theory, equation.lhs),
            _matrix(theory, equation.rhs),
            UNITARY_TOLERANCE,
        ):
            raise NotApplicable(
                f"equation {equation.id!r} is not matrix-sound "
                "up to global phase"
            )

    entangling = set(certificates)
    left = _normal_form(left_trace, entangling, deadline)
    right = _normal_form(right_trace, entangling, deadline)
    if left is None or right is None:
        return
    left_signature = (left["block_count"], tuple(left["signature"]))
    right_signature = (right["block_count"], tuple(right["signature"]))
    if left_signature == right_signature:
        return

    if left["block_count"] == right["block_count"]:
        difference = (
            f"{left['block_count']} blocks with different boundary types"
        )
    else:
        difference = (
            f"{left['block_count']} versus {right['block_count']} blocks"
        )
    yield DirectCertificate(
        description=(
            f"numerical three-wire irreducible block normal forms differ "
            f"({difference})"
        ),
        parameters={
            "wire_dimension": dimension,
            "retained_arity_bound": 2,
            "normal_form": "alternating adjacent blocks with lower-pair padding",
            "local_equivalence": (
                "discard scalar phases and absorb one-wire unitaries"
            ),
            "primitive_test": (
                "Brylinski local-or-local-after-SWAP operator-Schmidt test"
            ),
            "generator_rank_certificates": certificates,
            "unitary_tolerance": UNITARY_TOLERANCE,
            "local_tolerance": LOCAL_TOLERANCE,
            "entangling_tolerance": ENTANGLING_TOLERANCE,
        },
        lhs_value=left,
        rhs_value=right,
        certified=False,
    )
