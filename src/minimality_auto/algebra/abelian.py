"""Integer Smith reduction and exact modular separating weights."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from math import gcd

from ..search_types import Deadline


def smith_basis(
    relations: Sequence[Sequence[int]], columns: int, deadline: Deadline
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Return D and V with U A V = D for an integral unimodular U, V.

    D lists one diagonal entry per generator, including free (zero) entries.
    A count row c represents the quotient element c V, reduced modulo D.
    Only V is needed for evaluating circuits, so U is never allocated.
    """
    rows = [list(row) for row in relations if any(row)]
    if any(len(row) != columns for row in rows):
        raise ValueError("relation has the wrong number of generators")
    basis = [[int(i == j) for j in range(columns)] for i in range(columns)]

    def swap_columns(left: int, right: int) -> None:
        for row in chain(rows, basis):
            row[left], row[right] = row[right], row[left]

    for pivot in range(min(len(rows), columns)):
        deadline.check()
        position = next(
            ((i, j) for i in range(pivot, len(rows)) for j in range(pivot, columns) if rows[i][j]),
            None,
        )
        if position is None:
            break
        i, j = position
        rows[pivot], rows[i] = rows[i], rows[pivot]
        swap_columns(pivot, j)
        while True:
            deadline.check()
            i = next((i for i in range(pivot + 1, len(rows)) if rows[i][pivot]), None)
            if i is not None:
                quotient = rows[i][pivot] // rows[pivot][pivot]
                rows[i] = [a - quotient * b for a, b in zip(rows[i], rows[pivot], strict=True)]
                if rows[i][pivot]:
                    rows[pivot], rows[i] = rows[i], rows[pivot]
                continue
            j = next((j for j in range(pivot + 1, columns) if rows[pivot][j]), None)
            if j is not None:
                quotient = rows[pivot][j] // rows[pivot][pivot]
                for row in chain(rows, basis):
                    row[j] -= quotient * row[pivot]
                if rows[pivot][j]:
                    swap_columns(pivot, j)
                continue
            # Make the pivot divide the remaining block, giving invariant factors.
            i = next(
                (
                    i
                    for i in range(pivot + 1, len(rows))
                    if any(rows[i][j] % rows[pivot][pivot] for j in range(pivot + 1, columns))
                ),
                None,
            )
            if i is None:
                break
            rows[pivot] = [a + b for a, b in zip(rows[pivot], rows[i], strict=True)]
        if rows[pivot][pivot] < 0:
            rows[pivot] = [-value for value in rows[pivot]]

    diagonal = tuple(rows[i][i] if i < len(rows) else 0 for i in range(columns))
    return diagonal, tuple(tuple(row) for row in basis)


def separating_weights(
    relations: Sequence[Sequence[int]],
    target: Sequence[int],
    modulus: int,
    deadline: Deadline,
) -> tuple[int, ...] | None:
    """Find A w = 0 and target w != 0 modulo m, including composite m.

    In the Smith basis, each coordinate has kernel step m / gcd(diagonal, m).
    """
    if not any(value % modulus for value in target):
        return None
    # Prefer a single nonzero weight, trying the last coordinate first.
    for column in reversed(range(len(target))):
        deadline.check()
        step = modulus // gcd(modulus, *(row[column] for row in relations))
        if target[column] * step % modulus:
            return tuple(step if i == column else 0 for i in range(len(target)))
    diagonal, basis = smith_basis(
        [[value % modulus for value in row] for row in relations],
        len(target),
        deadline,
    )
    for column, value in enumerate(diagonal):
        coefficient = sum(a * row[column] for a, row in zip(target, basis, strict=True))
        step = modulus // gcd(value, modulus)
        if coefficient * step % modulus:
            return tuple(row[column] * step % modulus for row in basis)
    return None
