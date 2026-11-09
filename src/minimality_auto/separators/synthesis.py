"""Shared Z3 solving and time budgets for finite interpretation searches."""

from __future__ import annotations

import z3

from ..search_types import Deadline


def solve(solver: z3.Solver, deadline: Deadline) -> z3.ModelRef | None:
    """Unknown/attempt timeout leaves the equation unresolved; it proves nothing."""
    deadline.check()
    remaining = max(1, int(1000 * deadline.remaining))
    solver.set(timeout=min(remaining, 2**32 - 1), random_seed=0)
    return solver.model() if solver.check() == z3.sat else None
