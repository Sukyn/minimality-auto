from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

from ..search import CandidateModel, Deadline
from .common import circuit_analysis, primitive_names, wire_parity_is_preserved


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_modulus: int = 8,
    context: Any = None,
) -> Iterator[CandidateModel]:
    """Additive ``Z/m`` PROP models, optionally using the swap sign."""
    modulus = bound
    if modulus < 2 or modulus > max_modulus:
        return
    names = primitive_names(theory, target, deadline, context)
    nonzero_swap_is_sound = modulus % 2 == 0 and wire_parity_is_preserved(
        theory, target.lhs, target.rhs
    )
    swap_values = (0, modulus // 2) if nonzero_swap_is_sound else (0,)

    for values in product(range(modulus), repeat=len(names)):
        deadline.check()
        weights = dict(zip(names, values, strict=True))
        for swap in swap_values:
            if not any(values) and swap == 0:
                continue

            def evaluate(
                term: Any,
                weights: dict[str, int] = weights,
                swap: int = swap,
                modulus: int = modulus,
            ) -> int:
                analysis = circuit_analysis(term, context)
                counts = analysis.occurrence_counter()
                total = sum(counts[name] * value for name, value in weights.items())
                total += analysis.permutation_parity * swap
                return total % modulus

            nonzero = {name: value for name, value in weights.items() if value}
            description = f"count mod {modulus}: {nonzero or '{}'}"
            if swap:
                description += f", swap={swap}"
            yield CandidateModel(
                description=description,
                parameters={"modulus": modulus, "weights": nonzero, "swap": swap},
                evaluate=evaluate,
            )
