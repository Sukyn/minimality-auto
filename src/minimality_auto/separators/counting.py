from __future__ import annotations

from itertools import product
from typing import Any, Iterator

from ..core import primitive_occurrences, structural_permutation_parity
from ..search import CandidateModel, Deadline
from .presence import _primitive_names


def _wire_parity_is_preserved(theory: Any, *extra_terms: Any) -> bool:
    """Whether the sign character extends over every supplied generator.

    A nonzero swap value has order two.  Naturality around ``f: m -> n``
    requires ``m*s = n*s``, so preserving wire parity is necessary and
    sufficient; requiring every generator to be an endomorphism would be
    unnecessarily restrictive.
    """

    def preserves(node: Any) -> bool:
        if getattr(node, "kind", None) == "gen":
            if int(node.inputs) % 2 != int(node.outputs) % 2:
                return False
        return all(preserves(part) for part in getattr(node, "parts", ()))

    if any(
        int(generator.inputs) % 2 != int(generator.outputs) % 2
        for generator in theory.signature.values()
    ):
        return False
    for definition in theory.macros.values():
        body = getattr(definition, "body", definition)
        if not preserves(body):
            return False
    theory_terms_are_safe = all(
        preserves(side)
        for equation in theory.equations
        for side in (equation.lhs, equation.rhs)
    )
    return theory_terms_are_safe and all(preserves(term) for term in extra_terms)


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_modulus: int = 8,
    **_: Any,
) -> Iterator[CandidateModel]:
    """Additive ``Z/m`` PROP models, optionally using the swap sign."""
    modulus = bound
    if modulus < 2 or modulus > max_modulus:
        return
    names = _primitive_names(theory, target, deadline)
    nonzero_swap_is_sound = modulus % 2 == 0 and _wire_parity_is_preserved(
        theory, target.lhs, target.rhs
    )
    swap_values = (0, modulus // 2) if nonzero_swap_is_sound else (0,)
    assignments = product(range(modulus), repeat=len(names))
    for values in assignments:
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
                counts = primitive_occurrences(term, theory.macros)
                total = sum(counts.get(name, 0) * value for name, value in weights.items())
                total += structural_permutation_parity(term, theory.macros) * swap
                return total % modulus

            nonzero = {name: value for name, value in weights.items() if value}
            description = f"count mod {modulus}: {nonzero or '{}'}"
            if swap:
                description += f", swap={swap}"
            yield CandidateModel(
                kind="counting",
                description=description,
                parameters={"modulus": modulus, "weights": nonzero, "swap": swap},
                evaluator=evaluate,
                key=(modulus, values, swap),
            )
