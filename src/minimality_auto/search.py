"""Verify candidate interpretations and search the selected equation targets."""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import Any

from .context import SearchContext, TargetContext, relevant_equations, retained_equations
from .search_types import (
    CandidateModel,
    Deadline,
    DirectEvidence,
    NotApplicable,
    SearchConfigurationError,
    SearchReport,
    SearchTimeout,
    Separation,
)
from .strategies import (
    DEFAULT_STRATEGIES,
    DEFAULT_TIMEOUT,
    LIMIT_BY_NAME,
    LIMIT_SPECS,
    STRATEGY_SPECS,
    LimitSpec,
    StrategySpec,
    _ResolvedStrategy,
    _search_schedule,
    _strategy_specs,
)
from .syntax import Equation, Theory

__all__ = [
    "DEFAULT_STRATEGIES",
    "DEFAULT_TIMEOUT",
    "LIMIT_SPECS",
    "STRATEGY_SPECS",
    "CandidateModel",
    "Deadline",
    "DirectEvidence",
    "LimitSpec",
    "NotApplicable",
    "SearchConfigurationError",
    "SearchContext",
    "SearchReport",
    "SearchTimeout",
    "Separation",
    "StrategySpec",
    "TargetContext",
    "relevant_equations",
    "search_theory",
    "verify",
]


def verify(
    model: CandidateModel | DirectEvidence,
    theory: Theory,
    target: Equation,
    deadline: Deadline,
    context: SearchContext | None = None,
    *,
    strategy: str,
) -> Separation | None:
    equations = retained_equations(theory, target, context)
    if isinstance(model, DirectEvidence):
        # These strategies supply target values from their own separation argument.
        left, right = model.lhs_value, model.rhs_value
        equal = left == right
        if hasattr(equal, "all"):
            equal = bool(equal.all())
        if equal:
            raise ValueError(f"direct evidence from {strategy} has equal target values")
    else:

        def compare(equation: Equation) -> tuple[Any, Any, bool]:
            deadline.check()
            left = model.evaluate(equation.lhs)
            deadline.check()
            right = model.evaluate(equation.rhs)
            deadline.check()
            equal = model.equal(left, right)
            deadline.check()
            return left, right, equal

        for equation in equations:
            _, _, equal = compare(equation)
            if not equal:
                return None
        left, right, equal = compare(target)
        if equal:
            return None
    return Separation(
        equation=target.id,
        strategy=strategy,
        description=model.description,
        parameters=model.parameters,
        checked_equations=tuple(equation.id for equation in equations),
        lhs_value=left,
        rhs_value=right,
        certified=model.certified,
    )


def _integer_option(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool):
        raise SearchConfigurationError(f"{name} must be an integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise SearchConfigurationError(f"{name} must be an integer") from exc
    if parsed < minimum:
        qualifier = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise SearchConfigurationError(f"{name} must be {qualifier}")
    return parsed


def _candidate_witnesses(
    theory: Theory,
    target: Equation,
    strategy_name: str,
    spec: _ResolvedStrategy,
    bound: int,
    deadline: Deadline,
    context: SearchContext,
    seen: set[tuple[Hashable, bool]],
) -> Iterator[Separation]:
    try:
        models = spec.candidates(
            theory,
            target,
            bound=bound + spec.bound_offset,
            deadline=deadline,
            context=context,
        )
        for model in models:
            deadline.check()
            key = getattr(model, "key", None)
            seen_key = (key, model.certified) if key is not None else None
            if seen_key is not None and seen_key in seen:
                continue
            try:
                witness = verify(model, theory, target, deadline, context, strategy=strategy_name)
            except NotApplicable:
                continue
            if seen_key is not None:
                seen.add(seen_key)
            if witness is not None:
                yield witness
                if witness.certified:
                    return
    except NotApplicable:
        return


def search_theory(
    theory: Theory,
    *,
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    equation_ids: set[str] | None = None,
    max_arity: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    on_witness: Callable[[Separation], None] | None = None,
    **limits: int,
) -> SearchReport:
    strategies = tuple(strategies)
    unknown = sorted(set(strategies) - set(DEFAULT_STRATEGIES))
    if unknown:
        raise SearchConfigurationError(f"unknown separation strategy: {', '.join(unknown)}")
    try:
        timeout = float(timeout)
    except (TypeError, ValueError) as exc:
        raise SearchConfigurationError("timeout must be a finite non-negative number") from exc
    if not math.isfinite(timeout) or timeout < 0:
        raise SearchConfigurationError("timeout must be a finite non-negative number")
    if max_arity is not None:
        max_arity = _integer_option("max_arity", max_arity, 0)
    unknown_limits = limits.keys() - LIMIT_BY_NAME.keys()
    if unknown_limits:
        raise SearchConfigurationError(
            f"unknown search limit(s): {', '.join(sorted(unknown_limits))}"
        )
    candidate_options = {
        spec.name: _integer_option(
            spec.name,
            limits.get(spec.name, spec.default),
            spec.minimum,
        )
        for spec in LIMIT_SPECS
    }

    start = time.monotonic()
    deadline = Deadline.after(timeout)
    targets = tuple(
        equation
        for equation in theory.equations
        if (equation_ids is None or equation.id in equation_ids)
        and (max_arity is None or equation.arity <= max_arity)
    )
    target_ids = tuple(equation.id for equation in targets)
    missing = (equation_ids or set()) - set(target_ids)
    if missing:
        raise SearchConfigurationError(
            f"unknown or filtered equation(s): {', '.join(sorted(missing))}"
        )

    witnesses: dict[str, Separation] = {}
    pending = set(target_ids)
    seen: dict[tuple[str, str], set[tuple[Hashable, bool]]] = {}
    timed_out = False

    try:
        if targets and strategies:
            context = SearchContext.build(theory, targets, deadline)
            specs = _strategy_specs(strategies, context, targets, candidate_options)
            for bound, strategy_name, spec in _search_schedule(specs):
                deadline.check()
                for target in targets:
                    if target.id not in pending:
                        continue
                    bucket = seen.setdefault((target.id, strategy_name), set())
                    for witness in _candidate_witnesses(
                        theory,
                        target,
                        strategy_name,
                        spec,
                        bound,
                        deadline,
                        context,
                        bucket,
                    ):
                        previous = witnesses.get(target.id)
                        if previous is None or (not previous.certified and witness.certified):
                            witnesses[target.id] = witness
                            if on_witness is not None:
                                on_witness(witness)
                            if witness.certified:
                                pending.discard(target.id)
                if not pending:
                    break
    except SearchTimeout:
        timed_out = True

    unresolved = tuple(target_id for target_id in target_ids if target_id in pending)
    return SearchReport(
        theory=theory.name or "theory",
        witnesses=witnesses,
        unresolved=unresolved,
        elapsed_seconds=time.monotonic() - start,
        timed_out=timed_out,
    )
