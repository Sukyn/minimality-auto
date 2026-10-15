from __future__ import annotations

import importlib
import math
import operator
import time
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .core import (
    Circuit,
    CircuitAnalysis,
    Equation,
    Theory,
    analyze_circuit,
    contains_arity_change,
    iter_theory_terms,
)


class SearchTimeout(RuntimeError):
    pass


class SearchConfigurationError(ValueError):
    pass


class NotApplicable(RuntimeError):
    pass


@dataclass(frozen=True)
class Deadline:
    expires_at: float

    @classmethod
    def after(cls, seconds: float) -> Deadline:
        return cls(time.monotonic() + max(0.0, seconds))

    def check(self) -> None:
        if time.monotonic() >= self.expires_at:
            raise SearchTimeout


@dataclass
class CandidateModel:
    description: str
    parameters: dict[str, Any]
    evaluate: Callable[[Any], Any]
    equal: Callable[[Any, Any], bool] = operator.eq
    key: Hashable | None = None
    certified: bool = True


@dataclass(frozen=True)
class Separation:
    equation: str
    strategy: str
    description: str
    parameters: dict[str, Any]
    checked_equations: tuple[str, ...]
    lhs_value: Any
    rhs_value: Any
    certified: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "equation": self.equation,
            "strategy": self.strategy,
            "description": self.description,
            "parameters": _json_value(self.parameters),
            "checked_equations": list(self.checked_equations),
            "lhs_value": _json_value(self.lhs_value),
            "rhs_value": _json_value(self.rhs_value),
            "certified": self.certified,
        }


@dataclass(frozen=True)
class DirectEvidence:
    description: str
    parameters: dict[str, Any]
    lhs_value: Any
    rhs_value: Any
    certified: bool = False


@dataclass(frozen=True)
class SearchReport:
    theory: str
    witnesses: dict[str, Separation]
    unresolved: tuple[str, ...]
    elapsed_seconds: float
    timed_out: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "theory": self.theory,
            "witnesses": {name: value.as_dict() for name, value in self.witnesses.items()},
            "unresolved": list(self.unresolved),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "timed_out": self.timed_out,
        }


@dataclass(slots=True)
class TargetContext:
    equations: tuple[Equation, ...]
    primitive_names: tuple[str, ...]
    fixed_arity_presentation: Any = None


@dataclass(slots=True)
class SearchContext:
    targets: dict[Equation, TargetContext]
    active_primitive_count: int
    endomorphic: bool
    analyses: dict[Circuit, CircuitAnalysis]

    @classmethod
    def build(
        cls,
        theory: Theory,
        targets: Iterable[Equation],
        deadline: Deadline | None = None,
    ) -> SearchContext:
        targets = tuple(targets)
        if deadline is not None:
            deadline.check()
        checkpoint = deadline.check if deadline is not None else None
        terms = dict.fromkeys(iter_theory_terms(theory))
        analyses = {term: analyze_circuit(term, checkpoint) for term in terms}
        endomorphic = _theory_is_endomorphic(theory, deadline, analyses)
        target_contexts: dict[Equation, TargetContext] = {}
        active_names: set[str] = set()
        for target in targets:
            if deadline is not None:
                deadline.check()
            retained = tuple(
                _relevant_equations(
                    theory,
                    target,
                    endomorphic=endomorphic,
                    deadline=deadline,
                    analyses=analyses,
                )
            )
            names: set[str] = set()
            for equation in (*retained, target):
                for term in (equation.lhs, equation.rhs):
                    if deadline is not None:
                        deadline.check()
                    names.update(name for name, _ in analyses[term].occurrences)
            target_contexts[target] = TargetContext(retained, tuple(sorted(names)))
            active_names.update(names)
        return cls(
            target_contexts,
            len(active_names),
            endomorphic,
            analyses,
        )

    def for_target(self, target: Equation) -> TargetContext:
        return self.targets[target]

    def analysis_for(self, term: Circuit) -> CircuitAnalysis:
        analysis = self.analyses.get(term)
        if analysis is None:
            analysis = analyze_circuit(term)
            self.analyses[term] = analysis
        return analysis


@dataclass(frozen=True)
class LimitSpec:
    name: str
    default: int
    minimum: int
    help: str
    aliases: tuple[str, ...] = ()

    @property
    def flags(self) -> tuple[str, ...]:
        return (f"--{self.name.replace('_', '-')}", *self.aliases)


LIMIT_SPECS = (
    LimitSpec("max_modulus", 8, 2, "largest counting modulus"),
    LimitSpec("max_depth", 3, 0, "largest substitution word length"),
    LimitSpec("max_substitution_matrix_entries", 1_000_000, 1, "dense matrix entries"),
    LimitSpec(
        "max_permutation_degree",
        5,
        2,
        "largest permutation degree",
        ("--max_permutation_degree", "--max-degree"),
    ),
    LimitSpec("max_affine_lift_prime", 5, 2, "largest affine-lift characteristic"),
    LimitSpec("max_affine_lift_dimension", 128, 1, "largest affine-lift dimension"),
    LimitSpec("max_amalgam_prime", 11, 2, "largest amalgam characteristic"),
    LimitSpec("max_amalgam_order", 4096, 1, "largest finite factor order"),
    LimitSpec("max_amalgam_bridge_generators", 1, 1, "largest bridge set"),
    LimitSpec("max_amalgam_scalars", 3, 0, "largest template-scalar count"),
    LimitSpec("max_amalgam_matrix_dimension", 32, 1, "largest finite-field matrix"),
    LimitSpec("max_spin_matrix_dimension", 1024, 1, "largest dense Spin matrix"),
    LimitSpec("max_block_matrix_dimension", 512, 1, "largest dense block matrix"),
    LimitSpec("max_block_exact_states", 4096, 1, "largest exact block state space"),
)
LIMIT_BY_NAME = {spec.name: spec for spec in LIMIT_SPECS}
DEFAULT_TIMEOUT = 600.0


@dataclass(frozen=True)
class StrategySpec:
    name: str
    flags: tuple[str, ...]
    help: str
    bound: int | str
    option_names: tuple[str, ...] = ()
    bound_padding: int = 0
    candidate_offset: int = 0
    eager_bound: int = -1

    def maximum_bound(
        self,
        context: SearchContext,
        targets: Iterable[Equation],
        limits: Mapping[str, int],
    ) -> int:
        if isinstance(self.bound, int):
            base = self.bound
        elif self.bound == "active_primitive_count":
            base = max(1, context.active_primitive_count)
        elif self.bound == "target_arity":
            base = max(3, *(equation.arity for equation in targets))
        else:
            base = limits[self.bound]
        return base + self.bound_padding


@dataclass(frozen=True)
class _ResolvedStrategy:
    candidates: Callable[..., Iterator[CandidateModel | DirectEvidence]]
    max_bound: int
    bound_offset: int = 0
    options: Mapping[str, Any] = field(default_factory=dict)
    eager_bound: int = -1


# Eager bounds give each strategy a small search before larger bounds run.
STRATEGY_SPECS = (
    StrategySpec(
        "presence",
        ("--presence-checking", "--presence_checking", "--max-occurrence"),
        "run Boolean max/occurrence models",
        "active_primitive_count",
        eager_bound=4,
    ),
    StrategySpec(
        "counting",
        ("--counting",),
        "run modular counting models",
        "max_modulus",
        option_names=("max_modulus",),
        eager_bound=2,
    ),
    StrategySpec(
        "block_normal_form",
        ("--block-normal-form", "--block_normal_form", "--brylinski"),
        "run three-wire Brylinski block-normal-form evidence",
        3,
        option_names=("max_block_matrix_dimension", "max_block_exact_states"),
        eager_bound=3,
    ),
    StrategySpec(
        "finite_model",
        ("--finite-model", "--finite_model", "--permutation-model"),
        "search small permutation interpretations",
        "max_permutation_degree",
        option_names=("max_permutation_degree",),
        eager_bound=2,
    ),
    StrategySpec(
        "affine_lift",
        ("--affine-lift", "--affine_lift"),
        "search exact finite affine lifts of basis-permutation semantics",
        "max_affine_lift_prime",
        option_names=("max_affine_lift_prime", "max_affine_lift_dimension"),
        eager_bound=2,
    ),
    StrategySpec(
        "amalgam",
        ("--amalgam",),
        "search finite factors and free-product-with-amalgamation models",
        1,
        option_names=(
            "max_amalgam_prime",
            "max_amalgam_order",
            "max_amalgam_bridge_generators",
            "max_amalgam_scalars",
            "max_amalgam_matrix_dimension",
        ),
    ),
    StrategySpec(
        "spin",
        ("--spin-cover", "--spin_cover"),
        "run numerical Spin-cover commutator evidence",
        0,
        option_names=("max_spin_matrix_dimension",),
    ),
    StrategySpec(
        "determinant",
        ("--determinant",),
        "run numerical scaled determinant-phase models",
        "target_arity",
    ),
    StrategySpec(
        "substitution",
        ("--substitution",),
        "run numerical projective substitutions",
        "max_depth",
        option_names=("max_depth", "max_substitution_matrix_entries"),
        bound_padding=2,
        candidate_offset=-2,
    ),
)
_STRATEGY_BY_NAME = {spec.name: spec for spec in STRATEGY_SPECS}
DEFAULT_STRATEGIES = tuple(spec.name for spec in STRATEGY_SPECS)


def _theory_is_endomorphic(
    theory: Theory,
    deadline: Deadline | None = None,
    analyses: Mapping[Circuit, CircuitAnalysis] | None = None,
) -> bool:
    for generator in theory.signature.values():
        if deadline is not None:
            deadline.check()
        if generator.inputs != generator.outputs:
            return False
    if analyses is not None:
        return all(not analyses[term].changes_arity for term in iter_theory_terms(theory))
    checkpoint = deadline.check if deadline is not None else None
    return all(not contains_arity_change(term, checkpoint) for term in iter_theory_terms(theory))


def _relevant_equations(
    theory: Theory,
    target: Equation,
    *,
    endomorphic: bool,
    deadline: Deadline | None = None,
    analyses: Mapping[Circuit, CircuitAnalysis] | None = None,
) -> list[Equation]:
    others = [equation for equation in theory.equations if equation.id != target.id]
    if analyses is not None:
        target_changes_arity = (
            analyses[target.lhs].changes_arity or analyses[target.rhs].changes_arity
        )
    else:
        checkpoint = deadline.check if deadline is not None else None
        target_changes_arity = contains_arity_change(
            target.lhs, checkpoint
        ) or contains_arity_change(target.rhs, checkpoint)
    if not endomorphic or target_changes_arity:
        return others
    return [equation for equation in others if equation.arity <= target.arity]


def relevant_equations(theory: Theory, target: Equation) -> list[Equation]:
    """Select the retained equations required to validate a candidate.

    Endomorphic targets in endomorphic theories need equations up to the
    target arity; other cases need every retained equation.
    """
    return _relevant_equations(theory, target, endomorphic=_theory_is_endomorphic(theory))


def verify(
    model: CandidateModel,
    theory: Theory,
    target: Equation,
    deadline: Deadline,
    context: SearchContext | None = None,
    *,
    strategy: str,
) -> Separation | None:
    def compare(equation: Equation) -> tuple[Any, Any, bool]:
        deadline.check()
        left = model.evaluate(equation.lhs)
        deadline.check()
        right = model.evaluate(equation.rhs)
        deadline.check()
        equal = model.equal(left, right)
        deadline.check()
        return left, right, equal

    checked: list[str] = []
    equations = (
        context.for_target(target).equations
        if context is not None
        else relevant_equations(theory, target)
    )
    for equation in equations:
        _, _, equal = compare(equation)
        if not equal:
            return None
        checked.append(equation.id)

    left, right, equal = compare(target)
    if equal:
        return None
    return Separation(
        equation=target.id,
        strategy=strategy,
        description=model.description,
        parameters=model.parameters,
        checked_equations=tuple(checked),
        lhs_value=left,
        rhs_value=right,
        certified=model.certified,
    )


def _direct_separation(
    evidence: DirectEvidence,
    target: Equation,
    strategy: str,
    context: SearchContext,
) -> Separation:
    same = evidence.lhs_value == evidence.rhs_value
    if hasattr(same, "all"):
        same = bool(same.all())
    if same:
        raise ValueError(f"direct evidence from {strategy} has equal target values")
    return Separation(
        equation=target.id,
        strategy=strategy,
        description=evidence.description,
        parameters=evidence.parameters,
        checked_equations=tuple(equation.id for equation in context.for_target(target).equations),
        lhs_value=evidence.lhs_value,
        rhs_value=evidence.rhs_value,
        certified=evidence.certified,
    )


def _strategy_specs(
    names: Iterable[str],
    context: SearchContext,
    targets: Iterable[Equation],
    options: Mapping[str, int],
) -> dict[str, _ResolvedStrategy]:
    resolved: dict[str, _ResolvedStrategy] = {}
    for name in names:
        spec = _STRATEGY_BY_NAME[name]
        module = importlib.import_module(
            f".separators.{spec.name}",
            package=__package__,
        )
        resolved[name] = _ResolvedStrategy(
            candidates=module.candidates,
            max_bound=spec.maximum_bound(context, targets, options),
            bound_offset=spec.candidate_offset,
            options={key: options[key] for key in spec.option_names},
            eager_bound=spec.eager_bound,
        )
    return resolved


def _search_schedule(
    specs: Mapping[str, _ResolvedStrategy],
) -> Iterator[tuple[int, str, _ResolvedStrategy]]:
    """Run each strategy's eager bounds, then advance all strategies by bound."""

    scheduled: set[tuple[str, int]] = set()
    for name, spec in specs.items():
        for bound in range(min(spec.eager_bound, spec.max_bound) + 1):
            scheduled.add((name, bound))
            yield bound, name, spec

    rounds = max((spec.max_bound for spec in specs.values()), default=-1)
    for bound in range(rounds + 1):
        for name, spec in specs.items():
            if bound <= spec.max_bound and (name, bound) not in scheduled:
                yield bound, name, spec


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
            **spec.options,
        )
        for model in models:
            deadline.check()
            try:
                if isinstance(model, DirectEvidence):
                    witness = _direct_separation(model, target, strategy_name, context)
                else:
                    seen_key = (model.key, model.certified) if model.key is not None else None
                    if seen_key is not None and seen_key in seen:
                        continue
                    witness = verify(
                        model,
                        theory,
                        target,
                        deadline,
                        context,
                        strategy=strategy_name,
                    )
                    if seen_key is not None:
                        seen.add(seen_key)
            except NotApplicable:
                continue

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
    theory_name = theory.name or "theory"
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

    if not targets or not strategies:
        return SearchReport(
            theory=theory_name,
            witnesses={},
            unresolved=target_ids,
            elapsed_seconds=time.monotonic() - start,
            timed_out=False,
        )

    try:
        context = SearchContext.build(theory, targets, deadline)
    except SearchTimeout:
        return SearchReport(
            theory=theory_name,
            witnesses={},
            unresolved=target_ids,
            elapsed_seconds=time.monotonic() - start,
            timed_out=True,
        )
    specs = _strategy_specs(strategies, context, targets, candidate_options)
    witnesses: dict[str, Separation] = {}
    pending = set(target_ids)
    seen: dict[tuple[str, str], set[tuple[Hashable, bool]]] = {}
    timed_out = False

    try:
        for bound, strategy_name, spec in _search_schedule(specs):
            deadline.check()
            for target, target_id in zip(targets, target_ids, strict=True):
                if target_id not in pending:
                    continue
                bucket = seen.setdefault((target_id, strategy_name), set())
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
                    previous = witnesses.get(target_id)
                    if previous is None or (not previous.certified and witness.certified):
                        witnesses[target_id] = witness
                        if on_witness is not None:
                            on_witness(witness)
                        if witness.certified:
                            pending.discard(target_id)
            if not pending:
                break
    except SearchTimeout:
        timed_out = True

    unresolved = tuple(target_id for target_id in target_ids if target_id in pending)
    return SearchReport(
        theory=theory_name,
        witnesses=witnesses,
        unresolved=unresolved,
        elapsed_seconds=time.monotonic() - start,
        timed_out=timed_out,
    )


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, complex):
        return [_json_value(value.real), _json_value(value.imag)]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value
