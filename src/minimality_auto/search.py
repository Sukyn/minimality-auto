from __future__ import annotations

import importlib
import math
import operator
import time
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .core import (
    Circuit,
    Equation,
    Theory,
    contains_arity_change,
    iter_theory_terms,
    primitive_occurrences,
    structural_permutation_parity,
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
    kind: str
    description: str
    parameters: dict[str, Any]
    evaluator: Callable[[Any], Any]
    equality: Callable[[Any, Any], bool] = operator.eq
    key: Hashable | None = None
    certified: bool = True

    def evaluate(self, term: Any) -> Any:
        return self.evaluator(term)

    def equal(self, left: Any, right: Any) -> bool:
        return self.equality(left, right)


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
class DirectCertificate:
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


@dataclass(frozen=True)
class SearchContext:
    retained: Mapping[Equation, tuple[Equation, ...]]
    active_primitive_count: int
    endomorphic: bool
    term_occurrences: Mapping[Circuit, Counter[str]]
    term_parities: dict[Circuit, int]

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
        endomorphic = _theory_is_endomorphic(theory, deadline)
        retained: dict[Equation, tuple[Equation, ...]] = {}
        for target in targets:
            if deadline is not None:
                deadline.check()
            retained[target] = tuple(
                _relevant_equations(
                    theory,
                    target,
                    endomorphic=endomorphic,
                    deadline=deadline,
                )
            )
        terms: list[Circuit] = []
        seen_terms: set[Circuit] = set()
        for target in targets:
            for equation in (*retained[target], target):
                for term in (equation.lhs, equation.rhs):
                    if term not in seen_terms:
                        seen_terms.add(term)
                        terms.append(term)
        occurrences: dict[Circuit, Counter[str]] = {}
        active_names: set[str] = set()
        for term in terms:
            if deadline is not None:
                deadline.check()
            counts = primitive_occurrences(term, theory.macros)
            occurrences[term] = counts
            active_names.update(counts)
        return cls(
            retained,
            len(active_names),
            endomorphic,
            occurrences,
            {},
        )

    def equations_for(self, target: Equation) -> tuple[Equation, ...]:
        return self.retained[target]

    def occurrences_for(
        self,
        term: Circuit,
        macros: Mapping[str, Circuit],
    ) -> Counter[str]:
        counts = self.term_occurrences.get(term)
        return counts if counts is not None else primitive_occurrences(term, macros)

    def parity_for(
        self,
        term: Circuit,
        macros: Mapping[str, Circuit],
    ) -> int:
        if term not in self.term_parities:
            self.term_parities[term] = structural_permutation_parity(term, macros)
        return self.term_parities[term]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    flags: tuple[str, ...]
    help: str
    bound: int | str
    option_names: tuple[str, ...] = ()
    bound_padding: int = 0
    candidate_offset: int = 0

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

    def load_candidates(
        self,
    ) -> Callable[..., Iterator[CandidateModel | DirectCertificate]]:
        module = importlib.import_module(
            f".separators.{self.name}",
            package=__package__,
        )
        return module.candidates


@dataclass(frozen=True)
class _ResolvedStrategy:
    candidates: Callable[..., Iterator[CandidateModel | DirectCertificate]]
    max_bound: int
    bound_offset: int = 0
    options: Mapping[str, Any] = field(default_factory=dict)


STRATEGY_SPECS = (
    StrategySpec(
        "presence",
        ("--presence-checking", "--presence_checking", "--max-occurrence"),
        "run Boolean max/occurrence models",
        "active_primitive_count",
    ),
    StrategySpec(
        "counting",
        ("--counting",),
        "run modular counting models",
        "max_modulus",
        option_names=("max_modulus",),
    ),
    StrategySpec(
        "block_normal_form",
        ("--block-normal-form", "--block_normal_form", "--brylinski"),
        "run three-wire Brylinski block-normal-form evidence",
        3,
        option_names=("max_block_matrix_dimension",),
    ),
    StrategySpec(
        "finite_model",
        ("--finite-model", "--finite_model", "--permutation-model"),
        "search small permutation interpretations",
        "max_permutation_degree",
        option_names=("max_permutation_degree",),
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
) -> bool:
    for generator in theory.signature.values():
        if deadline is not None:
            deadline.check()
        if generator.inputs != generator.outputs:
            return False
    checkpoint = deadline.check if deadline is not None else None
    return all(
        not contains_arity_change(term, checkpoint)
        for term in iter_theory_terms(theory)
    )


def _relevant_equations(
    theory: Theory,
    target: Equation,
    *,
    endomorphic: bool,
    deadline: Deadline | None = None,
) -> list[Equation]:
    others = [
        equation
        for equation in theory.equations
        if equation.id != target.id
    ]
    checkpoint = deadline.check if deadline is not None else None
    target_changes_arity = contains_arity_change(
        target.lhs, checkpoint
    ) or contains_arity_change(target.rhs, checkpoint)
    if not endomorphic or target_changes_arity:
        return others
    return [equation for equation in others if equation.arity <= target.arity]


def relevant_equations(theory: Theory, target: Equation) -> list[Equation]:
    """Rules to validate, with an arity shortcut for endomorphic theories."""
    return _relevant_equations(
        theory, target, endomorphic=_theory_is_endomorphic(theory)
    )


def verify(
    model: CandidateModel,
    theory: Theory,
    target: Equation,
    deadline: Deadline,
    context: SearchContext | None = None,
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
        context.equations_for(target)
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
        strategy=model.kind,
        description=model.description,
        parameters=model.parameters,
        checked_equations=tuple(checked),
        lhs_value=left,
        rhs_value=right,
        certified=model.certified,
    )


def _direct_separation(
    certificate: DirectCertificate,
    target: Equation,
    strategy: str,
    context: SearchContext,
) -> Separation:
    same = certificate.lhs_value == certificate.rhs_value
    if hasattr(same, "all"):
        same = bool(same.all())
    if same:
        raise ValueError(f"direct certificate from {strategy} has equal target values")
    return Separation(
        equation=target.id,
        strategy=strategy,
        description=certificate.description,
        parameters=certificate.parameters,
        checked_equations=tuple(
            equation.id for equation in context.equations_for(target)
        ),
        lhs_value=certificate.lhs_value,
        rhs_value=certificate.rhs_value,
        certified=certificate.certified,
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
        resolved[name] = _ResolvedStrategy(
            spec.load_candidates(),
            spec.maximum_bound(context, targets, options),
            spec.candidate_offset,
            {key: options[key] for key in spec.option_names},
        )
    return resolved


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
        models = iter(
            spec.candidates(
                theory,
                target,
                bound=bound + spec.bound_offset,
                deadline=deadline,
                context=context,
                **spec.options,
            )
        )
    except NotApplicable:
        return

    while True:
        try:
            model = next(models)
        except StopIteration:
            return
        except NotApplicable:
            return

        deadline.check()
        try:
            if isinstance(model, DirectCertificate):
                witness = _direct_separation(model, target, strategy_name, context)
            else:
                seen_key = (
                    (model.key, model.certified)
                    if model.key is not None
                    else None
                )
                if seen_key is not None and seen_key in seen:
                    continue
                witness = verify(model, theory, target, deadline, context)
                if seen_key is not None:
                    seen.add(seen_key)
        except NotApplicable:
            continue

        if witness is not None:
            yield witness
            if witness.certified:
                return


def search_theory(
    theory: Theory,
    *,
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    equation_ids: set[str] | None = None,
    max_arity: int | None = None,
    timeout: float = 600.0,
    max_modulus: int = 8,
    max_depth: int = 3,
    max_substitution_matrix_entries: int = 1_000_000,
    max_permutation_degree: int = 5,
    max_amalgam_prime: int = 11,
    max_amalgam_order: int = 4096,
    max_amalgam_bridge_generators: int = 1,
    max_amalgam_scalars: int = 3,
    max_amalgam_matrix_dimension: int = 32,
    max_spin_matrix_dimension: int = 1024,
    max_block_matrix_dimension: int = 512,
) -> SearchReport:
    strategies = tuple(strategies)
    unknown = sorted(set(strategies) - set(DEFAULT_STRATEGIES))
    if unknown:
        raise SearchConfigurationError(
            f"unknown separation strategy: {', '.join(unknown)}"
        )
    try:
        timeout = float(timeout)
    except (TypeError, ValueError) as exc:
        raise SearchConfigurationError(
            "timeout must be a finite non-negative number"
        ) from exc
    if not math.isfinite(timeout) or timeout < 0:
        raise SearchConfigurationError("timeout must be a finite non-negative number")
    if max_arity is not None:
        max_arity = _integer_option("max_arity", max_arity, 0)
    candidate_options = {
        "max_modulus": _integer_option("max_modulus", max_modulus, 2),
        "max_depth": _integer_option("max_depth", max_depth, 0),
        "max_substitution_matrix_entries": _integer_option(
            "max_substitution_matrix_entries",
            max_substitution_matrix_entries,
            1,
        ),
        "max_permutation_degree": _integer_option(
            "max_permutation_degree", max_permutation_degree, 2
        ),
        "max_amalgam_prime": _integer_option(
            "max_amalgam_prime", max_amalgam_prime, 2
        ),
        "max_amalgam_order": _integer_option(
            "max_amalgam_order", max_amalgam_order, 1
        ),
        "max_amalgam_bridge_generators": _integer_option(
            "max_amalgam_bridge_generators",
            max_amalgam_bridge_generators,
            1,
        ),
        "max_amalgam_scalars": _integer_option(
            "max_amalgam_scalars", max_amalgam_scalars, 0
        ),
        "max_amalgam_matrix_dimension": _integer_option(
            "max_amalgam_matrix_dimension",
            max_amalgam_matrix_dimension,
            1,
        ),
        "max_spin_matrix_dimension": _integer_option(
            "max_spin_matrix_dimension", max_spin_matrix_dimension, 1
        ),
        "max_block_matrix_dimension": _integer_option(
            "max_block_matrix_dimension", max_block_matrix_dimension, 1
        ),
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
    rounds = max((spec.max_bound for spec in specs.values()), default=0)
    timed_out = False

    try:
        for bound in range(rounds + 1):
            deadline.check()
            for strategy_name, spec in specs.items():
                deadline.check()
                if bound > spec.max_bound:
                    continue
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
                        if previous is None or (
                            not previous.certified and witness.certified
                        ):
                            witnesses[target_id] = witness
                            if witness.certified:
                                pending.discard(target_id)
                if not pending:
                    break
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
