"""Strategy configuration, loading, and the bounded search schedule."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial

from .context import SearchContext
from .search_types import CandidateModel, DirectEvidence
from .syntax import Equation


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
    LimitSpec("max_matrix_dimension", 2, 1, "largest finite matrix model wire dimension"),
    LimitSpec("max_matrix_prime", 5, 2, "largest characteristic for finite matrix models"),
    LimitSpec("max_matrix_entries", 256, 1, "entry limit for each finite model matrix"),
    LimitSpec("max_transformation_degree", 4, 2, "largest transformation monoid degree"),
    LimitSpec("model_timeout_ms", 1000, 1, "time per finite synthesis attempt, in milliseconds"),
)


LIMIT_BY_NAME = {spec.name: spec for spec in LIMIT_SPECS}


DEFAULT_TIMEOUT = 600.0


@dataclass(frozen=True)
class StrategySpec:
    name: str
    flags: tuple[str, ...]
    help: str
    bound: int | str | tuple[str, ...]
    option_names: tuple[str, ...] = ()
    candidate_offset: int = 0
    eager_bound: int = -1
    implementation: str = ""

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
            names = (self.bound,) if isinstance(self.bound, str) else self.bound
            base = max(limits[name] for name in names)
        return base - self.candidate_offset


@dataclass(frozen=True)
class _ResolvedStrategy:
    candidates: Callable[..., Iterator[CandidateModel | DirectEvidence]]
    max_bound: int
    bound_offset: int = 0
    eager_bound: int = -1


_MATRIX_OPTIONS = (
    "max_matrix_dimension",
    "max_matrix_prime",
    "max_matrix_entries",
    "model_timeout_ms",
)


STRATEGY_SPECS = (
    StrategySpec(
        "presence",
        ("--presence-checking", "--presence_checking", "--max-occurrence"),
        "run Boolean max/occurrence models",
        "active_primitive_count",
        eager_bound=4,
    ),
    StrategySpec(
        "abelianisation",
        ("--abelianisation", "--abelianization"),
        "compute the full additive quotient over Z, including torsion",
        0,
        eager_bound=0,
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
        implementation="actions.permutations",
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
        "relations",
        ("--relations", "--boolean-matrices"),
        "synthesize finite relations / Boolean matrices",
        "max_matrix_dimension",
        option_names=_MATRIX_OPTIONS,
        implementation="matrices.relations",
    ),
    StrategySpec(
        "dual_numbers",
        ("--dual-numbers",),
        "synthesize matrices over F_p[epsilon]/(epsilon^2)",
        ("max_matrix_dimension", "max_matrix_prime"),
        option_names=_MATRIX_OPTIONS,
        implementation="matrices.dual_numbers",
    ),
    StrategySpec(
        "finite_field_matrices",
        ("--finite-field-matrices",),
        "synthesize independent rectangular matrices over prime fields",
        ("max_matrix_dimension", "max_matrix_prime"),
        option_names=_MATRIX_OPTIONS,
        implementation="matrices.finite_field_matrices",
    ),
    StrategySpec(
        "transformation_monoid",
        ("--transformation-monoid",),
        "synthesize finite actions, including noninvertible transformations",
        "max_transformation_degree",
        option_names=("max_transformation_degree", "model_timeout_ms"),
        implementation="actions.transformations",
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
        candidate_offset=-2,
    ),
)


_STRATEGY_BY_NAME = {spec.name: spec for spec in STRATEGY_SPECS}


DEFAULT_STRATEGIES = tuple(spec.name for spec in STRATEGY_SPECS)


def _strategy_specs(
    names: Iterable[str],
    context: SearchContext,
    targets: Iterable[Equation],
    options: Mapping[str, int],
) -> dict[str, _ResolvedStrategy]:
    resolved: dict[str, _ResolvedStrategy] = {}
    for name in names:
        spec = _STRATEGY_BY_NAME[name]
        module_name, _, function = (spec.implementation or f"{name}.candidates").rpartition(".")
        module = importlib.import_module(
            f".separators.{module_name}",
            package=__package__,
        )
        resolved[name] = _ResolvedStrategy(
            candidates=partial(
                getattr(module, function),
                **{key: options[key] for key in spec.option_names},
            ),
            max_bound=spec.maximum_bound(context, targets, options),
            bound_offset=spec.candidate_offset,
            eager_bound=spec.eager_bound,
        )
    return resolved


def _search_schedule(
    specs: Mapping[str, _ResolvedStrategy],
) -> Iterator[tuple[int, str, _ResolvedStrategy]]:
    """Run each strategy's eager bounds, then advance all strategies by bound."""

    for name, spec in specs.items():
        for bound in range(min(spec.eager_bound, spec.max_bound) + 1):
            yield bound, name, spec

    rounds = max((spec.max_bound for spec in specs.values()), default=-1)
    for bound in range(rounds + 1):
        for name, spec in specs.items():
            if spec.eager_bound < bound <= spec.max_bound:
                yield bound, name, spec
