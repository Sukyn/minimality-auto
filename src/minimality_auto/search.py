from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import operator
import time
from typing import Any, Callable, Iterable, Iterator, Protocol


DEFAULT_STRATEGIES = (
    "presence",
    "counting",
    "finite_model",
    "amalgam",
    "spin",
    "determinant",
    "substitution",
)


class SearchTimeout(RuntimeError):
    pass


class NotApplicable(RuntimeError):
    pass


@dataclass(frozen=True)
class Deadline:
    expires_at: float

    @classmethod
    def after(cls, seconds: float) -> "Deadline":
        return cls(time.monotonic() + max(0.0, seconds))

    @property
    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def check(self) -> None:
        if time.monotonic() >= self.expires_at:
            raise SearchTimeout


class Model(Protocol):
    kind: str
    description: str
    parameters: dict[str, Any]

    def evaluate(self, term: Any) -> Any: ...

    def equal(self, left: Any, right: Any) -> bool: ...


@dataclass
class CandidateModel:
    kind: str
    description: str
    parameters: dict[str, Any]
    evaluator: Callable[[Any], Any]
    equality: Callable[[Any, Any], bool] = field(default=lambda a, b: a == b)
    key: tuple[Any, ...] = ()

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

    def as_dict(self) -> dict[str, Any]:
        return {
            "equation": self.equation,
            "strategy": self.strategy,
            "description": self.description,
            "parameters": _json_value(self.parameters),
            "checked_equations": list(self.checked_equations),
            "lhs_value": _json_value(self.lhs_value),
            "rhs_value": _json_value(self.rhs_value),
        }


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


def equation_id(equation: Any) -> str:
    return str(getattr(equation, "id", getattr(equation, "name", "?")))


def equation_arity(equation: Any) -> int:
    explicit = getattr(equation, "arity", None)
    if explicit is not None:
        return int(explicit)
    source = getattr(equation, "source", None)
    if source is not None:
        return int(source)
    lhs = getattr(equation, "lhs", None)
    for attr in ("source", "inputs"):
        value = getattr(lhs, attr, None)
        if value is not None:
            return int(value)
    raise ValueError(f"cannot determine arity of equation {equation_id(equation)}")


def _contains_arity_change(node: Any) -> bool:
    inputs = getattr(node, "inputs", None)
    outputs = getattr(node, "outputs", None)
    if inputs is not None and outputs is not None and inputs != outputs:
        return True
    return any(_contains_arity_change(part) for part in getattr(node, "parts", ()))


def _theory_is_endomorphic(theory: Any) -> bool:
    signature = getattr(theory, "signature", ())
    if any(generator.inputs != generator.outputs for generator in signature.values()):
        return False
    for equation in theory.equations:
        if _contains_arity_change(equation.lhs) or _contains_arity_change(equation.rhs):
            return False
    macros = getattr(theory, "macros", {})
    for definition in macros.values():
        body = getattr(definition, "body", definition)
        if _contains_arity_change(body):
            return False
    return True


def _active_primitive_name_count(theory: Any) -> int:
    from .core import primitive_occurrences

    names: set[str] = set()
    for equation in theory.equations:
        names.update(primitive_occurrences(equation.lhs, theory.macros))
        names.update(primitive_occurrences(equation.rhs, theory.macros))
    return len(names)


def relevant_equations(theory: Any, target: Any) -> list[Any]:
    """Rules to validate, with an arity shortcut for endomorphic theories."""
    others = [
        equation
        for equation in theory.equations
        if equation_id(equation) != equation_id(target)
    ]
    target_changes_arity = _contains_arity_change(
        getattr(target, "lhs", None)
    ) or _contains_arity_change(getattr(target, "rhs", None))
    if not _theory_is_endomorphic(theory) or target_changes_arity:
        return others
    arity = equation_arity(target)
    return [equation for equation in others if equation_arity(equation) <= arity]


def verify(model: Model, theory: Any, target: Any, deadline: Deadline) -> Separation | None:
    checked: list[str] = []
    for equation in relevant_equations(theory, target):
        deadline.check()
        left = model.evaluate(equation.lhs)
        deadline.check()
        right = model.evaluate(equation.rhs)
        deadline.check()
        equal = model.equal(left, right)
        deadline.check()
        if not equal:
            return None
        checked.append(equation_id(equation))

    deadline.check()
    left = model.evaluate(target.lhs)
    deadline.check()
    right = model.evaluate(target.rhs)
    deadline.check()
    equal = model.equal(left, right)
    deadline.check()
    if equal:
        return None
    return Separation(
        equation=equation_id(target),
        strategy=model.kind,
        description=model.description,
        parameters=model.parameters,
        checked_equations=tuple(checked),
        lhs_value=left,
        rhs_value=right,
    )


def _validate_direct_separation(
    separation: Separation,
    theory: Any,
    target: Any,
    strategy: str,
) -> None:
    expected = tuple(
        equation_id(equation) for equation in relevant_equations(theory, target)
    )
    if separation.equation != equation_id(target) or separation.strategy != strategy:
        raise ValueError(f"invalid direct separation from {strategy}")
    if separation.checked_equations != expected:
        raise ValueError(
            f"direct separation from {strategy} did not check every retained equation"
        )
    same = separation.lhs_value == separation.rhs_value
    if hasattr(same, "all"):
        same = bool(same.all())
    if same:
        raise ValueError(f"direct separation from {strategy} has equal target values")


def _strategy_functions(
    names: Iterable[str],
) -> dict[str, Callable[..., Iterator[CandidateModel | Separation]]]:
    from .separators import (
        amalgam,
        counting,
        determinant,
        finite_model,
        presence,
        spin,
        substitution,
    )

    available = {
        "presence": presence.candidates,
        "counting": counting.candidates,
        "substitution": substitution.candidates,
        "determinant": determinant.candidates,
        "finite_model": finite_model.candidates,
        "amalgam": amalgam.candidates,
        "spin": spin.candidates,
    }
    requested = tuple(names)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown separation strategy: {', '.join(unknown)}")
    return {name: available[name] for name in requested}


def _integer_option(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        qualifier = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return parsed


def search_theory(
    theory: Any,
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
) -> SearchReport:
    try:
        timeout = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a finite non-negative number") from exc
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be a finite non-negative number")
    if max_arity is not None:
        max_arity = _integer_option("max_arity", max_arity, 0)
    max_modulus = _integer_option("max_modulus", max_modulus, 2)
    max_depth = _integer_option("max_depth", max_depth, 0)
    max_substitution_matrix_entries = _integer_option(
        "max_substitution_matrix_entries", max_substitution_matrix_entries, 1
    )
    max_permutation_degree = _integer_option(
        "max_permutation_degree", max_permutation_degree, 2
    )
    max_amalgam_prime = _integer_option(
        "max_amalgam_prime", max_amalgam_prime, 2
    )
    max_amalgam_order = _integer_option("max_amalgam_order", max_amalgam_order, 1)
    max_amalgam_bridge_generators = _integer_option(
        "max_amalgam_bridge_generators", max_amalgam_bridge_generators, 1
    )
    max_amalgam_scalars = _integer_option(
        "max_amalgam_scalars", max_amalgam_scalars, 0
    )
    max_amalgam_matrix_dimension = _integer_option(
        "max_amalgam_matrix_dimension", max_amalgam_matrix_dimension, 1
    )
    max_spin_matrix_dimension = _integer_option(
        "max_spin_matrix_dimension", max_spin_matrix_dimension, 1
    )

    start = time.monotonic()
    deadline = Deadline.after(timeout)
    functions = _strategy_functions(strategies)
    targets = [
        equation
        for equation in theory.equations
        if (equation_ids is None or equation_id(equation) in equation_ids)
        and (max_arity is None or equation_arity(equation) <= max_arity)
    ]
    missing = (equation_ids or set()) - {equation_id(equation) for equation in targets}
    if missing:
        raise ValueError(f"unknown or filtered equation(s): {', '.join(sorted(missing))}")

    witnesses: dict[str, Separation] = {}
    seen: dict[tuple[str, str], set[tuple[Any, ...]]] = {}
    round_limits = {
        "counting": max_modulus,
        "substitution": max_depth + 2,
        "finite_model": max_permutation_degree,
        "amalgam": 1,
        "spin": 0,
    }
    if "presence" in functions:
        round_limits["presence"] = max(1, _active_primitive_name_count(theory))
    if "determinant" in functions:
        round_limits["determinant"] = max(
            3, *(equation_arity(equation) for equation in targets)
        )
    rounds = max((round_limits[name] for name in functions), default=0)
    timed_out = False

    try:
        for bound in range(0, rounds + 1):
            deadline.check()
            for target in targets:
                target_id = equation_id(target)
                if target_id in witnesses:
                    continue
                for strategy_name, generate in functions.items():
                    deadline.check()
                    if bound > round_limits[strategy_name]:
                        continue
                    strategy_bound = bound - 2 if strategy_name == "substitution" else bound
                    try:
                        models = generate(
                            theory,
                            target,
                            bound=strategy_bound,
                            deadline=deadline,
                            max_modulus=max_modulus,
                            max_depth=max_depth,
                            max_substitution_matrix_entries=(
                                max_substitution_matrix_entries
                            ),
                            max_permutation_degree=max_permutation_degree,
                            max_amalgam_prime=max_amalgam_prime,
                            max_amalgam_order=max_amalgam_order,
                            max_amalgam_bridge_generators=max_amalgam_bridge_generators,
                            max_amalgam_scalars=max_amalgam_scalars,
                            max_amalgam_matrix_dimension=max_amalgam_matrix_dimension,
                            max_spin_matrix_dimension=max_spin_matrix_dimension,
                        )
                        for model in models:
                            deadline.check()
                            if isinstance(model, Separation):
                                _validate_direct_separation(
                                    model, theory, target, strategy_name
                                )
                                witnesses[target_id] = model
                                break
                            model_key = model.key or (model.description,)
                            bucket = seen.setdefault((target_id, strategy_name), set())
                            if model_key in bucket:
                                continue
                            bucket.add(model_key)
                            witness = verify(model, theory, target, deadline)
                            if witness is not None:
                                witnesses[target_id] = witness
                                break
                    except NotApplicable:
                        continue
                    if target_id in witnesses:
                        break
            if len(witnesses) == len(targets):
                break
    except SearchTimeout:
        timed_out = True

    unresolved = tuple(
        equation_id(target)
        for target in targets
        if equation_id(target) not in witnesses
    )
    return SearchReport(
        theory=str(getattr(theory, "name", None) or "theory"),
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
