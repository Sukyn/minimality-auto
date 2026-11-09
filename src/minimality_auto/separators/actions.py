"""Finite actions: permutation search and synthesis of arbitrary transformations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import cache, lru_cache
from itertools import permutations as all_permutations
from typing import Any

import z3

from .. import presentation as fixed_arity
from ..presentation import Token, Word
from ..search_types import CandidateModel, Deadline, SearchTimeout
from ..syntax import Circuit
from .common import require_endomorphic_theory
from .synthesis import solve

Action = tuple[int, ...]
MAX_DEGREE = 9


def _identity(degree: int) -> Action:
    return tuple(range(degree))


@lru_cache(maxsize=200_000)
def _compose(left: Action, right: Action) -> Action:
    """Return ``left`` after ``right``."""
    return tuple(left[right[index]] for index in range(len(left)))


def _evaluate(
    word: Word,
    assignment: Mapping[Token, Action],
    degree: int,
    deadline: Deadline | None = None,
) -> Action:
    result = _identity(degree)
    for token in word:
        if deadline is not None:
            deadline.check()
        result = _compose(assignment[token], result)
    return result


def _cycle_type(value: Action) -> tuple[int, ...]:
    unseen = set(range(len(value)))
    lengths: list[int] = []
    while unseen:
        start = min(unseen)
        current = start
        length = 0
        while current in unseen:
            unseen.remove(current)
            current = value[current]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def _conjugacy_representatives(
    values: tuple[Action, ...], deadline: Deadline
) -> tuple[Action, ...]:
    representatives: dict[tuple[int, ...], Action] = {}
    for index, value in enumerate(values):
        if index % 512 == 0:
            deadline.check()
        representatives.setdefault(_cycle_type(value), value)
    return tuple(representatives.values())


def _permutations(degree: int, deadline: Deadline) -> tuple[Action, ...]:
    values: list[Action] = []
    for index, value in enumerate(all_permutations(range(degree))):
        if index % 1024 == 0:
            deadline.check()
        values.append(value)
    return tuple(values)


def _holds(
    constraint: fixed_arity.Constraint,
    assignment: Mapping[Token, Action],
    degree: int,
    deadline: Deadline | None = None,
) -> bool:
    same = _evaluate(constraint.left, assignment, degree, deadline) == _evaluate(
        constraint.right, assignment, degree, deadline
    )
    return same == constraint.equal


def _validate_assignment(
    variables: tuple[Token, ...],
    constraints: tuple[fixed_arity.Constraint, ...],
    assignment: Mapping[Token, Action],
    degree: int,
    deadline: Deadline | None = None,
    *,
    bijective: bool = True,
) -> None:
    declared = frozenset(variables)
    required = frozenset(
        variable for constraint in constraints for variable in constraint.variables
    )
    if not required <= declared:
        missing = min(required - declared, key=str)
        raise ValueError(
            f"constraint uses undeclared interpretation {fixed_arity.display_name(missing)}"
        )
    expected = _identity(degree)
    for variable in variables:
        if deadline is not None:
            deadline.check()
        value = assignment.get(variable)
        if value is None:
            raise ValueError(f"missing interpretation for {fixed_arity.display_name(variable)}")
        valid = len(value) == degree and (
            tuple(sorted(value)) == expected
            if bijective
            else all(0 <= entry < degree for entry in value)
        )
        if not valid:
            kind = "permutation" if bijective else "transformation"
            raise ValueError(
                f"{fixed_arity.display_name(variable)} is not a degree-{degree} {kind}"
            )
    for constraint in constraints:
        if deadline is not None:
            deadline.check()
        if not _holds(constraint, assignment, degree, deadline):
            raise ValueError("finite action interpretation violates a required relation")


def _find_assignment(
    variables: tuple[Token, ...],
    constraints: tuple[fixed_arity.Constraint, ...],
    degree: int,
    deadline: Deadline,
) -> dict[Token, Action] | None:
    group = _permutations(degree, deadline)
    identity = _identity(degree)
    involutions: list[Action] = []
    for index, value in enumerate(group):
        if index % 512 == 0:
            deadline.check()
        if _compose(value, value) == identity:
            involutions.append(value)
    domains = {
        variable: tuple(involutions) if variable[0] == "swap" else group for variable in variables
    }
    by_variable = {
        variable: tuple(
            constraint for constraint in constraints if variable in constraint.variables
        )
        for variable in variables
    }
    empty = tuple(constraint for constraint in constraints if not constraint.variables)
    if any(not _holds(constraint, {}, degree) for constraint in empty):
        return None

    for variable in variables:
        local = tuple(
            constraint for constraint in by_variable[variable] if constraint.variables <= {variable}
        )
        if local:
            filtered: list[Action] = []
            for index, value in enumerate(domains[variable]):
                if index % 512 == 0:
                    deadline.check()
                if all(_holds(constraint, {variable: value}, degree) for constraint in local):
                    filtered.append(value)
            domains[variable] = tuple(filtered)
        if not domains[variable]:
            return None

    assignment: dict[Token, Action] = {}
    checks = 0

    def viable(variable: Token) -> tuple[Action, ...]:
        nonlocal checks
        available = frozenset((*assignment, variable))
        ready = tuple(
            constraint for constraint in by_variable[variable] if constraint.variables <= available
        )
        result: list[Action] = []
        for value in domains[variable]:
            checks += 1
            if checks % 512 == 0:
                deadline.check()
            assignment[variable] = value
            if all(_holds(constraint, assignment, degree) for constraint in ready):
                result.append(value)
            assignment.pop(variable)
        values = tuple(result)
        return _conjugacy_representatives(values, deadline) if not assignment else values

    def choose() -> tuple[Token, tuple[Action, ...]] | None:
        choices: list[tuple[int, int, int, str, Token, tuple[Action, ...]]] = []
        assigned = frozenset(assignment)
        for variable in variables:
            if variable in assignment:
                continue
            values = viable(variable)
            if not values:
                return None
            ready = sum(
                constraint.variables <= assigned | {variable}
                for constraint in by_variable[variable]
            )
            choices.append(
                (
                    len(values),
                    -ready,
                    -len(by_variable[variable]),
                    str(variable),
                    variable,
                    values,
                )
            )
        _, _, _, _, variable, values = min(choices, key=lambda choice: choice[:4])
        return variable, values

    # An explicit stack handles presentations with more generators than
    # Python's recursion limit.
    frames: list[tuple[Token, tuple[Action, ...], int]] = []

    def advance() -> bool:
        while frames:
            variable, values, next_index = frames[-1]
            assignment.pop(variable, None)
            if next_index < len(values):
                assignment[variable] = values[next_index]
                frames[-1] = variable, values, next_index + 1
                return True
            frames.pop()
        return False

    while len(assignment) < len(variables):
        deadline.check()
        choice = choose()
        if choice is not None:
            variable, values = choice
            frames.append((variable, values, 0))
        if not advance():
            return None
    return dict(assignment)


def permutations(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_permutation_degree: int = MAX_DEGREE,
    context: Any = None,
) -> Iterator[CandidateModel]:
    if bound < 2 or bound > max_permutation_degree:
        return
    deadline.check()
    require_endomorphic_theory(theory, deadline, context)

    presentation = fixed_arity.compile_presentation(
        theory,
        target,
        deadline,
        context,
    )
    constraints = presentation.separation_constraints
    solution = _find_assignment(
        presentation.variables,
        constraints,
        bound,
        deadline,
    )
    if solution is None:
        return
    _validate_assignment(
        presentation.variables,
        constraints,
        solution,
        bound,
        deadline,
    )

    yield _model(presentation, solution, bound, deadline, bijective=True)


def _model(
    presentation: fixed_arity.Presentation,
    assignment: Mapping[Token, Action],
    degree: int,
    deadline: Deadline,
    *,
    bijective: bool,
) -> CandidateModel:
    values = dict(assignment)
    for name in presentation.signature:
        values.setdefault(fixed_arity.generator(name), _identity(degree))
    generator_values = {
        name: list(values[fixed_arity.generator(name)]) for name in presentation.signature
    }
    swap_values = {
        fixed_arity.display_name(fixed_arity.swap(index)): list(values[fixed_arity.swap(index)])
        for index in range(presentation.arity - 1)
    }

    def render(values: Mapping[str, list[int]]) -> str:
        return ", ".join(f"{name}={tuple(value)}" for name, value in values.items())

    kind = "permutation" if bijective else "transformation monoid"
    description = f"degree-{degree} {kind} interpretation"
    if bijective:
        description += f": generators {{{render(generator_values)}}}"
        if swap_values:
            description += f"; structural swaps {{{render(swap_values)}}}"

    @cache
    def evaluate(term: Circuit) -> Action:
        return _evaluate(
            presentation.compile(term, deadline),
            values,
            degree,
            deadline,
        )

    return CandidateModel(
        description=description,
        parameters={
            "arity": presentation.arity,
            "degree": degree,
            "interpretation": {
                "generators": generator_values,
                "structural_swaps": swap_values,
            },
        },
        evaluate=evaluate,
    )


def _select(table: tuple[Any, ...], index: Any) -> Any:
    if isinstance(index, int):
        return table[index]
    result = table[-1]
    for i in range(len(table) - 2, -1, -1):
        result = z3.If(index == i, table[i], result)
    return result


def _find_transformations(
    presentation: fixed_arity.Presentation,
    degree: int,
    deadline: Deadline,
) -> dict[Token, Action] | None:
    solver = z3.Solver()
    assignment = {}
    for index, token in enumerate(presentation.variables):
        deadline.check()
        table = tuple(z3.Int(f"t{index}_{i}") for i in range(degree))
        solver.add(*(z3.And(value >= 0, value < degree) for value in table))
        assignment[token] = table

    @cache
    def evaluate(word: Word) -> tuple[Any, ...]:
        result = _identity(degree)
        for token in word:
            deadline.check()
            result = tuple(_select(assignment[token], i) for i in result)
        return result

    for constraint in presentation.separation_constraints:
        deadline.check()
        same = z3.And(
            *(
                a == b
                for a, b in zip(
                    evaluate(constraint.left),
                    evaluate(constraint.right),
                    strict=True,
                )
            )
        )
        solver.add(same if constraint.equal else z3.Not(same))
    solution = solve(solver, deadline)
    if solution is None:
        return None
    values = {
        token: tuple(solution.eval(item, model_completion=True).as_long() for item in table)
        for token, table in assignment.items()
    }
    # Retained equations and separation are checked by the common verifier.
    _validate_assignment(
        presentation.variables,
        presentation.prop,
        values,
        degree,
        deadline,
        bijective=False,
    )
    return values


def transformations(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_transformation_degree: int = 4,
    model_timeout_ms: int = 1000,
    context: Any = None,
) -> Iterator[CandidateModel]:
    if bound < 2 or bound > max_transformation_degree:
        return
    require_endomorphic_theory(theory, deadline, context)
    presentation = fixed_arity.compile_presentation(theory, target, deadline, context)
    try:
        assignment = _find_transformations(
            presentation,
            bound,
            deadline.capped(model_timeout_ms / 1000),
        )
    except SearchTimeout:
        deadline.check()
        return
    if assignment is not None:
        yield _model(presentation, assignment, bound, deadline, bijective=False)
