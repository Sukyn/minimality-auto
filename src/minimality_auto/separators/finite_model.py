"""Generic fixed-arity permutation interpretations for endomorphic PROPs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import lru_cache
from itertools import permutations
from typing import Any

from ..core import Circuit
from ..search import (
    CandidateModel,
    Deadline,
    NotApplicable,
)
from . import fixed_arity
from .common import require_endomorphic_theory
from .fixed_arity import Token, Word

Permutation = tuple[int, ...]
MAX_DEGREE = 9


def _identity(degree: int) -> Permutation:
    return tuple(range(degree))


@lru_cache(maxsize=200_000)
def _compose(left: Permutation, right: Permutation) -> Permutation:
    """Return ``left`` after ``right``."""
    return tuple(left[right[index]] for index in range(len(left)))


def _evaluate(
    word: Word,
    assignment: Mapping[Token, Permutation],
    degree: int,
) -> Permutation:
    result = _identity(degree)
    for token in word:
        result = _compose(assignment[token], result)
    return result


def _cycle_type(value: Permutation) -> tuple[int, ...]:
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
    values: tuple[Permutation, ...], deadline: Deadline
) -> tuple[Permutation, ...]:
    representatives: dict[tuple[int, ...], Permutation] = {}
    for index, value in enumerate(values):
        if index % 512 == 0:
            deadline.check()
        representatives.setdefault(_cycle_type(value), value)
    return tuple(representatives.values())


def _permutations(degree: int, deadline: Deadline) -> tuple[Permutation, ...]:
    values: list[Permutation] = []
    for index, value in enumerate(permutations(range(degree))):
        if index % 1024 == 0:
            deadline.check()
        values.append(value)
    return tuple(values)


def _holds(
    constraint: fixed_arity.Constraint,
    assignment: Mapping[Token, Permutation],
    degree: int,
) -> bool:
    same = _evaluate(constraint.left, assignment, degree) == _evaluate(
        constraint.right, assignment, degree
    )
    return same == constraint.equal


def _validate_assignment(
    variables: tuple[Token, ...],
    constraints: tuple[fixed_arity.Constraint, ...],
    assignment: Mapping[Token, Permutation],
    degree: int,
    deadline: Deadline | None = None,
) -> None:
    declared = frozenset(variables)
    required = frozenset(
        variable for constraint in constraints for variable in constraint.variables
    )
    if not required <= declared:
        missing = min(required - declared, key=str)
        raise ValueError(
            "constraint uses undeclared interpretation "
            f"{fixed_arity.display_name(missing)}"
        )
    expected = _identity(degree)
    for variable in variables:
        if deadline is not None:
            deadline.check()
        value = assignment.get(variable)
        if value is None:
            raise ValueError(
                f"missing interpretation for {fixed_arity.display_name(variable)}"
            )
        if len(value) != degree or tuple(sorted(value)) != expected:
            raise ValueError(
                f"{fixed_arity.display_name(variable)} is not a degree-{degree} "
                "permutation"
            )
    for constraint in constraints:
        if deadline is not None:
            deadline.check()
        if not _holds(constraint, assignment, degree):
            raise ValueError("permutation interpretation violates a required relation")


def _validate_prop_assignment(
    arity: int,
    generators: tuple[tuple[str, int], ...],
    assignment: Mapping[Token, Permutation],
    degree: int,
) -> None:
    variables = tuple(fixed_arity.generator(name) for name, _ in generators) + tuple(
        fixed_arity.swap(index) for index in range(arity - 1)
    )
    _validate_assignment(
        variables,
        fixed_arity.prop_constraints(arity, generators),
        assignment,
        degree,
    )


def _find_assignment(
    variables: tuple[Token, ...],
    constraints: tuple[fixed_arity.Constraint, ...],
    degree: int,
    deadline: Deadline,
) -> dict[Token, Permutation] | None:
    group = _permutations(degree, deadline)
    identity = _identity(degree)
    involutions: list[Permutation] = []
    for index, value in enumerate(group):
        if index % 512 == 0:
            deadline.check()
        if _compose(value, value) == identity:
            involutions.append(value)
    domains = {
        variable: tuple(involutions) if variable[0] == "swap" else group
        for variable in variables
    }
    by_variable = {
        variable: tuple(
            constraint for constraint in constraints if variable in constraint.variables
        )
        for variable in variables
    }
    empty = tuple(
        constraint for constraint in constraints if not constraint.variables
    )
    if any(not _holds(constraint, {}, degree) for constraint in empty):
        return None

    for variable in variables:
        local = tuple(
            constraint
            for constraint in by_variable[variable]
            if constraint.variables <= {variable}
        )
        if local:
            filtered: list[Permutation] = []
            for index, value in enumerate(domains[variable]):
                if index % 512 == 0:
                    deadline.check()
                if all(
                    _holds(constraint, {variable: value}, degree)
                    for constraint in local
                ):
                    filtered.append(value)
            domains[variable] = tuple(filtered)
        if not domains[variable]:
            return None

    assignment: dict[Token, Permutation] = {}
    checks = 0

    def viable(variable: Token) -> tuple[Permutation, ...]:
        nonlocal checks
        available = frozenset((*assignment, variable))
        ready = tuple(
            constraint
            for constraint in by_variable[variable]
            if constraint.variables <= available
        )
        result: list[Permutation] = []
        for value in domains[variable]:
            checks += 1
            if checks % 512 == 0:
                deadline.check()
            assignment[variable] = value
            if all(_holds(constraint, assignment, degree) for constraint in ready):
                result.append(value)
            assignment.pop(variable)
        values = tuple(result)
        return (
            _conjugacy_representatives(values, deadline)
            if not assignment
            else values
        )

    def choose() -> tuple[Token, tuple[Permutation, ...]] | None:
        choices: list[tuple[int, int, int, str, Token, tuple[Permutation, ...]]] = []
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

    # Use an explicit backtracking stack.  Large, otherwise straightforward
    # presentations can contain more generators than Python's recursion limit.
    frames: list[tuple[Token, tuple[Permutation, ...], int]] = []

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


def candidates(
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

    identity = _identity(bound)
    full_assignment = {
        fixed_arity.generator(name): solution.get(
            fixed_arity.generator(name), identity
        )
        for name in presentation.signature
    }
    full_assignment.update(
        {
            fixed_arity.swap(index): solution[fixed_arity.swap(index)]
            for index in range(presentation.arity - 1)
        }
    )
    generator_values = {
        name: list(full_assignment[fixed_arity.generator(name)])
        for name in presentation.signature
    }
    swap_values = {
        fixed_arity.display_name(fixed_arity.swap(index)): list(
            full_assignment[fixed_arity.swap(index)]
        )
        for index in range(presentation.arity - 1)
    }

    def render(values: Mapping[str, list[int]]) -> str:
        return ", ".join(f"{name}={tuple(value)}" for name, value in values.items())

    rendered = f"generators {{{render(generator_values)}}}"
    if swap_values:
        rendered += f"; structural swaps {{{render(swap_values)}}}"

    def evaluate(term: Circuit) -> Permutation:
        if term.inputs != term.outputs or term.inputs > presentation.arity:
            raise NotApplicable(
                "permutation interpretation supports endomorphisms through arity "
                f"{presentation.arity}"
            )
        return _evaluate(
            fixed_arity.compile_word(
                term,
                presentation.signature,
                presentation.arity,
            ),
            full_assignment,
            bound,
        )

    yield CandidateModel(
        description=f"degree-{bound} permutation interpretation: {rendered}",
        parameters={
            "arity": presentation.arity,
            "degree": bound,
            "interpretation": {
                "generators": generator_values,
                "structural_swaps": swap_values,
            },
        },
        evaluator=evaluate,
    )
