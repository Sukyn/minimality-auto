"""Generic fixed-arity permutation interpretations for endomorphic PROPs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from typing import Any, Iterator, Mapping

from ..core import Circuit, expand_macros
from ..search import (
    CandidateModel,
    Deadline,
    NotApplicable,
    equation_arity,
    relevant_equations,
)


Permutation = tuple[int, ...]
Token = tuple[str, str | int]
Word = tuple[Token, ...]
MAX_DEGREE = 9


def _generator(name: str) -> Token:
    return "generator", name


def _swap(index: int) -> Token:
    return "swap", index


def _identity(degree: int) -> Permutation:
    return tuple(range(degree))


@lru_cache(maxsize=200_000)
def _compose(left: Permutation, right: Permutation) -> Permutation:
    """Return ``left`` after ``right``."""
    return tuple(left[right[index]] for index in range(len(left)))


def _inverse(order: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(order)
    for output, source in enumerate(order):
        result[source] = output
    return tuple(result)


def _adjacent_word(
    order: tuple[int, ...],
    offset: int = 0,
    deadline: Deadline | None = None,
) -> Word:
    current = list(range(len(order)))
    result: list[Token] = []
    for position, wanted in enumerate(order):
        if deadline is not None and position % 256 == 0:
            deadline.check()
        found = current.index(wanted, position)
        while found > position:
            current[found - 1], current[found] = current[found], current[found - 1]
            result.append(_swap(offset + found - 1))
            if deadline is not None and len(result) % 1024 == 0:
                deadline.check()
            found -= 1
    assert tuple(current) == order
    return tuple(result)


def _placed(
    token: Token,
    selected: tuple[int, ...],
    total: int,
    deadline: Deadline | None = None,
) -> Word:
    chosen = frozenset(selected)
    if (
        total < 0
        or len(chosen) != len(selected)
        or any(wire < 0 or wire >= total for wire in chosen)
    ):
        raise ValueError("selected wires must be distinct positions in the ambient arity")
    rest = tuple(wire for wire in range(total) if wire not in chosen)
    order = selected + rest
    return (
        _adjacent_word(order, deadline=deadline)
        + (token,)
        + _adjacent_word(_inverse(order), deadline=deadline)
    )


def _word(
    node: Circuit,
    signature: Mapping[str, int],
    total: int,
    *,
    offset: int = 0,
    deadline: Deadline | None = None,
) -> Word:
    if deadline is not None:
        deadline.check()
    if (
        total < 0
        or offset < 0
        or node.inputs != node.outputs
        or offset + node.inputs > total
    ):
        raise NotApplicable("permutation search requires endomorphisms")
    if node.kind == "id":
        return ()
    if node.kind == "gen":
        if node.name not in signature or signature[node.name] != node.inputs:
            raise NotApplicable(f"unsupported generator {node.name!r}")
        selected = tuple(range(offset, offset + node.inputs))
        return _placed(_generator(node.name), selected, total, deadline)
    if node.kind == "perm":
        return _adjacent_word(node.order, offset, deadline)
    if node.kind == "compose":
        return tuple(
            token
            for part in node.parts
            for token in _word(
                part, signature, total, offset=offset, deadline=deadline
            )
        )
    if node.kind == "tensor":
        result: list[Token] = []
        position = offset
        for part in node.parts:
            result.extend(
                _word(
                    part,
                    signature,
                    total,
                    offset=position,
                    deadline=deadline,
                )
            )
            position += part.inputs
        return tuple(result)
    raise NotApplicable(f"unsupported circuit node {node.kind!r}")


def _evaluate(
    word: Word,
    assignment: Mapping[Token, Permutation],
    degree: int,
) -> Permutation:
    result = _identity(degree)
    for token in word:
        result = _compose(assignment[token], result)
    return result


@dataclass(frozen=True)
class _Constraint:
    left: Word
    right: Word
    equal: bool = True

    @property
    def variables(self) -> frozenset[Token]:
        return frozenset((*self.left, *self.right))


def _prop_constraints(
    arity: int,
    generators: tuple[tuple[str, int], ...],
    deadline: Deadline | None = None,
) -> tuple[_Constraint, ...]:
    if deadline is not None:
        deadline.check()
    if arity < 0:
        raise ValueError("PROP arity must be non-negative")
    names = tuple(name for name, _ in generators)
    if len(set(names)) != len(names):
        raise ValueError("PROP generator names must be distinct")
    if any(width < 0 or width > arity for _, width in generators):
        raise ValueError("PROP generator arities must lie in the fixed-arity fragment")

    constraints: list[_Constraint] = []
    for index in range(arity - 1):
        if deadline is not None and index % 256 == 0:
            deadline.check()
        token = _swap(index)
        constraints.append(_Constraint((token, token), ()))
    for index in range(arity - 2):
        if deadline is not None and index % 256 == 0:
            deadline.check()
        left, right = _swap(index), _swap(index + 1)
        constraints.append(_Constraint((left, right, left), (right, left, right)))
    for left_index in range(arity - 1):
        if deadline is not None:
            deadline.check()
        for right_index in range(left_index + 2, arity - 1):
            left, right = _swap(left_index), _swap(right_index)
            constraints.append(_Constraint((left, right), (right, left)))

    for name, width in generators:
        if deadline is not None:
            deadline.check()
        token = _generator(name)
        for index in range(width, arity - 1):
            swap = _swap(index)
            constraints.append(_Constraint((token, swap), (swap, token)))

    checked_pairs = 0
    for index, (left_name, left_width) in enumerate(generators):
        if deadline is not None:
            deadline.check()
        for right_name, right_width in generators[index:]:
            checked_pairs += 1
            if deadline is not None and checked_pairs % 1024 == 0:
                deadline.check()
            if left_width + right_width > arity:
                continue
            left = (_generator(left_name),)
            selected = tuple(range(left_width, left_width + right_width))
            right = _placed(_generator(right_name), selected, arity, deadline)
            constraints.append(_Constraint(left + right, right + left))

    unique: dict[tuple[Word, Word], _Constraint] = {}
    for constraint in constraints:
        if constraint.left != constraint.right:
            unique.setdefault((constraint.left, constraint.right), constraint)
    return tuple(unique.values())


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
    constraint: _Constraint,
    assignment: Mapping[Token, Permutation],
    degree: int,
) -> bool:
    same = _evaluate(constraint.left, assignment, degree) == _evaluate(
        constraint.right, assignment, degree
    )
    return same == constraint.equal


def _validate_assignment(
    variables: tuple[Token, ...],
    constraints: tuple[_Constraint, ...],
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
            f"constraint uses undeclared interpretation {_display_name(missing)}"
        )
    expected = _identity(degree)
    for variable in variables:
        if deadline is not None:
            deadline.check()
        value = assignment.get(variable)
        if value is None:
            raise ValueError(f"missing interpretation for {_display_name(variable)}")
        if len(value) != degree or tuple(sorted(value)) != expected:
            raise ValueError(
                f"{_display_name(variable)} is not a degree-{degree} permutation"
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
    variables = tuple(_generator(name) for name, _ in generators) + tuple(
        _swap(index) for index in range(arity - 1)
    )
    _validate_assignment(
        variables,
        _prop_constraints(arity, generators),
        assignment,
        degree,
    )


def _find_assignment(
    variables: tuple[Token, ...],
    constraints: tuple[_Constraint, ...],
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


def _display_name(token: Token) -> str:
    if token[0] == "generator":
        return str(token[1])
    index = int(token[1])
    return f"swap[{index},{index + 1}]"


def _contains_arity_change(
    node: Circuit, deadline: Deadline | None = None
) -> bool:
    pending = [node]
    visited = 0
    while pending:
        if deadline is not None and visited % 1024 == 0:
            deadline.check()
        current = pending.pop()
        if current.inputs != current.outputs:
            return True
        pending.extend(current.parts)
        visited += 1
    return False


def _require_endomorphic_theory(theory: Any, deadline: Deadline) -> None:
    """Reject inputs for which fixed-wire truncation is not a sound invariant."""

    for index, generator in enumerate(theory.signature.values()):
        if index % 1024 == 0:
            deadline.check()
        if generator.inputs != generator.outputs:
            raise NotApplicable(
                "fixed-arity PROP search requires an endomorphic signature"
            )

    for index, equation in enumerate(theory.equations):
        if index % 256 == 0:
            deadline.check()
        if any(
            _contains_arity_change(term, deadline)
            for term in (equation.lhs, equation.rhs)
        ):
            raise NotApplicable(
                "fixed-arity PROP search requires endomorphic signature terms and macros"
            )
    for definition in getattr(theory, "macros", {}).values():
        deadline.check()
        body = getattr(definition, "body", definition)
        if _contains_arity_change(body, deadline):
            raise NotApplicable(
                "fixed-arity PROP search requires endomorphic signature terms and macros"
            )


def _register_inline_generators(
    term: Circuit,
    signature: dict[str, int],
    deadline: Deadline,
) -> None:
    """Add explicitly typed primitives omitted from the top-level signature."""
    pending = [term]
    visited = 0
    while pending:
        if visited % 1024 == 0:
            deadline.check()
        node = pending.pop()
        if node.kind == "gen":
            name = node.name or ""
            width = int(node.inputs)
            existing = signature.get(name)
            if existing is not None and existing != width:
                raise NotApplicable(
                    f"generator {name!r} is used at inconsistent arities"
                )
            signature[name] = width
        pending.extend(node.parts)
        visited += 1


def candidates(
    theory: Any,
    target: Any,
    *,
    bound: int,
    deadline: Deadline,
    max_permutation_degree: int = MAX_DEGREE,
    **_: Any,
) -> Iterator[CandidateModel]:
    if bound < 2 or bound > max_permutation_degree:
        return
    deadline.check()
    _require_endomorphic_theory(theory, deadline)

    arity = equation_arity(target)
    if arity < 0:
        raise NotApplicable("permutation search requires a non-negative target arity")
    declared_signature = {
        name: generator.inputs
        for name, generator in theory.signature.items()
    }
    for definition in theory.macros.values():
        deadline.check()
        body = getattr(definition, "body", definition)
        _register_inline_generators(body, declared_signature, deadline)
    for equation in theory.equations:
        deadline.check()
        _register_inline_generators(equation.lhs, declared_signature, deadline)
        _register_inline_generators(equation.rhs, declared_signature, deadline)
    _register_inline_generators(target.lhs, declared_signature, deadline)
    _register_inline_generators(target.rhs, declared_signature, deadline)
    others = relevant_equations(theory, target)
    equations = (*others, target)
    expanded: dict[str, tuple[Circuit, Circuit]] = {}
    for equation in equations:
        deadline.check()
        left = expand_macros(equation.lhs, theory.macros)
        right = expand_macros(equation.rhs, theory.macros)
        _register_inline_generators(left, declared_signature, deadline)
        _register_inline_generators(right, declared_signature, deadline)
        expanded[str(equation.id)] = left, right
    signature = {
        name: width
        for name, width in declared_signature.items()
        if width <= arity
    }
    words: dict[str, tuple[Word, Word]] = {}
    for equation in equations:
        deadline.check()
        left, right = expanded[str(equation.id)]
        words[str(equation.id)] = (
            _word(left, signature, arity, deadline=deadline),
            _word(right, signature, arity, deadline=deadline),
        )

    target_id = str(target.id)
    retained_pairs = tuple(
        words[str(equation.id)]
        for equation in others
        if words[str(equation.id)][0] != words[str(equation.id)][1]
    )
    used = {
        token
        for pair in (*retained_pairs, words[target_id])
        for word in pair
        for token in word
        if token[0] == "generator"
    }
    generator_specs = tuple(
        (name, width)
        for name, width in signature.items()
        if _generator(name) in used
    )
    constraints = list(_prop_constraints(arity, generator_specs, deadline))
    for left, right in retained_pairs:
        constraints.append(_Constraint(left, right))
    target_left, target_right = words[target_id]
    constraints.append(_Constraint(target_left, target_right, equal=False))

    variables = tuple(_generator(name) for name, _ in generator_specs) + tuple(
        _swap(index) for index in range(arity - 1)
    )
    solution = _find_assignment(variables, tuple(constraints), bound, deadline)
    if solution is None:
        return
    _validate_assignment(variables, tuple(constraints), solution, bound, deadline)

    identity = _identity(bound)
    full_assignment = {
        _generator(name): solution.get(_generator(name), identity)
        for name in signature
    }
    full_assignment.update(
        {_swap(index): solution[_swap(index)] for index in range(arity - 1)}
    )
    generator_values = {
        name: list(full_assignment[_generator(name)]) for name in signature
    }
    swap_values = {
        _display_name(_swap(index)): list(full_assignment[_swap(index)])
        for index in range(arity - 1)
    }

    def render(values: Mapping[str, list[int]]) -> str:
        return ", ".join(f"{name}={tuple(value)}" for name, value in values.items())

    rendered = f"generators {{{render(generator_values)}}}"
    if swap_values:
        rendered += f"; structural swaps {{{render(swap_values)}}}"

    def evaluate(term: Circuit) -> Permutation:
        expanded = expand_macros(term, theory.macros)
        if expanded.inputs != expanded.outputs or expanded.inputs > arity:
            raise NotApplicable(
                f"permutation interpretation supports endomorphisms through arity {arity}"
            )
        return _evaluate(_word(expanded, signature, arity), full_assignment, bound)

    yield CandidateModel(
        kind="finite_model",
        description=f"degree-{bound} permutation interpretation: {rendered}",
        parameters={
            "arity": arity,
            "degree": bound,
            "interpretation": {
                "generators": generator_values,
                "structural_swaps": swap_values,
            },
        },
        evaluator=evaluate,
        key=(
            arity,
            bound,
            *(full_assignment[variable] for variable in full_assignment),
        ),
    )
