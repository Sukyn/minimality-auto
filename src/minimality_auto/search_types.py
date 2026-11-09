"""Shared candidate models, results, exceptions, and search deadlines."""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np


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

    @property
    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def capped(self, seconds: float) -> Deadline:
        """Limit an attempt to this duration and the enclosing deadline."""
        self.check()
        return Deadline(min(self.expires_at, time.monotonic() + max(0.0, seconds)))


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
        return _json_value(self)


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
        return {**_json_value(self), "elapsed_seconds": round(self.elapsed_seconds, 6)}


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        return _json_value(value.tolist())
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, complex):
        return [_json_value(value.real), _json_value(value.imag)]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value
