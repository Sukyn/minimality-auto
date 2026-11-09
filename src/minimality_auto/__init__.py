"""Independence searches for PROP presentations."""

from .parsing import load_theory
from .syntax import PropError, Theory, ValidationError

__all__ = [
    "PropError",
    "Theory",
    "ValidationError",
    "load_theory",
]
