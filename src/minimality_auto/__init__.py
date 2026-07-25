"""CLI-first independence-separation searches for PROP presentations."""

from .core import PropError, Theory, ValidationError, load_theory

__all__ = [
    "PropError",
    "Theory",
    "ValidationError",
    "load_theory",
]

__version__ = "0.1.0"
