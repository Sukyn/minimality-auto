"""Typed circuits and independence-separation searches for PROPs."""

from .core import (
    Circuit,
    Equation,
    Generator,
    PropError,
    Signature,
    Theory,
    ValidationError,
    expand_macros,
    inversion_count,
    load_json,
    load_theory,
    parse_complex,
    permutation_parity,
    primitive_occurrences,
    structural_permutation_parity,
)
from .matrix import MatrixSemantics, evaluate_matrix, parse_complex_matrix

__all__ = [
    "Circuit",
    "Equation",
    "Generator",
    "MatrixSemantics",
    "PropError",
    "Signature",
    "Theory",
    "ValidationError",
    "evaluate_matrix",
    "expand_macros",
    "inversion_count",
    "load_json",
    "load_theory",
    "parse_complex",
    "parse_complex_matrix",
    "permutation_parity",
    "primitive_occurrences",
    "structural_permutation_parity",
]

__version__ = "0.1.0"
