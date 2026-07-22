"""Typed circuits and independence-separation searches for PROPs."""

from .core import (
    Circuit,
    Equation,
    Generator,
    MacroDef,
    MatrixSemantics,
    PropError,
    Signature,
    Theory,
    ValidationError,
    evaluate_matrix,
    expand_macros,
    inversion_count,
    load_json,
    load_theory,
    parse_complex,
    parse_complex_matrix,
    permutation_parity,
    primitive_occurrences,
    structural_permutation_parity,
)

__all__ = [
    "Circuit",
    "Equation",
    "Generator",
    "MacroDef",
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

