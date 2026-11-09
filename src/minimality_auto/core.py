"""Public circuit API; implementation lives in syntax, parsing, and analysis."""

from .analysis import (
    CircuitAnalysis,
    analyze_circuit,
    contains_arity_change,
    iter_circuit_nodes,
    iter_theory_terms,
    permutation_parity,
)
from .parsing import MacroResolver, load_json, load_theory, parse_complex
from .syntax import Circuit, Equation, Generator, PropError, Signature, Theory, ValidationError

__all__ = [
    "Circuit", "CircuitAnalysis", "Equation", "Generator", "MacroResolver",
    "PropError", "Signature", "Theory", "ValidationError", "analyze_circuit",
    "contains_arity_change", "iter_circuit_nodes", "iter_theory_terms",
    "load_json", "load_theory", "parse_complex", "permutation_parity",
]
