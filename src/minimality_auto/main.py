from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .core import load_theory
from .search import DEFAULT_STRATEGIES, search_theory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minimality-auto",
        description=(
            "Search for PROP-compatible invariants separating equations from the other axioms."
        ),
    )
    parser.add_argument("theory", type=Path, help="JSON equational theory")
    parser.add_argument(
        "--auto", action="store_true", help="run every available heuristic (default)"
    )
    parser.add_argument(
        "--presence-checking",
        "--presence_checking",
        "--max-occurrence",
        dest="presence",
        action="store_true",
        help="run Boolean max/occurrence models",
    )
    parser.add_argument("--counting", action="store_true", help="run modular counting models")
    parser.add_argument("--substitution", action="store_true", help="run projective substitutions")
    parser.add_argument(
        "--determinant",
        action="store_true",
        help="run scaled determinant-phase models",
    )
    parser.add_argument(
        "--amalgam",
        action="store_true",
        help="search finite factors and free-product-with-amalgamation models",
    )
    parser.add_argument(
        "--max-amalgam-prime",
        type=int,
        default=11,
        help="largest finite-field characteristic for amalgam search (default: 11)",
    )
    parser.add_argument(
        "--max-amalgam-order",
        type=int,
        default=4096,
        help="largest finite factor or shared-subgroup order (default: 4096)",
    )
    parser.add_argument(
        "--max-amalgam-bridge-generators",
        type=int,
        default=1,
        help="largest active-generator bridge set (default: 1)",
    )
    parser.add_argument(
        "--max-amalgam-scalars",
        type=int,
        default=3,
        help="largest number of finite-field template scalars (default: 3)",
    )
    parser.add_argument(
        "--max-amalgam-matrix-dimension",
        type=int,
        default=32,
        help="largest ambient finite-field matrix dimension (default: 32)",
    )
    parser.add_argument(
        "--spin-cover",
        "--spin_cover",
        dest="spin",
        action="store_true",
        help="run Spin-cover commutator certificates",
    )
    parser.add_argument(
        "--finite-model",
        "--finite_model",
        "--permutation-model",
        dest="finite_model",
        action="store_true",
        help="search small permutation interpretations",
    )
    parser.add_argument(
        "--equation", action="append", help="search only this equation ID; repeatable"
    )
    parser.add_argument("--max-arity", type=int, help="search only equations at most this arity")
    parser.add_argument(
        "--max-modulus",
        type=int,
        default=8,
        help="largest counting modulus (default: 8)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="largest substitution word length (default: 3)",
    )
    parser.add_argument(
        "--max-substitution-matrix-entries",
        type=int,
        default=1_000_000,
        help="largest dense substitution matrix footprint (default: 1000000)",
    )
    parser.add_argument(
        "--max-permutation-degree",
        "--max_permutation_degree",
        "--max-degree",
        type=int,
        default=5,
        help="largest permutation degree (default: 5)",
    )
    parser.add_argument(
        "--max-spin-matrix-dimension",
        type=int,
        default=1024,
        help="largest dense Spin matrix dimension (default: 1024)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="global timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="emit machine-readable results",
    )
    return parser


def _selected_strategies(args: argparse.Namespace) -> tuple[str, ...]:
    if args.auto:
        return DEFAULT_STRATEGIES
    chosen = tuple(
        name for name in DEFAULT_STRATEGIES if getattr(args, name)
    )
    return chosen or DEFAULT_STRATEGIES


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        theory = load_theory(args.theory)
        report = search_theory(
            theory,
            strategies=_selected_strategies(args),
            equation_ids=set(args.equation) if args.equation else None,
            max_arity=args.max_arity,
            timeout=args.timeout,
            max_modulus=args.max_modulus,
            max_depth=args.max_depth,
            max_substitution_matrix_entries=args.max_substitution_matrix_entries,
            max_permutation_degree=args.max_permutation_degree,
            max_amalgam_prime=args.max_amalgam_prime,
            max_amalgam_order=args.max_amalgam_order,
            max_amalgam_bridge_generators=args.max_amalgam_bridge_generators,
            max_amalgam_scalars=args.max_amalgam_scalars,
            max_amalgam_matrix_dimension=args.max_amalgam_matrix_dimension,
            max_spin_matrix_dimension=args.max_spin_matrix_dimension,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    if args.json_output:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"{report.theory}: {len(report.witnesses)} separator(s) found")
        for equation, witness in report.witnesses.items():
            print(f"  [found] {equation}: {witness.description}")
        for equation in report.unresolved:
            print(f"  [open]  {equation}")
        suffix = " (timeout)" if report.timed_out else ""
        print(f"elapsed: {report.elapsed_seconds:.3f}s{suffix}")

    if report.timed_out:
        return 124
    return 0 if not report.unresolved else 1


if __name__ == "__main__":
    sys.exit(main())
