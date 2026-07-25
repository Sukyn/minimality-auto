from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .core import PropError, Theory, load_theory
from .search import (
    DEFAULT_STRATEGIES,
    STRATEGY_SPECS,
    SearchConfigurationError,
    SearchReport,
    Separation,
    search_theory,
)


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
    for spec in STRATEGY_SPECS:
        parser.add_argument(
            *spec.flags,
            dest=spec.name,
            action="store_true",
            help=spec.help,
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
        "--max-block-matrix-dimension",
        type=int,
        default=512,
        help="largest dense block-normal-form matrix dimension (default: 512)",
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


def _search_worker(sender: Any, theory: Theory, options: dict[str, Any]) -> None:
    try:
        report = search_theory(
            theory,
            **options,
            on_witness=lambda witness: sender.send(("witness", witness)),
        )
        sender.send(("result", report))
    # The parent must receive internal failures instead of mistaking them for
    # a timeout. This process boundary is the one intentional broad catch.
    except Exception as error:  # noqa: BLE001
        sender.send(("error", error))
    finally:
        sender.close()


def _timed_search(
    theory: Theory,
    options: dict[str, Any],
    on_witness: Callable[[Separation], None] | None = None,
) -> SearchReport:
    """Run the CLI search in a process so its timeout can interrupt native code."""

    timeout = float(options["timeout"])
    if not math.isfinite(timeout) or timeout < 0:
        return search_theory(theory, **options)
    preflight_options = {**options, "timeout": 0.0}
    preflight = search_theory(theory, **preflight_options)
    if timeout == 0 or not preflight.timed_out:
        if on_witness is not None:
            for witness in preflight.witnesses.values():
                on_witness(witness)
        return preflight

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_search_worker,
        args=(sender, theory, options),
        daemon=True,
    )
    start = time.monotonic()
    process.start()
    sender.close()
    witnesses: dict[str, Separation] = {}
    try:
        while receiver.poll(max(0.0, timeout - (time.monotonic() - start))):
            try:
                event, value = receiver.recv()
            except EOFError as exc:
                process.join()
                raise PropError(
                    "search worker exited without returning a result "
                    f"(exit code {process.exitcode})"
                ) from exc
            if event == "witness":
                witnesses[value.equation] = value
                if on_witness is not None:
                    on_witness(value)
                continue
            process.join()
            if event == "result":
                return value
            if event == "error":
                raise value
            raise PropError(f"unknown search worker event: {event!r}")
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        receiver.close()

    return SearchReport(
        theory.name or "theory",
        witnesses,
        tuple(
            equation
            for equation in preflight.unresolved
            if equation not in witnesses or not witnesses[equation].certified
        ),
        time.monotonic() - start,
        True,
    )


def _print_witness(witness: Separation) -> None:
    label = "found" if witness.certified else "numerical"
    print(
        f"  [{label}] {witness.equation}: {witness.description}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    hard_timeout = argv is None
    args = parser.parse_args(argv)

    try:
        theory = load_theory(args.theory)
        options = {
            "strategies": _selected_strategies(args),
            "equation_ids": set(args.equation) if args.equation else None,
            "max_arity": args.max_arity,
            "timeout": args.timeout,
            "max_modulus": args.max_modulus,
            "max_depth": args.max_depth,
            "max_substitution_matrix_entries": (
                args.max_substitution_matrix_entries
            ),
            "max_permutation_degree": args.max_permutation_degree,
            "max_amalgam_prime": args.max_amalgam_prime,
            "max_amalgam_order": args.max_amalgam_order,
            "max_amalgam_bridge_generators": (
                args.max_amalgam_bridge_generators
            ),
            "max_amalgam_scalars": args.max_amalgam_scalars,
            "max_amalgam_matrix_dimension": args.max_amalgam_matrix_dimension,
            "max_spin_matrix_dimension": args.max_spin_matrix_dimension,
            "max_block_matrix_dimension": args.max_block_matrix_dimension,
        }
        emitted: set[str] = set()

        def emit(witness: Separation) -> None:
            _print_witness(witness)
            emitted.add(witness.equation)

        on_witness = None if args.json_output else emit
        report = (
            _timed_search(theory, options, on_witness)
            if hard_timeout
            else search_theory(theory, **options, on_witness=on_witness)
        )
    except (OSError, PropError, SearchConfigurationError) as error:
        parser.exit(2, f"error: {error}\n")

    if args.json_output:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        certified = sum(
            witness.certified for witness in report.witnesses.values()
        )
        numerical = len(report.witnesses) - certified
        summary = f"{report.theory}: {certified} separator(s) found"
        if numerical:
            summary += f", {numerical} numerical candidate(s)"
        print(summary)
        for equation, witness in report.witnesses.items():
            if equation not in emitted:
                _print_witness(witness)
        for equation in report.unresolved:
            if equation not in report.witnesses:
                print(f"  [open]  {equation}")
        suffix = " (timeout)" if report.timed_out else ""
        print(f"elapsed: {report.elapsed_seconds:.3f}s{suffix}")

    if report.timed_out:
        return 124
    return 0 if not report.unresolved else 1


if __name__ == "__main__":
    sys.exit(main())
