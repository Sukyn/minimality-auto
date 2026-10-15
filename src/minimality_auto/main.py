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
    DEFAULT_TIMEOUT,
    LIMIT_SPECS,
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
        "--equation", action="append", help="search only this equation ID; repeatable"
    )
    parser.add_argument("--max-arity", type=int, help="search only equations at most this arity")
    for limit in LIMIT_SPECS:
        parser.add_argument(
            *limit.flags,
            dest=limit.name,
            type=int,
            default=limit.default,
            help=f"{limit.help} (default: {limit.default})",
        )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"global timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
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
    chosen = tuple(name for name in DEFAULT_STRATEGIES if getattr(args, name))
    return chosen or DEFAULT_STRATEGIES


def _search_worker(sender: Any, theory: Theory, options: dict[str, Any]) -> None:
    try:
        report = search_theory(
            theory,
            **options,
            on_witness=lambda witness: sender.send(("witness", witness)),
        )
        sender.send(("result", report))
    # Forward worker exceptions so the parent can report the search failure.
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
            **{limit.name: getattr(args, limit.name) for limit in LIMIT_SPECS},
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
        certified = sum(witness.certified for witness in report.witnesses.values())
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
