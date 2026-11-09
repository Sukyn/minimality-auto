"""Run searches in a separate process to enforce a hard timeout."""

from __future__ import annotations

import math
import multiprocessing
import time
from collections.abc import Callable
from typing import Any

from .search import search_theory
from .search_types import Deadline, SearchReport, Separation
from .syntax import Theory


def _search_worker(sender: Any, theory: Theory, options: dict[str, Any]) -> None:
    with sender:
        report = search_theory(
            theory,
            **options,
            on_witness=lambda witness: sender.send(("witness", witness)),
        )
        sender.send(("result", report))


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
    deadline = Deadline(start + timeout)
    process.start()
    sender.close()
    witnesses: dict[str, Separation] = {}
    try:
        while receiver.poll(deadline.remaining):
            event, value = receiver.recv()
            if event == "result":
                return value
            witnesses[value.equation] = value
            if on_witness is not None:
                on_witness(value)
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
