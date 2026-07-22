from __future__ import annotations

import json

import numpy as np
import pytest

from minimality_auto.core import Theory
from minimality_auto.search import Separation, search_theory
from minimality_auto.separators import counting, determinant, spin


def _theory() -> Theory:
    return Theory.from_json(
        {
            "generators": {"g": [1, 1]},
            "equations": [
                {"id": "target", "lhs": {"gen": "g"}, "rhs": {"id": 1}}
            ],
        }
    )


def test_programmatic_search_rejects_unknown_strategies_cleanly():
    with pytest.raises(ValueError, match="unknown separation strategy: mystery"):
        search_theory(_theory(), strategies=("mystery",))


def test_search_runs_only_bounds_needed_by_selected_strategies(monkeypatch):
    bounds: list[int] = []

    def candidates(*_, bound, **__):
        bounds.append(bound)
        yield from ()

    monkeypatch.setattr(spin, "candidates", candidates)
    search_theory(_theory(), strategies=("spin",), timeout=2)

    assert bounds == [0]


def test_each_strategy_stops_at_its_own_bound(monkeypatch):
    bounds: dict[str, list[int]] = {"counting": [], "determinant": []}

    def recorder(name):
        def candidates(*_, bound, **__):
            bounds[name].append(bound)
            yield from ()

        return candidates

    monkeypatch.setattr(counting, "candidates", recorder("counting"))
    monkeypatch.setattr(determinant, "candidates", recorder("determinant"))
    search_theory(
        _theory(),
        strategies=("counting", "determinant"),
        max_modulus=7,
        timeout=2,
    )

    assert bounds == {
        "counting": list(range(8)),
        "determinant": list(range(4)),
    }


@pytest.mark.parametrize(
    "option",
    [
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": -1},
        {"max_arity": -1},
        {"max_modulus": 1},
        {"max_depth": -1},
        {"max_substitution_matrix_entries": 0},
        {"max_permutation_degree": 1},
        {"max_amalgam_prime": 1},
        {"max_amalgam_order": 0},
        {"max_amalgam_bridge_generators": 0},
        {"max_amalgam_scalars": -1},
        {"max_amalgam_matrix_dimension": 0},
        {"max_spin_matrix_dimension": 0},
        {"max_depth": True},
    ],
)
def test_programmatic_search_rejects_invalid_resource_bounds(option):
    with pytest.raises(ValueError):
        search_theory(_theory(), strategies=("presence",), **option)


def test_report_serialization_recurses_through_numpy_and_complex_values():
    separation = Separation(
        equation="target",
        strategy="test",
        description="nested values",
        parameters={"matrix": np.asarray([[1 + 2j, complex(float("nan"), 0)]])},
        checked_equations=(),
        lhs_value=np.asarray([1 + 0j]),
        rhs_value={"values": (2 - 3j,)},
    )

    payload = separation.as_dict()
    assert payload["parameters"]["matrix"][0][0] == [1.0, 2.0]
    assert payload["parameters"]["matrix"][0][1] == ["nan", 0.0]
    assert payload["lhs_value"] == [[1.0, 0.0]]
    assert payload["rhs_value"] == {"values": [[2.0, -3.0]]}
    json.dumps(payload, allow_nan=False)
