from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import minimality_auto.main as main_module
from minimality_auto.main import main
from minimality_auto.search import SearchReport, Separation


def _write_theory(path):
    path.write_text(
        json.dumps(
            {
                "name": "cli_example",
                "generators": {"a": [1, 1]},
                "equations": [
                    {"id": "drop", "lhs": {"gen": "a"}, "rhs": {"id": 1}}
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("flag", ["--presence-checking", "--presence_checking"])
def test_presence_cli_aliases(tmp_path, capsys, flag):
    path = tmp_path / "theory.json"
    _write_theory(path)
    assert main([str(path), flag, "--equation", "drop", "--timeout", "2"]) == 0
    assert "[found] drop" in capsys.readouterr().out


def test_cli_json_output(tmp_path, capsys):
    path = tmp_path / "theory.json"
    path.write_text(
        json.dumps(
            {
                "name": "cli_json",
                "generators": {"a": [1, 1]},
                "equations": [
                    {"id": "drop", "lhs": {"gen": "a"}, "rhs": {"id": 1}}
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main([str(path), "--presence-checking", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["witnesses"]["drop"]["strategy"] == "presence"


@pytest.mark.parametrize("degree_flag", ["--max-permutation-degree", "--max-degree"])
def test_finite_model_cli_alias_and_degree_bound(tmp_path, capsys, degree_flag):
    path = tmp_path / "finite.json"
    path.write_text(
        json.dumps(
            {
                "generators": {"H": [1, 1], "Z": [1, 1]},
                "equations": [
                    {
                        "id": "1",
                        "lhs": {"compose": [{"gen": "H"}, {"gen": "H"}]},
                        "rhs": {"id": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(
        [str(path), "--finite-model", degree_flag, "3", "--timeout", "2"]
    ) == 0
    output = capsys.readouterr().out
    assert "degree-3 permutation interpretation: generators {H=" in output


def test_cli_labels_numerical_evidence_without_reporting_success(
    tmp_path, capsys, monkeypatch
):
    path = tmp_path / "theory.json"
    _write_theory(path)
    evidence = Separation(
        equation="drop",
        strategy="determinant",
        description="floating-point evidence",
        parameters={},
        checked_equations=(),
        lhs_value=0,
        rhs_value=1,
        certified=False,
    )
    monkeypatch.setattr(
        main_module,
        "search_theory",
        lambda *_args, **_kwargs: SearchReport(
            "cli_example", {"drop": evidence}, ("drop",), 0.0, False
        ),
    )

    assert main([str(path), "--determinant"]) == 1
    output = capsys.readouterr().out
    assert "[numerical] drop" in output
    assert "[found] drop" not in output
    assert "[open]  drop" not in output


def test_cli_does_not_mask_internal_value_errors(tmp_path, monkeypatch):
    path = tmp_path / "theory.json"
    _write_theory(path)

    def fail(*_args, **_kwargs):
        raise ValueError("internal bug")

    monkeypatch.setattr(main_module, "search_theory", fail)
    with pytest.raises(ValueError, match="internal bug"):
        main([str(path), "--presence-checking"])


def test_cli_reports_search_configuration_errors(tmp_path, capsys):
    path = tmp_path / "theory.json"
    _write_theory(path)
    with pytest.raises(SystemExit) as error:
        main([str(path), "--equation", "missing"])
    assert error.value.code == 2
    assert "unknown or filtered equation" in capsys.readouterr().err


def test_real_cli_enforces_a_hard_timeout(tmp_path):
    path = tmp_path / "theory.json"
    _write_theory(path)
    root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "minimality_auto",
            str(path),
            "--presence-checking",
            "--timeout",
            "0.001",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 124
    assert "(timeout)" in result.stdout


def test_hard_timeout_reports_worker_exit_without_a_result(monkeypatch):
    theory = main_module.Theory.from_json(
        {
            "generators": {"a": 1},
            "equations": [
                {"id": "drop", "lhs": {"gen": "a"}, "rhs": {"id": 1}}
            ],
        }
    )

    class Receiver:
        closed = False

        def poll(self, _timeout):
            return True

        def recv(self):
            raise EOFError

        def close(self):
            self.closed = True

    class Sender:
        def close(self):
            pass

    class Process:
        exitcode = 7
        joins = 0

        def start(self):
            pass

        def join(self):
            self.joins += 1

        def is_alive(self):
            return False

        def terminate(self):
            raise AssertionError("completed worker must not be terminated")

    receiver = Receiver()
    process = Process()

    class Context:
        def Pipe(self, *, duplex):
            assert not duplex
            return receiver, Sender()

        def Process(self, **_kwargs):
            return process

    monkeypatch.setattr(
        main_module,
        "search_theory",
        lambda *_args, **_kwargs: SearchReport(
            "theory", {}, ("drop",), 0.0, True
        ),
    )
    monkeypatch.setattr(main_module.multiprocessing, "get_context", lambda _mode: Context())

    with pytest.raises(main_module.PropError, match="exit code 7"):
        main_module._timed_search(
            theory,
            {
                "timeout": 1.0,
                "equation_ids": None,
                "max_arity": None,
            },
        )

    assert receiver.closed
    assert process.joins >= 1
