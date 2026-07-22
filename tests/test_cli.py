from __future__ import annotations

import json

import pytest

from minimality_auto.main import main


@pytest.mark.parametrize("flag", ["--presence-checking", "--presence_checking"])
def test_presence_cli_aliases(tmp_path, capsys, flag):
    path = tmp_path / "theory.json"
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
