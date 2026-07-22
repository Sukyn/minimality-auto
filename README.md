# minimality-auto

Search for PROP-compatible invariants separating one equation from the other equations in a JSON equational theory. The tool assumes the supplied theory is complete.

## Install

```powershell
python -m pip install -e ".[test]"
```

## Run

```powershell
# Run every heuristic.
python -m minimality_auto THEORY.json --auto

# Run one heuristic.
python -m minimality_auto THEORY.json --presence-checking
python -m minimality_auto THEORY.json --counting --max-modulus 8
python -m minimality_auto THEORY.json --finite-model --max-permutation-degree 5
python -m minimality_auto THEORY.json --amalgam --max-amalgam-prime 11
python -m minimality_auto THEORY.json --spin-cover
python -m minimality_auto THEORY.json --determinant
python -m minimality_auto THEORY.json --substitution --max-depth 3

# Restrict the search and request JSON output.
python -m minimality_auto THEORY.json --equation RULE_ID --timeout 60 --json
```

`--equation` may be repeated. The timeout is in seconds and defaults to 600.

Run `python -m minimality_auto --help` for every option.

## JSON input

```json
{
  "name": "example",
  "generators": {
    "H": [1, 1]
  },
  "equations": [
    {
      "id": "H2",
      "lhs": {"compose": [{"gen": "H"}, {"gen": "H"}]},
      "rhs": {"id": 1}
    }
  ]
}
```

Generator types are `[inputs, outputs]`. Terms use `gen`, `macro`, `id`, `perm`, `compose`, or `tensor`; composition is listed in execution order. Matrices are optional and are used by the amalgam, determinant, substitution, and Spin-cover heuristics. More complete inputs are available in [`theories/`](theories).

The finite-permutation heuristic currently supports endomorphic signatures only. Its PROP constraints are summarized in [`docs/permutation-models.md`](docs/permutation-models.md).

## Test

```powershell
python -m pytest
```
