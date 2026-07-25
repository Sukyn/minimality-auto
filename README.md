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
python -m minimality_auto THEORY.json --block-normal-form
python -m minimality_auto THEORY.json --determinant
python -m minimality_auto THEORY.json --substitution --max-depth 3

# Restrict the search and request JSON output.
python -m minimality_auto THEORY.json --equation RULE_ID --timeout 60 --json
```

`--equation` may be repeated. The timeout is in seconds, defaults to 600, and
is enforced by the CLI in a separate process.

Run `python -m minimality_auto --help` for every option.

## JSON input

```json
{
  "name": "example",
  "generators": {
    "H": 1
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

The type `a` is shorthand for the endomorphism `[a, a]`; use
`[inputs, outputs]` for arity-changing generators. Macros map names directly
to circuit bodies, with inferred types. Terms use `gen`, `macro`, `id`,
`perm`, `compose`, or `tensor`; composition is listed in execution order.
Matrices are optional and may be dense arrays or `{"diagonal":[...]}`. They
are used by the amalgam, block-normal-form, determinant, substitution, and
Spin-cover heuristics. More complete inputs are available in
[`theories/`](theories).

In compact `{"wires":...,"ops":[...]}` circuits, `on` is optional for
nullary and whole-register operations; a unary placement may use `on: 2`
instead of `on: [2]`.

Floating-point results are shown as `[numerical]` and serialized with
`"certified": false`; exact combinatorial results are shown as `[found]`.

TODO: optionally support slower SymPy arithmetic for exact matrix certificates.

The finite-permutation heuristic currently supports endomorphic signatures only. Its PROP constraints are summarized in [`docs/permutation-models.md`](docs/permutation-models.md).

## Test

```powershell
python -m ruff check src tests
python -m pytest
```
