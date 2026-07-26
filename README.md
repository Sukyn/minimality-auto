# minimality-auto

Command-line search for PROP-compatible invariants separating one equation from
the other equations in a JSON equational theory. The tool assumes the supplied
theory is complete.

## Install

```powershell
python -m pip install -e .
```

## Run

```powershell
# Run every heuristic.
minimality-auto THEORY.json --auto

# Run one heuristic.
minimality-auto THEORY.json --presence-checking
minimality-auto THEORY.json --counting --max-modulus 8
minimality-auto THEORY.json --finite-model --max-permutation-degree 5
minimality-auto THEORY.json --affine-lift --max-affine-lift-prime 5
minimality-auto THEORY.json --amalgam --max-amalgam-prime 11
minimality-auto THEORY.json --spin-cover
minimality-auto THEORY.json --block-normal-form
minimality-auto THEORY.json --determinant
minimality-auto THEORY.json --substitution --max-depth 3

# Restrict the search and request JSON output.
minimality-auto THEORY.json --equation RULE_ID --timeout 60 --json
```

`--equation` may be repeated. The timeout is in seconds, defaults to 600, and
is enforced by the CLI in a separate process.

Run `minimality-auto --help` for every option. `python -m minimality_auto` is
equivalent.

## JSON input

```json
{
  "name": "example",
  "wire_dimension": 2,
  "generators": {
    "Z": {
      "type": 1,
      "matrix": {"diagonal": [1, -1]}
    }
  },
  "macros": {
    "Z2": {
      "compose": [{"gen": "Z"}, {"gen": "Z"}]
    }
  },
  "equations": [
    {
      "id": "Z2",
      "lhs": {"macro": "Z2"},
      "rhs": {"id": 1}
    }
  ]
}
```

`generators` and `macros` are objects. A generator type `a` means the
endomorphism `[a, a]`; use `[inputs, outputs]` only for an arity-changing
generator. Use the bare type when no matrix is supplied. Each macro value is
its circuit body, whose type is inferred.
Terms use `gen`, `macro`, `id`, `perm`, `compose`, or `tensor`; composition is
listed in execution order.

Matrices are optional. They are dense arrays, `{"diagonal": [...]}`, or
`{"permutation": [...]}` where entry `i` is the output basis index of input
basis vector `i`. Entries may be numbers or complex strings such as `"1j"`.
Matrices are used by the affine-lift, amalgam, block-normal-form, determinant,
substitution, and Spin-cover heuristics. More examples are in
[`theories/`](theories).

In compact `{"wires":...,"ops":[...]}` circuits, `on` is optional for
nullary and whole-register operations; a unary placement may use `on: 2`
instead of `on: [2]`. An endomorphic generator, macro, or permutation may use
`"repeat": n`.

Floating-point results are shown as `[numerical]` and serialized with
`"certified": false`; exact combinatorial results are shown as `[found]`.
The affine-lift heuristic requires exact `0`/`1` basis-permutation matrices.

TODO: optionally support slower SymPy arithmetic for exact matrix certificates.

The finite-permutation and affine-lift heuristics currently support endomorphic
signatures only. Their fixed-arity PROP constraints are summarized in
[`docs/permutation-models.md`](docs/permutation-models.md).

## Test

```powershell
python -m pip install -e ".[test]"
python -m ruff check src tests
python -m pytest
```
