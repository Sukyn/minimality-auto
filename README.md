# minimality-auto

Command-line search for PROP-compatible invariants separating one equation from
the other equations in a JSON equational theory.

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
minimality-auto THEORY.json --abelianisation
minimality-auto THEORY.json --finite-model --max-permutation-degree 5
minimality-auto THEORY.json --transformation-monoid --max-transformation-degree 4
minimality-auto THEORY.json --relations --max-matrix-dimension 2
minimality-auto THEORY.json --finite-field-matrices --max-matrix-prime 5
minimality-auto THEORY.json --dual-numbers --max-matrix-prime 5
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

The finite-permutation, transformation-monoid, and affine-lift heuristics support
endomorphic signatures only. The three new matrix searches support arbitrary
input/output arities and do not require supplied matrices.

## Exact finite models and abelianisation

`--dual-numbers` searches matrices over `F_p[epsilon]/(epsilon^2)`. Entries are
pairs `[a,b]` meaning `a + b*epsilon`; multiplication uses `epsilon^2 = 0`.
`--finite-field-matrices` searches arbitrary matrices over `F_p`, independently
for each primitive generator. The base fields currently have prime order.
`--relations` uses Boolean matrices: composition is OR of ANDs, and tensor is
the Cartesian product of relations. Empty, partial, and multivalued relations
are allowed.

For these three searches a generator `m -> n` receives a matrix with `d^n` rows
and `d^m` columns. The model wire dimension `d` is independent of the input's
`wire_dimension`. Identities and permutations use their standard matrices.
These constructions therefore satisfy the symmetric PROP laws, including
naturality for arity-changing maps.

`--transformation-monoid` searches arbitrary maps of a finite set, including
nonbijective maps. Like the permutation search, it interprets a fixed ambient
arity through the word compiler. Structural swaps remain invertible; the
solver enforces all compiled PROP constraints and retained equations.

Z3 solves the finite interpretation constraints. The common verifier checks
each candidate with exact Python integer/Boolean arithmetic; transformation
models also check the compiled PROP laws. JSON output includes the active
generator interpretation and, for transformation models, the structural swaps.

The defaults are `--max-matrix-dimension 2`, `--max-matrix-prime 5`,
`--max-matrix-entries 256`, `--max-transformation-degree 4`, and
`--model-timeout-ms 1000`. The entry cap applies to intermediate matrices too.
The per-attempt timeout covers constraint construction and solving;
`--timeout` limits the entire search, including verification. Increase these
bounds for larger models, for example:

```powershell
minimality-auto THEORY.json --dual-numbers --max-matrix-dimension 4 --max-matrix-prime 5 --max-matrix-entries 4096 --model-timeout-ms 10000
```

All allowed matrix entries and transformations are solver variables, rather
than a list of supplied patterns. Matrix search advances by increasing
`max(p,d)` (by `d` for relations); transformation degree increases from 2.
An attempt that times out or returns unknown leaves the equation unresolved.
Failure within these finite bounds is not a dependence proof.

`--abelianisation` computes the full additive quotient of primitive occurrence
counts by retained equation differences using integer Smith reduction. It
keeps both free integer coordinates and torsion invariant factors, without a
modulus ceiling. It also includes the swap sign with order two when every
generator preserves wire parity. JSON gives coordinate moduli (`0` for a free
coordinate) and each generator's image. This is complete for this additive
invariant family; it does not capture all noncommutative consequences of a
theory. The older bounded `--counting` option remains available.

All five methods are included in `--auto`. Abelianisation runs after the
initial presence attempts. Finite synthesis is interleaved with the remaining
bounded strategies, with a separate time cap on each synthesis attempt.

Small distinguishing examples are in [`theories/finite_models/`](theories/finite_models).

## Code layout

- `syntax.py`, `parsing.py`, and `analysis.py`: typed circuits, JSON input, and
  structural analysis. `core.py` preserves the public circuit imports.
- `search_types.py`, `context.py`, and `strategies.py`: candidate/result types,
  deadlines, cached target information, configuration, and scheduling.
  `search.py` verifies candidates and runs the search.
- `presentation.py`: shared fixed-arity compilation, arity checks, and PROP constraints.
- `algebra/`: finite-field and semiring matrices, shared integer Smith reduction
  for counting and abelianisation, shared finite-field row reduction for inverses
  and affine lifts, finite groups, and amalgam normal forms.
- `separators/`: candidate construction. `matrices.py` contains the three finite
  matrix methods; `actions.py` shares evaluation, validation, and certificates
  between permutation and transformation searches. `synthesis.py` provides
  shared Z3 time budgets and solving.
- `main.py` and `runner.py`: CLI handling and process-based timeouts.

Validation focuses on JSON input, mathematical assumptions, witness verification,
and search budgets. Internal helpers trust data built by the application;
unexpected programming errors raise normally.

## Test

```powershell
python -m pip install -e ".[test]"
python -m ruff check src tests
python -m pytest
```
