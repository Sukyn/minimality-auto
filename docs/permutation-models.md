# Permutation heuristic

For a target on `n` wires, this heuristic searches for an interpretation in a finite symmetric group `S_d`. It derives all variables and constraints from the JSON theory and tries degrees in increasing order.

The heuristic applies only when every primitive generator is an endomorphism. Wire count is then preserved by rewriting, so an `n`-wire target needs only axioms of arity at most `n`.

For each degree, the search assigns a permutation to every relevant primitive generator and adjacent structural swap. It enforces:

- the Coxeter relations for structural swaps;
- routing independence on unused wires;
- interchange for disjoint generator placements, including centrality of scalars;
- every non-target axiom usable at the target arity;
- inequality of the target terms.

Assignments are searched with constraint filtering, a most-constrained-variable order, and simultaneous-conjugacy reduction. Reported JSON separates primitive `generators` from `structural_swaps` and retains the generator names from the input.

Use:

```powershell
python -m minimality_auto THEORY.json --finite-model --max-permutation-degree 5
```

The search is factorial in the degree and obeys the global `--timeout`. Failure within a finite degree bound is not a derivability proof. Arity-changing signatures are rejected by this heuristic.
