# TODO / future ideas

## `pretty_indices` and `generate_einsum` rename indices twice

`substitute_dummies_double_vac(expr, pretty_indices=...)` and
`code_generator.generate_einsum` both rename indices, and the second rename is
applied on top of the first. `_psi4numpy_indices` is only injective on the
canonical names (a, b, i, j, p, q), so anything else is corrupted silently:

| custom name | after `_psi4numpy_indices` |
|---|---|
| `r` (particle) | `r` |
| `a` (hole) | `r` — collides with the above |
| `alpha` | `rlphr` |
| `vir` | `var` |

Naming the dummies psi4numpy-style up front is the obvious thing to reach for
when the goal is psi4numpy-style code, and it is exactly what breaks:

```python
expr = substitute_dummies_double_vac(expr, pretty_indices={
    "above_molA": "r", "above_molB": "s",
    "below_molA": "a", "below_molB": "b",
})
generate_einsum(expr)      # +np.einsum("rsrs,rsrs", t_rsrs, v_rsrs)
```

Four distinct indices collapse to two and the einsum contracts the wrong pairs.
Nothing raises. With the default names the same expression gives the correct
`+np.einsum("rsab,abrs", t_rsab, v_abrs)`.

Pinned by the `xfail` `test_psi4_indices_name_collision`. Documented in the
docstrings of `substitute_dummies_double_vac`, `generate_einsum` and
`_psi4numpy_indices`, and in `CLAUDE.md`, but not yet fixed. Worth deciding
between: rejecting non-canonical names in `generate_einsum` (cheap, turns a
silent wrong answer into an error), keying the rename off the index assumptions
rather than the name string (robust, no longer depends on the name at all), or
making the psi4numpy convention a `generate_einsum` option so the two renames
are never both applied.

Two details for whoever picks this up. The `xfail` is strict, so a fix — whether
deliberate or incidental — turns the suite red with `[XPASS(strict)]`; that is
the signal to drop the marker and the warnings in the three docstrings and in
`CLAUDE.md`. And `PRETTY_INDICES` no longer carries `i`/`j` rows, because
`_psi4numpy_indices` maps `i -> a` and `j -> b` before the table is consulted;
they have to come back if the psi4numpy rename becomes optional.


## UHF spin-summation

Adding spin summation rules for UHF refference.


## `substitute_dummies_double_vac` is not deterministic

`get_R_nm(2, 0, get_a_operator(n=2))` usually returns a 4-term expression but
intermittently returns **`Zero`**, across fresh processes. The zero rate is
itself unstable: three separate 25-30 run batches measured 20%, 17% and 7%.

Localized by bisection:

| Stage | Stable? |
|---|---|
| `wicks_double_vac(..., substitute_dummies=False)` | yes — 20/20 runs give the same 4 terms |
| `_use_symmetries` / the `e` denominator index order | yes — identical in 20/20 runs |
| `substitute_dummies_double_vac` (`double_fermi_vac.py:516`) | **no** — intermittently collapses the 4 terms to `Zero` |

`get_R_nm` at `(1,0)`, `(0,1)`, `(1,1)` and `get_R_nm(2, 0, V)` were each stable
over 25 runs, so this is specific to the `n ≥ 2` path, whose indices carry a
monomer tag but no Fermi-level assumption.

### Root cause

`_get_ordered_dummies_double_vac` ends with `sorted(term.atoms(Dummy),
key=_get_key)`. `atoms` returns a **set**, and `_get_key` is not a total order:
two dummies of the same type sitting in equivalent positions get the same key.
`sorted` is stable, so ties keep the set's iteration order — which is random
per process. Two independent randomness sources feed it, which is why pinning
only one made this look like "not simple hash ordering":

- the ordinary string hash (`PYTHONHASHSEED`),
- sympy's `Dummy._base_dummy_index`, drawn once per process from
  `random.Random().randint(10**6, 9*10**6)` and mixed into every dummy's hash.

Pinning both makes the derivation completely deterministic:

| pinned | distinct results / `Zero`, out of 40 runs |
|---|---|
| nothing | many / 4 |
| `PYTHONHASHSEED` only | 11 / 1 |
| `Dummy._base_dummy_index` only | many / 4 |
| both | **1 / 0** (and 1 / 0 over 60 further runs) |

When a tie breaks the other way, a term is canonicalized onto the form of a
sibling term and the two cancel — which is why the failure mode is `Zero`
rather than a wrong-but-nonzero answer.

The fix is a total ordering key: break ties on something intrinsic and
process-independent (the dummy's `name`, its position in the term) instead of
letting the set decide. The same key degeneracy is pinned from another angle by
the `xfail` `test_dummy_ordering_uses_normal_ordered_operators`, where the
operators inside a normal ordering bracket contribute nothing to the key at all.

**Ruled out:** the temporary-symbol branch at `double_fermi_vac.py:667-679`,
the earlier prime suspect. Its guard is `if v in subsdict`, where `v` is a
replacement dummy created fresh inside the call while the keys are the
expression's own dummies; `Dummy` equality includes `dummy_index`, so the guard
is never true and the branch is dead (coverage agrees — `:666` and `:671-679`
never execute). Substitution cannot alias two dummies either: every dummy in
`ordered` takes its own `next(...)` from a fresh iterator.


## Tests worth adding

Taken from `coverage report --show-missing`, re-measured on 2026-09-02 after
the round below landed. Total coverage is now 92% against `fail_under = 83`.

`code_generator.py` and `sapt_utils.py` are at 100% and are off the list:
coefficient formatting (including the `Float == int` tripwire for sympy ≥1.13),
the `Add` branch, the unsupported-type fallback, both `pretty_indices` paths,
and `get_a_operator` / `get_b_operator` are all covered.

### `double_fermi_vac.py` — 88%

- `commutator` / `anticommutator` (`:693-696`, `:704-707`) — reached only by
      `examples/sapt_exch11.py`, which runs under `--slow`, outside the job
      that measures coverage
- `NO` branch of `_get_ordered_dummies_double_vac` (`:469-494`) — dead code:
      `NO.args` holds a single `Mul`, so none of its `isinstance` checks can
      ever match. Pinned from the outside by the `xfail`
      `test_dummy_ordering_uses_normal_ordered_operators`; the lines stay
      uncovered until the branch is repaired or dropped
- substitution-cycle branch (`:666`, `:671-679`) — unreachable by
      construction, see the section above. A skipped template with the
      argument is in `tests/test_double_fermi_vac.py`

### `sinfinitizer.py` — 83%, now the least-covered module

- the un-hit arms of the index-pair dispatch in `_get_tensor_symbol`
      (`:47-48`, `:71-72`, `:95-96`, `:119-120`, `:127-128`, `:151-152`) and
      its `NotImplementedError` fallback (`:161-170`) — a table-driven test
      over the index-type combinations would close all of it at once

### `operators.py` — 91%

- the four `__repr__` methods (`:27`, `:49`, `:71`, `:93`) — only cosmetic,
      but they are four one-line assertions

### Regression guards

Skipped templates are in `tests/test_get_R_nm.py`, both blocked on the
nondeterminism above:

- `test_R_20_is_deterministic`
- `test_R_20_denominator_carries_the_full_permutation_symmetry`
