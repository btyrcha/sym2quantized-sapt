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
Fixing `PYTHONHASHSEED` does not stabilize it, so this is not simple hash
ordering.

Localized by bisection:

| Stage | Stable? |
|---|---|
| `wicks_double_vac(..., substitute_dummies=False)` | yes — 20/20 runs give the same 4 terms |
| `_use_symmetries` / the `e` denominator index order | yes — identical in 20/20 runs |
| `substitute_dummies_double_vac` (`double_fermi_vac.py:516`) | **no** — intermittently collapses the 4 terms to `Zero` |

`get_R_nm` at `(1,0)`, `(0,1)`, `(1,1)` and `get_R_nm(2, 0, V)` were each stable
over 25 runs, so this is specific to the `n ≥ 2` path, whose indices carry a
monomer tag but no Fermi-level assumption.

Prime suspect, stated as a hypothesis rather than a conclusion: the
substitution-ordering logic at `double_fermi_vac.py:662-682` — the `subslist` /
`final_subs` split, whose temporary-symbol branch (lines 671-679) is the one
this report already flags as never executed. If two distinct dummies are aliased
onto the same symbol by an unlucky substitution order, terms that should survive
cancel to zero.

That exact failure mode turned out to be real in `code_generator`, where the
replacement pool did not exclude names already in use and two distinct dummies
collapsed onto one index — see `_replace_indices_names` and its regression
guard `test_index_name_collision`. It is circumstantial, but it is the same
shape of bug in the same kind of code, which makes the hypothesis worth testing
rather than merely plausible.


## Tests worth adding

This is the shortlist, taken from the current
`coverage report --show-missing` and from what the mutation probes could not
kill. Line numbers are as measured on 2026-09-02; total coverage is 87%
against `fail_under = 83`.

### `code_generator.py` — 88%

- coefficient formatting, general (non-±1) case (`:128-136`) — this is where
      the sympy ≥1.13 fix has to go, and nothing exercises it
- the `Add` branch (`:239`) — multi-term expressions; the existing tests
      only ever generate a single term
- unsupported-type fallback (`:250`) — returns `""`, silently dropping terms
- `pretty_indices=True` for a bare `TensorSymbol` (`:160`) — only the `Mul`
      route through `_get_code_str` is covered
- the `pretty_indices` rejection path (`:61`) — the new `ValueError` for an
      index outside `PRETTY_INDICES` has no test

### `double_fermi_vac.py` — 84%

- `get_fully_contracted` (`:352-374`) — public, called by nothing
- substitution-cycle branch (`:671-679`) — never executed, and the prime
      suspect for the nondeterminism
- `NO` branch of `_get_ordered_dummies_double_vac` (`:469-494`) — the one
      mutation still surviving; needs a partially-contracted expression
      (`keep_only_fully_contracted=False`)

### `sapt_utils.py` — 80%, now the least-covered module

- `get_a_operator` / `get_b_operator` (`:39-52`, `:70-83`) — never executed
      by any test or example

### Regression guards

- `get_R_nm(2, 0, ...)` non-determinism
- the `n >= 2` denominator symmetry set
