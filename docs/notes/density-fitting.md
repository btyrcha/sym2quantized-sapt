# Density fitting in `generate_einsum`

`generate_einsum(expr, density_fitting=True)` replaces every intermolecular
two-electron integral by its density-fitted factorization, so the generated
code contracts three-index arrays instead of four-index ERIs:

```
v^{p r}_{q s} = (p q | r s) = sum_Q B^{Q}_{q p} B^{Q}_{s r}
```

```python
generate_einsum(t * v)                        # +np.einsum("rsab,abrs", t_rsab, v_abrs)
generate_einsum(t * v, density_fitting=True)  # +np.einsum("rsab,Qar,Qbs", t_rsab, Qar, Qbs)
```

Introduced as a prototype in `97c5081`; the contract below and the tests in
`tests/test_code_generator.py` were added on top of it.

## What is factorized

Only `v` — the intermolecular interaction integral `get_V_operator` builds.
Everything else is emitted as usual: the monomer potentials `(v_A)` / `(v_B)`
(which print as `v_A` / `v_B`, so they do not match the `v` test), the overlap
`s`, the resolvent denominator `e`, and every amplitude. With no `v` in the
expression the output is byte-identical to the non-fitted one.

## The two arrays

`v` carries `upper = (p in A, r in B)` and `lower = (q in A, s in B)`. The
lower indices come first in a variable name, so the four slots are
`(q, s, p, r)` and the split pairs slot 0 with slot 2 and slot 1 with slot 3 —
one array per monomer. For the canonical dispersion ERI `v^{a b}_{i j}` that
gives `Qar` and `Qbs`, matching the psi4numpy letters: A occupied `a`,
A virtual `r`, B occupied `b`, B virtual `s`.

The `Q` in the array name spells out `B^{Q}`; it is part of the name, not the
subscript. Both ERIs of E_disp(20) read the same `Qar` / `Qbs` arrays even
though their subscripts use different auxiliary letters.

The pairing is positional, so it is only the right factorization when each
slot pair sits on one monomer. `_check_eri_indices` enforces that whenever the
indices carry `is_molA` / `is_molB`, and raises rather than emit arrays
straddling both monomers. Indices without a monomer assumption keep the
positional pairing unchecked.

## One auxiliary index per ERI

Each ERI is a sum over the auxiliary basis of its own. Two ERIs sharing a label
would be fused into a single sum:

```
sum_Q B B B B          !=          (sum_Q B B) (sum_P B B)
```

The prototype emitted the literal `Q` for every ERI, so E_disp(20) came out as
`np.einsum("rsab,Qar,Qbs,Qra,Qsb", ...)` — numerically wrong, silently, for
every term second order in `V`. It now reads
`np.einsum("rsab,Qar,Qbs,Pra,Psb", ...)`.

Naming happens in two stages, because the letters an auxiliary index may use
are not known until the ordinary indices have been renamed:

- `_expand_tensor` writes a per-ERI sentinel from `_AUX_SENTINELS`
  (`#`, `$`, `%`, …). None of them is matched by the `[a-z](?:_\d+)?` pattern
  the two renamers key off, nor drawn from the ASCII letter pool
  `_replace_indices_names` hands out, so they pass through untouched.
- `_assign_auxiliary_names` runs last, inside `_get_code_str`, and swaps each
  sentinel for a letter of `_AUX_NAMES` (`Q` first) that the finished subscript
  has not already spent.

Doing it last is what keeps `pretty_indices` working: that table renames `q_1`
to `Q` and `p_1` to `P`, so the auxiliary index steps aside to `R`.

Eight sentinels means eight ERIs per term; beyond that `generate_einsum` raises
`IndexError("Too many ERIs!!! Not enough auxiliary indices for them.")` rather
than reuse a label.

## Known gap

`2.0 * v * v` — a *squared* ERI — is a `Pow`, not a `Mul` of two
`TensorSymbol`s, so no argument of the term is recognised and the whole factor
is dropped: the line comes out as `+2 * np.einsum("", )`. This is not specific
to density fitting; the non-fitted route drops it identically. Same defect
family as `test_unsupported_term_is_not_dropped_silently`. Pinned by the strict
`xfail` `test_density_fitting_does_not_drop_a_squared_eri` and tracked in
`TODO.md`.
