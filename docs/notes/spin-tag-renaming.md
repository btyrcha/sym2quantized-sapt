# Renaming spin-tagged indices

`substitute_dummies_double_vac` renames summed indices to canonical names so
that terms differing only in index names merge. With spin tags
(`is_alpha` / `is_beta`, the spin-tag route in `open_shell.py`) it used to
rename α indices onto β ones.

## The bug

Replacement indices came from one pool per monomer and Fermi level. Each
replacement copied the assumptions of whichever original index it was made
from, taken in set order, so the "hole-A" pool held α and β replacements in a
per-process random order. Every index then took the next replacement of its
pool, whatever its spin. An α index could therefore become a β one, which
changes the expression.

| probe | before the fix |
|---|---|
| `t_ab(i_α,i_β; a_α,a_β) · x_ab(…)`, 6 `PYTHONHASHSEED`s | spins right in 1 of 6 |
| `x_a(i_α; a_α) + y_b(i_β; a_β)`, 8 fresh processes | wrong in 8 of 8 |
| tagged UMP2 from `examples/ump2_uhf.py`, 3 seeds | 36/36 terms with tags contradicting the tensor names; `array_table` raises |

The UMP2 *energies* still came out right, because the example's evaluator
reads each index's spin from the tensor name (`v_ab`, `e_ab_ab`), never from
the tag. A numeric check alone could not see this bug.

The example passes `substitute_dummies=False` to `wicks_double_vac`, which is
why the spin-tag route worked at all. `get_R_nm` renames indices internally,
so before the fix it could not be used with tagged input.

## The fix

An index is renamed only onto one with the same assumptions: the same
monomer, Fermi level and spin, so the same orbitals. Each class of index has
its own pool, as large as the most indices of that class in one term, and the
pools are built in a fixed order.

**Putting spin in the sort key was not the fix.** With a spin-aware sort key
and the old shared pool, the output was still wrong under every seed: the
order in which the indices *asked* for replacements was deterministic, but
the spins of the pool entries they received were not. With per-spin pools,
the key only orders indices within one pool, all of the same spin, so
`_get_ordered_dummies_double_vac` is unchanged.

RHF output is byte-identical: the whole suite, which compares exact `latex()`
strings, and every example print the same.

## Names: distinct, with the bar at print time only

α and β replacements must get different `.name`s. `code_generator` builds
einsum subscripts from the name strings, so an α `i` and a β `i` in one term
become one letter:

```
same names:      np.einsum("rraa,aarr", ...)   # a diagonal: silently wrong
distinct names:  np.einsum("rcad,adrc", ...)   # correct
```

So the kinds share one name counter across their spins, untagged first, then
α, then β: α holes `i`, `i_1`, β holes `i_2`, `i_3`. The bar over a β index is
added only when printing LaTeX (`open_shell.index_latex`, used by
`DoubleVacuumTensorSymbol` and the four operator classes): `\bar{i}_2`. It is
never stored in `.name`, which stays within the `[a-z](?:_\d+)?` form that
code generation parses.

The rejected alternative was to give both spins the same names (β restarting
at `\bar{i}`, `\bar{i}_1`) and make code generation separate them by spin
before choosing einsum letters. That reads better, but it changes code
generation's string pipeline, whose tests compare exact strings and where a
mistake is silent, as shown above.

## Not covered

`KroneckerDelta` and `str()` print through SymPy's own printer and show no
bar. Deltas are normally evaluated away.

Pinned by `tests/test_open_shell.py`:
- `test_renaming_keeps_every_index_in_its_spin` fails in every process
  without the fix, so it needs no seeds or subprocesses (see
  `dummy-ordering.md` on why cross-process tests were dropped);
- `test_alpha_and_beta_indices_are_renamed_apart` pins the distinct names
  and the einsum string;
- `test_beta_indices_print_with_a_bar` pins the LaTeX.
