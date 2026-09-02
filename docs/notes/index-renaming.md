# `pretty_indices` and `generate_einsum` rename indices twice

Open defect. Tracked in `TODO.md`.

`substitute_dummies_double_vac(expr, pretty_indices=...)` and
`code_generator.generate_einsum` both rename indices, and the second rename is
applied on top of the first. `_psi4numpy_indices` is injective only on the
canonical names (a, b, i, j, p, q); anything else is corrupted silently.

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

Pinned by the `xfail` `test_psi4_indices_name_collision`; warned about in the
docstrings of `substitute_dummies_double_vac`, `generate_einsum` and
`_psi4numpy_indices`, and in `CLAUDE.md`.

## Options

- Reject non-canonical names in `generate_einsum`. Cheap; converts a silent
  wrong answer into an error.
- Key the rename off the index assumptions rather than the name string. Robust;
  removes the dependency on the name entirely.
- Make the psi4numpy convention a `generate_einsum` option, so the two renames
  are never both applied.

## Notes for whoever takes it

The `xfail` is strict, so any fix — deliberate or incidental — turns the suite
red with `[XPASS(strict)]`. That is the signal to drop the marker along with the
warnings in the three docstrings and in `CLAUDE.md`.

`PRETTY_INDICES` carries no `i`/`j` rows, because `_psi4numpy_indices` maps
`i -> a` and `j -> b` before the table is consulted. They must be restored if
the psi4numpy rename becomes optional.
