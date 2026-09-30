# The spin-tag route (UHF)

The supported way to derive open-shell (UHF) expressions. Indices carry
`is_alpha=True` or `is_beta=True`, the spin analogue of the monomer tags
`is_molA` / `is_molB`, and Wick's theorem does the spin bookkeeping itself:
nothing contracts across opposite spins, so no spin-summation step follows.
The other route, per-loop summation, is frozen (`uhf-spin-summation.md`).

The predicates and the printer live in `open_shell.py`; the core consults them:

| where | what the tags do |
|---|---|
| `contraction_double_vac` | a contraction across opposite spins is 0 (`opposite_spins`) |
| `evaluate_deltas_double_vac` | an opposite-spin delta is 0, and a delta's survivor keeps the tag (`_merge_delta_indices`, `shared_spin_tag`) |
| `substitute_dummies_double_vac` | an index is renamed only onto one of its own spin |
| `spin_integration` | raises `ValueError` on tagged input (`has_spin_tags`): the tags already resolve the spin, and RHF integration would count every loop twice |
| `code_generator` | one spin letter per array axis |

## Deltas

The survivor of a delta is chosen by Fermi-level information alone
(`killable_index` / `preferred_index`), so it can be the untagged index:
`δ(p_α, i)` keeps the hole `i`. A plain substitution would drop the tag and
leave the term summed over both spins. `_merge_delta_indices` instead replaces
both indices with a copy of the survivor that carries the tag.

That also makes chains vanish. In `δ(p_α, q) δ(q, r_β) y(p_α; r_β)` the first
merge tags `q` as α, so the second delta is opposite-spin and the term is 0.

Two kinds of delta stay unevaluated:
- **The survivor occurs only once in the term**: a free index under the
  Einstein convention, whether it is a `Symbol` or a `Dummy` whose other
  occurrences are not in this expression yet. Tagging it means renaming it,
  which would cut it off from those occurrences.
- **A bare delta**, not in a product, as with the monomer tags.

## Renaming

`substitute_dummies_double_vac` keeps a separate pool per spin, and the spins
of one kind share the name counter: α holes `i`, `i_1`; β holes `i_2`, `i_3`.
`.name` never repeats across spins, because code generation identifies indices
by name. The bar over a β index (`\bar{i}_2`, `index_latex`) is added in LaTeX
only. → `spin-tag-renaming.md`

## Building an expression

Write the perturbation as its spin sectors, one spin per slot pair (the pair a
line runs through). For the fluctuation potential of monomer A:

```python
# DVT = DoubleVacuumTensorSymbol; A, Ad from operators
for s1, s2 in product(("is_alpha", "is_beta"), repeat=2):
    p, q = symbols("p q", is_molA=True, cls=Dummy, **{s1: True})
    p2, q2 = symbols("p' q'", is_molA=True, cls=Dummy, **{s2: True})
    sectors.append(
        Rational(1, 2) * DVT("v", (p, p2), (q, q2)) * Ad(q) * Ad(q2) * A(p2) * A(p)
    )
```

The library has no builder for W; `examples/ump2_uhf.py` has one.

**`get_R_nm` works unchanged.** Its own indices carry no tag, so each one runs
over both spins, like a spin-orbital index. Inside `get_R_nm`, their deltas with
the operator's tagged indices stay unevaluated, because the resolvent's indices
occur only once there (see above). The outer Wick step resolves them, and the
tag reaches every occurrence, the denominator `e` included. So its single
normalisation `1/(n! m!)^2` is the right one. Written out by sector instead, the
resolvent needs `1/(2!)^2` for each same-spin pair and `1` for the mixed pair,
with α before β. After canonicalization the two forms are the same expression.

Name tensors plainly (`t`, `v`): the tags carry the spin. Permutation
symmetries may be declared, since the spin moves with the index. Per-sector
names such as `t_ab` are an obsolete workaround from when array names were
spin-blind.

## Code generation

`generate_einsum` appends one spin letter per axis to the array name, in the
order of the index letters (lower indices first). `t` with upper `i_α i_β`
and lower `a_α a_β` is held in `t_rraa_abab`. A density-fitting factor gets
the spins of its slot pair: `Qar_aa`.

A tensor with both tagged and untagged indices raises `ValueError`: under UHF
an untagged axis has no single orbital range. `array_table` reads each axis's
spin from the tags and never parses a tagged tensor's name. Its keys are exactly the
code's array names, since both take them from `_array_names`.
→ `density-fitting.md` for the `Q` factors.

## Validation

There is no symbolic gate like the per-loop route's `rhf_collapse`. One would
bring tensor-naming conventions back, while on this route the tags are the
truth. Validation is numeric, as `GETTING_STARTED.md` §6 asks:
- `examples/ump2_uhf.py` evaluates tagged UMP2 on random UHF-like data, with
  different α and β orbital counts. It prints the result next to textbook UMP2,
  sector by sector, and the two agree to machine precision.
- The by-sector form was checked downstream against psi4's conventional
  UHF-MP2, to ~1e-16 per spin channel.

`tests/test_open_shell.py` pins the rules symbolically:
- contractions and deltas: `test_can_evaluate_opposite_spin_delta`,
  `test_delta_survivor_inherits_the_spin_tag`,
  `test_opposite_spins_through_untagged_index_give_zero`,
  `test_delta_with_free_untagged_survivor_is_kept`;
- renaming and printing;
- the `spin_integration` guard;
- `get_R_nm` against the by-sector form
  (`test_get_R_nm_works_unchanged_on_spin_tagged_input`);
- array names.

## Not yet checked, and known costs

- **Resolvents:** only R(2,0) on monomer A has been checked with tags.
  `get_R_nm(n, m)` with m > 0 (both monomers) and nested `get_R_nm` (UMP3) are
  untested, although the mechanism doesn't depend on n, m or nesting.
- **UMP3:**
  - `<W R W R W>`, measured on the spatial route, has 328 terms, and 136 of them
    contain a self-contracted `v` (an index in both rows of one tensor).
  - With a canonical HF reference the perturbation is the normal-ordered W, so
    those terms must be dropped. `NO(...)` can't do this, because
    `wicks_double_vac` sets `NO` factors aside and never contracts them.
  - The tagged expansion has more terms: one per spin sector.
- **Cost:** `_merge_delta_indices` substitutes with `expr.subs`, which dominates
  `get_R_nm` on tagged input: about 1300 calls and a few seconds for UMP2.
- **Printing:** `KroneckerDelta` and `str()` print through SymPy's own printer,
  without the bar.
