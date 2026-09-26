# UHF spin summation

**Status: frozen proof of concept, not maintained (since 2026-09-26).**
The supported UHF route is spin tags on the indices (`open_shell.py`;
`GETTING_STARTED.md` §4, `spin-tag-renaming.md`). This page stays as the
design record of the per-loop route; its known limitations, listed at
the end, will not be fixed.

(Implemented in `sym2quantized_sapt/spin_integrator/uhf.py`; it shares
the loop counter in `spin_integrator/__init__.py` with the RHF path.
The spin-tag alternative is `open_shell.py`.)

`spin_integration_uhf` extends the package to an unrestricted
reference.  Design and the reasoning behind it.

## Spin is a per-loop label, not a per-index one

Spin is constant along a Goldstone loop, so the unit of spin
bookkeeping is the loop, not the index.  `_loop_partition` (the
refactored core of `_count_loops`; identical counts, now also the
membership) partitions a term's slot pairs into loops; UHF summation
assigns one label per loop and enumerates:

- **RHF** (`spin_integration`): both labels of every loop give the
  same spatial term → factor `2**loops`.
- **UHF** (`spin_integration_uhf`): the labels differ → the term
  becomes an explicit sum of `2**loops` spin-blocked copies.

Enumerating per loop rather than per index is what keeps the term
count at its physical minimum.

Loops are traced only through the vertices of the term's Goldstone /
Hugenholtz graph.  A tensor built with `is_graph_vertex=False` — the
resolvent denominator from `get_R_nm` — is not a vertex: it weights
the lines crossing a cut between vertices.  Its slot pairing is not a
line, and tracing it merges loops (see the bug below).

## Blocks are labelled per slot pair

A particle-conserving tensor carries one spin per `(upper_k, lower_k)`
pair — one line through the vertex — so blocks get one label per pair:
`t_ab[i, j, a, b]` has the `(i, a)` line alpha and `(j, b)` beta;
`v_ba` likewise.  These are the genuinely unrestricted arrays
(alpha and beta spatial ranges differ), which downstream code must
supply as distinct tensors, not views of one.

A tensor that is not a vertex (`is_graph_vertex=False`) has no
lines through it, hence no pairs to label, and its canonical pairing can cross the loops: in
`e^{i i1}_{a a1}` the indices `i` and `a1` may lie on one loop.  It
is labelled per index instead, upper spins then lower, and keeps its
spatial layout: `e_ab_ba[i, i1, a, a1]` has `i, a1` alpha and
`i1, a` beta.  `array_table` reads both forms.

Declared permutation symmetries are **not** carried onto blocks: the
doubles pair symmetry maps *between* blocks (`t_ab ↔ t_ba`) under an
unrestricted reference, and declaring it on one block would
canonicalize within it — silently wrong for mixed spin.  The
inter-block relation shows up naturally instead (the UMP2 mixed-spin
terms couple `v_ab` with `v_ba`).

## The RHF collapse gate

`rhf_collapse` strips the block labels; with all blocks equal the
copies merge and

    rhf_collapse(spin_integration_uhf(e)) == spin_integration(e)

holds **exactly**, term by term.  It is a *consistency* gate, not a
correctness one — the bug below passed it while halving an energy.

## The denominator bug the psi4 benchmark caught

`<W R_(2,0) W>` spin-summed per loop gave exactly half the
opposite-spin MP2 energy — closed-shell limit `1.5A − B` instead of
`2A − B` — while the same-spin channel and the RHF collapse gate were
perfectly consistent.  Numerics against psi4 (UHF/cc-pVDZ, OH radical)
exposed it.  It was first read as a limit of the per-loop route (a
multi-pair projector's permutation multiplicity supposedly flowing
through contractions that spatial terms cannot carry) and pinned as
such.

The cause was the loop counter, and the RHF `spin_integration` had
the same bug.  `get_R_nm` multiplies its result by the denominator
`e^{i i1}_{a a1}`, which was traced like any other tensor.  Its slot
pairing is not a line: `e` is declared symmetric under independent
permutations of its upper and its lower row, so canonicalization
picks a pairing that has nothing to do with the lines.  Where that
pairing crossed the two loops — 4 of the 8 direct MP2 terms — `e`
merged them into one, and those terms got a factor 2 instead of 4.
With one index pair per space (`R_(1,0)`, `R_(1,1)`) there is only one
pairing, which is why the SAPT results were unaffected.

The fix: `DoubleVacuumTensorSymbol(..., is_graph_vertex=False)` marks
a tensor that is not a graph vertex, `get_R_nm` sets it on `e`, and
both loop counters skip such tensors.  The flag is stored in the tensor's args
and is part of its equality, so it survives `subs`, `xreplace` and
dummy substitution.  A denominator entered by division (`v / e`, a
`Pow`) was never counted and needs no flag.

With the fix, the per-loop and spin-tagged UMP2 expressions agree
numerically in every spin sector to machine precision (relative
~1e-15), and both match a conventional UMP2 evaluation — on random
UHF-like integrals with different alpha and beta orbital counts.
Putting `e` back into the loop graph reproduces the halved
opposite-spin sector exactly.  `tests/test_spin_integrator.py` pins
the closed-shell `2A − B`.

Both routes therefore hold for MP-n.  **Spin tags** are the general
mechanism, and the supported route: indices carry `is_alpha` / `is_beta`, the contraction rule
vanishes across them (the same mechanism as the monomer tags), the
fluctuation operator enters as its four spin sectors, and each
resolvent sector gets its own normalisation — `1/(2!)^2` for two
indistinguishable same-spin pairs, `1` for the distinguishable mixed
pair.  That route agrees with psi4's conventional UHF-MP2 to ~1e-16
per spin channel (derivation and numeric check live in the downstream
application: `derive_ump2_uhf.py` / `run_ump2_uhf_check.py`).
**Per-loop labels** were the cheaper route for spin-free spatial
expressions; the route is now frozen.

## Open lines, and what is deliberately out of scope

External (uncontracted) lines are enumerated like closed loops, so an
operator-valued expression returns one term per spin block of the
result; select by the external labels if one block is wanted.  Out of
scope here: ROHF / spin adaptation (a different reference structure,
not a summation rule), and code generation for the blocked tensors —
`code_generator`'s four-space naming needs a deliberate extension to
spin-split spaces before `generate_einsum` can emit these terms.

## Known limitations (won't fix)

Recorded when the route was frozen. None raises an error.

- **Spatial input only.** Spin tags already on the indices are ignored:
  a tagged term is blocked as if it were spatial, so the blocks can
  contradict the tags (`x_a_b` holding α-tagged indices).
- **Expanded input only.** Factors of a `Mul` that are not tensors,
  including `Add` and `Pow`, are treated as coefficients: `(x + z) * y`
  blocks only `y`, and `x**2 * y` leaves `x` spatial. The spatial
  `<W R_(2,0) W R_(2,0) W>` contains such `e**2` factors.
- **Index-less tensors get a trailing separator.** `_blocked` always
  appends it, so `V_0` becomes the different symbol `V_0_`, and code
  generation emits `V_0__`.
- **The block label is a name suffix.** `rhf_collapse` strips a genuine
  `_a` suffix from a spatial tensor (`u_a` → `u`); multi-character
  `labels` make it a no-op; blocking twice nests the labels (`x_a_b`).
- **The `rhf_collapse` gate and `Float` coefficients.** `get_R_nm` and
  `get_Pn_operator` produce `Float`s, and adding the `2**loops` copies
  one at a time can round differently from `2**loops * c` (e.g. `1/36`
  from `get_R_nm(3, 0)`), so the exact gate could report a false
  failure. Not reproduced.
