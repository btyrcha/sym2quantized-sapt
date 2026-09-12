# UHF spin summation

`spin_integration_uhf` extends the package to an unrestricted
reference.  Design and the reasoning behind it; the RHF path is
untouched.

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

## Blocks are labelled per slot pair

A particle-conserving tensor carries one spin per `(upper_k, lower_k)`
pair — one line through the vertex — so blocks get one label per pair:
`t_ab[i, j, a, b]` has the `(i, a)` line alpha and `(j, b)` beta;
`w_ba`, `e_ab` likewise.  These are the genuinely unrestricted arrays
(alpha and beta spatial ranges differ), which downstream code must
supply as distinct tensors, not views of one.

Declared permutation symmetries are **not** carried onto blocks: the
doubles pair symmetry maps *between* blocks (`t_ab ↔ t_ba`) under an
unrestricted reference, and declaring it on one block would
canonicalize within it — silently wrong for mixed spin.  The
inter-block relation shows up naturally instead (the UMP2 mixed-spin
terms couple `w_ab` with `w_ba`).

## The RHF collapse gate

`rhf_collapse` strips the block labels; with all blocks equal the
copies merge and

    rhf_collapse(spin_integration_uhf(e)) == spin_integration(e)

holds **exactly**, term by term.  Every UHF derivation should pass
this gate before its blocks are trusted; the tests and the
`ump2_ump3_uhf.py` example enforce it for E_disp(20), UMP2
(16 spatial → 40 blocked terms; the opposite-spin block provably
exchange-free) and UMP3 (336 → 1136; the RSPT
`<WRWRW> − E1<WRRW>` form).

## Open lines, and what is deliberately out of scope

External (uncontracted) lines are enumerated like closed loops, so an
operator-valued expression returns one term per spin block of the
result; select by the external labels if one block is wanted.  Out of
scope here: ROHF / spin adaptation (a different reference structure,
not a summation rule), and code generation for the blocked tensors —
`code_generator`'s four-space naming needs a deliberate extension to
spin-split spaces before `generate_einsum` can emit these terms.
