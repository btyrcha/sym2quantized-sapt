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

holds **exactly**, term by term.  It is a *consistency* gate, not a
correctness one — the benchmark below caught a formula that passes it
and is still wrong.

## The limitation the psi4 benchmark caught

The per-loop route is only as correct as the spatial expression under
it, and spatial Wick + `2**loops` is **invalid for a resolvent (or
projector) carrying two or more index pairs in one space**: part of
its permutation multiplicity flows through exchange-wired
contractions, which spatial terms can only carry with same-spin
labels.  `<W R_(2,0) W>` treated this way halves the opposite-spin
MP2 energy — closed-shell limit `1.5A − B` instead of `2A − B` —
while the same-spin channel and the RHF collapse gate are perfectly
consistent.  Numerics against psi4 (UHF/cc-pVDZ, OH radical) exposed
it.

The correct open-shell route for MP-n is **spin tags**: indices carry
`is_alpha` / `is_beta`, the contraction rule vanishes across them
(the same mechanism as the monomer tags), the fluctuation operator
enters as its four spin sectors, and each resolvent sector gets its
own normalisation — `1/(2!)^2` for two indistinguishable same-spin
pairs, `1` for the distinguishable mixed pair.  With that, UMP2
agrees with psi4's conventional UHF-MP2 to ~1e-16 per spin channel
(derivation and numeric check live in the downstream application:
`derive_ump2_uhf.py` / `run_ump2_uhf_check.py`).  Per-loop labels
remain valid — and cheap — for single-pair-per-space projections:
`R_(1,1)` dispersion and the eq 46 dressings.  UMP3 via spin tags is
the natural follow-up; the `ump2_ump3_uhf.py` example pins the
per-loop halving so the limitation cannot be forgotten silently.

## Open lines, and what is deliberately out of scope

External (uncontracted) lines are enumerated like closed loops, so an
operator-valued expression returns one term per spin block of the
result; select by the external labels if one block is wanted.  Out of
scope here: ROHF / spin adaptation (a different reference structure,
not a summation rule), and code generation for the blocked tensors —
`code_generator`'s four-space naming needs a deliberate extension to
spin-split spaces before `generate_einsum` can emit these terms.
