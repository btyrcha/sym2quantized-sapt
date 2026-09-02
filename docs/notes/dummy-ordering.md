# Dummy ordering and the `Zero` probe

Both defects described here are fixed. The record is kept because the symptom
was misleading: two independent bugs were stacked, so each one alone appeared to
explain it and neither did.

1. `_get_ordered_dummies_double_vac` let a Python set decide the canonical index
   order, so derivations differed between processes.
2. `get_a_operator` / `get_b_operator` built their generated indices as `Dummy`,
   that is as summation indices, which made the operator identically zero for
   n >= 2.

Baseline for every measurement below is 507e35a.

## 1. Ordering nondeterminism

`_get_ordered_dummies_double_vac` ended with `sorted(term.atoms(Dummy),
key=_get_key)`. `_get_key` is not a total order — two dummies of the same type
in equivalent positions score equal — and `sorted` is stable, so ties fell back
to the iteration order of the set `atoms` returns. Two independent sources
randomize that order per process: the string hash (`PYTHONHASHSEED`) and sympy's
`Dummy._base_dummy_index`, drawn from `random.Random().randint(10**6, 9*10**6)`.
Pinning either one alone left the derivation random, which is why this did not
look like plain hash ordering.

The fix has two parts. Dummies are collected in `preorder_traversal` order into
a `dict.fromkeys`, so a tie is broken by position in the term. And the `NO`
branch of `_get_tensors_with_dummy` unpacks the single `Mul` that `NO.args`
holds, so operators inside a normal ordering bracket contribute to the key —
previously none of its `isinstance` checks could match and the branch was dead.

Distinct results over 8 fresh processes:

| probe | before | after |
|---|---|---|
| `R(1,0) V`, `R(0,1) V`, `R(1,1) V`, `R(2,0) V` | 1 | 1 |
| `R(1,0)` of `get_a_operator(n=1)` | 2 | 1 |
| `R(2,0)` of `get_a_operator(n=2)` | 7 (13 over 30 runs) | 1 |
| `R(0,2)` of `get_b_operator(n=2)` | 5 | 1 |
| `R(1,1)` of `a₁·b₁` | 4 | 1 |
| all eight `examples/` scripts, 10 runs each | 1 | 1 |

`_get_key` remains degenerate. The fix makes the tie-break reproducible; it does
not make the key total. Do not restore `term.atoms(Dummy)` — the obvious
spelling reintroduces the defect silently.

### Ruled out

`_use_symmetries` sorts by `Basic.compare`, which for same-class symbols
compares names rather than `dummy_index`, and is therefore process-stable.

The `(x, y) -> (y, x)` branch of `substitute_dummies_double_vac` was an early
suspect. Its guard is `if v in subsdict`, where `v` is a replacement dummy
created fresh inside the call while the keys are the expression's own dummies.
`Dummy` equality includes `dummy_index`, so the guard is never true and the
branch is unreachable; coverage agrees. Substitution cannot alias two dummies
either — every dummy in `ordered` takes its own `next(...)` from a fresh
iterator. `test_substitution_cycle_needs_a_temporary_symbol` carries the
argument; the branch should be reached or deleted.

## 2. The `Zero` probe

`get_R_nm(2, 0, get_a_operator(n=2))` was the probe used to track down the
ordering bug, and its intermittent `Zero` was recorded as the failure mode. The
`Zero` was correct. `get_a_operator` and `get_b_operator` built the indices `n`
generates as `Dummy`; they should be free, because the exchange operator is
contracted with overlap integrals by its caller.

Summed, the operator is identically zero for n >= 2: `Σ_{p₀p₁} a†_{p₀} a†_{p₁}`
is an antisymmetric product over a symmetric index range. The four terms
`substitute_dummies_double_vac` received were relabelings of one another that
cancel in pairs — swapping the summed `q₀ ↔ q₁` in the third turns it into minus
the first, touching only deltas. What the pre-fix code returned, four terms in
13 different labelings, was that same zero left unreduced.

Dropping `cls=Dummy` from the two builders fixes it at the source.
`get_R_nm(2, 0, get_a_operator(n=2))` now returns the four-term antisymmetrized
expression, as do `R(0,2)` and `R(2,1)`. `get_R_nm(2, 0, get_V_operator())`
remains `Zero`, correctly: V acts on one electron per monomer and cannot doubly
excite A. Indices that should stay `Dummy` were left alone — `get_Pn_operator`
contracts its own against `s` tensors, and `get_R_nm` sums over its excitation
indices.

### A misleading relabeling

Relabeling `i₁ ↔ i₂` rather than `q₀ ↔ q₁` reaches the same terms through the
operator string, requires the anticommutator to reorder it, and reads as though
the substitution had dropped a sign. It has not. Both relabelings are legal and
appear to disagree only because each term is separately zero: `e` is symmetric
in the index pair — `_use_symmetries` reorders indices without attaching a sign,
which is correct for an energy denominator — while `a_{i₂} a_{i₁}` is
antisymmetric in it.

## Why sympy needs no sign tracking

sympy's `substitute_dummies` is also a plain `.subs()` of index names with no
parity bookkeeping, and needs none: its index carriers re-canonicalize
themselves, with the sign, inside `.subs`.

| carrier | swapping two indices | sign emitted |
|---|---|---|
| `AntiSymmetricTensor("t", (a,b), (i,j))` | `-AntiSymmetricTensor(t, (a,b), (i,j))` | yes |
| `NO(Fd(a)*Fd(b)*F(j)*F(i))` | re-sorted, `-NO(…)` → `+NO(…)` | yes |
| bare `Mul` of `Fd`/`F` | `Fd(b)*Fd(a)*F(j)*F(i)` — unchanged | no |

Both self-canonicalizing cases do the work in `__new__`: `AntiSymmetricTensor`
sorts its index tuples and extracts the parity, `NO` runs
`_sort_anticommuting_fermions` on its operator string.

sympy's pipeline never produces the third row — after
`wicks(..., keep_only_fully_contracted=True)` no operators remain, and survivors
live inside `NO`. This package produces one deliberately: `get_R_nm` builds
`coeff * tensors * Dagger(a_part) * Dagger(b_part) * denom`, where
`Dagger(a_part)` is that bare `Mul`, because the excitation operator must stay
uncontracted. `DoubleVacuumTensorSymbol` is a second instance — its symmetries
are unsigned by design, so unlike `AntiSymmetricTensor` it carries no sign.

Nothing is known to be wrong as a result, but this is the invariant to check
first if the n >= 2 path is extended.

## Guards

`test_generated_indicies_are_free`, `test_R_20_of_a_double_excitation_does_not_vanish`
and `test_R_20_denominator_carries_the_full_permutation_symmetry` cover the
index fix; all three fail against 507e35a.

`test_dummy_ordering_uses_normal_ordered_operators` is the only guard on the
ordering fix, so it repeats its comparison `_ORDERING_REPEATS` times with fresh
dummies. A single comparison catches a regression in about 8 runs in 10 — the
tie is decided per `Dummy` pair, and a pair can come out right by chance.
Repeating separates the cases completely:

| | disagreements out of 10 |
|---|---|
| fix present | 0, in each of 30 processes |
| fix reverted | at least 7, in each of 20 processes |

A cross-process `test_R_20_is_deterministic` was drafted and dropped as vacuous.
It would have been the natural guard while `get_a_operator` still built summed
indices, but with free indices every `get_R_nm` probe gives one distinct result
over 12 processes whether or not the ordering fix is present. A cross-process
guard would need an expression retaining two interchangeable dummies of the same
type in one term; the `NO` pair above is the only one known to qualify.
