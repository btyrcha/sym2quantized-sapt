# TODO / future ideas

Goals and future ideas for the repo.

## Correctness

- **Stop `pretty_indices` and `generate_einsum` renaming indices twice.**
  Custom index names are silently corrupted before they reach the generated
  einsum, so the wrong indices get contracted and nothing raises. Pinned by the
  strict `xfail` `test_psi4_indices_name_collision`.
  → [`docs/notes/index-renaming.md`](docs/notes/index-renaming.md)

- **Settle the substitution-cycle branch in `substitute_dummies_double_vac`.**
  Inherited from sympy and dead by construction here — either build an
  expression that reaches it, or delete it and its `final_subs` bookkeeping.
  Template in `tests/test_double_fermi_vac.py`.
  → [`docs/notes/dummy-ordering.md`](docs/notes/dummy-ordering.md)

## Features

- **UHF spin summation.** Add spin summation rules for a UHF reference.
