# Coverage gaps

Total is 93% against `fail_under = 83` in `.coveragerc`. At 100%:
`code_generator.py`, `sapt_utils.py`, `diagrams.py`, `spin_integrator.py`.
`tensors.py` is at 97%, partial branches only, and is not worth chasing.

Regenerate before acting on this list; line numbers go stale.

```shell
python3 -m coverage run --source=src/sym2quantized_sapt -m pytest tests/
python3 -m coverage report --show-missing
```

CI measures coverage on the fast suite only, so anything reached solely by an
`examples/` script — those run under `--slow`, in a separate job — never counts.

## `sinfinitizer.py` — 83%, lowest in the package

The unhit arms of the index-pair dispatch in `_get_tensor_symbol`, and its
`NotImplementedError` fallback. A table-driven test over the index-type
combinations closes all of it at once.

## `double_fermi_vac.py` — 91%

- `commutator` / `anticommutator` — reached only by `examples/sapt_exch11.py`,
  which runs under `--slow`.
- The two `TypeError` guards in the `NO` branch of
  `_get_ordered_dummies_double_vac` — defensive. `NO` always wraps a single
  `Mul`, and `NO(one_operator)` returns the bare operator rather than an `NO`,
  so neither guard has a known trigger. The branch itself is live; its four
  `isinstance` arms are covered by
  `test_dummy_ordering_uses_normal_ordered_operators`.
- The substitution-cycle branch — unreachable by construction; see
  `dummy-ordering.md`. `test_substitution_cycle_needs_a_temporary_symbol`
  carries the argument.

## `operators.py` — 91%

The four `__repr__` methods. Cosmetic, but four one-line assertions.

## `utils.py` — 83%

The body of the `timeit` decorator, which runs only when an `examples/` script
calls it.
