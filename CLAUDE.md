# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A SymPy-based package for second-quantized operator algebra in **SAPT** (Symmetry-Adapted
Perturbation Theory of intermolecular interactions). It extends SymPy's `sympy.physics.secondquant`
module to handle the **double Fermi vacuum** — the product of two independent Fermi vacuums, one per
monomer (A and B) of a dimer/complex. Standard Wick's theorem assumes a single vacuum; the core of
this package is a generalized Wick's theorem and dummy-index machinery for the two-vacuum case, used
to derive closed-form SAPT energy expressions.

The importable package name is `sym2quantized_sapt` (distribution name `py-quantized-sapt`).
The stray `src/py_quantized_sapt.egg-info/` is leftover build metadata, not a second package.

## Commands

```shell
# Environment + editable install (setup.py is the single source of truth;
# the `dev` extra carries pytest/coverage/black/pylint/pre-commit/build)
python3 -m pip install -e ".[dev]"

# or: conda env create -f environment.yml && conda activate sym2quantized-sapt

# Fast tests (the example-runner and other heavy cases are marked `slow` and skipped by default)
python3 -m pytest ./tests/

# Full suite including slow tests — note the custom `--slow` flag (see tests/conftest.py),
# NOT pytest's built-in `-m` marker selection
python3 -m pytest ./tests/ --slow

# Single test
python3 -m pytest tests/test_sapt_disp.py::test_can_evaluate_sapt_disp_20_energy

# Run a derivation script directly
python3 examples/sapt_pol20.py

# Lint / format (black line length is 79; both run in CI via pre-commit)
pre-commit run -a
python3 -m pylint --rcfile=.pylintrc src/
```

CI runs on Python 3.8 (GitHub Actions, `.github/workflows/ci.yml`). Supported range is `>=3.8,<3.13`.

## Version sensitivity

This package subclasses and reuses SymPy `secondquant` internals (`NO`, `contraction`,
`TensorSymbol`, `Dagger`, dummy-substitution logic), so it is tightly coupled to SymPy's
implementation. `setup.py` caps it at `sympy>=1.8.0,<1.13.0` — 1.13 changed `Float`/`int`
equality and breaks coefficient formatting in `code_generator`. Treat SymPy upgrades as breaking
until proven otherwise. Tests assert on exact `latex()` output strings, so they double as
regression guards against SymPy behavior changes.

## Index convention (load-bearing)

Everything keys off SymPy `Dummy` symbols carrying assumptions. Two orthogonal axes:

- **Monomer:** `is_molA=True` or `is_molB=True`. Contractions and KroneckerDeltas across monomers
  evaluate to zero; this is what makes the double-vacuum algebra work.
- **Fermi level:** `above_fermi=True` (particle/virtual), `below_fermi=True` (hole/occupied), or
  neither (general).

Canonical dummy names produced by `substitute_dummies_double_vac`:
`a`=particle-A, `i`=hole-A, `p`=general-A; `b`=particle-B, `j`=hole-B, `q`=general-B (with
`_1, _2, …` suffixes). Create indices with e.g. `symbols("a", is_molA=True, above_fermi=True, cls=Dummy)`.

## Typical derivation pipeline

1. Build an operator expression from `DoubleVacuumTensorSymbol` (amplitudes/integrals like
   `v^{pr}_{qs}`) times fermion operators `a, ad, b, bd` (imported from `operators`).
   `sapt_utils` provides ready-made builders (`get_V_operator`, `get_Pn_operator`, `get_R_nm`, …).
2. `wicks_double_vac(expr, keep_only_fully_contracted=True)` — apply generalized Wick's theorem,
   evaluate deltas, and canonicalize dummies so equivalent terms collapse.
3. `spin_integration(expr)` — RHF spin integration (multiplies each term by `2**(#loops)`).
4. Output: `latex(expr)` / `utils.format_expr` for formulas, or `code_generator.generate_einsum`
   for runnable `np.einsum` code (uses psi4numpy index naming: a→r, b→s, i→a, j→b).

`examples/sapt_pol20.py` is the canonical end-to-end demonstration (E_pol(10), E_ind(20), E_disp(20)).

## Module map (`src/sym2quantized_sapt/`)

- `operators.py` — `AnnihilateFermion_A/B`, `CreateFermion_A/B` subclassing SymPy fermion ops +
  a `DoubleFermiVaccum` mixin carrying `is_molA`/`is_molB`. Exported aliases `a, ad, b, bd`.
- `double_fermi_vac.py` — **the core.** `wicks_double_vac` (main entry), `NO_double_vac`
  (normal-ordering split into A/B parts), `contraction_double_vac` (same-monomer contractions only),
  `evaluate_deltas_double_vac` (Einstein-summation delta evaluation respecting monomer tags),
  `substitute_dummies_double_vac` (term canonicalization — key to collapsing equivalent terms),
  `get_fully_contracted`, `commutator`, `anticommutator`.
- `tensors.py` — `DoubleVacuumTensorSymbol` (symbol + upper/lower index tuples + optional
  permutation symmetries applied at construction).
- `sapt_utils.py` — operator builders: interaction `V`, exchange operators `get_a/b_operator`,
  permutation operators `get_P2/P4/Pn_operator`, resolvent superoperator `get_R_nm`.
- `spin_integrator.py` — `spin_integration` + `_count_loops` (Goldstone-diagram loop counting).
- `sinfinitizer.py` — `sinfinitizer`: expands overlap integrals (S^∞), wiring tensors together in
  all ways and assigning signs from loop/hole-line parity.
- `diagrams.py` — `get_only_linked`: keeps only connected (linked) terms via graph traversal.
- `code_generator.py` — `generate_einsum`: SymPy expression → `np.einsum` source string.
- `utils.py` — `format_expr` (LaTeX align formatting), `timeit` decorator.

## Conventions

- `examples/*.py` are executable derivation scripts; `tests/test_examples.py` runs each (except the
  `EXAMPLES_BLACKLIST`, currently `coupled_cluster.py`) as a subprocess under the `slow` marker and
  asserts a zero exit code. Adding an example automatically adds it to that slow test.
- The Polish-flavored spellings in identifiers/docstrings (`DoubleFermiVaccum`, `Prepers`,
  `indicies`, `coresponding`) are established API names — match existing spelling rather than
  "correcting" it, or you will break imports and tests.
