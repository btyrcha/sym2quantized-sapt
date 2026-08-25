# Contributing

Thanks for your interest in contributing to `sym2quantized-sapt`.

## Development setup

Python 3.8 is the reference interpreter (it is what CI runs); 3.9-3.12 also
work. `setup.py` is the single source of truth for dependencies - the `dev`
extra adds the test, lint and format tooling.

```shell
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Or with conda:

```shell
conda env create -f environment.yml
conda activate sym2quantized-sapt
```

## Running the tests

```shell
# fast tests (example runners and other heavy cases are marked `slow`)
python3 -m pytest ./tests/

# full suite, including the slow tests — note the custom `--slow` flag
python3 -m pytest ./tests/ --slow

# a single test
python3 -m pytest tests/test_sapt_disp.py::test_can_evaluate_sapt_disp_20_energy
```

Many tests assert exact `sympy.latex(...)` output, so they are sensitive to the
SymPy version. `setup.py` caps it at `sympy>=1.8.0,<1.13.0`: 1.13 changed
`Float`/`int` equality and breaks coefficient formatting in `code_generator`.

## Style and checks

- Formatting is enforced with `black` (line length **79**).
- `pre-commit` runs black plus whitespace/line-ending hooks:

  ```shell
  pre-commit install   # once
  pre-commit run -a    # check everything
  ```

- `pylint` (config in `.pylintrc`) runs in CI as a non-blocking job.

CI (GitHub Actions, `.github/workflows/ci.yml`) runs the same checks on
Python 3.8 for every pull request against `main`.

## Notes for new derivations

Add executable derivation scripts under `examples/`. Each `examples/*.py` is run
by `tests/test_examples.py` (under the `slow` marker) and must exit cleanly.
