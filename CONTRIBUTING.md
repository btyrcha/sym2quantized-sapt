# Contributing

Thanks for your interest in contributing to `sym2quantized-sapt`.

## Development setup

Python 3.8 is the reference interpreter (the pinned dependencies and the
string-exact tests target it). At least Python 3.8 is required.

```shell
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt -r requirements-dev.txt
python3 -m pip install -e .
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
SymPy version — keep `sympy==1.8.0` (see `requirements.txt`) when running them.

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
