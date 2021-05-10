import os
import pathlib
import shutil
import subprocess
from typing import Generator

import pytest


EXAMPLES_DIR = (
    pathlib.Path(__file__).parent.absolute() / ".." / "examples"
).absolute()
EXAMPLES_BLACKLIST = ["coupled_cluster.py"]


def collect_example_files(
    examples: pathlib.Path,
) -> Generator[pathlib.Path, None, None]:
    """yields all *.py under $GIT_ROOT/examples/ directory"""
    if not examples.exists():
        raise RuntimeError(f" {EXAMPLES_DIR} not directory found!")

    for example in examples.iterdir():
        if example.is_file() and example.suffix == ".py":
            yield example.absolute()


@pytest.fixture(autouse=True)
def _change_cwd(tmp_path):
    """helper fixture that changes CWD to tmp_path
    and restores it back to the original, for each test case
    """
    # get last cwd
    last_cwd = os.getcwd()

    # change cwd
    os.chdir(tmp_path)

    # yield set up phase
    yield

    # restore cwd in tear down
    os.chdir(last_cwd)


EXAMPLES = [
    example
    for example in collect_example_files(EXAMPLES_DIR)
    if example.name not in EXAMPLES_BLACKLIST
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "example_file",
    [example.absolute() for example in EXAMPLES],
    ids=[example.name for example in EXAMPLES],
)
def test_run_example(example_file: pathlib.Path, tmp_path: pathlib.Path):
    """runs all files in $GIT_ROOT/examples/*.py and checks the return code.

    Args:
        example_file (pathlib.Path): example file Path object
        tmp_path (pathlib.Path): pytest's fixture
    """

    source_file = example_file
    target_file = tmp_path / example_file.name

    # copy the file to tmp_path
    shutil.copyfile(source_file, target_file)

    # run as a subprocess call
    completed_run = subprocess.run(
        ["python3", target_file], stdout=subprocess.PIPE, check=True
    )
    # assert script didn't crash
    completed_run.check_returncode()
