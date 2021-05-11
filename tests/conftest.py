import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", help="run the slow test cases as well"
    )


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--slow"):
        pytest.skip("need --slow option to run this test")
