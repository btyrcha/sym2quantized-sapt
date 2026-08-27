from sympy import symbols, S

from sym2quantized_sapt.utils import format_expr


def test_format_single_symbol():
    x = symbols("x")
    expr = x

    assert format_expr(expr) == "x"


def test_format_simple_mul():
    x, y = symbols("x y")
    expr = 2 * x * y

    assert format_expr(expr) == "2 x y"


def test_format_zero():
    assert format_expr(S.Zero) == "0"
