import pytest

from sympy import Dummy, KroneckerDelta, symbols, latex

from sym2quantized_sapt.double_fermi_vac import (
    a,
    ad,
    b,
    bd,
    evaluate_deltas_double_vac,
)


def test_can_evaluate_simple_delta():
    """
    simple test checking if general p, q indices can be evaluted
    using evalute_deltas_double_vac
    """
    reference_latex = r"a^\dagger_{p} a_{p}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)

    # build the expression using abstact operators
    expr = ad(p) * a(q) * KroneckerDelta(p, q)

    # evaluate the expression using our tested function
    expr = evaluate_deltas_double_vac(expr)

    # get comperable representation of the result
    tested_expr = latex(expr)

    # assert if  the result matches our expectation
    assert reference_latex == tested_expr
