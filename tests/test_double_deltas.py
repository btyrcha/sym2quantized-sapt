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
    Simple test checking if general p, q indices can be evaluted
    using evalute_deltas_double_vac.
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


def test_can_evaluate_cross_monomer_delta():
    """
    Test checking if cross monomer delta is evaluated to zero.
    """
    reference_latex = r"0"

    a1 = symbols("a1", is_molA=True, above_fermi=True, cls=Dummy)
    b1 = symbols("b1", is_molB=True, above_fermi=True, cls=Dummy)

    # building expression
    expr = ad(a1) * a(a1) * bd(b1) * b(b1) * KroneckerDelta(a1, b1)

    # evaluation using tested function
    expr = evaluate_deltas_double_vac(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_two_deltas():
    """
    Test checking if expression with two deltas is evaluated.
    """
    reference_latex = r"a^\dagger_{a1} a_{a1} b^\dagger_{b1} b_{b1}"

    p = symbols("p", is_molA=True, cls=Dummy)
    q = symbols("q", is_molB=True, cls=Dummy)
    a1 = symbols("a1", is_molA=True, above_fermi=True, cls=Dummy)
    b1 = symbols("b1", is_molB=True, above_fermi=True, cls=Dummy)

    expr = (
        ad(a1)
        * a(p)
        * KroneckerDelta(a1, p)
        * bd(b1)
        * b(q)
        * KroneckerDelta(b1, q)
    )

    expr = evaluate_deltas_double_vac(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_hole_particle_delta():
    """
    Test checking if hole-index, particle-index delta is evaluated to zero.
    """
    reference_latex = r"0"

    a1 = symbols("a1", is_molA=True, above_fermi=True, cls=Dummy)
    i1 = symbols("i1", is_molA=True, below_fermi=True, cls=Dummy)

    expr = ad(a1) * a(i1) * KroneckerDelta(a1, i1)

    expr = evaluate_deltas_double_vac(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_no_changes():
    """
    Test checking if the correct result is given when there is nothing to do.
    """
    reference_latex = r"\delta_{a_{1} p} a^\dagger_{a1}"

    p = symbols("p", is_molA=True, cls=Dummy)
    a1 = symbols("a1", is_molA=True, above_fermi=True, cls=Dummy)

    expr = ad(a1) * KroneckerDelta(a1, p)

    expr = evaluate_deltas_double_vac(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr
