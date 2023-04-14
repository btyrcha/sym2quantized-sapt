import pytest

from sympy import Dummy, symbols, latex

from sym2quantized_sapt.double_fermi_vac import (
    contraction_double_vac,
)

from sym2quantized_sapt.operators import a, ad, b, bd


def test_can_evaluate_hole_contraction():
    reference_latex = r"\delta_{i_{1} i_{2}}"

    i1, i2 = symbols("i1 i2", is_molA=True, below_fermi=True, cls=Dummy)
    # ad(i1) a(i2)
    expr = contraction_double_vac(ad(i1), a(i2))
    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_particle_contraction():
    reference_latex = r"\delta_{b_{1} b_{2}}"

    b1, b2 = symbols("b1 b2", is_molB=True, above_fermi=True, cls=Dummy)
    # b(b1) bd(b2)
    expr = contraction_double_vac(b(b1), bd(b2))
    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_zero_contraction_1():
    reference_latex = r"0"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    # ad(p) ad(q)
    expr = contraction_double_vac(ad(p), ad(q))
    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_zero_contraction_2():
    reference_latex = r"0"

    r, s = symbols("r s", is_molB=True, cls=Dummy)
    # b(r) b(s)
    expr = contraction_double_vac(b(r), b(s))
    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_cross_monomer_contraction():
    reference_latex = r"0"

    p = symbols("p", is_molA=True, cls=Dummy)
    r = symbols("r", is_molB=True, cls=Dummy)
    # a(p) b(r)
    expr = contraction_double_vac(a(p), b(r))
    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_general_indicies_contraction_1():
    reference_latex = r"\delta_{i q} \delta_{p q}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    # ad(p) a(q)
    expr = contraction_double_vac(ad(p), a(q))
    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_general_indicies_contraction_2():
    reference_latex = r"\delta_{b s} \delta_{r s}"

    r, s = symbols("r s", is_molB=True, cls=Dummy)
    # b(r) bd(s)
    expr = contraction_double_vac(b(r), bd(s))
    tested_expr = latex(expr)

    assert reference_latex == tested_expr
