import pytest

from sympy import symbols, Dummy, latex

from sympy.physics.secondquant import Dagger

from sym2quantized_sapt.double_fermi_vac import (
    TensorDoubleVac,
    ad,
    a,
    bd,
    b,
)


def test_can_evaluate_simple_dagger_molA():

    reference_latex = r"a^\dagger_{i} a_{a}"

    a1 = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i1 = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    expr = ad(a1) * a(i1)
    expr = Dagger(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_simple_dagger_molB():

    reference_latex = r"b^\dagger_{j} b_{b}"

    b1 = symbols("b", is_molB=True, below_fermi=True, cls=Dummy)
    j1 = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)

    expr = bd(b1) * b(j1)
    expr = Dagger(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_four_operator_dagger_molA():

    reference_latex = r"a^\dagger_{q} a^\dagger_{q_1} a_{p_1} a_{p}"

    p, p1, q, q1 = symbols("p p_1 q q_1", is_molA=True, cls=Dummy)

    expr = ad(p) * ad(p1) * a(q1) * a(q)
    expr = Dagger(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_four_operator_dagger_molB():

    reference_latex = r"b^\dagger_{s} b^\dagger_{s_1} b_{r_1} b_{r}"

    r, r1, s, s1 = symbols("r r_1 s s_1", is_molB=True, cls=Dummy)

    expr = bd(r) * bd(r1) * b(s1) * b(s)
    expr = Dagger(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_tensor_dagger():

    reference_latex = r"v^{qs}_{pr}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    v = TensorDoubleVac("v", (p, r,), (q, s,))

    expr = Dagger(v)
    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_mixed_dagger():

    reference_latex = r"v^{qs}_{pr} b^\dagger_{r} b_{s} a^\dagger_{p} a_{q}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    v = TensorDoubleVac("v", (p, r,), (q, s,))

    expr = v * ad(q) * a(p) * bd(s) * b(r)
    expr = Dagger(expr)
    tested_expr = latex(expr)

    assert reference_latex == tested_expr
