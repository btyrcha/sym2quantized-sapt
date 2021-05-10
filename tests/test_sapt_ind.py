import pytest

from sympy import symbols, Dummy, latex

from sym2quantized_sapt.double_fermi_vac import (
    TensorDoubleVac,
    ad,
    a,
    bd,
    b,
    wicks_double_vac,
    get_fully_contracted,
    spin_integration,
)


def test_can_evaluate_sapt_indA_20_energy():

    reference_latex = r"\frac{4 o_B^{i}_{a} v^{aj}_{ij}}{e^{a}_{i}} + \frac{2 o_B^{i}_{a} v_B^{a}_{i}}{e^{a}_{i}}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    a1 = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i1 = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    v = TensorDoubleVac("v", (p, r,), (q, s,))
    vA = TensorDoubleVac("(v_A)", (r,), (s,))
    vB = TensorDoubleVac("(v_B)", (p,), (q,))
    V0 = symbols("V_0")

    V = (
        v * ad(q) * a(p) * bd(s) * b(r)
        + vA * bd(s) * b(r)
        + vB * ad(q) * a(p)
        + V0
    )

    T10 = (
        TensorDoubleVac("o_B", (i1,), (a1,))
        / TensorDoubleVac("e", (a1,), (i1,))
        * ad(a1)
        * a(i1)
    )

    expr = V * T10
    expr = wicks_double_vac(expr, keep_only_fully_contracted=True)
    # expr = get_fully_contracted(expr)
    expr = spin_integration(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_sapt_indB_20_energy():

    reference_latex = r"\frac{4 o_A^{j}_{b} v^{ib}_{ij}}{e^{b}_{j}} + \frac{2 o_A^{j}_{b} v_A^{b}_{j}}{e^{b}_{j}}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    b1 = symbols("b", is_molB=True, above_fermi=True, cls=Dummy)
    j1 = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)

    v = TensorDoubleVac("v", (p, r,), (q, s,))
    vA = TensorDoubleVac("(v_A)", (r,), (s,))
    vB = TensorDoubleVac("(v_B)", (p,), (q,))
    V0 = symbols("V_0")

    V = (
        v * ad(q) * a(p) * bd(s) * b(r)
        + vA * bd(s) * b(r)
        + vB * ad(q) * a(p)
        + V0
    )

    T01 = (
        TensorDoubleVac("o_A", (j1,), (b1,))
        / TensorDoubleVac("e", (b1,), (j1,))
        * bd(b1)
        * b(j1)
    )

    expr = V * T01
    expr = wicks_double_vac(expr, simplify_kronecker_deltas=True)
    expr = get_fully_contracted(expr)
    expr = spin_integration(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr
