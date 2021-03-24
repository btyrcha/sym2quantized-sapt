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


def test_can_evaluate_sapt_exch_10_energy():

    reference_latex = r"- 2 s^{i}_{b} s^{j}_{i} v_A^{b}_{j} - 2 s^{i}_{j} s^{j}_{a} v_B^{a}_{i} - 4 s^{i}_{b} s^{j}_{i} v^{i_1b}_{i_1j} - 4 s^{i}_{j} s^{j}_{a} v^{aj_1}_{ij_1} - 2 s^{i}_{b} s^{j}_{a} v^{ab}_{ij}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    a1 = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i1 = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

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

    P10 = (
        -TensorDoubleVac("s", (j1,), (a1,))
        * TensorDoubleVac("s", (i1,), (j1,))
        * ad(a1)
        * a(i1)
    )

    P01 = (
        -TensorDoubleVac("s", (j1,), (i1,))
        * TensorDoubleVac("s", (i1,), (b1,))
        * bd(b1)
        * b(j1)
    )

    P11 = (
        -TensorDoubleVac("s", (j1,), (a1,))
        * TensorDoubleVac("s", (i1,), (b1,))
        * ad(a1)
        * a(i1)
        * bd(b1)
        * b(j1)
    )

    P = P10 + P01 + P11

    expr = V * P
    expr = wicks_double_vac(expr, simplify_kronecker_deltas=True)
    expr = get_fully_contracted(expr)
    expr = spin_integration(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr
