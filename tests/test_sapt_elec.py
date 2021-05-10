import pytest

from sympy import symbols, Dummy, latex

from sym2quantized_sapt.double_fermi_vac import (
    TensorDoubleVac,
    ad,
    a,
    bd,
    b,
    wicks_double_vac,
)

from sym2quantized_sapt import spin_integration


def test_can_evaluate_sapt_pol_10_energy():

    reference_latex = r"V_{0} + 4 v^{ij}_{ij} + 2 v_A^{j}_{j} + 2 v_B^{i}_{i}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    v = TensorDoubleVac(
        "v",
        (p, r),
        (q, s),
    )
    vA = TensorDoubleVac("(v_A)", (r,), (s,))
    vB = TensorDoubleVac("(v_B)", (p,), (q,))
    V0 = symbols("V_0")

    V = (
        v * ad(q) * a(p) * bd(s) * b(r)
        + vA * bd(s) * b(r)
        + vB * ad(q) * a(p)
        + V0
    )

    expr = V
    expr = wicks_double_vac(expr, keep_only_fully_contracted=True)
    expr = spin_integration(expr)
    tested_expr = latex(expr)
    assert reference_latex == tested_expr
