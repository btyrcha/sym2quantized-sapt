import pytest

from sympy import symbols, Dummy, latex

from sym2quantized_sapt.double_fermi_vac import (
    wicks_double_vac,
)

from sym2quantized_sapt.operators import A, Ad, B, Bd
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


def test_can_evaluate_sapt_disp_20_energy():
    """calculates e20_disp energy expression"""

    reference_latex = r"\frac{4 v^{ab}_{ij} v^{ij}_{ab}}{e^{ab}_{ij}}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    a1 = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i1 = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    b1 = symbols("b", is_molB=True, above_fermi=True, cls=Dummy)
    j1 = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)

    v = DoubleVacuumTensorSymbol(
        "v",
        (p, r),
        (q, s),
    )
    vA = DoubleVacuumTensorSymbol("(v_A)", (r,), (s,))
    vB = DoubleVacuumTensorSymbol("(v_B)", (p,), (q,))
    V0 = symbols("V_0")

    V = (
        v * Ad(q) * A(p) * Bd(s) * B(r)
        + vA * Bd(s) * B(r)
        + vB * Ad(q) * A(p)
        + V0
    )

    T11 = (
        DoubleVacuumTensorSymbol(
            "v",
            (i1, j1),
            (a1, b1),
        )
        / DoubleVacuumTensorSymbol(
            "e",
            (a1, b1),
            (i1, j1),
        )
        * Ad(a1)
        * A(i1)
        * Bd(b1)
        * B(j1)
    )

    expr = V * T11
    expr = wicks_double_vac(expr, keep_only_fully_contracted=True)
    expr = spin_integration(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr
