import pytest

from sympy import symbols, Dummy, latex

from sympy.physics.secondquant import Dagger

from sym2quantized_sapt.operators import A, Ad, B, Bd
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


def test_can_evaluate_simple_dagger_molA():
    reference_latex = r"a^\dagger_{i} a_{a}"

    a1 = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i1 = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    expr = Ad(a1) * A(i1)
    expr = Dagger(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_simple_dagger_molB():
    reference_latex = r"b^\dagger_{j} b_{b}"

    b1 = symbols("b", is_molB=True, below_fermi=True, cls=Dummy)
    j1 = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)

    expr = Bd(b1) * B(j1)
    expr = Dagger(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_four_operator_dagger_molA():
    reference_latex = r"a^\dagger_{q} a^\dagger_{q_1} a_{p_1} a_{p}"

    p, p1, q, q1 = symbols("p p_1 q q_1", is_molA=True, cls=Dummy)

    expr = Ad(p) * Ad(p1) * A(q1) * A(q)
    expr = Dagger(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_four_operator_dagger_molB():
    reference_latex = r"b^\dagger_{s} b^\dagger_{s_1} b_{r_1} b_{r}"

    r, r1, s, s1 = symbols("r r_1 s s_1", is_molB=True, cls=Dummy)

    expr = Bd(r) * Bd(r1) * B(s1) * B(s)
    expr = Dagger(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_tensor_dagger():
    reference_latex = r"v^{qs}_{pr}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    v = DoubleVacuumTensorSymbol(
        "v",
        (
            p,
            r,
        ),
        (
            q,
            s,
        ),
    )

    expr = Dagger(v)
    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_mixed_dagger():
    reference_latex = r"v^{qs}_{pr} b^\dagger_{r} b_{s} a^\dagger_{p} a_{q}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    v = DoubleVacuumTensorSymbol(
        "v",
        (
            p,
            r,
        ),
        (
            q,
            s,
        ),
    )

    expr = v * Ad(q) * A(p) * Bd(s) * B(r)
    expr = Dagger(expr)
    tested_expr = latex(expr)

    assert reference_latex == tested_expr
