import pytest

from sympy import Dummy, symbols, latex, expand

from sym2quantized_sapt.double_fermi_vac import (
    a,
    ad,
    b,
    bd,
    _get_contractions_double_vac,
    evaluate_deltas_double_vac,
)


def test_can_evaluate_one_mol_contraction():

    reference_latex = r"- \delta_{i_{1} p} \delta_{i_{1} q} - \delta_{i_{1} p} \left\{a^\dagger_{q} a_{s}\right\} + \delta_{i_{1} p} \delta_{i_{1} q} + \delta_{i_{1} p} \left\{a^\dagger_{q} a_{r}\right\} + \delta_{i_{1} q} \left\{a^\dagger_{p} a_{s}\right\} - \delta_{i_{1} q} \left\{a^\dagger_{p} a_{r}\right\} + \left\{a^\dagger_{p} a^\dagger_{q} a_{r} a_{s}\right\}"

    p, q, r, s = symbols("p q r s", is_molA=True, cls=Dummy)

    expr = ad(p) * ad(q) * a(r) * a(s)
    expr = _get_contractions_double_vac(expr.args)
    expr = expand(expr)
    expr = evaluate_deltas_double_vac(expr)
    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_two_mol_contraction():

    reference_latex = r"\delta_{i_{1} q} \delta_{j_{1} s} + \delta_{i_{1} q} \left\{b^\dagger_{s} b_{r}\right\} + \delta_{j_{1} s} \left\{a^\dagger_{q} a_{p}\right\} + \left\{a^\dagger_{q} a_{p}\right\} \left\{b^\dagger_{s} b_{r}\right\}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    expr = ad(q) * a(p) * bd(s) * b(r)
    expr = _get_contractions_double_vac(expr.args)
    expr = expand(expr)
    expr = evaluate_deltas_double_vac(expr)
    tested_expr = latex(expr)

    assert reference_latex == tested_expr
