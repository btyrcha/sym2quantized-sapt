import pytest

from sympy import Dummy, KroneckerDelta, symbols, latex

from sym2quantized_sapt.double_fermi_vac import (
    evaluate_deltas_double_vac,
)

from sym2quantized_sapt.operators import A, Ad, B, Bd


def test_can_evaluate_simple_delta():
    """
    Simple test checking if general p, q indices can be evaluted
    using evalute_deltas_double_vac.
    """
    reference_latex = r"a^\dagger_{p} a_{p}"

    p, q = symbols("p q", is_molA=True, cls=Dummy)

    # build the expression using abstact operators
    expr = Ad(p) * A(q) * KroneckerDelta(p, q)

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
    expr = Ad(a1) * A(a1) * Bd(b1) * B(b1) * KroneckerDelta(a1, b1)

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
        Ad(a1)
        * A(p)
        * KroneckerDelta(a1, p)
        * Bd(b1)
        * B(q)
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

    expr = Ad(a1) * A(i1) * KroneckerDelta(a1, i1)

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

    expr = Ad(a1) * KroneckerDelta(a1, p)

    expr = evaluate_deltas_double_vac(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_can_evaluate_opposite_spin_delta():
    """
    Test checking if opposite spin delta is evaluated to zero.
    """
    reference_latex = r"0"

    i_alpha = symbols(
        "i", is_molA=True, below_fermi=True, is_alpha=True, cls=Dummy
    )
    i_beta = symbols(
        "i", is_molA=True, below_fermi=True, is_beta=True, cls=Dummy
    )

    expr = Ad(i_alpha) * A(i_beta) * KroneckerDelta(i_alpha, i_beta)

    expr = evaluate_deltas_double_vac(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


@pytest.mark.xfail(
    strict=True,
    reason="the index that survives a delta is picked by Fermi-level "
    "information alone, so the alpha-tagged general `p` is replaced by the "
    "untagged hole `i` and the spin tag is dropped: the term then sums over "
    "both spins. Through an untagged index this also hides an opposite-spin "
    "pair from the zero check: d(p_a, q) d(q, r_b) comes back nonzero",
)
def test_delta_survivor_inherits_the_spin_tag():
    p_alpha = symbols("p", is_molA=True, is_alpha=True, cls=Dummy)
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    expr = Ad(p_alpha) * A(i) * KroneckerDelta(p_alpha, i)

    result = evaluate_deltas_double_vac(expr)

    tags = [
        index.assumptions0.get("is_alpha") for index in result.atoms(Dummy)
    ]

    assert tags and all(tags)
