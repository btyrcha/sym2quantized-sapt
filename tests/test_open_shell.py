"""
The spin-tag route (``open_shell``): ``is_alpha`` / ``is_beta`` on the
indices, consulted by Wick's theorem. These tests are what the route
guarantees: nothing contracts or survives a delta across opposite spins,
and no index that stands in for a tagged one loses the tag.
"""

from sympy import Dummy, KroneckerDelta, symbols, latex

from sym2quantized_sapt.double_fermi_vac import (
    evaluate_deltas_double_vac,
    wicks_double_vac,
)
from sym2quantized_sapt.operators import A, Ad
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


def test_fallback_summation_dummy_inherits_the_spin_tag():
    # a bubble: two general same-spin indices contract, and the fresh
    # particle/hole summation dummy must range over THAT spin only

    p = symbols("p", is_molA=True, is_alpha=True, cls=Dummy)
    q = symbols("q", is_molA=True, is_alpha=True, cls=Dummy)
    u = DoubleVacuumTensorSymbol("u_a", (p,), (q,))

    result = wicks_double_vac(
        u * Ad(q) * A(p),
        keep_only_fully_contracted=True,
        substitute_dummies=False,
    )

    tags = [
        index.assumptions0.get("is_alpha") for index in result.atoms(Dummy)
    ]

    assert tags and all(tags)


def test_contraction_with_untagged_hole_keeps_the_spin_tag():
    # a tagged general index meets an untagged hole: the contraction gives
    # d(p_a, i), and the hole that survives it must range over alpha only

    p = symbols("p", is_molA=True, is_alpha=True, cls=Dummy)
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)
    x = DoubleVacuumTensorSymbol("x", (p,), (i,))

    result = wicks_double_vac(
        x * Ad(p) * A(i),
        keep_only_fully_contracted=True,
        substitute_dummies=False,
    )

    tags = [
        index.assumptions0.get("is_alpha") for index in result.atoms(Dummy)
    ]

    assert tags and all(tags)


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


def test_delta_survivor_inherits_the_spin_tag():
    """
    Test checking if the index that survives a delta keeps the spin tag of
    the one it replaces: the hole i beats the general p_a on Fermi-level
    information, and must come out alpha-tagged.
    """
    p_alpha = symbols("p", is_molA=True, is_alpha=True, cls=Dummy)
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    expr = Ad(p_alpha) * A(i) * KroneckerDelta(p_alpha, i)

    result = evaluate_deltas_double_vac(expr)

    tags = [
        index.assumptions0.get("is_alpha") for index in result.atoms(Dummy)
    ]

    assert tags and all(tags)


def test_opposite_spins_through_untagged_index_give_zero():
    """
    Test checking if d(p_a, q) d(q, r_b) is zero: the untagged q takes the
    spin of whichever index it meets first, so the pair still clashes.
    """
    reference_latex = r"0"

    p_alpha = symbols("p", is_molA=True, is_alpha=True, cls=Dummy)
    q = symbols("q", is_molA=True, cls=Dummy)
    r_beta = symbols("r", is_molA=True, is_beta=True, cls=Dummy)

    expr = (
        Ad(p_alpha)
        * A(r_beta)
        * KroneckerDelta(p_alpha, q)
        * KroneckerDelta(q, r_beta)
    )

    expr = evaluate_deltas_double_vac(expr)

    tested_expr = latex(expr)

    assert reference_latex == tested_expr


def test_delta_with_free_untagged_survivor_is_kept():
    """
    Test checking if the delta is kept when the index that would survive
    is free and untagged: the result is nonzero only for one spin of it,
    which renaming a free index cannot express.
    """
    p_alpha = symbols("p", is_molA=True, is_alpha=True, cls=Dummy)
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    expr = Ad(p_alpha) * KroneckerDelta(p_alpha, i)

    assert evaluate_deltas_double_vac(expr) == expr
