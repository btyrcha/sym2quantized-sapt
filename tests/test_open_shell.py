"""
The spin-tag route (``open_shell``): ``is_alpha`` / ``is_beta`` on the
indices, consulted by Wick's theorem. These tests are what the route
guarantees: nothing contracts or survives a delta across opposite spins,
no index that stands in for a tagged one loses the tag (renaming
included), beta indices print with a bar, a tagged result cannot be
spin-integrated a second time by mistake, and generated code gives every
spin its own arrays.

Tensors here are named plainly (``t``, ``x``): on this route the tags
carry the spin, so a per-sector name like ``t_ab`` would only repeat it.
"""

import re

import pytest
from sympy import Add, Dummy, KroneckerDelta, Rational, Symbol, symbols, latex

from sym2quantized_sapt.code_generator import array_table, generate_einsum
from sym2quantized_sapt.double_fermi_vac import (
    evaluate_deltas_double_vac,
    substitute_dummies_double_vac,
    wicks_double_vac,
)
from sym2quantized_sapt.open_shell import has_spin_tags
from sym2quantized_sapt.operators import A, Ad
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


def test_fallback_summation_dummy_inherits_the_spin_tag():
    # a bubble: two general same-spin indices contract, and the fresh
    # particle/hole summation dummy must range over THAT spin only

    p = symbols("p", is_molA=True, is_alpha=True, cls=Dummy)
    q = symbols("q", is_molA=True, is_alpha=True, cls=Dummy)
    u = DoubleVacuumTensorSymbol("u", (p,), (q,))

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


def _tagged(name, spin, **assumptions):
    tag = {"is_alpha": True} if spin == "a" else {"is_beta": True}
    return symbols(name, is_molA=True, cls=Dummy, **tag, **assumptions)


def _slot_spins(tensor):
    return "".join(
        "a" if index.assumptions0.get("is_alpha") else "b"
        for index in (*tensor.upper, *tensor.lower)
    )


def test_renaming_keeps_every_index_in_its_spin():
    # an alpha-only and a beta-only term: both draw the first hole and the
    # first particle from the renaming pools, so a pool shared by the two
    # spins hands one of the terms the wrong spin - in every process
    x = DoubleVacuumTensorSymbol(
        "x",
        (_tagged("i", "a", below_fermi=True),),
        (_tagged("a", "a", above_fermi=True),),
    )
    y = DoubleVacuumTensorSymbol(
        "y",
        (_tagged("i", "b", below_fermi=True),),
        (_tagged("a", "b", above_fermi=True),),
    )

    result = substitute_dummies_double_vac(x + y)

    spins = {str(t.symbol): _slot_spins(t) for t in Add.make_args(result)}

    assert spins == {"x": "aa", "y": "bb"}


def test_alpha_and_beta_indices_are_renamed_apart():
    # generate_einsum identifies indices by name, so an alpha and a beta
    # hole must not both be renamed to `i`: that would give the same
    # einsum letter to two different indices
    i_alpha = _tagged("i", "a", below_fermi=True)
    i_beta = _tagged("i", "b", below_fermi=True)
    a_alpha = _tagged("a", "a", above_fermi=True)
    a_beta = _tagged("a", "b", above_fermi=True)

    t = DoubleVacuumTensorSymbol("t", (i_alpha, i_beta), (a_alpha, a_beta))
    x = DoubleVacuumTensorSymbol("x", (a_alpha, a_beta), (i_alpha, i_beta))

    result = substitute_dummies_double_vac(t * x)

    assert [_slot_spins(f) for f in result.args] == ["abab", "abab"]
    assert len({index.name for index in result.atoms(Dummy)}) == 4
    assert (
        generate_einsum(result).strip()
        == '+np.einsum("rcad,adrc", t_rraa_abab, x_aarr_abab)'
    )


def test_beta_indices_print_with_a_bar():
    i_alpha = _tagged("i", "a", below_fermi=True)
    i_beta = _tagged("i_1", "b", below_fermi=True)
    a_alpha = _tagged("a", "a", above_fermi=True)
    a_beta = _tagged("a_1", "b", above_fermi=True)

    t = DoubleVacuumTensorSymbol("t", (i_alpha, i_beta), (a_alpha, a_beta))

    assert latex(t) == r"t^{i\bar{i}_1}_{a\bar{a}_1}"
    assert latex(Ad(i_beta)) == r"a^\dagger_{\bar{i}_1}"
    assert latex(A(a_alpha)) == r"a_{a}"


def test_has_spin_tags():
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)
    a = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    spatial = DoubleVacuumTensorSymbol("x", (i,), (a,))

    assert not has_spin_tags(Rational(1, 2) * spatial)

    # one tagged index is enough, summed or free
    i_beta = _tagged("i", "b", below_fermi=True)
    assert has_spin_tags(
        spatial * DoubleVacuumTensorSymbol("y", (a,), (i_beta,))
    )
    assert has_spin_tags(Ad(Symbol("p", is_molA=True, is_alpha=True)))


def test_spin_integration_rejects_spin_tagged_input():
    # an alpha-only loop is already resolved by spin: RHF spin integration
    # would multiply it by 2 and count the loop a second time
    i = _tagged("i", "a", below_fermi=True)
    a = _tagged("a", "a", above_fermi=True)
    term = DoubleVacuumTensorSymbol("x", (i,), (a,)) * (
        DoubleVacuumTensorSymbol("y", (a,), (i,))
    )

    with pytest.raises(ValueError, match="spin tags"):
        spin_integration(term)


def _code_arrays(code):
    """The array names a block of generated einsum lines refers to."""
    return {
        name
        for arguments in re.findall(r'np\.einsum\("[^"]*", ([^)]*)\)', code)
        for name in arguments.split(", ")
    }


def test_plain_tensor_names_get_one_array_per_spin():
    # with a spin-blind array name the alpha and the beta term would both
    # refer to `t_ra` and `v_ar`: one array for two different ones
    i_alpha = _tagged("i", "a", below_fermi=True)
    a_alpha = _tagged("a", "a", above_fermi=True)
    i_beta = _tagged("i_1", "b", below_fermi=True)
    a_beta = _tagged("a_1", "b", above_fermi=True)

    def term(i, a):
        return DoubleVacuumTensorSymbol(
            "t", (i,), (a,)
        ) * DoubleVacuumTensorSymbol("v", (a,), (i,))

    code = generate_einsum(term(i_alpha, a_alpha) + term(i_beta, a_beta))

    assert code.splitlines() == [
        '+np.einsum("ra,ar", t_ra_aa, v_ar_aa)',
        '+np.einsum("cd,dc", t_ra_bb, v_ar_bb)',
    ]


def test_array_table_takes_spins_from_the_tags():
    i_alpha = _tagged("i", "a", below_fermi=True)
    i_beta = _tagged("i_1", "b", below_fermi=True)
    a_alpha = _tagged("a", "a", above_fermi=True)
    a_beta = _tagged("a_1", "b", above_fermi=True)

    # beta slots first: any order is valid, the tags say which is which
    t = DoubleVacuumTensorSymbol("t", (i_beta, i_alpha), (a_beta, a_alpha))

    table = array_table(t)

    assert list(table) == ["t_rraa_baba"]
    assert table["t_rraa_baba"]["base"] == "t"
    assert table["t_rraa_baba"]["spin_block"] == "baba"
    assert [axis["spin"] for axis in table["t_rraa_baba"]["axes"]] == [
        "b",
        "a",
        "b",
        "a",
    ]

    # per-sector names are not the convention on this route, but a name
    # that looks like a per-loop block label must not be read as one
    t_ab = DoubleVacuumTensorSymbol(
        "t_ab", (i_beta, i_alpha), (a_beta, a_alpha)
    )

    (entry,) = array_table(t_ab).values()

    assert entry["base"] == "t_ab" and entry["spin_block"] == "baba"


def test_array_table_keys_match_the_code_for_tagged_input():
    # the table must describe exactly the arrays the generated code refers
    # to: the monomer-potential rename and the density-fitting factors
    # included
    i = _tagged("i", "a", below_fermi=True)
    a = _tagged("a", "a", above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True, is_beta=True, cls=Dummy)
    b = symbols("b", is_molB=True, above_fermi=True, is_beta=True, cls=Dummy)

    potential = DoubleVacuumTensorSymbol(
        "v_A", (i,), (a,)
    ) * DoubleVacuumTensorSymbol("t", (a,), (i,))
    eri = DoubleVacuumTensorSymbol(
        "v", (a, b), (i, j)
    ) * DoubleVacuumTensorSymbol("t", (i, j), (a, b))

    for expr, density_fitting in (
        (potential, False),
        (eri, False),
        (eri, True),
    ):
        code = generate_einsum(expr, density_fitting=density_fitting)
        table = array_table(expr, density_fitting=density_fitting)

        assert set(table) == _code_arrays(code)

    # each density-fitting factor carries the spins of its slot pair
    assert {"Qar_aa", "Qbs_bb"} <= _code_arrays(
        generate_einsum(eri, density_fitting=True)
    )


def test_partly_tagged_tensor_is_rejected_by_code_generation():
    # an untagged axis next to tagged ones has no single orbital range
    # under UHF, so it cannot become one array axis
    p = symbols("p", is_molA=True, cls=Dummy)
    x = DoubleVacuumTensorSymbol(
        "x", (_tagged("i", "a", below_fermi=True),), (p,)
    )

    with pytest.raises(ValueError, match="both spin-tagged and untagged"):
        generate_einsum(x)

    with pytest.raises(ValueError, match="both spin-tagged and untagged"):
        array_table(x)
