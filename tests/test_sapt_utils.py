from sympy import Dummy, symbols, latex

from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.sapt_utils import (
    get_V_operator,
    get_a_operator,
    get_b_operator,
)
from sym2quantized_sapt.spin_integrator import spin_integration


def test_get_a_operator_for_one_electron():
    """a^{p_0}_{q_0} = ad(p_0) a(q_0)"""
    reference = r"a^\dagger_{p_0} a_{q_0}"

    tested_str = latex(get_a_operator(n=1))

    assert reference == tested_str


def test_get_a_operator_reverses_the_annihilators():
    """the annihilation part runs backwards, so the two electron operator
    is ad(p_0) ad(p_1) a(q_1) a(q_0) and not ... a(q_0) a(q_1)"""
    reference = r"a^\dagger_{p_0} a^\dagger_{p_1} a_{q_1} a_{q_0}"

    tested_str = latex(get_a_operator(n=2))

    assert reference == tested_str


def test_get_b_operator_for_one_electron():
    """b^{r_0}_{s_0} = bd(r_0) b(s_0)"""
    reference = r"b^\dagger_{r_0} b_{s_0}"

    tested_str = latex(get_b_operator(n=1))

    assert reference == tested_str


def test_get_b_operator_reverses_the_annihilators():
    reference = r"b^\dagger_{r_0} b^\dagger_{r_1} b_{s_1} b_{s_0}"

    tested_str = latex(get_b_operator(n=2))

    assert reference == tested_str


def test_get_a_operator_with_explicit_indicies():
    """`n` wins over the explicit indicies, so they are only used when it
    is falsy - this is what the "if given ignors" in the docstring means"""
    reference = r"a^\dagger_{p} a^\dagger_{r} a_{s} a_{q}"

    p, q, r, s = symbols("p q r s", is_molA=True)

    tested_str = latex(get_a_operator([p, r], [q, s], n=0))

    assert reference == tested_str


def test_get_b_operator_with_explicit_indicies():
    reference = r"b^\dagger_{p} b_{q}"

    p, q = symbols("p q", is_molB=True)

    tested_str = latex(get_b_operator([p], [q], n=0))

    assert reference == tested_str


def test_get_a_operator_ignores_indicies_when_n_is_given():
    """the explicit indicies are silently discarded, not merged"""
    p, q = symbols("p q", is_molA=True)

    with_indicies = latex(get_a_operator([p], [q], n=1))
    without_indicies = latex(get_a_operator(n=1))

    assert with_indicies == without_indicies


def test_generated_indicies_are_free():
    """
    The indicies `n` generates are plain Symbols, not Dummies.

    A `Dummy` index is a summation index, and these are not.
    """
    assert not get_a_operator(n=2).atoms(Dummy)
    assert not get_b_operator(n=2).atoms(Dummy)


def test_every_call_builds_its_own_summation_indices():
    """
    `get_V_operator` has to hand out fresh dummies on every call.

    The summation indices are `Dummy` objects, and two factors holding the
    same `Dummy` are summed over one index rather than two - see
    `test_reusing_one_operator_object_in_a_product_collapses_it`.
    """
    first = get_V_operator()
    second = get_V_operator()

    assert first.atoms(Dummy)
    assert not first.atoms(Dummy) & second.atoms(Dummy)


def test_reusing_one_operator_object_in_a_product_collapses_it():
    """
    An operator that appears twice in a product needs to be built twice.

    `V * V` from a single object shares `p, q, r, s` between the two
    factors, so every index is summed once instead of twice: the product
    degenerates into a product of traces, and terms such as the dispersion
    -shaped `v^{ab}_{ij} v^{ij}_{ab}` are missing outright. Nothing raises.
    """
    V = get_V_operator()

    reused = spin_integration(
        wicks_double_vac(V * V, keep_only_fully_contracted=True)
    )
    fresh = spin_integration(
        wicks_double_vac(
            get_V_operator() * get_V_operator(),
            keep_only_fully_contracted=True,
        )
    )

    assert reused != fresh
    assert "v^{ab}_{ij} v^{ij}_{ab}" in latex(fresh)
    assert "v^{ab}_{ij} v^{ij}_{ab}" not in latex(reused)
