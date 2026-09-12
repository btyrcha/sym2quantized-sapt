import pytest
from sympy import Add, Mul, symbols

from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.sapt_utils import get_R_nm, get_V_operator
from sym2quantized_sapt.open_shell import (
    rhf_collapse,
    spin_integration_uhf,
)
from sym2quantized_sapt.spin_integrator import (
    _loop_partition,
    spin_integration,
)
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


def _disp20_indices():
    a = symbols("a", is_molA=True, above_fermi=True)
    i = symbols("i", is_molA=True, below_fermi=True)
    b = symbols("b", is_molB=True, above_fermi=True)
    j = symbols("j", is_molB=True, below_fermi=True)
    return a, i, b, j


def _t_and_v():
    a, i, b, j = _disp20_indices()
    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b))
    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    return t, v


def test_loop_partition_of_a_two_loop_term():
    t, v = _t_and_v()

    upper, lower = [], []
    for tensor in (t, v):
        upper += list(tensor.upper())
        lower += list(tensor.lower())
    loops = _loop_partition(upper, lower)

    # two loops, each threading one slot pair of t and one of v
    assert len(loops) == 2
    assert sorted(len(loop) for loop in loops) == [2, 2]


def test_uhf_blocks_a_two_loop_term_four_ways():
    t, v = _t_and_v()
    a, i, b, j = _disp20_indices()

    blocked = spin_integration_uhf(t * v)

    def pair(label):
        t_block = DoubleVacuumTensorSymbol("t_" + label, (i, j), (a, b))
        v_block = DoubleVacuumTensorSymbol("v_" + label, (a, b), (i, j))
        return t_block * v_block

    assert blocked == Add(*[pair(l) for l in ("aa", "ab", "ba", "bb")])


def test_rhf_collapse_recovers_spin_integration():
    t, v = _t_and_v()

    assert rhf_collapse(spin_integration_uhf(t * v)) == spin_integration(
        t * v
    )


def test_uhf_on_derived_e_disp20():
    E = wicks_double_vac(
        get_V_operator() * get_R_nm(1, 1, get_V_operator()),
        keep_only_fully_contracted=True,
    )

    blocked = spin_integration_uhf(E)

    # one spatial term, two loops: exactly the four spin cases, with
    # consistent labels on the denominator and both integrals
    assert len(blocked.args) == 4
    names = sorted(
        str(factor.symbol())
        for term in blocked.args
        for factor in term.args
        if isinstance(factor, DoubleVacuumTensorSymbol)
        and str(factor.symbol()).startswith("e_")
    )
    assert names == ["e_aa", "e_ab", "e_ba", "e_bb"]
    assert rhf_collapse(blocked) == spin_integration(E)


def test_blocked_symmetries_are_not_carried_over():
    # the pair symmetry of a doubles amplitude maps BETWEEN spin
    # blocks, so the blocked tensors must be built without it
    a, i, b, j = _disp20_indices()
    pair_symmetry = (((0, 1), (0, 1)), ((1, 0), (1, 0)))
    t = DoubleVacuumTensorSymbol("t", (i, j), (a, b), pair_symmetry)
    v = DoubleVacuumTensorSymbol("v", (a, b), (i, j))

    blocked = spin_integration_uhf(t * v)

    for term in blocked.args:
        for factor in term.args:
            if isinstance(factor, DoubleVacuumTensorSymbol):
                assert not factor.get_symmetries()


def test_unbalanced_tensor_is_rejected():
    a, i, b, j = _disp20_indices()
    lopsided = DoubleVacuumTensorSymbol("x", (i, j), (a,))

    with pytest.raises(ValueError, match="particle-conserving"):
        spin_integration_uhf(Mul(2, lopsided))


def test_numbers_pass_through():
    assert spin_integration_uhf(symbols("V_0")) == symbols("V_0")


def test_array_table_defines_blocked_arrays():
    from sym2quantized_sapt.code_generator import array_table

    t, v = _t_and_v()
    table = array_table(spin_integration_uhf(t * v))

    # four blocks of each tensor, all defined
    assert len(table) == 8
    entry = table["t_ab_rsab"]
    assert entry["base"] == "t" and entry["spin_block"] == "ab"
    # storage order: lower (a, b) then upper (i, j)
    assert [axis["space"] for axis in entry["axes"]] == ["v", "v", "o", "o"]
    assert [axis["role"] for axis in entry["axes"]] == ["l", "l", "u", "u"]
    # spin follows the slot pair: lower_k and upper_k share pair k
    assert [axis["spin"] for axis in entry["axes"]] == ["a", "b", "a", "b"]
    assert [axis["monomer"] for axis in entry["axes"]] == ["A", "B", "A", "B"]


def test_array_table_leaves_unblocked_tensors_unlabelled():
    from sym2quantized_sapt.code_generator import array_table

    t, v = _t_and_v()
    table = array_table(t * v)

    assert table["t_rsab"]["spin_block"] == ""
    assert all(axis["spin"] == "" for axis in table["t_rsab"]["axes"])


def test_fallback_summation_dummy_inherits_the_spin_tag():
    # a bubble: two general same-spin indices contract, and the fresh
    # particle/hole summation dummy must range over THAT spin only
    from sympy import Dummy, symbols as sy
    from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
    from sym2quantized_sapt.operators import a, ad

    p = sy("p", is_molA=True, is_alpha=True, cls=Dummy)
    q = sy("q", is_molA=True, is_alpha=True, cls=Dummy)
    u = DoubleVacuumTensorSymbol("u_a", (p,), (q,))

    result = wicks_double_vac(
        u * ad(q) * a(p), keep_only_fully_contracted=True,
        substitute_dummies=False,
    )

    tags = [
        index.assumptions0.get("is_alpha")
        for index in result.atoms(Dummy)
    ]
    assert tags and all(tags)
