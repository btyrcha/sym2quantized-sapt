from pytest import approx
from sympy import Dummy, Mul, Rational, symbols
from sympy.physics.secondquant import TensorSymbol

from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.operators import A, Ad
from sym2quantized_sapt.sapt_utils import get_R_nm
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


def crossed_mp2_term(is_graph_vertex=False):
    """e^{i i1}_{a a1} v^{a a1}_{i1 i} v^{i i1}_{a1 a}.

    The v's form two loops, through (a, i1) and (a1, i); the slot pairs
    of e, (i, a) and (i1, a1), cross them.  `get_R_nm` produces exactly
    this layout, since canonicalization permutes e's rows independently.
    """
    a, a1 = symbols("a a1", is_molA=True, above_fermi=True, cls=Dummy)
    i, i1 = symbols("i i1", is_molA=True, below_fermi=True, cls=Dummy)

    return (
        DoubleVacuumTensorSymbol(
            "e", (i, i1), (a, a1), is_graph_vertex=is_graph_vertex
        )
        * DoubleVacuumTensorSymbol("v", (a, a1), (i1, i))
        * DoubleVacuumTensorSymbol("v", (i, i1), (a1, a))
    )


def mp2_fluctuation_operator():
    p, q = symbols("p q", is_molA=True, cls=Dummy)
    p2, q2 = symbols("p' q'", is_molA=True, cls=Dummy)

    return (
        Rational(1, 2)
        * DoubleVacuumTensorSymbol("v", (p, p2), (q, q2))
        * Ad(q)
        * Ad(q2)
        * A(p2)
        * A(p)
    )


def test_denominator_is_not_a_graph_vertex():
    term = crossed_mp2_term()

    assert spin_integration(term) == 4 * term


def test_vertex_with_the_same_pairing_joins_loops():
    # the contrast: were e a vertex, its slot pairs would be lines and
    # merge the two loops into one
    term = crossed_mp2_term(is_graph_vertex=True)

    assert spin_integration(term) == 2 * term


def test_closed_shell_mp2_is_2A_minus_B():
    """
    <W R_(2,0) W> has 8 direct terms (2 loops each) and 8 exchange terms
    (1 loop each), all with coefficient +-1/16, so the RHF result sums to
    8 * 4/16 - 8 * 2/16 = 1, i.e. 2A - B.  Counting the denominator
    gave half the direct terms one loop and a sum of 1/2 (1.5A - B).
    """
    E2 = wicks_double_vac(
        mp2_fluctuation_operator()
        * get_R_nm(2, 0, mp2_fluctuation_operator()),
        keep_only_fully_contracted=True,
    )

    coefficients = [
        Mul(*[f for f in term.args if not isinstance(f, TensorSymbol)])
        for term in spin_integration(E2).args
    ]

    assert len(coefficients) == 16
    assert float(sum(coefficients)) == approx(1.0)
