from sympy import symbols, Dummy, Add, latex
from sympy.physics.secondquant import Dagger

from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.operators import A, Ad
from sym2quantized_sapt.double_fermi_vac import (
    NO_double_vac,
    wicks_double_vac,
    commutator,
)
from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.spin_integrator import spin_integration

v_SYMMETRIES = [((0, 1), (0, 1)), ((1, 0), (1, 0))]


def get_CA_operator():
    """
    Prepares C_A operator.
    """

    i = symbols("i", below_fermi=True, is_molA=True, cls=Dummy)
    a = symbols("a", above_fermi=True, is_molA=True, cls=Dummy)

    return DoubleVacuumTensorSymbol("c", (i,), (a,)) * Ad(a) * A(i)


def get_WA_operator():
    """
    Prepare W_A operator.
    """
    p1, p2, p3, p4 = symbols("p_1 p_2 p_3 p_4", is_molA=True, cls=Dummy)

    vv = DoubleVacuumTensorSymbol(
        "v",
        (
            p3,
            p4,
        ),
        (
            p1,
            p2,
        ),
        symmetries=v_SYMMETRIES,
    ) - DoubleVacuumTensorSymbol(
        "v",
        (
            p4,
            p3,
        ),
        (
            p1,
            p2,
        ),
        symmetries=v_SYMMETRIES,
    )

    return 0.25 * vv * NO_double_vac(Ad(p1) * Ad(p2) * A(p4) * A(p3)).doit()


def get_T10_operator():
    """
    Prepares the T10 excitation operator.
    """

    a = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    return DoubleVacuumTensorSymbol("t_B", (i,), (a,)) * Ad(a) * A(i)


def test_wa_ca_commutator_collapses_under_v_symmetry():
    """<i_3 a_3| [W_A, C_A - C_A^dag] T_10> derived end to end.

    v carries the coupled symmetry v^{pq}_{rs} = v^{qp}_{sr}, so two
    terms differing only by swapping both index pairs are the same term
    and have to collapse. 16 terms with coefficients of 2 or 4 is the
    collapsed form; canonicalizing the upper and lower rows independently
    left 20 terms with the coefficients split instead (0.5 + 1.5 where
    -2.0 belongs), because the independently sorted layout is not in the
    orbit of a symmetry that couples the two rows.
    """
    reference_latex = (
        r"4.0 c^{a}_{i} t_B^{i}_{a_1} v^{a_1i_3}_{aa_3} "
        r"- 2.0 c^{a}_{i} t_B^{i}_{a_1} v^{a_1i_3}_{a_3a} "
        r"- 4.0 c^{a}_{i} t_B^{i_1}_{a} v^{ii_3}_{i_1a_3} "
        r"+ 2.0 c^{a}_{i} t_B^{i_1}_{a} v^{ii_3}_{a_3i_1} "
        r"- 4.0 c^{a}_{i} t_B^{i_1}_{a_3} v^{ii_3}_{ai_1} "
        r"+ 2.0 c^{a}_{i} t_B^{i_1}_{a_3} v^{ii_3}_{i_1a} "
        r"- 2.0 c^{a}_{i} t_B^{i_3}_{a_1} v^{a_1i}_{aa_3} "
        r"+ 4.0 c^{a}_{i} t_B^{i_3}_{a_1} v^{a_1i}_{a_3a} "
        r"- 4.0 c^{i}_{a} t_B^{i_1}_{a_3} v^{ai_3}_{ii_1} "
        r"+ 2.0 c^{i}_{a} t_B^{i_1}_{a_3} v^{ai_3}_{i_1i} "
        r"+ 4.0 c^{i}_{a} t_B^{i_3}_{a_1} v^{aa_1}_{ia_3} "
        r"- 2.0 c^{i}_{a} t_B^{i_3}_{a_1} v^{aa_1}_{a_3i} "
        r"+ 2.0 c^{i}_{a_3} t_B^{i_1}_{a} v^{ai_3}_{ii_1} "
        r"- 4.0 c^{i}_{a_3} t_B^{i_1}_{a} v^{ai_3}_{i_1i} "
        r"- 2.0 c^{i_3}_{a} t_B^{i}_{a_1} v^{aa_1}_{ia_3} "
        r"+ 4.0 c^{i_3}_{a} t_B^{i}_{a_1} v^{aa_1}_{a_3i}"
    )

    expr = (
        Ad(symbols("i_3", below_fermi=True, is_molA=True))
        * A(symbols("a_3", above_fermi=True, is_molA=True))
        * commutator(
            get_WA_operator(), get_CA_operator() - Dagger(get_CA_operator())
        )
        * get_T10_operator()
    )

    expr = wicks_double_vac(expr, keep_only_fully_contracted=True)
    expr = get_only_linked(expr)
    expr = spin_integration(expr)

    assert len(Add.make_args(expr)) == 16
    assert latex(expr) == reference_latex
