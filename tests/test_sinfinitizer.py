from sympy import Add, Dummy, Mul, latex, symbols

from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.double_fermi_vac import substitute_dummies_double_vac
from sym2quantized_sapt.sinfinitizer import (
    count_hole_lines,
    count_loops,
    sinfinitizer,
)
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.utils import format_expr


PRETTY_INDICES = {
    "above_molA": "r",
    "above_molB": "s",
    "below_molA": "a",
    "below_molB": "b",
}


def _mol_indices():
    """fresh (a, i, b, j) dummies: particle/hole on monomer A and B"""
    a = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)
    b = symbols("b", is_molB=True, above_fermi=True, cls=Dummy)
    j = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)
    return a, i, b, j


def test_count_loops_closed_pair():
    """v^{ab}_{ij} t^{ij}_{ab} closes into two Goldstone loops."""
    a, i, b, j = _mol_indices()
    expr = DoubleVacuumTensorSymbol(
        "v", (a, b), (i, j)
    ) * DoubleVacuumTensorSymbol("t", (i, j), (a, b))

    assert count_loops(expr) == 2


def test_count_hole_lines_counts_only_below_fermi():
    """Only below_fermi indices are hole lines."""
    a, i, b, j = _mol_indices()

    assert count_hole_lines([a, i, b, j]) == 2
    assert count_hole_lines([a, b]) == 0
    assert count_hole_lines([]) == 0


def test_minimal_expansion_is_single_negative_term():
    """o_B^{a}_{i} closed by one S-integral, with the sign derived.

    sinfinitizer assigns (-1) ** (l + h + hidden_holes). The single
    generated integral here is E^{i}_{a} (monomer A, hole from
    particle), which _get_tensor_symbol counts as one hidden hole line:

        l = 1, h = 1, hidden_holes = 1  ->  (-1) ** 3 = -1

    l and h are asserted independently below, so this pins the rule
    rather than just the answer.
    """
    a, i, _, _ = _mol_indices()
    oB_ia = DoubleVacuumTensorSymbol("o_B", (a,), (i,))

    result = sinfinitizer((oB_ia,), ())

    # exactly one permutation of one index -> one term
    assert isinstance(result, Mul)
    assert count_loops(result) == 1
    assert count_hole_lines([a, i]) == 1

    coeff, _ = result.as_coeff_Mul()
    assert coeff == -1
    assert latex(result) == r"- E^{i}_{a} o_B^{a}_{i}"


def test_expansion_is_not_sign_uniform():
    """The sign rule must actually vary across an expansion.

    Collapsing (-1) ** (l + h + hidden_holes) to a constant makes every
    term share a sign. This catches that without needing a reference
    value for the expansion itself.
    """
    a, i, _, _ = _mol_indices()
    oB_ia = DoubleVacuumTensorSymbol("o_B", (a,), (i,))
    tB_ai = DoubleVacuumTensorSymbol("t_B", (i,), (a,))

    result = sinfinitizer((oB_ia,), (tB_ai,))

    assert isinstance(result, Add)
    signs = {term.as_coeff_Mul()[0] for term in result.args}
    assert signs == {-1, 1}


EXCH_IND20_A_REFERENCE = (
    r"& - 2 G^{r}_{s} H^{b}_{a} (o_A)^{s}_{b} (t_B)^{a}_{r} \\ " + "\n"
    r"& + 2 B^{a}_{a_1} C^{r_1}_{r} (o_B)^{r}_{a} (t_B)^{a_1}_{r_1} \\ " + "\n"
    r"& - 4 B^{a_1}_{a} C^{r}_{r_1} F^{b}_{s} (t_B)^{a}_{r} v^{r_1s}_{a_1b} \\ "
    + "\n"
    r"& - 2 I^{b}_{r_1} B^{a_1}_{a} G^{r}_{s} (t_B)^{a}_{r} v^{r_1s}_{a_1b} \\ "
    + "\n"
    r"& + 2 C^{r}_{r_1} H^{b}_{a} J^{a_1}_{s} (t_B)^{a}_{r} v^{r_1s}_{a_1b} \\ "
    + "\n"
    r"& + 4 E^{a}_{r} G^{r_1}_{s} H^{b}_{a_1} (t_B)^{a_1}_{r_1} v^{rs}_{ab} \\ "
    + "\n"
)


def _exch_ind20_A_expansion():
    """The E(20)_exch-ind(A) pipeline from examples/sinfinitizer.py."""
    a = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)
    b = symbols("b", is_molB=True, above_fermi=True, cls=Dummy)
    j = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)
    a_2 = symbols("a_2", is_molA=True, above_fermi=True, cls=Dummy)
    i_2 = symbols("i_2", is_molA=True, below_fermi=True, cls=Dummy)

    oB_ia = DoubleVacuumTensorSymbol("o_B", (a,), (i,))
    oA_jb = DoubleVacuumTensorSymbol("o_A", (b,), (j,))
    v_ijab = DoubleVacuumTensorSymbol("v", (a, b), (i, j))
    tB_ai = DoubleVacuumTensorSymbol("t_B", (i_2,), (a_2,))

    expr = (
        sinfinitizer((oA_jb,), (tB_ai,))
        + sinfinitizer((oB_ia,), (tB_ai,))
        + sinfinitizer((v_ijab,), (tB_ai,))
    )
    return spin_integration(get_only_linked(expr))


def test_exch_ind20_A_expansion():
    """E(20)_exch-ind(A) without the single-exchange approximation."""
    result = _exch_ind20_A_expansion()

    formatted = format_expr(
        substitute_dummies_double_vac(result, pretty_indices=PRETTY_INDICES)
    )
    assert formatted == EXCH_IND20_A_REFERENCE
