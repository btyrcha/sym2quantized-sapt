"""
Example on how to use sinfinitizer to generate second-quantized SAPT expressions
without the single-exchange approximation.

The example is based on the expression for E(20)_exch-ind(A) from the paper (look: equation 79):
B. Tyrcha, F. Brzęk, and P. S. Żuchowski, “Second quantization-based symmetry-adapted
perturbation theory: Generalizing exchange beyond single electron pair approximation,”
The Journal of Chemical Physics, vol. 160, no. 4, p. 044118, Jan. 2024, doi: 10.1063/5.0184750.

Definitions of tensors A-J can be found in:
B. Tyrcha, T. Gupta, K. Patkowski, and P. S. Żuchowski, “Analytical derivatives
of symmetry-adapted perturbation theory corrections for interaction-induced properties,”
Feb. 13, 2025, ChemRxiv. doi: 10.26434/chemrxiv-2025-91mps.
"""

from sympy import symbols, Dummy

from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.code_generator import generate_einsum
from sym2quantized_sapt.utils import format_expr
from sym2quantized_sapt.double_fermi_vac import substitute_dummies_double_vac
from sym2quantized_sapt.sinfinitizer import sinfinitizer


PRETTY_INDICES = {
    "above_molA": "r",
    "above_molB": "s",
    "below_molA": "a",
    "below_molB": "b",
}


def print_results(expression, code=True):
    """
    Print latex formated expression and the corresponding
    einsum code.
    """

    print("Expression:")
    print(
        format_expr(
            substitute_dummies_double_vac(
                expression, pretty_indices=PRETTY_INDICES
            )
        )
    )

    if code:
        print("\n")
        print("Einsum code:")
        print(generate_einsum(expression), "\n")


def get_oB_ia():
    """
    Prepares (omega_B)_ia tensor
    """
    i = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)
    a = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)

    return DoubleVacuumTensorSymbol("o_B", (a,), (i,))


def get_oA_jb():
    """
    Prepares (omega_A)_jb tensor
    """
    j = symbols("j", cls=Dummy, is_molB=True, below_fermi=True)
    b = symbols("b", cls=Dummy, is_molB=True, above_fermi=True)

    return DoubleVacuumTensorSymbol("o_A", (b,), (j,))


def get_v_ijab():
    """
    Prepares v_ijab tensor
    """
    i = symbols("i", cls=Dummy, is_molA=True, below_fermi=True)
    j = symbols("j", cls=Dummy, is_molB=True, below_fermi=True)
    a = symbols("a", cls=Dummy, is_molA=True, above_fermi=True)
    b = symbols("b", cls=Dummy, is_molB=True, above_fermi=True)

    return DoubleVacuumTensorSymbol(
        "v",
        (
            a,
            b,
        ),
        (
            i,
            j,
        ),
    )


def get_tB_ai():
    """
    Prepares (t_B)_ai tensor
    """
    a = symbols("a_2", is_molA=True, above_fermi=True, cls=Dummy)
    i = symbols("i_2", is_molA=True, below_fermi=True, cls=Dummy)

    return DoubleVacuumTensorSymbol("t_B", (i,), (a,))


if __name__ == "__main__":
    # prepare expression elements
    oB_ia = get_oB_ia()
    oA_jb = get_oA_jb()
    v_ijab = get_v_ijab()

    tB_ai = get_tB_ai()

    # NOTE: the comas in the arguments are important
    terms_omegaB = sinfinitizer(
        (oB_ia,),  # <- vertex/vertices that close the diagram from the top
        # all possible contractions with S matrix elements enter in between
        (tB_ai,),  # <- vertex/vertices that close the diagram from the bottom
    )
    terms_omegaA = sinfinitizer(
        (oA_jb,),
        (tB_ai,),
    )
    terms_v = sinfinitizer(
        (v_ijab,),
        (tB_ai,),
    )

    # NOTE: the tuples in the arguments of sinfinitizer can be of any length, also 0
    # if there are no vertices to close the diagram from the top or bottom -
    # - the diagram is closed with S matrix elements

    expr = terms_omegaA + terms_omegaB + terms_v
    expr = get_only_linked(expr)
    expr = spin_integration(expr)

    print("Numer of linked terms:", len(expr.args), "\n")
    print_results(expr)
