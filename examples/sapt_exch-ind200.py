"""
Derives second-order exchange inuction correction in S^2 approximation
without intramonomer contributions (i.e. on HF level description of monomers).
"""

from sympy import symbols, Dummy

from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.operators import a, ad
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.sapt_utils import get_V_operator, get_P2_operator
from sym2quantized_sapt.utils import format_expr


def get_T10_operator():
    """
    Prepares the T10 excitation operator.
    """

    a1 = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i1 = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    # (t_B)^{i1}_{a1} = (\omega_B)^{i1}_{a1} / (\epsilon_{i1} - \epsilon_{a1})
    t_B = DoubleVacuumTensorSymbol("t_B", (i1,), (a1,))

    return t_B * ad(a1) * a(i1)


V = get_V_operator()
P2 = get_P2_operator()
T10 = get_T10_operator()

expr = V * P2 * T10
expr = wicks_double_vac(expr, keep_only_fully_contracted=True)
expr = get_only_linked(expr)
expr = spin_integration(expr)

expr_str = format_expr(expr)
print(expr_str)
