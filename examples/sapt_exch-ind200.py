"""
This is a test example - it probably does not give a correct answer.
"""

from sympy import symbols, Dummy

from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.operators import a, ad
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.sapt_utils import get_V_operator, get_P2_operator
from sym2quantized_sapt.utils import format_expr


def get_SA_operator():

    a1 = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
    i1 = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

    SA_operator = (
        DoubleVacuumTensorSymbol(
            "o_B", (i1,), (a1,)
        )  # (o_B)^{i1}_{a1} = (\omega_B)^{i1}_{a1} / (\epsilon_{i1} - \epsilon_{a1})
        * ad(a1)
        * a(i1)
    )

    return SA_operator


V = get_V_operator()
P2 = get_P2_operator()
SA = get_SA_operator()

expr = V * P2 * SA
expr = wicks_double_vac(expr, keep_only_fully_contracted=True)
expr = get_only_linked(expr)
expr = spin_integration(expr)

expr_str = format_expr(expr)
print(expr_str)
