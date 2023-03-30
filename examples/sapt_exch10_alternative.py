"""
Derives the first-order exchange correction in S^2 approximation as only linked
parts of <V P2> term.
"""

from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.sapt_utils import get_V_operator, get_P2_operator
from sym2quantized_sapt.utils import format_expr

V = get_V_operator()
P = get_P2_operator()

expr = V * P
expr = wicks_double_vac(
    expr, simplify_kronecker_deltas=True, keep_only_fully_contracted=True
)
expr = spin_integration(expr)

print("< V P > =")
print(format_expr(expr))

print("Only linked terms of < V P >:")
print("< V P >_L =")
expr = get_only_linked(expr)
print(format_expr(expr))
