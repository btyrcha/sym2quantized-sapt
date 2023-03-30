"""
Derives formula for first-order SAPT exchange energy
in second-quantzation approach.

Based on:
Moszynski, Robert, Bogumil Jeziorski, and Krzysztof Szalewicz.
"Many-body theory of exchange effects in intermolecular interactions.
Second-quantization approach and comparison with full configuration interaction
results", The Journal of chemical physics 100 (2), 1312 (1994).
"""

from sympy import symbols, Dummy, latex
from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.operators import a, ad, b, bd
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol

p, q = symbols("p q", is_molA=True, cls=Dummy)
r, s = symbols("r s", is_molB=True, cls=Dummy)

a1 = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
i1 = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

b1 = symbols("b", is_molB=True, above_fermi=True, cls=Dummy)
j1 = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)

v = DoubleVacuumTensorSymbol("v", (p, r,), (q, s,),)
vA = DoubleVacuumTensorSymbol("(v_A)", (r,), (s,))
vB = DoubleVacuumTensorSymbol("(v_B)", (p,), (q,))
V0 = symbols("V_0")

V = (
    v * ad(q) * a(p) * bd(s) * b(r)
    + vA * bd(s) * b(r)
    + vB * ad(q) * a(p)
    + V0
)

P10 = (
    -DoubleVacuumTensorSymbol("s", (j1,), (a1,))
    * DoubleVacuumTensorSymbol("s", (i1,), (j1,))
    * ad(a1)
    * a(i1)
)

P01 = (
    -DoubleVacuumTensorSymbol("s", (j1,), (i1,))
    * DoubleVacuumTensorSymbol("s", (i1,), (b1,))
    * bd(b1)
    * b(j1)
)

P11 = (
    -DoubleVacuumTensorSymbol("s", (j1,), (a1,))
    * DoubleVacuumTensorSymbol("s", (i1,), (b1,))
    * ad(a1)
    * a(i1)
    * bd(b1)
    * b(j1)
)

P = P10 + P01 + P11

print("First order MBPT - SAPT Exchange Energy is defined as:")
print("E(10)_exch = < Phi |V P| Phi >", "\n")

print("where")
print("V =", latex(V), "\n")
print("and assuming S^2 approximation")
print("P =", latex(P), "\n")

expr = V * P
expr = wicks_double_vac(
    expr, simplify_kronecker_deltas=True, keep_only_fully_contracted=True
)
print("It can be written in using only one- and two-electron integrals as:")
print("E(10)_exch =", latex(expr), "\n")


expr = spin_integration(expr)
print("After performing spin integration in RHF case it takes form:")
print("E(10)_exch =", latex(expr), "\n")
