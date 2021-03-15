"""
Derives formula for first order SAPT exchange energy
in secondquantzation approach.

Based on:
Moszynski, Robert, Bogumil Jeziorski, and Krzysztof Szalewicz. 
"Many‐body theory of exchange effects in intermolecular interactions. 
Second‐quantization approach and comparison with full configuration interaction 
results." The Journal of chemical physics 100.2 (1994): 1312-1325.
"""

from sympy import symbols, Dummy, latex, Add, S
from sympy.physics.secondquant import NO
from sym2quantized_sapt.double_fermi_vac import (
    TensorDoubleVac,
    ad,
    a,
    bd,
    b,
    wicks_double_vac,
    evaluate_deltas_double_vac,
    get_fully_contracted,
    substitute_dummies_double_vac,
)


p, q = symbols("p q", is_molA=True, cls=Dummy)
r, s = symbols("r s", is_molB=True, cls=Dummy)

a1, a2 = symbols("a_{1} a_{2}", is_molA=True, above_fermi=True, cls=Dummy)
i1, i2 = symbols("i_{1} i_{2}", is_molA=True, below_fermi=True, cls=Dummy)

b1, b2 = symbols("b_{1} b_{2}", is_molB=True, above_fermi=True, cls=Dummy)
j1, j2 = symbols("j_{1} j_{2}", is_molB=True, below_fermi=True, cls=Dummy)

v = TensorDoubleVac("v", (p, r,), (q, s,))
vA = TensorDoubleVac("(v_A)", (r,), (s,))
vB = TensorDoubleVac("(v_B)", (p,), (q,))
V0 = symbols("V_0")

V = (
    v * ad(q) * a(p) * bd(s) * b(r)
    + vA * bd(s) * b(r)
    + vB * ad(q) * a(p)
    + V0
)

# This fragment calculates and prints normal ordered operator V
# and separately its fully contracted terms.
"""
expr = wicks_double_vac(V)
expr = evaluate_deltas_double_vac(expr)
print(latex(expr))

expr = get_fully_contracted(expr)
print("\n", latex(expr), "\n")
"""

P10 = (
    S.NegativeOne
    * TensorDoubleVac("s", (j1,), (a1,))
    * TensorDoubleVac("s", (i1,), (j1,))
    * ad(a1)
    * a(i1)
)

P01 = (
    S.NegativeOne
    * TensorDoubleVac("s", (j1,), (i1,))
    * TensorDoubleVac("s", (i1,), (b1,))
    * bd(b1)
    * b(j1)
)

P11 = (
    S.NegativeOne
    * TensorDoubleVac("s", (j1,), (a1,))
    * TensorDoubleVac("s", (i1,), (b1,))
    * ad(a1)
    * a(i1)
    * bd(b1)
    * b(j1)
)

P = P10 + P01 + P11

# print(latex(P10), "\n")
# print(latex(P01), "\n")
# print(latex(P11), "\n")

expr = V * P
expr = wicks_double_vac(expr)
expr = evaluate_deltas_double_vac(expr)

# print(latex(expr), "\n")

expr = get_fully_contracted(expr)
print("\n", latex(expr), "\n")

expr = substitute_dummies_double_vac(expr)
print(latex(expr), "\n")
