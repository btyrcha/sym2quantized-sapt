"""
Derives formula for first-order SAPT intramonomer correlation
correction for exchange energy E^{(11)}_{exch}.

Based on:
Moszynski, Robert, Bogumil Jeziorski, and Krzysztof Szalewicz.
"Many-body theory of exchange effects in intermolecular interactions.
Second-quantization approach and comparison with full configuration interaction
results", The Journal of chemical physics 100 (2), 1312 (1994).
"""

# There is an issue with simplification of Add expresions.
# Expresions of type:
# 0.5 x - 1.0 x
# are not simplified to -0.5 x.

from sympy import symbols, Dummy
from sympy import expand as sy_expand
from sym2quantized_sapt.double_fermi_vac import wicks_double_vac, commutator
from sym2quantized_sapt.operators import a, ad, b, bd
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.utils import format_expr


p, p1, q, q1 = symbols("p p_1 q q_1", is_molA=True, cls=Dummy)
r, r1, s, s1 = symbols("r r_1 s s_1", is_molB=True, cls=Dummy)

a1, a2, a3, a4 = symbols(
    "a a_1 a_2 a_3", is_molA=True, above_fermi=True, cls=Dummy
)
i1, i2, i3, i4 = symbols(
    "i i_1 i_2 i_3", is_molA=True, below_fermi=True, cls=Dummy
)

b1, b2 = symbols("b b_1", is_molB=True, above_fermi=True, cls=Dummy)
j1, j2 = symbols("j j_1", is_molB=True, below_fermi=True, cls=Dummy)


v = DoubleVacuumTensorSymbol("v", (q, s,), (p, r,),)
vA = DoubleVacuumTensorSymbol("(v_A)", (s,), (r,))
vB = DoubleVacuumTensorSymbol("(v_B)", (q,), (p,))
V0 = DoubleVacuumTensorSymbol("V_0", (), ())

V_dagger = (
    v * ad(p) * a(q) * bd(r) * b(s)
    + vA * bd(r) * b(s)
    + vB * ad(p) * a(q)
    + V0
)


V10_dash_dagger = (
    DoubleVacuumTensorSymbol("o_B", (a2,), (i2,)) * ad(i2) * a(a2)
)

V01_dash_dagger = (
    DoubleVacuumTensorSymbol("o_A", (b2,), (j2,)) * bd(j2) * b(b2)
)

V11_dash_dagger = (
    DoubleVacuumTensorSymbol("v", (a2, b2,), (i2, j2,),)
    * ad(i2)
    * a(a2)
    * bd(j2)
    * b(b2)
)

V_dash_dagger = V10_dash_dagger + V01_dash_dagger + V11_dash_dagger


P = (
    -DoubleVacuumTensorSymbol("s", (r1,), (q1,))
    * DoubleVacuumTensorSymbol("s", (p1,), (s1,))
    * ad(q1)
    * a(p1)
    * bd(s1)
    * b(r1)
)


P10_dash = (
    -DoubleVacuumTensorSymbol("s", (j1,), (a1,))
    * DoubleVacuumTensorSymbol("s", (i1,), (j1,))
    * ad(a1)
    * a(i1)
)

P01_dash = (
    -DoubleVacuumTensorSymbol("s", (j1,), (i1,))
    * DoubleVacuumTensorSymbol("s", (i1,), (b1,))
    * bd(b1)
    * b(j1)
)

P11_dash = (
    -DoubleVacuumTensorSymbol("s", (j1,), (a1,))
    * DoubleVacuumTensorSymbol("s", (i1,), (b1,))
    * ad(a1)
    * a(i1)
    * bd(b1)
    * b(j1)
)

P_dash = P10_dash + P01_dash + P11_dash


T20_part1 = (
    1
    / 4
    * DoubleVacuumTensorSymbol("t", (i3, i4,), (a3, a4,),)
    * ad(a3)
    * ad(a4)
    * a(i4)
    * a(i3)
)

T20_part2 = (
    -1
    / 4
    * DoubleVacuumTensorSymbol("t", (i4, i3,), (a3, a4,),)
    * ad(a3)
    * ad(a4)
    * a(i4)
    * a(i3)
)

T20 = T20_part1 + T20_part2

T20_part1_dagger = (
    1
    / 4
    * DoubleVacuumTensorSymbol("t", (a3, a4,), (i3, i4,),)
    * ad(i3)
    * ad(i4)
    * a(a4)
    * a(a3)
)

T20_part2_dagger = (
    -1
    / 4
    * DoubleVacuumTensorSymbol("t", (a3, a4,), (i4, i3,),)
    * ad(i3)
    * ad(i4)
    * a(a4)
    * a(a3)
)

T20_dagger = T20_part1_dagger + T20_part2_dagger


# E(11)_exch = E(110)_exch + E(101)_exch

expr = (
    V_dash_dagger * commutator(P, T20)
    - commutator(V_dagger, T20_dagger) * P_dash
)  # [A, B]^\dagger = -[A^\dagger, B^\dagger]
expr = sy_expand(expr)
expr = wicks_double_vac(
    expr, simplify_kronecker_deltas=True, keep_only_fully_contracted=True
)
expr = spin_integration(expr)


expr_str = format_expr(expr)
print(expr_str)
