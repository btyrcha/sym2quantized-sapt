"""
This is a test example - it probably does not give a correct answer.
"""

from sympy import symbols, Dummy, latex, expand

from sympy.physics.secondquant import Dagger

from sym2quantized_sapt.double_fermi_vac import (
    TensorDoubleVac,
    ad,
    a,
    bd,
    b,
    commutator,
    wicks_double_vac,
    get_fully_contracted,
    spin_integration,
)


def lprint(expresion):
    return print(latex(expresion))


def lprint_args(expresion):
    print(expresion.func)
    for arg in expresion.args:
        print(latex(arg))


p, p1, q, q1 = symbols("p p_1 q q_1", is_molA=True, cls=Dummy)
r, r1, s, s1 = symbols("r r_1 s s_1", is_molB=True, cls=Dummy)

a1, a2 = symbols("a a_1", is_molA=True, above_fermi=True, cls=Dummy)
i1, i2 = symbols("i i_1", is_molA=True, below_fermi=True, cls=Dummy)

b1, b2 = symbols("b b_1", is_molB=True, above_fermi=True, cls=Dummy)
j1, j2 = symbols("j j_1", is_molB=True, below_fermi=True, cls=Dummy)

v = TensorDoubleVac(
    "v",
    (
        p,
        r,
    ),
    (
        q,
        s,
    ),
)
vA = TensorDoubleVac("(v_A)", (r,), (s,))
vB = TensorDoubleVac("(v_B)", (p,), (q,))
V0 = symbols("V_0")

V = (
    v * ad(q) * a(p) * bd(s) * b(r)
    + vA * bd(s) * b(r)
    + vB * ad(q) * a(p)
    + V0
)

P = (
    -TensorDoubleVac("s", (r1,), (q1,))
    * TensorDoubleVac("s", (p1,), (s1,))
    * ad(q1)
    * a(p1)
    * bd(s1)
    * b(r1)
)

T10 = (
    TensorDoubleVac("o_B", (i2,), (a2,))
    / TensorDoubleVac("e", (a2,), (i2,))
    * ad(a2)
    * a(i2)
)

E_exch_ind200_A = V * P * T10
E_exch_ind200_A = wicks_double_vac(E_exch_ind200_A)
E_exch_ind200_A = get_fully_contracted(E_exch_ind200_A)
E_exch_ind200_A = spin_integration(E_exch_ind200_A)
print(latex(E_exch_ind200_A))
