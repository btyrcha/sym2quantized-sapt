from sympy import symbols, Dummy, latex

from sym2quantized_sapt.double_fermi_vac import (
    TensorDoubleVac,
    ad,
    a,
    bd,
    b,
    wicks_double_vac,
    get_fully_contracted,
    spin_integration,
)

from sym2quantized_sapt.diagrams import get_olnly_linked


def lprint_args(expresion):
    for arg in expresion.args:
        if latex(arg)[0] == "-":
            print("&", latex(arg), "\\\\")
        else:
            print("&", "+", latex(arg), "\\\\")
    print("\n")


p, p1, p2, q, q1, q2 = symbols("p p_1 p_2 q q_1 q_2", is_molA=True, cls=Dummy)
r, r1, r2, s, s1, s2 = symbols("r r_1 r_2 s s_1 s_2", is_molB=True, cls=Dummy)


v = TensorDoubleVac("v", (p, r,), (q, s,))
vA = TensorDoubleVac("(v_A)", (r,), (s,))
vB = TensorDoubleVac("(v_B)", (p,), (q,))
V0 = TensorDoubleVac("V_0", (), ())

V = (
    v * ad(q) * a(p) * bd(s) * b(r)
    + vA * bd(s) * b(r)
    + vB * ad(q) * a(p)
    + V0
)


P_tensor = (
    -TensorDoubleVac("s", (r1,), (q1,))
    * TensorDoubleVac("s", (r2,), (q2,))
    * TensorDoubleVac("s", (p1,), (s1,))
    * TensorDoubleVac("s", (p2,), (s2,))
)

a_part = ad(q1) * ad(q2) * a(p2) * a(p1)
b_part = bd(s1) * bd(s2) * b(r2) * b(r1)

P4 = P_tensor * a_part * b_part


expr = V * P4
expr = wicks_double_vac(expr, simplify_kronecker_deltas=True)
expr = get_fully_contracted(expr)
expr = spin_integration(expr)
expr = get_olnly_linked(expr)

print("< V P4 >_L =")
lprint_args(expr)
