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

from sym2quantized_sapt.diagrams import get_only_linked


def lprint_args(expresion):
    for arg in expresion.args:
        print(latex(arg))
    print("\n")


p, q = symbols("p q", is_molA=True, cls=Dummy)
r, s = symbols("r s", is_molB=True, cls=Dummy)

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
V0 = TensorDoubleVac("V_0", (), ())

V = (
    v * ad(q) * a(p) * bd(s) * b(r)
    + vA * bd(s) * b(r)
    + vB * ad(q) * a(p)
    + V0
)


p1, q1 = symbols("p q", is_molA=True, cls=Dummy)
r1, s1 = symbols("r s", is_molB=True, cls=Dummy)

P = (
    -TensorDoubleVac("s", (r1,), (q1,))
    * TensorDoubleVac("s", (p1,), (s1,))
    * ad(q1)
    * a(p1)
    * bd(s1)
    * b(r1)
)

expr = V * P
expr = wicks_double_vac(expr, simplify_kronecker_deltas=True)
expr = get_fully_contracted(expr)
expr = spin_integration(expr)

print("< V P > =")
lprint_args(expr)

print("Only linked terms of < V P >:")
print("< V P >_L =")
expr = get_only_linked(expr)
lprint_args(expr)
