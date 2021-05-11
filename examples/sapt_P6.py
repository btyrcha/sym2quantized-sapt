import time

from sympy import symbols, Dummy, latex

from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.operators import a, ad, b, bd
from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


def lprint_args(expresion):
    for arg in expresion.args:
        if latex(arg)[0] == "-":
            print("&", latex(arg), "\\\\")
        else:
            print("&", "+", latex(arg), "\\\\")
    print("\n")


def lwrite_args(expresion, file):
    for arg in expresion.args:
        if latex(arg)[0] == "-":
            file.write("& " + latex(arg) + " \\\\")
            file.write("\n")
        else:
            file.write("& " + "+ " + latex(arg) + " \\\\")
            file.write("\n")
    file.write("\n")


def get_V_operator():
    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    v = DoubleVacuumTensorSymbol(
        "v",
        (p, r),
        (q, s),
    )
    vA = DoubleVacuumTensorSymbol("(v_A)", (r,), (s,))
    vB = DoubleVacuumTensorSymbol("(v_B)", (p,), (q,))
    V0 = DoubleVacuumTensorSymbol("V_0", (), ())

    V = (
        v * ad(q) * a(p) * bd(s) * b(r)
        + vA * bd(s) * b(r)
        + vB * ad(q) * a(p)
        + V0
    )

    return V


def get_P6_operator():
    p, p1, p2, q, q1, q2 = symbols(
        "p p_1 p_2 q q_1 q_2", is_molA=True, cls=Dummy
    )
    r, r1, r2, s, s1, s2 = symbols(
        "r r_1 r_2 s s_1 s_2", is_molB=True, cls=Dummy
    )

    P_tensor = (
        -1
        / 9
        * DoubleVacuumTensorSymbol("s", (r,), (q,))
        * DoubleVacuumTensorSymbol("s", (r1,), (q1,))
        * DoubleVacuumTensorSymbol("s", (r2,), (q2,))
        * DoubleVacuumTensorSymbol("s", (p,), (s,))
        * DoubleVacuumTensorSymbol("s", (p1,), (s1,))
        * DoubleVacuumTensorSymbol("s", (p2,), (s2,))
    )

    a_part = ad(q) * ad(q1) * ad(q2) * a(p2) * a(p1) * a(p)
    b_part = bd(s) * bd(s1) * bd(s2) * b(r2) * b(r1) * b(r)

    return P_tensor * a_part * b_part


### Start measuring run time
start_time = time.time()

print("Getting second-quantized operators...")
V = get_V_operator()
P6 = get_P6_operator()

expr = V * P6

print("Calculating Wicks theorem form...")
expr = wicks_double_vac(expr, keep_only_fully_contracted=True)

print("Spin integrating....")
expr = spin_integration(expr)

print("Getting only linked terms...")
expr_linked = get_only_linked(expr)

### Stop measuring run time
run_time = time.time() - start_time


with open("sapt_P6.out", "w") as f:

    f.write("Program took %s seconds to run\n" % run_time)

    # f.write("\n< V P6 > =\n")
    # lwrite_args(expr, f)

    f.write("\n< V P6 >_L =\n")
    lwrite_args(expr_linked, f)
