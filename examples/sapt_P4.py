import time

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

    return V


def get_P4_operator():
    p1, p2, q1, q2 = symbols("p_1 p_2 q_1 q_2", is_molA=True, cls=Dummy)
    r1, r2, s1, s2 = symbols("r_1 r_2 s_1 s_2", is_molB=True, cls=Dummy)

    P_tensor = (
        0.25
        * TensorDoubleVac("s", (r1,), (q1,))
        * TensorDoubleVac("s", (r2,), (q2,))
        * TensorDoubleVac("s", (p1,), (s1,))
        * TensorDoubleVac("s", (p2,), (s2,))
    )

    a_part = ad(q1) * ad(q2) * a(p2) * a(p1)
    b_part = bd(s1) * bd(s2) * b(r2) * b(r1)

    return P_tensor * a_part * b_part


if __name__ == "__main__":

    ### Start measuring run time
    start_time = time.time()

    V = get_V_operator()
    P4 = get_P4_operator()

    expr = V * P4
    expr = wicks_double_vac(expr, keep_only_fully_contracted=True)
    expr = spin_integration(expr)
    expr_linked = get_olnly_linked(expr)

    ### Stop measuring run time
    run_time = time.time() - start_time


with open("sapt_P4.out", "w") as f:

    f.write("Program took %s seconds to run\n" % run_time)

    f.write("\n< V P4 > =\n")
    lwrite_args(expr, f)

    f.write("\n< V P4 >_L =\n")
    lwrite_args(expr_linked, f)
