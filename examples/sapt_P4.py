import time
from typing import Tuple

from sympy import symbols, Dummy
from sympy.core import Expr

from sym2quantized_sapt.double_fermi_vac import (
    TensorDoubleVac,
    ad,
    a,
    bd,
    b,
    wicks_double_vac,
    spin_integration,
)

from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.utils import timeit, format_expr


def get_V_operator() -> Expr:
    """prepares V-tilde operator

    Returns:
        Expr: SymPy Expr encoding V-tilde (dimer interaction operator)
    """
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

    return V


def get_P4_operator() -> Expr:
    """constructs exchange-S^4  - P4 operator

    Returns:
        Expr: SymPy Expr encoding P4 operator (S^4 exchange operator)
    """
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


@timeit
def compute_exch10_s4() -> Tuple[Expr, Expr]:
    """computes Exch10(S^4) contribution: <V P4>

    Returns:
        Tuple[Expr, Expr]: <V P4> and <V P4>_L
    """
    V = get_V_operator()
    P4 = get_P4_operator()

    # create operator to be averaged
    expr = V * P4

    # perform wicks
    expr = wicks_double_vac(expr, keep_only_fully_contracted=True)

    # spin integrate for RHF
    expr_all = spin_integration(expr)

    # extract linked only
    expr_linked = get_only_linked(expr_all)

    # return both
    return expr_all, expr_linked


if __name__ == "__main__":

    ### Start measuring run time
    start_time = time.time()

    expr, expr_linked = compute_exch10_s4()

    ### Stop measuring run time
    run_time = time.time() - start_time

    plain_expr_str = format_expr(expr)
    linked_expr_str = format_expr(expr_linked)
    # write out the results
    with open("sapt_P4.out", "w") as f:
        f.write(f"Program took {run_time} seconds to run\n")

        # write-down <V P4>
        f.write("\n< V P4 > =\n")
        f.write(plain_expr_str)

        # write-down <V P4>_L
        f.write("\n< V P4 >_L =\n")
        f.write(linked_expr_str)
