import time
from typing import Tuple

from sympy.core import Expr

from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.utils import timeit, format_expr
from sym2quantized_sapt.sapt_utils import get_V_operator, get_P4_operator


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
