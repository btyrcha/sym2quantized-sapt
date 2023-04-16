"""
Script calculates the  <V P_(2) C_{A}> expectation value,
where C_{A} = C^{x}_{y} a^{\\dagger}_{y} a_{x} is an operator
of single excitation of monomer A.
"""

from time import time
from sympy import symbols, expand

from sym2quantized_sapt.operators import (
    CreateFermion_A,
    CreateFermion_B,
    AnnihilateFermion_A,
    AnnihilateFermion_B,
)
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.double_fermi_vac import (
    wicks_double_vac,
    substitute_dummies_double_vac,
    evaluate_deltas_double_vac,
)
from sym2quantized_sapt.sapt_utils import get_P2_operator, get_V_operator
from sym2quantized_sapt.utils import format_expr
from sym2quantized_sapt.diagrams import get_only_linked
from sym2quantized_sapt.code_generator import generate_einsum


Ad = CreateFermion_A
A = AnnihilateFermion_A
Bd = CreateFermion_B
B = AnnihilateFermion_B


def get_C_operator():
    """
    Prepares C_{A} operator.
    """
    x = symbols("x", is_molA=True, below_fermi=True)
    y = symbols("y", is_molA=True, above_fermi=True)

    C_xy = DoubleVacuumTensorSymbol("C", (x,), (y,))

    return C_xy * Ad(y) * A(x)


V = get_V_operator()
P2 = get_P2_operator()
C = get_C_operator()

start_time = time()

expr = V * P2 * C
expr = expand(expr)

expr = wicks_double_vac(
    expr,
    keep_only_fully_contracted=True,
    substitute_dummies=False,
    simplify_kronecker_deltas=False,
)
expr = evaluate_deltas_double_vac(expr)
expr = substitute_dummies_double_vac(expr)


expr_linked = get_only_linked(expr)

### Calculating unlinked element explicitely
exp_V = wicks_double_vac(
    V,
    keep_only_fully_contracted=True,
    substitute_dummies=False,
    simplify_kronecker_deltas=False,
)
exp_P2 = wicks_double_vac(
    P2,
    keep_only_fully_contracted=True,
    substitute_dummies=False,
    simplify_kronecker_deltas=False,
)
exp_P2C = wicks_double_vac(
    P2 * C,
    keep_only_fully_contracted=True,
    substitute_dummies=False,
    simplify_kronecker_deltas=False,
)
exp_VC = wicks_double_vac(
    V * C,
    keep_only_fully_contracted=True,
    substitute_dummies=False,
    simplify_kronecker_deltas=False,
)


unlinked = -(exp_V * exp_P2C + exp_VC * exp_P2)
unlinked = expand(unlinked)
unlinked = evaluate_deltas_double_vac(unlinked)
unlinked = substitute_dummies_double_vac(unlinked)


run_time = time() - start_time
print(f"Calculations took: {run_time:.2f} s")

### Formating and saving
formated_expr = format_expr(expr)
with open("exch10_C_A.out", "w") as f:
    f.write(formated_expr)

formated_expr_linked = format_expr(expr_linked)
with open("exch10_C_A_linked.out", "w") as f:
    f.write(formated_expr_linked)

formated_unlinked = format_expr(unlinked)
with open("exch10_C_A_unlinked.out", "w") as f:
    f.write(formated_unlinked)

### Code generation
einsum_code = generate_einsum(expr)
with open("exch10_C_A_code.out", "w") as f:
    f.write(einsum_code)
