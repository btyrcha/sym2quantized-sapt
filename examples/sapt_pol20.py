"""
Derives formula for first-order SAPT electrostatcic energy as well as
formulas for second-order SAPT induction and dispertion energies.

Based on:
Stanislaw Rybak, Bogumil Jeziorski, Krzysztof Szalewicz,
"Many-body symmetry-adapted perturbation theory of intermolecular interactions.
H_2O and HF dimers", The Journal of Chemical Physics 95, 6576 (1991).
"""

from sympy import symbols, Dummy, latex
from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.operators import a, ad, b, bd
from sym2quantized_sapt.spin_integrator import spin_integration
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol

p, q = symbols("p q", is_molA=True, cls=Dummy)
r, s = symbols("r s", is_molB=True, cls=Dummy)

a1 = symbols("a", is_molA=True, above_fermi=True, cls=Dummy)
i1 = symbols("i", is_molA=True, below_fermi=True, cls=Dummy)

b1 = symbols("b", is_molB=True, above_fermi=True, cls=Dummy)
j1 = symbols("j", is_molB=True, below_fermi=True, cls=Dummy)

v = DoubleVacuumTensorSymbol(
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
vA = DoubleVacuumTensorSymbol("(v_A)", (r,), (s,))
vB = DoubleVacuumTensorSymbol("(v_B)", (p,), (q,))
V0 = symbols("V_0")

V = (
    v * ad(q) * a(p) * bd(s) * b(r)
    + vA * bd(s) * b(r)
    + vB * ad(q) * a(p)
    + V0
)

T10 = (
    DoubleVacuumTensorSymbol("o_B", (i1,), (a1,))
    / DoubleVacuumTensorSymbol("e", (a1,), (i1,))
    * ad(a1)
    * a(i1)
)

T01 = (
    DoubleVacuumTensorSymbol("o_A", (j1,), (b1,))
    / DoubleVacuumTensorSymbol("e", (b1,), (j1,))
    * bd(b1)
    * b(j1)
)

T11 = (
    DoubleVacuumTensorSymbol(
        "v",
        (
            i1,
            j1,
        ),
        (
            a1,
            b1,
        ),
    )
    / DoubleVacuumTensorSymbol(
        "e",
        (
            a1,
            b1,
        ),
        (
            i1,
            j1,
        ),
    )
    * ad(a1)
    * a(i1)
    * bd(b1)
    * b(j1)
)

E_pol_10 = V
E_pol_10 = wicks_double_vac(
    E_pol_10, simplify_kronecker_deltas=True, keep_only_fully_contracted=True
)
E_pol_10 = spin_integration(E_pol_10)
print("E_pol_10 =", latex(E_pol_10), "\n")


E_indA_20 = V * T10
E_indA_20 = wicks_double_vac(
    E_indA_20, simplify_kronecker_deltas=True, keep_only_fully_contracted=True
)
E_indA_20 = spin_integration(E_indA_20)
print("E(20)_ind_A =", latex(E_indA_20), "\n")

E_indB_20 = V * T01
E_indB_20 = wicks_double_vac(
    E_indB_20, simplify_kronecker_deltas=True, keep_only_fully_contracted=True
)
E_indB_20 = spin_integration(E_indB_20)
print("E(20)_ind_B =", latex(E_indB_20), "\n")

E_disp_20 = V * T11
E_disp_20 = wicks_double_vac(
    E_disp_20, simplify_kronecker_deltas=True, keep_only_fully_contracted=True
)
E_disp_20 = spin_integration(E_disp_20)
print("E(20)_disp =", latex(E_disp_20), "\n")
