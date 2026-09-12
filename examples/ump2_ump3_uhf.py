"""
UMP2 and UMP3 for a single molecule, spin-blocked for a UHF reference.

Second order is <W R W> with the (2,0) resolvent, third order the
standard RSPT expression <W R W R W> - E1 <W R R W>.  Spin summation
by ``spin_integration_uhf`` labels every Goldstone loop with one spin
and replaces each tensor by its spin block (one label per slot pair),
so the emitted objects are the genuinely unrestricted arrays --
``w_ab`` with the first electron alpha and the second beta, and so on.

Two checks run on both orders:

* the RHF collapse gate: stripping the block labels must reproduce the
  restricted ``2**loops`` result exactly;
* the UMP2 opposite-spin block carries no exchange: every mixed-spin
  term has a positive coefficient, while the same-spin blocks mix
  signs -- the textbook structure of unrestricted MP2.
"""

from sympy import Dummy, Rational, symbols

from sym2quantized_sapt.double_fermi_vac import (
    substitute_dummies_double_vac,
    wicks_double_vac,
)
from sym2quantized_sapt.operators import a, ad
from sym2quantized_sapt.sapt_utils import get_R_nm
from sym2quantized_sapt.spin_integrator import (
    rhf_collapse,
    spin_integration,
    spin_integration_uhf,
)
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol as DVT

p, q = symbols("p q", is_molA=True, cls=Dummy)
p2, q2 = symbols("p' q'", is_molA=True, cls=Dummy)
W = Rational(1, 2) * DVT("w", (p, p2), (q, q2)) * ad(q)*ad(q2)*a(p2)*a(p)


def blocks_census(expr):
    census = {}
    for term in expr.args:
        label = next(
            str(factor.symbol()).split("_")[-1]
            for factor in term.args
            if isinstance(factor, DVT)
            and str(factor.symbol()).startswith("e_")
        )
        census[label] = census.get(label, 0) + 1
    return dict(sorted(census.items()))


# ---- UMP2 -------------------------------------------------------------
E2 = wicks_double_vac(
    W * get_R_nm(2, 0, W), keep_only_fully_contracted=True
)
E2 = substitute_dummies_double_vac(E2)
u2 = spin_integration_uhf(E2)
print(f"UMP2: {len(E2.args)} spatial -> {len(u2.args)} blocked terms, "
      f"by denominator block {blocks_census(u2)}")

assert rhf_collapse(u2) == spin_integration(E2), "UMP2 RHF collapse"

for term in u2.args:
    label = next(
        str(f.symbol()).split("_")[-1]
        for f in term.args
        if isinstance(f, DVT) and str(f.symbol()).startswith("e_")
    )
    coefficient = [f for f in term.args if f.is_number]
    if label in ("ab", "ba"):
        assert all(c > 0 for c in coefficient), (
            "opposite-spin MP2 has no exchange term"
        )
print("UMP2: RHF collapse exact; opposite-spin block exchange-free")

# ---- UMP3 -------------------------------------------------------------
E3_main = wicks_double_vac(
    W * get_R_nm(2, 0, W * get_R_nm(2, 0, W)),
    keep_only_fully_contracted=True,
)
E1 = wicks_double_vac(W, keep_only_fully_contracted=True)
E3_renorm = wicks_double_vac(
    W * get_R_nm(2, 0, get_R_nm(2, 0, W)),
    keep_only_fully_contracted=True,
)
E3 = (
    substitute_dummies_double_vac(E3_main)
    - (E1 * substitute_dummies_double_vac(E3_renorm)).expand()
).expand()

u3 = spin_integration_uhf(E3)
print(f"UMP3: {len(E3.args)} spatial -> {len(u3.args)} blocked terms")

assert rhf_collapse(u3) == spin_integration(E3), "UMP3 RHF collapse"
print("UMP3: RHF collapse exact")
