"""
UMP2 with explicit spin tags -- and the cautionary tale that led there.

Spin tags (``is_alpha`` / ``is_beta``) are the open-shell analogue of
the monomer tags: contractions vanish across them, so Wick's theorem
does the UHF bookkeeping exactly.  The fluctuation operator enters as
its four spin sectors and the (2,0) resolvent as its three, with the
per-sector normalisation the tags make explicit: 1/(2!)^2 for two
same-spin (indistinguishable) pairs, 1 for the distinguishable
alpha+beta pair.  The result is numerically exact against psi4's
conventional UHF-MP2 (checked downstream to ~1e-16 per spin channel).

The cautionary tale: the same energy derived spatially and spin-summed
AFTERWARDS with ``spin_integration_uhf`` (one label per Goldstone
loop) passes its RHF-collapse gate yet is WRONG -- the mixed-spin
sector comes out exactly half, because part of the resolvent's
permutation multiplicity flows through exchange-wired contractions
that spatial terms can only carry with same-spin labels.  The collapse
gate checks internal consistency, not correctness; only a numerical
benchmark caught this.  See the warning on ``spin_integration_uhf``.
This script asserts the halving explicitly, so the limitation stays
pinned.
"""

from itertools import product

from sympy import Add, Dummy, Rational, nsimplify, symbols
from sympy.physics.secondquant import Dagger

from sym2quantized_sapt.double_fermi_vac import (
    substitute_dummies_double_vac,
    wicks_double_vac,
)
from sym2quantized_sapt.operators import a, ad
from sym2quantized_sapt.sapt_utils import get_R_nm
from sym2quantized_sapt.open_shell import spin_integration_uhf
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol as DVT

SPIN = {"a": {"is_alpha": True}, "b": {"is_beta": True}}
_counter = [0]


def index(letter, spin, **kw):
    _counter[0] += 1
    return symbols(
        f"{letter}_{_counter[0]}", is_molA=True, cls=Dummy,
        **SPIN[spin], **kw,
    )


def w_tagged():
    """W = 1/2 sum over spin sectors of w_{s1 s2}."""
    sectors = []
    for s1, s2 in product("ab", repeat=2):
        p, q = index("p", s1), index("q", s1)
        p2, q2 = index("p", s2), index("q", s2)
        sectors.append(
            Rational(1, 2)
            * DVT(f"w_{s1}{s2}", (p, p2), (q, q2))
            * ad(q) * ad(q2) * a(p2) * a(p)
        )
    return Add(*sectors)


def r20_tagged(operator):
    """(2,0) resolvent by spin sector: 1/4 same-spin, 1 mixed."""
    pieces = []
    for spins, coefficient in (
        (("a", "a"), Rational(1, 4)),
        (("b", "b"), Rational(1, 4)),
        (("a", "b"), 1),
    ):
        holes = [index("i", s, below_fermi=True) for s in spins]
        particles = [index("a", s, above_fermi=True) for s in spins]
        excitation = (
            ad(holes[0]) * ad(holes[1]) * a(particles[1]) * a(particles[0])
        )
        amplitude = wicks_double_vac(
            excitation * operator,
            keep_only_fully_contracted=True,
            substitute_dummies=False,
        )
        denominator = DVT(
            "e_" + "".join(spins), tuple(holes), tuple(particles)
        )
        pieces.append(
            coefficient * amplitude * Dagger(excitation) * denominator
        )
    return Add(*pieces)


def sector_weight(expr, block):
    """Sum of |coefficients| of the terms carrying denominator block."""
    total = 0
    for term in expr.args:
        for factor in term.args:
            if isinstance(factor, DVT) and str(factor.symbol()) == (
                "e_" + block
            ):
                total += abs(
                    nsimplify(
                        [f for f in term.args if f.is_number][0]
                        if any(f.is_number for f in term.args)
                        else 1
                    )
                )
    return total


# ---- the exact route: spin tags ---------------------------------------
E2_tagged = wicks_double_vac(
    (w_tagged() * r20_tagged(w_tagged())).expand(),
    keep_only_fully_contracted=True,
    substitute_dummies=False,
).expand()
print(f"UMP2, spin-tagged: {len(E2_tagged.args)} terms "
      f"(validated against psi4 downstream)")

# ---- the per-loop route, and its pinned failure ------------------------
p, q = symbols("p q", is_molA=True, cls=Dummy)
p2, q2 = symbols("p' q'", is_molA=True, cls=Dummy)
W_spatial = (
    Rational(1, 2) * DVT("w", (p, p2), (q, q2))
    * ad(q) * ad(q2) * a(p2) * a(p)
)
E2_spatial = substitute_dummies_double_vac(
    wicks_double_vac(
        W_spatial * get_R_nm(2, 0, W_spatial),
        keep_only_fully_contracted=True,
    )
)
E2_perloop = spin_integration_uhf(E2_spatial)

# the mixed sector: tagged route carries weight 2 (per the sector
# coefficient 1 entering twice, bra and ket), the per-loop route only 1
tagged_mixed = sector_weight(E2_tagged, "ab")
perloop_mixed = sector_weight(E2_perloop, "ab") + sector_weight(
    E2_perloop, "ba"
)
print(f"mixed-sector weight: tagged {tagged_mixed}, "
      f"per-loop {perloop_mixed}")
assert tagged_mixed == 2 * perloop_mixed, (
    "the per-loop route halves the opposite-spin sector - if this "
    "ever stops holding, revisit the warning on spin_integration_uhf"
)
print("per-loop opposite-spin halving: pinned")
