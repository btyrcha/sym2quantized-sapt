"""
UMP2 by both open-shell routes, compared with conventional UMP2.

**Spin tags** are the supported route.  The per-loop column is a frozen
proof of concept, not maintained, and is shown here only for comparison.

**Spin tags** (``is_alpha`` / ``is_beta``) are the open-shell analogue
of the monomer tags: contractions vanish across them, so Wick's theorem
does the UHF bookkeeping exactly.  The fluctuation operator enters as
its four spin sectors, written out here (the library has no monomer W
builder).  The (2,0) resolvent is the library's ``get_R_nm``, unchanged:
its indices carry no tag, so they run over both spins and take the tag
of the W index they contract with, and its one normalisation 1/(2!)^2
is the right one for that sum.  Tensors keep plain names (``v``); the
tags carry the spin.  With the resolvent written out by sector
instead (1/(2!)^2 same-spin, 1 for alpha before beta), this route was
also checked downstream against psi4's conventional UHF-MP2, to ~1e-16
per spin channel.

**Per loop** (frozen proof of concept): the energy is derived once with
spatial indices and then spin-summed with ``spin_integration_uhf``, one
spin label per Goldstone loop.  Loops run through graph vertices only.  The resolvent
denominator ``e`` is not a vertex (``is_graph_vertex=False``), so it
takes its spins from the loops its indices lie on and is labelled per
index: ``e_ab_ba``.  When ``e`` was still traced as a vertex, this
route gave half the opposite-spin energy while passing its
RHF-collapse gate -- see ``docs/notes/uhf-spin-summation.md``.

Both routes are evaluated on small random UHF-like data and printed
next to the textbook UMP2 formula, sector by sector: the spin-tag
route's spins are read off the index tags, the per-loop route's off its
block names.  A self-consistency gate cannot certify a formula; an
independent reference can.
"""

import random
from itertools import product

from sympy import Add, Dummy, Mul, Rational, symbols

from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.open_shell import has_spin_tags, index_spin
from sym2quantized_sapt.operators import A, Ad
from sym2quantized_sapt.sapt_utils import get_R_nm
from sym2quantized_sapt.spin_integrator.uhf import spin_integration_uhf
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol as DVT

SPIN = {"a": {"is_alpha": True}, "b": {"is_beta": True}}


def get_w_tagged():
    """W as its four spin sectors: each (upper, lower) slot pair of v,
    (p, q) and (p', q'), carries one spin."""
    sectors = []

    for s1, s2 in product("ab", repeat=2):
        p, q = symbols("p q", is_molA=True, cls=Dummy, **SPIN[s1])
        p2, q2 = symbols("p' q'", is_molA=True, cls=Dummy, **SPIN[s2])
        sectors.append(
            Rational(1, 2)
            * DVT("v", (p, p2), (q, q2))
            * Ad(q)
            * Ad(q2)
            * A(p2)
            * A(p)
        )

    return Add(*sectors)


def get_w_spatial():
    p, q = symbols("p q", is_molA=True, cls=Dummy)
    p2, q2 = symbols("p' q'", is_molA=True, cls=Dummy)

    return (
        Rational(1, 2)
        * DVT("v", (p, p2), (q, q2))
        * Ad(q)
        * Ad(q2)
        * A(p2)
        * A(p)
    )


# ---- spin tags ----------------------------------------------------------
R20_W_tagged = get_R_nm(2, 0, get_w_tagged())

E2_tagged = wicks_double_vac(
    get_w_tagged() * R20_W_tagged,
    keep_only_fully_contracted=True,
)

print(f"UMP2, spin-tagged: {len(E2_tagged.args)} terms")

# ---- per loop -----------------------------------------------------------
R20_W_spatial = get_R_nm(2, 0, get_w_spatial())

E2_spatial = wicks_double_vac(
    get_w_spatial() * R20_W_spatial,
    keep_only_fully_contracted=True,
)

E2_perloop = spin_integration_uhf(E2_spatial)

print(
    f"UMP2, per loop: {len(E2_spatial.args)} spatial terms, "
    f"{len(E2_perloop.args)} spin-blocked"
)

# ---- numbers ------------------------------------------------------------
# Random UHF-like data, small enough for plain Python loops.  Alpha and
# beta get different orbital counts, so a spin label on the wrong index
# changes an index range instead of going unnoticed.  The integrals are
# built density-fitting style, (pq|rs) = sum_P B[P][p][q] B[P][r][s]
# with each B[P] symmetric, so they have the symmetry of real ones.

rng = random.Random(7)
N_OCC = {"a": 3, "b": 2}
N_VIR = {"a": 2, "b": 3}
N_AUX = 3

EPS = {
    s: [-1 - rng.random() for _ in range(N_OCC[s])]
    + [1 + rng.random() for _ in range(N_VIR[s])]
    for s in "ab"
}

B = {}
for s in "ab":
    n = N_OCC[s] + N_VIR[s]
    B[s] = [[[0.0] * n for _ in range(n)] for _ in range(N_AUX)]
    for P, p in product(range(N_AUX), range(n)):
        for q in range(p, n):
            B[s][P][p][q] = B[s][P][q][p] = rng.uniform(-1, 1)


def eri(s1, p, q, s2, r, s):
    """(pq|rs): p and q of spin s1, r and s of spin s2."""
    return sum(B[s1][P][p][q] * B[s2][P][r][s] for P in range(N_AUX))


def orbitals(space, spin):
    if space == "o":
        return range(N_OCC[spin])

    return range(N_OCC[spin], N_OCC[spin] + N_VIR[spin])


def index_spins(tensor):
    """(index, spin) for every upper, then lower, index of ``tensor``.

    A spin-tagged tensor carries them on its indices.  A per-loop block
    carries them in its name: one label per slot pair for a vertex
    (``v_ab``), upper then lower labels per index for a non-vertex
    (``e_ab_ba``)."""
    if has_spin_tags(tensor):
        return [(i, index_spin(i)) for i in (*tensor.upper, *tensor.lower)]

    labels = str(tensor.symbol).split("_")[1:]

    if tensor.is_graph_vertex:
        (pair_spins,) = labels
        spins = pair_spins + pair_spins
    else:
        spins = "".join(labels)

    return list(zip((*tensor.upper, *tensor.lower), spins))


def tensor_value(tensor, spin, orbital):
    upper = [(spin[i], orbital[i]) for i in tensor.upper]
    lower = [(spin[i], orbital[i]) for i in tensor.lower]

    if str(tensor.symbol).split("_")[0] == "v":
        # v^{p p2}_{q q2} = (p q | p2 q2): pairs (p, q) and (p2, q2)
        (s1, p), (s2, p2) = upper
        (_, q), (_, q2) = lower
        return eri(s1, p, q, s2, p2, q2)

    # e^{i j}_{a b} = 1 / (e_i + e_j - e_a - e_b)
    return 1 / (
        sum(EPS[s][o] for s, o in upper) - sum(EPS[s][o] for s, o in lower)
    )


def energy_by_sector(expr):
    """Evaluate a spin-resolved energy, keyed by the spins of e's
    holes."""
    sectors = {}

    for term in Add.make_args(expr):
        tensors = [f for f in term.args if isinstance(f, DVT)]
        coefficient = float(
            Mul(*[f for f in term.args if not isinstance(f, DVT)])
        )

        spin = {}
        for tensor in tensors:
            spin.update(index_spins(tensor))

        indices = list(spin)
        spaces = [
            "o" if idx.assumptions0.get("below_fermi") else "v"
            for idx in indices
        ]

        total = 0.0
        for values in product(
            *(
                orbitals(space, spin[idx])
                for idx, space in zip(indices, spaces)
            )
        ):
            orbital = dict(zip(indices, values))
            contribution = coefficient
            for tensor in tensors:
                contribution *= tensor_value(tensor, spin, orbital)
            total += contribution

        (denominator,) = [t for t in tensors if not t.is_graph_vertex]
        sector = "".join(sorted(spin[i] for i in denominator.upper))
        sectors[sector] = sectors.get(sector, 0.0) + total

    return sectors


def conventional_ump2():
    """Textbook UMP2 from the same integrals, keyed like energy_by_sector."""
    sectors = {}

    for s1, s2 in product("ab", repeat=2):
        total = 0.0
        for i, j, a, b in product(
            orbitals("o", s1),
            orbitals("o", s2),
            orbitals("v", s1),
            orbitals("v", s2),
        ):
            direct = eri(s1, i, a, s2, j, b)  # <ij|ab>
            denominator = EPS[s1][i] + EPS[s2][j] - EPS[s1][a] - EPS[s2][b]

            if s1 == s2:
                exchange = eri(s1, i, b, s2, j, a)  # <ij|ba>
                total += 0.25 * (direct - exchange) ** 2 / denominator
            else:
                # alpha-beta and beta-alpha orderings carry half each
                total += 0.5 * direct**2 / denominator

        sector = "".join(sorted(s1 + s2))
        sectors[sector] = sectors.get(sector, 0.0) + total

    return sectors


reference = conventional_ump2()
routes = {
    "spin tags": energy_by_sector(E2_tagged),
    "per loop": energy_by_sector(E2_perloop),
}

print(f"\n{'sector':8s}{'conventional':>16s}", end="")
print("".join(f"{route:>16s}" for route in routes))
for sector in sorted(reference):
    print(f"{sector:8s}{reference[sector]:16.10f}", end="")
    print("".join(f"{routes[r][sector]:16.10f}" for r in routes))
