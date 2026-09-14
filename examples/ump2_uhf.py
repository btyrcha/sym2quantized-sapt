"""
UMP2 by both open-shell routes, compared with conventional UMP2.

**Spin tags** (``is_alpha`` / ``is_beta``) are the open-shell analogue
of the monomer tags: contractions vanish across them, so Wick's theorem
does the UHF bookkeeping exactly.  The fluctuation operator enters as
its four spin sectors and the (2,0) resolvent as its three, with the
per-sector normalisation the tags make explicit: 1/(2!)^2 for two
same-spin (indistinguishable) pairs, 1 for the distinguishable
alpha+beta pair.  This route also matches psi4's conventional UHF-MP2
(checked downstream to ~1e-16 per spin channel).

**Per loop**: the energy is derived once with spatial indices and then
spin-summed with ``spin_integration_uhf``, one spin label per Goldstone
loop.  Loops run through graph vertices only.  The resolvent
denominator ``e`` is not a vertex (``is_graph_vertex=False``), so it
takes its spins from the loops its indices lie on and is labelled per
index: ``e_ab_ba``.  When ``e`` was still traced as a vertex, this
route gave half the opposite-spin energy while passing its
RHF-collapse gate -- see ``docs/notes/uhf-spin-summation.md``.

Both routes are evaluated on small random UHF-like data and printed
next to the textbook UMP2 formula, sector by sector.  A
self-consistency gate cannot certify a formula; an independent
reference can.
"""

import random
from itertools import product

from sympy import Add, Dummy, Mul, Rational, symbols
from sympy.physics.secondquant import Dagger

from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.operators import A, Ad
from sym2quantized_sapt.sapt_utils import get_R_nm
from sym2quantized_sapt.open_shell import spin_integration_uhf
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol as DVT

SPIN = {"a": {"is_alpha": True}, "b": {"is_beta": True}}
_counter = [0]


def index(letter, spin, **kw):
    _counter[0] += 1
    return symbols(
        f"{letter}_{_counter[0]}",
        is_molA=True,
        cls=Dummy,
        **SPIN[spin],
        **kw,
    )


def get_w_tagged():
    """W = 1/2 sum over spin sectors of v_{s1 s2}."""
    sectors = []

    for s1, s2 in product("ab", repeat=2):
        p, q = index("p", s1), index("q", s1)
        p2, q2 = index("p", s2), index("q", s2)
        sectors.append(
            Rational(1, 2)
            * DVT(f"v_{s1}{s2}", (p, p2), (q, q2))
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
            Ad(holes[0]) * Ad(holes[1]) * A(particles[1]) * A(particles[0])
        )
        amplitude = wicks_double_vac(
            excitation * operator,
            keep_only_fully_contracted=True,
            substitute_dummies=False,
        )
        # not a graph vertex, so labelled per index (holes, then
        # particles) -- the names spin_integration_uhf gives it
        denominator = DVT(
            "_".join(("e", "".join(spins), "".join(spins))),
            tuple(holes),
            tuple(particles),
            is_graph_vertex=False,
        )

        piece = coefficient * amplitude * Dagger(excitation) * denominator

        pieces.append(piece)

    return Add(*pieces).expand()


# ---- spin tags ----------------------------------------------------------
R20_W_tagged = r20_tagged(get_w_tagged())

E2_tagged = wicks_double_vac(
    get_w_tagged() * R20_W_tagged,
    keep_only_fully_contracted=True,
    substitute_dummies=False,
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
    """(index, spin) for every upper, then lower, index of a spin-blocked
    tensor, read off its name: one label per slot pair for a vertex
    (``v_ab``), upper then lower labels per index for a non-vertex
    (``e_ab_ba``)."""
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

    if str(tensor.symbol).startswith("v_"):
        # v^{p p2}_{q q2} = (p q | p2 q2): pairs (p, q) and (p2, q2)
        (s1, p), (s2, p2) = upper
        (_, q), (_, q2) = lower
        return eri(s1, p, q, s2, p2, q2)

    # e^{i j}_{a b} = 1 / (e_i + e_j - e_a - e_b)
    return 1 / (
        sum(EPS[s][o] for s, o in upper) - sum(EPS[s][o] for s, o in lower)
    )


def energy_by_sector(expr):
    """Evaluate a spin-blocked energy, keyed by the spins of e's holes."""
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
