from functools import cmp_to_key
from sympy import Basic, Tuple, sympify
from sympy.physics.secondquant import TensorSymbol


class DoubleVacuumTensorSymbol(TensorSymbol):
    """
    Tensor symbols abstraction in double fermi vacuum
    """

    def __new__(cls, symbol, upper, lower, symmetries=None):
        symbol = sympify(symbol)
        upper = Tuple(*upper)
        lower = Tuple(*lower)

        if symmetries:
            symmetries = Tuple(*symmetries)

            # check if correct format of symmteries was given
            for symmetry in symmetries:
                if len(symmetry) != 2:
                    raise IndexError(
                        f"Symmetry must be (upper, lower) permutation pair, was {symmetry}!"
                    )

            upper, lower = _use_symmetries(upper, lower, symmetries)
        else:
            symmetries = Tuple()

        return TensorSymbol.__new__(cls, symbol, upper, lower, symmetries)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def upper(self):
        return self.args[1]

    @property
    def lower(self):
        return self.args[2]

    @property
    def symmetries(self):
        return self.args[3]

    def _dagger_(self):
        return DoubleVacuumTensorSymbol(
            self.symbol,
            self.lower,
            self.upper,
            self.symmetries,
        )

    def _latex(self, printer):
        latex_str = "%s" % (self.symbol)

        if len(self.upper):
            latex_str += "^{%s}" % "".join([i.name for i in self.upper])

        if len(self.lower):
            latex_str += "_{%s}" % "".join([i.name for i in self.lower])

        return latex_str

    def __str__(self):
        return f"{self.symbol}({self.upper},{self.lower})"

    def _hashable_content(self):
        return (self.symbol, self.upper, self.lower)


_basic_sortkey = cmp_to_key(Basic.compare)


def _idx_sortkey(idx):
    """
    Order indices by name, then by SymPy's own total order.

    Basic.compare orders by class before content, so every free Symbol
    sorts ahead of every Dummy whatever it is called. Keying on the name
    first keeps the canonical order the same whether an index is free or
    summed - otherwise a tensor mixing the two gets a sort target that no
    permutation in its symmetry group can reach, and silently stays
    uncanonicalized. The Basic.compare tiebreak keeps the order total.

    An index slot does not always hold a Symbol: subs(simultaneous=True)
    passes a Mul sentinel through it while resolving swaps, so fall back
    to str() rather than assuming .name exists.
    """
    return (getattr(idx, "name", str(idx)), _basic_sortkey(idx))


def _orbit_key(upper, lower):
    """
    Total, deterministic order on a whole (upper, lower) index layout.
    """
    return tuple(_idx_sortkey(idx) for idx in tuple(upper) + tuple(lower))


def _use_symmetries(upper, lower, symmetries):
    """
    Canonicalize the index layout by taking the smallest one the
    symmetry group can reach.

    Sorting upper and lower independently and then hunting for a
    permutation pair that produces exactly that layout only works when
    the group can permute the two rows independently. A symmetry that
    couples them - v^{pq}_{rs} = v^{qp}_{sr}, where a row swap upstairs
    forces one downstairs - usually cannot reach the independently
    sorted target at all, and the tensor would then be left alone in
    whichever layout it happened to be built with. Two spellings of the
    same tensor stay distinct and their terms never collapse.

    Minimizing over the orbit is well defined for any group: every
    spelling of the same tensor reaches the same representative, and the
    result does not depend on the order the symmetries are listed in.

    A symmetry permutes positions within a row and can never move an
    index between the rows. That is deliberate, not a gap: the upper and
    lower slot is what carries the direction of a contraction line, so
    bra-ket symmetry (v^{pq}_{rs} = v^{rs}_{pq}) would break the Einstein
    summation convention the rest of the package reads - the once-up,
    once-down pairing in evaluate_deltas_double_vac, the loop matching in
    _count_loops, and the in/out lines in sinfinitizer.
    """
    upper = Tuple(*upper)
    lower = Tuple(*lower)

    best_upper, best_lower = upper, lower
    best_key = _orbit_key(upper, lower)

    for sym in symmetries:
        upper_sym = sym[0]
        lower_sym = sym[1]

        new_upper = Tuple(*(upper[idx] for idx in upper_sym))
        new_lower = Tuple(*(lower[idx] for idx in lower_sym))
        key = _orbit_key(new_upper, new_lower)

        if key < best_key:
            best_upper, best_lower, best_key = new_upper, new_lower, key

    return best_upper, best_lower
