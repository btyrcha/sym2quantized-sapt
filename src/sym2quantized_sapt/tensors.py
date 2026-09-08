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


def _use_symmetries(upper, lower, symmetries):
    _idx_sortkey = cmp_to_key(Basic.compare)
    upper_sorted = Tuple(*sorted(upper, key=_idx_sortkey))
    lower_sorted = Tuple(*sorted(lower, key=_idx_sortkey))

    if upper != upper_sorted or lower != lower_sorted:
        for sym in symmetries:
            upper_sym = sym[0]
            lower_sym = sym[1]

            new_upper = Tuple(*(upper[idx] for idx in upper_sym))
            new_lower = Tuple(*(lower[idx] for idx in lower_sym))

            if new_upper == upper_sorted and new_lower == lower_sorted:
                return new_upper, new_lower

    return upper, lower
