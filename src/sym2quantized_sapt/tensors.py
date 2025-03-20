from sympy.physics.secondquant import TensorSymbol, ViolationOfPauliPrinciple
from sympy import Tuple, sympify, S, Dummy


class DoubleVacuumTensorSymbol(TensorSymbol):
    """
    Tensor symbols abstraction in double fermi vacuum
    """

    def __new__(cls, symbol, upper, lower):
        symbol = sympify(symbol)
        upper = Tuple(*upper)
        lower = Tuple(*lower)

        return TensorSymbol.__new__(cls, symbol, upper, lower)

    def symbol(self):
        return self.args[0]

    def upper(self):
        return self.args[1]

    def lower(self):
        return self.args[2]

    def _dagger_(self):
        return DoubleVacuumTensorSymbol(
            self.args[0], self.args[2], self.args[1]
        )

    def _latex(self, printer):
        latex_str = "%s" % (self.args[0])

        if len(self.args[1]):
            latex_str += "^{%s}" % "".join([i.name for i in self.args[1]])

        if len(self.args[2]):
            latex_str += "_{%s}" % "".join([i.name for i in self.args[2]])

        return latex_str

    def __str__(self):
        return "%s(%s,%s)" % self.args


def _get_key(i):

    if isinstance(i, Dummy):
        return "_".join([i.name, str(i.dummy_index)])

    return i.name


def sort_fermionic_indices(indices):
    """
    Sort fermionic indices from one monomer
    and figure out parity of permutaiton of sorted indices.
    """

    # original order
    idx_num = dict(zip(indices, range(len(indices))))

    keys = [_get_key(i) for i in indices]
    indices = dict(zip(keys, indices))

    # sort it by keys
    sorted_keys = sorted(keys)
    sorted_indices = [indices[key] for key in sorted_keys]

    # figure out parity
    permutation = [idx_num[i] for i in sorted_indices]

    par = 0
    length = len(permutation)
    for i in range(length):

        for j in range(i + 1, length):

            if permutation[i] > permutation[j]:
                par += 1

    par = par % 2

    return sorted_indices, par


class AntiSymmetricTensorSymbol(DoubleVacuumTensorSymbol):
    """
    Anti-symmetric tensor symbol in double fermi vacuum
    """

    def __new__(cls, symbol, upper, lower):

        try:
            upper, par_up = sort_fermionic_indices(upper)
            lower, par_lo = sort_fermionic_indices(lower)

        except ViolationOfPauliPrinciple:
            return S.Zero

        if (par_up + par_lo) % 2:
            return -DoubleVacuumTensorSymbol.__new__(cls, symbol, upper, lower)

        return DoubleVacuumTensorSymbol.__new__(cls, symbol, upper, lower)
