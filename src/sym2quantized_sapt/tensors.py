from sympy.physics.secondquant import TensorSymbol
from sympy import Tuple, sympify


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
