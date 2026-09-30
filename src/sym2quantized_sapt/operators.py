from sympy.physics.secondquant import AnnihilateFermion, CreateFermion

from sym2quantized_sapt.open_shell import index_latex


class DoubleFermiVaccum:
    is_molA = False
    is_molB = False


class AnnihilateFermion_A(AnnihilateFermion, DoubleFermiVaccum):
    """
    Fermionic annihilation operator coresponding
    to molecule A (part A of a complex/dimer).

    Allows distinguishing creation and annihilation
    operators corresponding to A or B part of the complex/
    /dimer.
    """

    is_molA = True

    op_symbol = "a"

    def _dagger_(self):
        return CreateFermion_A(self.state)

    def __repr__(self):
        return "AnnihilateFermion_A(%s)" % self.state

    def _latex(self, printer):
        return "a_{%s}" % index_latex(self.state)


class CreateFermion_A(CreateFermion, DoubleFermiVaccum):
    """
    Fermionic creation operator coresponding
    to molecule A.

    See also AnnihilateFermion_A.
    """

    is_molA = True

    op_symbol = "a+"

    def _dagger_(self):
        return AnnihilateFermion_A(self.state)

    def __repr__(self):
        return "CreateFermion_A(%s)" % self.state

    def _latex(self, printer):
        return "a^\\dagger_{%s}" % index_latex(self.state)


class AnnihilateFermion_B(AnnihilateFermion, DoubleFermiVaccum):
    """
    Fermionic annihilation operator coresponding
    to molecule B.

    See also AnnihilateFermion_A.
    """

    is_molB = True

    op_symbol = "b"

    def _dagger_(self):
        return CreateFermion_B(self.state)

    def __repr__(self):
        return "AnnihilateFermion_B(%s)" % self.state

    def _latex(self, printer):
        return "b_{%s}" % index_latex(self.state)


class CreateFermion_B(CreateFermion, DoubleFermiVaccum):
    """
    Fermionic creation operator coresponding
    to molecule B.

    See also AnnihilateFermion_A.
    """

    is_molB = True

    op_symbol = "b+"

    def _dagger_(self):
        return AnnihilateFermion_B(self.state)

    def __repr__(self):
        return "CreateFermion_B(%s)" % self.state

    def _latex(self, printer):
        return "b^\\dagger_{%s}" % index_latex(self.state)


# importable operators classes
A = AnnihilateFermion_A
Ad = CreateFermion_A  # a_dagger
B = AnnihilateFermion_B
Bd = CreateFermion_B  # b_dagger
