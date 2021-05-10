from sympy.physics.secondquant import AnnihilateFermion, CreateFermion


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
        return "a_{%s}" % self.state.name


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
        return "a^\\dagger_{%s}" % self.state.name


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
        return "b_{%s}" % self.state.name


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
        return "b^\\dagger_{%s}" % self.state.name


# importable operators classes
a = AnnihilateFermion_A
ad = CreateFermion_A  # a_dagger
b = AnnihilateFermion_B
bd = CreateFermion_B  # b_dagger
