from sympy.physics.secondquant import (
    FockStateFermionKet,
    FockStateFermionBra,
    AnnihilateFermion,
    CreateFermion,
    AntiSymmetricTensor,
    apply_operators,
    NO,
    wicks,
    contraction,
    evaluate_deltas,
)

from sympy import (
    S,
    Symbol,
    symbols,
    Add,
    Mul,
    KroneckerDelta,
    Dummy,
    sympify,
    expand,
    pretty_print,
    latex,
)


class DoubleFermiVaccum:

    is_molA = False
    is_molB = False


class AnnihilateFermion_A(AnnihilateFermion, DoubleFermiVaccum):
    """
    Fermionic annihilation operator coresponding
    to molecule A (part A of a complex/dimer).

    Allows distinguishing creation and annihiltaion
    operators corresponding to A or B part of the complex/
    /dimer.
    """

    is_molA = True

    op_symbol = "a"

    def _dagger_(self):
        return CreateFermion_A(*self.state)

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
        return AnnihilateFermion_A(*self.state)

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
        return CreateFermion_A(*self.state)

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
        return AnnihilateFermion_B(*self.state)

    def __repr__(self):
        return "CreateFermion_B(%s)" % self.state

    def _latex(self, printer):
        return "b^\\dagger_{%s}" % self.state.name


a = AnnihilateFermion_A
ad = CreateFermion_A
b = AnnihilateFermion_B
bd = CreateFermion_B


class NO_double_vac:
    """
    Normal ordering brackets for double Fermi vaccum.

    Assumes arg consits only is_molA, is_molB or commuting
    parts.
    """

    def __new__(cls, arg):

        arg = sympify(arg)
        arg = arg.expand()
        if arg.is_Add:
            return Add(*[NO_double_vac(elem) for elem in arg.args])

        elif arg.is_Mul:

            # separating coefficient from arg
            comuting_part, seq = arg.args_cnc()
            if comuting_part:
                coeff = Mul(*comuting_part)
                if not seq:
                    return coeff
            else:
                coeff = S.One

            part_A = []
            part_B = []
            for elem in seq:
                if elem.is_molA:
                    part_A.append(elem)
                elif elem.is_molB:
                    part_B.append(elem)
                # elem neither is_molA, is_molB nor commuting
                else:
                    print("Something went wrong:")
                    print(elem, "coresponds to neither molecule A nor B!")

            # apply normal ordering brackets to parts A and B separately
            return coeff * NO(Mul(*part_A)) * NO(Mul(*part_B))

        # arg is neither Add nor Mul
        else:
            return arg


def contraction_double_vac(X, Y):
    """
    Calculates contraction for operators corresponding
    to either molecule A or molecule B.
    """

    if isinstance(X, DoubleFermiVaccum) and isinstance(Y, DoubleFermiVaccum):

        if isinstance(X, AnnihilateFermion_A) and isinstance(
            Y, CreateFermion_A
        ):
            if Y.state.assumptions0.get("below_fermi"):
                return S.Zero
            if X.state.assumptions0.get("below_fermi"):
                return S.Zero
            if Y.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(X.state, Y.state)
            if X.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(X.state, Y.state)

            return KroneckerDelta(X.state, Y.state) * KroneckerDelta(
                Y.state, Dummy("a1", is_molA=True, above_fermi=True)
            )

        if isinstance(X, CreateFermion_A) and isinstance(
            Y, AnnihilateFermion_A
        ):
            if Y.state.assumptions0.get("above_fermi"):
                return S.Zero
            if X.state.assumptions0.get("above_fermi"):
                return S.Zero
            if Y.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(X.state, Y.state)
            if X.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(X.state, Y.state)

            return KroneckerDelta(X.state, Y.state) * KroneckerDelta(
                Y.state, Dummy("i1", is_molA=True, below_fermi=True)
            )

        if isinstance(X, AnnihilateFermion_B) and isinstance(
            Y, CreateFermion_B
        ):
            if Y.state.assumptions0.get("below_fermi"):
                return S.Zero
            if X.state.assumptions0.get("below_fermi"):
                return S.Zero
            if Y.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(X.state, Y.state)
            if X.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(X.state, Y.state)

            return KroneckerDelta(X.state, Y.state) * KroneckerDelta(
                Y.state, Dummy("b1", is_molB=True, above_fermi=True)
            )

        if isinstance(X, CreateFermion_B) and isinstance(
            Y, AnnihilateFermion_B
        ):
            if Y.state.assumptions0.get("above_fermi"):
                return S.Zero
            if X.state.assumptions0.get("above_fermi"):
                return S.Zero
            if Y.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(X.state, Y.state)
            if X.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(X.state, Y.state)

            return KroneckerDelta(X.state, Y.state) * KroneckerDelta(
                Y.state, Dummy("j1", is_molB=True, below_fermi=True)
            )

        else:
            return S.Zero

    else:
        return contraction(X, Y)


def evaluate_deltas_double_vac(expr):
    """
    Function evaluating KroneckerDelta symbols in the expression assuming
    Einstein summation (the sum is over repeated index).

    Substiution are evaluated for double Fermi vaccum case. Therefore indicies
    in KronecerDelta should have an assumptions:
    - is_molA=True if this index applies only to part A of the complex,
    - is_molB=True if this index applies only to part B of the complex.
    """

    if isinstance(expr, Add):
        return Add(*[evaluate_deltas_double_vac(arg) for arg in expr.args])

    elif isinstance(expr, Mul):
        deltas = []
        indicies = {}
        for elem in expr.args:
            for s in elem.free_symbols:
                if s in indicies:
                    indicies[s] += 1
                else:
                    indicies[s] = 0
            if isinstance(elem, KroneckerDelta):
                deltas.append(elem)

        for d in deltas:
            # Now we have to check if killable and preferred apply
            # to the same part of the complex.
            killable_molA = d.killable_index.assumptions0.get("is_molA")
            killable_molB = d.killable_index.assumptions0.get("is_molB")
            preferred_molA = d.preferred_index.assumptions0.get("is_molA")
            preferred_molB = d.preferred_index.assumptions0.get("is_molB")

            if (killable_molA and preferred_molA) or (
                killable_molB and preferred_molB
            ):

                if d.killable_index.is_Symbol and indicies[d.killable_index]:
                    # Method killabel_index returns index containing less information
                    # regarding fermi level. If both contain the same amount of information
                    # alphabetical order is used to determine wich is preferred.
                    expr = expr.subs(d.killable_index, d.preferred_index)
                    if len(deltas) > 1:
                        return evaluate_deltas_double_vac(expr)

                elif (
                    d.preferred_index.is_Symbol
                    and indicies[d.preferred_index]
                    and d.indices_contain_equal_information
                ):
                    # Here we have situation where the preferred_index appers somewhere
                    # else in the expression. We can change
                    expr = expr.subs(d.preferred_index, d.killable_index)
                    if len(deltas) > 1:
                        return evaluate_deltas_double_vac(expr)

                else:
                    pass
            # Indicies correspond to diffrent parts of the complex! Delta is zero.
            else:
                return S.Zero

        return expr
    # Not Mul nor Add.
    else:
        return expr


if __name__ == "__main__":
    # Some debugs and checks (will be deleted in final version).
    # There is a plan to add some example files instead.

    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    a1, a2 = symbols("a1 a2", is_molA=True, above_fermi=True, cls=Dummy)
    i1, i2 = symbols("i1 i2", is_molA=True, below_fermi=True, cls=Dummy)

    b1, b2 = symbols("b1 b2", is_molB=True, above_fermi=True, cls=Dummy)
    j1, j2 = symbols("j1 j2", is_molB=True, below_fermi=True, cls=Dummy)

    v = AntiSymmetricTensor("v", (p, r,), (q, s,))
    vA = AntiSymmetricTensor("(v_A)", (r,), (s,))
    vB = AntiSymmetricTensor("(v_B)", (p,), (q,))
    V0 = symbols("V_0")

    V = (
        v * ad(q) * a(p) * bd(s) * b(r)
        + vA * bd(s) * b(r)
        + vB * ad(p) * a(q)
        + V0
    )

    expr = NO_double_vac(V)
    print(latex(expr))
    print()

    expr = NO(ad(p) * a(q)).doit(wicks=True)
    expr = expand(expr)

    expr = wicks(ad(p) * a(q))
    exor = evaluate_deltas_double_vac(expr)
    print(latex(expr))
