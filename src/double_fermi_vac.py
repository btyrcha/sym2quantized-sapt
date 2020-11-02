from sympy.physics.secondquant import (FockStateFermionKet, FockStateFermionBra,
    AnnihilateFermion, CreateFermion, AntiSymmetricTensor, apply_operators, NO,
    wicks, contraction, evaluate_deltas)
from sympy import (S, Symbol, symbols, Add, Mul, Dummy, sympify, expand,
    pretty_print, latex)

class DoubleFermiVaccum():

    is_molA=False
    is_molB=False


class AnnihilateFermion_A(AnnihilateFermion, DoubleFermiVaccum):
    """
    Fermionic annihilation operator coresponding
    to molecule A (part A of a complex/dimer).

    Allows distinguishing creation and annihiltaion
    operators corresponding to A or B part of the complex/
    /dimer.

    """
    is_molA = True

    op_symbol = 'a'

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

    op_symbol = 'a+'

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

    op_symbol = 'b'

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

    op_symbol = 'b+'

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


class NO_double_vac():
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

        if X.is_molA and Y.is_molA:
            return contraction(X, Y)

        elif X.is_molB and Y.is_molB:
            return contraction(X, Y)
        
        else:
            return S.Zero

    else:
        return contraction(X, Y)




p, q = symbols('p q', cls=Dummy, is_mol_A=True)
r, s = symbols('r s', cls=Dummy, is_mol_B=True)

v = AntiSymmetricTensor('v', (p, r,), (q, s,))
vA = AntiSymmetricTensor('(v_A)', (r,), (s,))
vB = AntiSymmetricTensor('(v_B)', (p,), (q,))
V0 = symbols('V_0')

V = v*ad(q)*a(p)*bd(s)*b(r) + vA*bd(s)*b(r) + vB*ad(p)*a(q) + V0

expr = NO_double_vac(V)
print(latex(expr))
print()


#d, e = symbols('d e', above_fermi=True, cls=Dummy)
#i, j = symbols('i j', below_fermi=True, cls=Dummy) 

#Fd = CreateFermion
#F = AnnihilateFermion

#print(contraction_double_vac(b(p), b(q)))
