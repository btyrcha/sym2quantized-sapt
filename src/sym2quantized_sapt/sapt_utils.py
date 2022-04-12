from sympy import symbols, Dummy
from sympy.core import Expr, Mul, S
from sym2quantized_sapt.operators import ad, a, bd, b
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from math import factorial


def get_a_operator(upper_indicies=None, lower_indicies=None, n=1) -> Expr:
    """
    Returns exchange operator for n electrons on monomer A in form:
    a^{p_0 p_1 ... p_n}_{q_0 q_1 ... q_n} =
        = ad(p_0) ad(p_1) ... ad(p_n) a(q_n) ... a(q_1) a(q_0)

    Args:
        upper_indicies (iterable): list of upper indicies
        lower_indicies (iterable): list of lower indicies
        n (int): number of electrons if given ignors upper_indicies and lower_indicies

    Returns:
        Expr: expression representing operator
    """

    if n:
        upper_indicies = []
        lower_indicies = []
        for i in range(n):
            upper_indicies.append(
                symbols("p_{0}".format(i), is_molA=True, cls=Dummy)
            )
            lower_indicies.append(
                symbols("q_{0}".format(i), is_molA=True, cls=Dummy)
            )

    creation_part = [ad(index) for index in upper_indicies]
    annihilation_part = [a(index) for index in reversed(lower_indicies)]
    return Mul(*creation_part, *annihilation_part)


def get_b_operator(upper_indicies=None, lower_indicies=None, n=1) -> Expr:
    """
    Returns exchange operator for n electrons on monomer B in form:
    b^{r_0 r_1 ... r_n}_{s_0 s_1 ... s_n} =
        = bd(r_0) bd(r_1) ... bd(r_n) b(s_n) ... b(s_1) b(s_0)

    Args:
        upper_indicies (iterable): list of upper indicies
        lower_indicies (iterable): list of lower indicies
        n (int): number of electrons if given ignors upper_indicies and lower_indicies

    Returns:
        Expr: expression representing operator
    """

    if n:
        upper_indicies = []
        lower_indicies = []
        for i in range(n):
            upper_indicies.append(
                symbols("r_{0}".format(i), is_molB=True, cls=Dummy)
            )
            lower_indicies.append(
                symbols("s_{0}".format(i), is_molB=True, cls=Dummy)
            )

    creation_part = [bd(index) for index in upper_indicies]
    annihilation_part = [b(index) for index in reversed(lower_indicies)]
    return Mul(*creation_part, *annihilation_part)


def get_V_operator() -> Expr:
    """prepares intermolecular interaction operator V

    Returns:
        Expr: SymPy Expr encoding V (dimer interaction operator)
    """
    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    v = DoubleVacuumTensorSymbol(
        "v",
        (
            p,
            r,
        ),
        (
            q,
            s,
        ),
    )
    vA = DoubleVacuumTensorSymbol("(v_A)", (r,), (s,))
    vB = DoubleVacuumTensorSymbol("(v_B)", (p,), (q,))
    V0 = DoubleVacuumTensorSymbol("V_0", (), ())

    V = (
        v * ad(q) * a(p) * bd(s) * b(r)
        + vA * bd(s) * b(r)
        + vB * ad(q) * a(p)
        + V0
    )

    return V


def get_P2_operator() -> Expr:
    """
    Prepers P2 permutation operator of one pair of electrons
    interchanging between monomers A and B.

    Returns:
        Expr: SymPy Expr encoding P2 permutation operator
    """
    p, q = symbols("p q", is_molA=True, cls=Dummy)
    r, s = symbols("r s", is_molB=True, cls=Dummy)

    P2 = (
        -DoubleVacuumTensorSymbol("s", (r,), (q,))
        * DoubleVacuumTensorSymbol("s", (p,), (s,))
        * ad(q)
        * a(p)
        * bd(s)
        * b(r)
    )

    return P2


def get_P4_operator() -> Expr:
    """
    Prepers P4 permutation operator of two pairs of electrons
    interchanging between monomers A and B.

    Returns:
        Expr: SymPy Expr encoding P4 permutation operator
    """
    p1, p2, q1, q2 = symbols("p_1 p_2 q_1 q_2", is_molA=True, cls=Dummy)
    r1, r2, s1, s2 = symbols("r_1 r_2 s_1 s_2", is_molB=True, cls=Dummy)

    P_tensor = (
        0.25
        * DoubleVacuumTensorSymbol("s", (r1,), (q1,))
        * DoubleVacuumTensorSymbol("s", (r2,), (q2,))
        * DoubleVacuumTensorSymbol("s", (p1,), (s1,))
        * DoubleVacuumTensorSymbol("s", (p2,), (s2,))
    )

    a_part = ad(q1) * ad(q2) * a(p2) * a(p1)
    b_part = bd(s1) * bd(s2) * b(r2) * b(r1)

    return P_tensor * a_part * b_part


def get_Pn_operator(n: int) -> Expr:
    """
    Prepers Pn operator - permutation operator of n pairs of electrons
    interchanging between monomers A and B.

    Args:
        n (int): number of electrons interchanging between monomers

    Returns:
        Expr: SymPy Expr encoding Pn permutation operator
    """

    m = n // 2

    upper_indicies_A = []
    lower_indicies_A = []
    upper_indicies_B = []
    lower_indicies_B = []

    for i in range(m):
        upper_indicies_A.append(
            symbols("p_{0}".format(i), is_molA=True, cls=Dummy)
        )
        lower_indicies_A.append(
            symbols("q_{0}".format(i), is_molA=True, cls=Dummy)
        )
        upper_indicies_B.append(
            symbols("r_{0}".format(i), is_molB=True, cls=Dummy)
        )
        lower_indicies_B.append(
            symbols("s_{0}".format(i), is_molB=True, cls=Dummy)
        )

    coeff = ((-1) ** m) / (factorial(m) ** 2)
    if coeff == -1:
        P_tensor = -1 * S.One
    else:
        P_tensor = coeff

    for i in range(m):
        P_tensor *= DoubleVacuumTensorSymbol(
            "s", (upper_indicies_B[i],), (lower_indicies_A[i],)
        )
    for i in range(m):
        P_tensor *= DoubleVacuumTensorSymbol(
            "s", (upper_indicies_A[i],), (lower_indicies_B[i],)
        )

    creation_part_A = [ad(index) for index in lower_indicies_A]
    annihilation_part_A = [a(index) for index in reversed(upper_indicies_A)]
    a_part = Mul(*creation_part_A, *annihilation_part_A)

    creation_part_B = [bd(index) for index in lower_indicies_B]
    annihilation_part_B = [b(index) for index in reversed(upper_indicies_B)]
    b_part = Mul(*creation_part_B, *annihilation_part_B)

    return P_tensor * a_part * b_part
