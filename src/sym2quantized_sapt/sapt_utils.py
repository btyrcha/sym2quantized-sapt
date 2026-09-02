import itertools
from sympy import symbols, Dummy, factorial
from sympy.core import Expr, Mul, S
from sympy.physics.secondquant import Dagger
from sym2quantized_sapt.operators import (
    CreateFermion_A,
    CreateFermion_B,
    AnnihilateFermion_A,
    AnnihilateFermion_B,
)
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.double_fermi_vac import (
    wicks_double_vac,
    substitute_dummies_double_vac,
)


A = AnnihilateFermion_A
Ad = CreateFermion_A
B = AnnihilateFermion_B
Bd = CreateFermion_B


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
            upper_indicies.append(symbols("p_{0}".format(i), is_molA=True))
            lower_indicies.append(symbols("q_{0}".format(i), is_molA=True))

    creation_part = [Ad(index) for index in upper_indicies]
    annihilation_part = [A(index) for index in reversed(lower_indicies)]
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
            upper_indicies.append(symbols("r_{0}".format(i), is_molB=True))
            lower_indicies.append(symbols("s_{0}".format(i), is_molB=True))

    creation_part = [Bd(index) for index in upper_indicies]
    annihilation_part = [B(index) for index in reversed(lower_indicies)]
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
        v * Ad(q) * A(p) * Bd(s) * B(r)
        + vA * Bd(s) * B(r)
        + vB * Ad(q) * A(p)
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
        * Ad(q)
        * A(p)
        * Bd(s)
        * B(r)
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

    a_part = Ad(q1) * Ad(q2) * A(p2) * A(p1)
    b_part = Bd(s1) * Bd(s2) * B(r2) * B(r1)

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

    for i in range(1, m + 1):
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
    coeff = float(coeff)
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

    creation_part_A = [Ad(index) for index in lower_indicies_A]
    annihilation_part_A = [A(index) for index in reversed(upper_indicies_A)]
    a_part = Mul(*creation_part_A, *annihilation_part_A)

    creation_part_B = [Bd(index) for index in lower_indicies_B]
    annihilation_part_B = [B(index) for index in reversed(upper_indicies_B)]
    b_part = Mul(*creation_part_B, *annihilation_part_B)

    return P_tensor * a_part * b_part


def get_R_nm(n: int, m: int, operator: Expr) -> Expr:
    """
    Calculates the resolvent of excitations (n,m) acting on the `operator`.
    """

    coeff = 1 / (factorial(n) * factorial(m)) ** 2
    coeff = float(coeff)

    particle_indicies_A = []
    hole_indicies_A = []
    particle_indicies_B = []
    hole_indicies_B = []

    for i in range(1, n + 1):
        particle_indicies_A.append(
            symbols(
                "a_{0}".format(i), above_fermi=True, is_molA=True, cls=Dummy
            )
        )
        hole_indicies_A.append(
            symbols(
                "i_{0}".format(i), below_fermi=True, is_molA=True, cls=Dummy
            )
        )

    for i in range(1, m + 1):
        particle_indicies_B.append(
            symbols(
                "b_{0}".format(i), above_fermi=True, is_molB=True, cls=Dummy
            )
        )
        hole_indicies_B.append(
            symbols(
                "j_{0}".format(i), below_fermi=True, is_molB=True, cls=Dummy
            )
        )

    creation_part_A = [Ad(index) for index in hole_indicies_A]
    annihilation_part_A = [A(index) for index in reversed(particle_indicies_A)]
    a_part = Mul(*creation_part_A, *annihilation_part_A)

    creation_part_B = [Bd(index) for index in hole_indicies_B]
    annihilation_part_B = [B(index) for index in reversed(particle_indicies_B)]
    b_part = Mul(*creation_part_B, *annihilation_part_B)

    tensors = wicks_double_vac(
        a_part * b_part * operator,
        keep_only_fully_contracted=True,
        substitute_dummies=False,
    )

    # figure out denominator symmetries
    perm_A = itertools.permutations(range(0, n))
    perm_B = itertools.permutations(range(n, n + m))

    perms = [
        tuple([*perm[0]] + [*perm[1]])
        for perm in itertools.product(perm_A, perm_B)
    ]

    e_symmetries = tuple(itertools.product(perms, perms))

    denom = DoubleVacuumTensorSymbol(
        "e",
        hole_indicies_A + hole_indicies_B,
        particle_indicies_A + particle_indicies_B,
        e_symmetries,
    )

    expr = coeff * tensors * Dagger(a_part) * Dagger(b_part) * denom
    expr = substitute_dummies_double_vac(expr)

    return expr
