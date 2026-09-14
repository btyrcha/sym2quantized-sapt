from sympy import Add, Mul
from sympy.core import Expr
from sympy.physics.secondquant import TensorSymbol


def _loop_partition(upper, lower):
    """The Goldstone loops of a term, as a partition of slot positions.

    Position ``i`` is the i-th ``(upper, lower)`` slot pair of the
    concatenated tensor lists (both lists are built per tensor in slot
    order, so a position identifies one tensor's k-th pair -- one
    particle/hole line passing through that tensor).  Two positions
    join a loop when one's upper index is the other's lower index.

    Spin is constant along a loop, which is what makes this partition
    the unit of spin bookkeeping: RHF sums each loop's spin to a factor
    2 (:func:`spin_integration`); the unrestricted counterpart in
    :mod:`sym2quantized_sapt.open_shell` enumerates it instead.
    """
    n = len(upper)
    graph = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if upper[i] == lower[j]:
                graph[i][j] = 1
                graph[j][i] = 1

    visited = [False] * n
    loops = []

    def _collect(i, members):
        visited[i] = True
        members.append(i)
        for j in range(n):
            if graph[i][j] and not visited[j]:
                _collect(j, members)

    for i in range(n):
        if not visited[i]:
            members = []
            _collect(i, members)
            loops.append(tuple(members))

    return loops


def _count_loops(upper: Expr, lower: Expr) -> int:
    """
    Helper function for spin integration.

    Returns number of loops in a corresponding Goldstone diagram.
    """
    return len(_loop_partition(upper, lower))


def spin_integration(expr: Expr) -> Expr:
    """
    Integrates expresion over spin variables assuming
    the Restricted Hartree-Fock case.

    Indices in returned expression refer to orbitals.
    """

    if isinstance(expr, Add):
        return Add(*[spin_integration(arg) for arg in expr.args])

    elif isinstance(expr, Mul):
        upper = []
        lower = []
        for elem in expr.args:
            if isinstance(elem, TensorSymbol):
                upper += [index for index in elem.upper]
                lower += [index for index in elem.lower]

        l = _count_loops(upper, lower)

        return Mul(2 ** (l), expr)

    elif isinstance(expr, TensorSymbol):
        upper = expr.upper
        lower = expr.lower

        l = _count_loops(upper, lower)

        return Mul(2 ** (l), expr)

    else:
        return expr
