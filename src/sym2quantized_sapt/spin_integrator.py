from sympy import Add, Mul
from sympy.core import Expr
from sympy.physics.secondquant import TensorSymbol


def _count_loops(upper: Expr, lower: Expr) -> int:
    """
    Helper function for spin integration.

    Returns number of loops in a corresponding Goldstone diagram.
    """

    l = 0
    N = len(upper)
    visited = [0] * N
    graph = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(N):
            if upper[i] == lower[j]:
                graph[i][j] = 1
                graph[j][i] = 1

    def _DFS(i):
        visited[i] = 1
        for j in range(N):
            if graph[i][j] and not visited[j]:
                _DFS(j)

    for i in range(N):
        if not visited[i]:
            _DFS(i)
            l += 1

    return l


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
                upper += [index for index in elem.upper()]
                lower += [index for index in elem.lower()]

        l = _count_loops(upper, lower)

        return Mul(2 ** (l), expr)

    elif isinstance(expr, TensorSymbol):
        upper = expr.upper()
        lower = expr.lower()

        l = _count_loops(upper, lower)

        return Mul(2 ** (l), expr)

    else:
        return expr
