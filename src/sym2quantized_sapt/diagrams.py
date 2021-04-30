from sympy.physics.secondquant import TensorSymbol

from sympy import (
    S,
    Symbol,
    symbols,
    Add,
    Mul,
    KroneckerDelta,
    Dummy,
    expand,
    latex,
)


def _make_simplified_graph(expr):
    """
    Assumes type Mul of argument expr.

    Returns:
    graph - a graph in a matrix represantation,
    n - number of verticies in a graph.
    """

    edges = []
    for elem in expr.args:
        if isinstance(elem, TensorSymbol):
            upper = [index for index in elem.upper()]
            lower = [index for index in elem.lower()]
            edges.append(upper + lower)

    n = len(edges)
    graph = [[0] * n for i in range(n)]
    for i in range(n):
        for k in range(len(edges[i])):
            for j in range(n):
                if (i != j) and (edges[i][k] in edges[j]):
                    graph[i][j] = 1
                    graph[j][i] = 1

    return graph, n


def get_olnly_linked(expr):

    if isinstance(expr, Add):
        return Add(*[get_olnly_linked(arg) for arg in expr.args])

    elif isinstance(expr, Mul):
        graph, n = _make_simplified_graph(expr)

        visited = [False] * n
        count = 0
        stack = [0]
        visited[0] = True
        while len(stack):
            i = stack.pop()
            count += 1
            for j in range(n):
                if graph[i][j] and not visited[j]:
                    visited[j] = True
                    stack.append(j)

        if count == n:
            return expr

        else:
            return S.Zero

    else:
        return expr
