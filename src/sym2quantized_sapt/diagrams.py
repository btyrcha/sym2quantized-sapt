from sympy.physics.secondquant import TensorSymbol

from sympy import (
    S,
    Add,
    Mul,
)


def _make_simplified_graph(expr):
    """
    Assumes type Mul of argument expr.

    Returns:
    graph - a graph in a matrix representation,
    n - number of vertices in a graph.
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


def get_only_linked(expr):
    """[summary]

    Args:
        expr ([type]): [description]

    Returns:
        [type]: [description]
    """

    if isinstance(expr, Add):
        return Add(*[get_only_linked(arg) for arg in expr.args])

    if isinstance(expr, Mul):
        # NOTE: The `Mul` check is not enough - has to be product of tensors.
        # NOTE: A scalar is linked.

        args_is_tensor = [isinstance(arg, TensorSymbol) for arg in expr.args]

        if bool(sum(args_is_tensor)):
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

            return S.Zero

        return expr

    return expr
