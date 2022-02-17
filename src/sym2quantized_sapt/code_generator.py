from sympy import Add, Mul, Expr
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol


def _psi4numpy_indices(index: str) -> str:

    if "a" in index:
        index = index.replace("a", "r")

    elif "b" in index:
        index = index.replace("b", "s")

    elif "i" in index:
        index = index.replace("i", "a")

    elif "j" in index:
        index = index.replace("j", "b")

    return index


def _replace_indices_names(indices: str) -> str:

    if "a_1" in indices:
        indices = indices.replace("a_1", "A")

    if "b_1" in indices:
        indices = indices.replace("b_1", "B")

    if "r_1" in indices:
        indices = indices.replace("r_1", "R")

    if "s_1" in indices:
        indices = indices.replace("s_1", "S")

    # TODO replace other possible index names

    return indices


def _get_einsum_for_term(term: Mul) -> str:

    coeff = term.args[0]

    indices = []
    variables = []

    for arg in term.args:

        if isinstance(arg, DoubleVacuumTensorSymbol):

            indices_names = [
                _psi4numpy_indices(idx.name)
                for idx in (*arg.lower(), *arg.upper())
            ]
            indices.append("".join(indices_names))

            var_indices = [
                _psi4numpy_indices(idx.name[0])
                for idx in (*arg.lower(), *arg.upper())
            ]
            var_indices = "".join(var_indices)
            variables.append("_".join((str(arg.symbol()), var_indices)))

    indices = ",".join(indices)
    indices = _replace_indices_names(indices)

    code_str = 'np.einsum("{0}", {1})'.format(indices, ", ".join(variables))

    if coeff == 1:
        pass
    elif coeff == -1:
        code_str = "".join(("-", code_str))
    else:
        if coeff == int(coeff):
            coeff_str = str(int(coeff))
        else:
            coeff_str = str(coeff)
        code_str = " * ".join((coeff_str, code_str))

    return code_str


def generate_einsum(expr: Expr) -> str:
    """
    Generates string containing numpy einsum code of given expression.

    Args:
        expr (Expr): SymPy expression for code generation

    Returns:
        str: string with numpy code
    """

    if isinstance(expr, Add):
        return "\n".join([generate_einsum(arg) for arg in expr.args])

    if isinstance(expr, Mul):
        return _get_einsum_for_term(expr)

    # expr is neither Mul nor Add:
    return ""
