import string
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


def _pretty_indices_names(indices: str) -> str:

    if "a_1" in indices:
        indices = indices.replace("a_1", "A")

    if "b_1" in indices:
        indices = indices.replace("b_1", "B")

    if "r_1" in indices:
        indices = indices.replace("r_1", "R")

    if "s_1" in indices:
        indices = indices.replace("s_1", "S")

    if "a_2" in indices:
        indices = indices.replace("a_2", "c")

    if "b_2" in indices:
        indices = indices.replace("b_2", "d")

    if "a_3" in indices:
        indices = indices.replace("a_3", "C")

    if "b_3" in indices:
        indices = indices.replace("b_3", "D")

    return indices


def _replace_indices_names(indices: str) -> str:

    new_names = list(string.ascii_lowercase) + list(string.ascii_uppercase)

    used_names = ["a", "b", "r", "s"]
    for elem in used_names:
        new_names.remove(elem)

    new_indices = indices
    for i, elem in enumerate(indices):
        if elem == "_":
            try:
                old_name = "".join([indices[i - 1], elem, indices[i + 1]])
                new_indices = new_indices.replace(old_name, new_names.pop(0))
            except:
                print("Too many indices!!!")

    return new_indices


def _get_einsum_for_term(term: Mul, pretty_indices=False) -> str:

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
    if pretty_indices:
        indices = _pretty_indices_names(indices)
    else:
        indices = _replace_indices_names(indices)

    variables = ", ".join(variables)
    variables = variables.replace("v_A", "vA")
    variables = variables.replace("v_B", "vB")

    code_str = 'np.einsum("{0}", {1})'.format(indices, variables)

    if coeff == 1:
        code_str = "".join(("+", code_str))
    elif coeff == -1:
        code_str = "".join(("-", code_str))
    else:
        if coeff == int(coeff):
            coeff_str = str(int(coeff))
        else:
            coeff_str = str(coeff)
        code_str = " * ".join((coeff_str, code_str))

        # extra plus in front
        if coeff > 0:
            code_str = "".join(("+", code_str))

    return code_str


def generate_einsum(expr: Expr, pretty_indices=False) -> str:
    """
    Generates string containing numpy einsum code of given expression.

    Args:
        expr (Expr): SymPy expression for code generation

    Returns:
        str: string with numpy code
    """

    if isinstance(expr, Add):
        return "\n".join(
            [
                generate_einsum(arg, pretty_indices=pretty_indices)
                for arg in expr.args
            ]
        )

    if isinstance(expr, Mul):
        return _get_einsum_for_term(expr, pretty_indices=pretty_indices)

    # expr is neither Mul nor Add:
    return ""
