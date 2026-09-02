import string
import re
from sympy import Add, Mul, Expr, expand
from sympy.physics.secondquant import TensorSymbol


def _psi4numpy_indices(index: str) -> str:
    """
    Renames a canonical dummy to the psi4numpy convention:
    a -> r, b -> s, i -> a, j -> b (p and q are left alone).

    NOTE: this is only injective on the canonical names produced by
    `substitute_dummies_double_vac`. Every occurrence of the first
    matching letter is replaced, so a name that already lives in the
    target alphabet is mapped a second time: `r` stays `r` while a hole
    named `a` also becomes `r`, and the two indices collapse into one.
    """
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
    PRETTY_INDICES = {
        "a": "a",
        "b": "b",
        "r": "r",
        "s": "s",
        "p": "p",
        "q": "q",
        "a_1": "A",
        "b_1": "B",
        "r_1": "R",
        "s_1": "S",
        "p_1": "P",
        "q_1": "Q",
    }
    REPLACABLE_INDICES = set(PRETTY_INDICES.keys())
    merged = "".join(indices.replace("->", ",").split(","))
    unique_indices = set(re.findall(r"[a-z](?:_\d+)?", merged))

    if unique_indices.issubset(REPLACABLE_INDICES):
        pattern = re.compile(
            "|".join(
                re.escape(index_key)
                for index_key in sorted(PRETTY_INDICES, key=len, reverse=True)
            )
        )
        return pattern.sub(lambda m: PRETTY_INDICES[m.group(0)], indices)

    raise ValueError("Code generator: `pretty_indices` cannot be applied!")


def _replace_indices_names(indices: str) -> str:
    """
    Renames every subscripted index (`a_1`, `p_10`, ...) to a single unused
    letter, so that an einsum subscript stays one character per index.

    Three rules keep the renaming faithful:
    - the pool of new names excludes every name already present in
      `indices`, so a dummy is never aliased onto an index in use,
    - the longest names are substituted first, so `a_1` is not substituted
      inside `a_10`,
    - the candidates keep their first-appearance order, so the generated
      code does not depend on the hash seed.
    """
    new_names = list(string.ascii_lowercase) + list(string.ascii_uppercase)

    used_names = {"a", "b", "r", "s"}
    used_names |= {i for i in indices if i in new_names}
    for elem in used_names:
        new_names.remove(elem)

    merged = "".join(indices.replace("->", ",").split(","))
    unique_indices = list(dict.fromkeys(re.findall(r"[a-z](?:_\d+)?", merged)))
    to_substitute = sorted(
        (e for e in unique_indices if "_" in e), key=len, reverse=True
    )

    new_indices = indices
    for elem in to_substitute:
        try:
            old_name = elem
            new_indices = new_indices.replace(old_name, new_names.pop(0))
        except IndexError as exc:
            raise IndexError(
                "Too many indices!!! Not enough names for them."
            ) from exc

    return new_indices


def _get_code_str(
    coeff, cont_ind, uncont_ind, variables, pretty_indices=False
) -> str:
    indices = ",".join(cont_ind)
    if uncont_ind:
        indices += "->" + "".join(uncont_ind)

    if pretty_indices:
        indices = _pretty_indices_names(indices)
    else:
        indices = _replace_indices_names(indices)

    variables = ", ".join(variables)
    variables = variables.replace("v_A", "vA")
    variables = variables.replace("v_B", "vB")
    variables = variables.replace("o_A", "omegaA")
    variables = variables.replace("o_B", "omegaB")

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


def _variable_name(tensor: TensorSymbol) -> str:
    """
    Name of the numpy array holding `tensor`: the tensor symbol followed by
    one letter per index, lower indices first.

    Only the leading letter of every index survives, so `t^{i_1 j}_{a_1 b}`
    and `t^{i j}_{a b}` share the array `t_rsab`. Both code paths
    (`_get_einsum_for_Tensor` and `_get_einsum_for_Mul`) name their arrays
    here, so a tensor keeps the same name whether it stands alone or sits
    in a product.
    """
    var_indices = [
        _psi4numpy_indices(idx.name[0])
        for idx in (*tensor.lower(), *tensor.upper())
    ]

    return "_".join((str(tensor.symbol()), "".join(var_indices)))


def _get_einsum_for_Tensor(tensor: TensorSymbol, pretty_indices=True) -> str:
    upper = [_psi4numpy_indices(idx.name) for idx in tensor.upper()]
    lower = [_psi4numpy_indices(idx.name) for idx in tensor.lower()]

    indices = "".join(lower + upper)
    indices_raw = lower + upper

    variable = _variable_name(tensor)

    # check for uncontracted indicies
    uncont_ind = []
    for elem in indices_raw:
        if (elem not in lower) or (elem not in upper):
            uncont_ind.append(elem)

    if uncont_ind:
        indices += "->" + "".join(uncont_ind)

    if pretty_indices:
        indices = _pretty_indices_names(indices)
    else:
        indices = _replace_indices_names(indices)

    return '+np.einsum("{0}", {1})'.format(indices, variable)


def _get_einsum_for_Mul(term: Mul, pretty_indices=False) -> str:
    if isinstance(term.args[0], TensorSymbol):
        coeff = 1
    else:
        coeff = term.args[0]

    indices = []
    indices_raw = []
    upper = []
    lower = []
    variables = []

    for arg in term.args:
        if isinstance(arg, TensorSymbol):
            upper += [_psi4numpy_indices(idx.name) for idx in arg.upper()]
            lower += [_psi4numpy_indices(idx.name) for idx in arg.lower()]

            arg_indices = [
                _psi4numpy_indices(idx.name)
                for idx in (*arg.lower(), *arg.upper())
            ]
            indices.append("".join(arg_indices))
            indices_raw += arg_indices

            variables.append(_variable_name(arg))

    # check for uncontracted indicies
    uncont_ind = []
    for elem in indices_raw:
        if (elem not in lower) or (elem not in upper):
            uncont_ind.append(elem)

    return _get_code_str(
        coeff, indices, uncont_ind, variables, pretty_indices=pretty_indices
    )


def generate_einsum(expr: Expr, pretty_indices=False) -> str:
    """
    Generates string containing numpy einsum code of given expression.

    The expression has to carry the canonical dummy names (a, b, i, j, p, q
    with the `_1, _2, ...` suffixes), because the indices are renamed to the
    psi4numpy convention here. Do NOT rename them beforehand with
    `substitute_dummies_double_vac(expr, pretty_indices=...)` - see the NOTE
    in that function.

    Args:
        expr (Expr): SymPy expression for code generation
        pretty_indices (bool): name the indices after the fixed table in
            `_pretty_indices_names` instead of renaming every subscripted
            index to an unused letter

    Returns:
        str: string with numpy code

    Raises:
        IndexError: the expression has more indices than there are names
        ValueError: `pretty_indices` is set and an index is outside the table
    """

    expr = expand(expr)

    if isinstance(expr, TensorSymbol):
        return _get_einsum_for_Tensor(expr, pretty_indices=pretty_indices)

    if isinstance(expr, Add):
        return "\n".join(
            [
                generate_einsum(arg, pretty_indices=pretty_indices)
                for arg in expr.args
            ]
        )

    if isinstance(expr, Mul):
        return _get_einsum_for_Mul(expr, pretty_indices=pretty_indices)

    # expr is neither Mul, Add nor TensorSymbol:
    return ""
