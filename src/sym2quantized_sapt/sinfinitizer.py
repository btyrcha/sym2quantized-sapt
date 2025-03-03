from itertools import permutations

from sympy import Mul, Add, Expr, S

from sympy.physics.secondquant import TensorSymbol
from sym2quantized_sapt.tensors import DoubleVacuumTensorSymbol
from sym2quantized_sapt.spin_integrator import _count_loops


def count_loops(expr: Mul) -> int:
    """
    Cunts loops in a given expression.
    """
    upper = []
    lower = []
    for elem in expr.args:
        if isinstance(elem, TensorSymbol):
            upper += [index for index in elem.upper()]
            lower += [index for index in elem.lower()]
    return _count_loops(upper, lower)


def count_hole_lines(indices: list) -> int:
    """
    Counts hole indices in an given list.
    """
    h = 0
    for i in indices:
        if i.assumptions0.get("below_fermi"):
            h += 1

    return h


def _get_tensor_symbol(upper, lower):
    """
    Returns a symbol for a given type of contracted
    S-intergrals tensor together with a number
    of "hidden" hole lines.
    """
    if (
        lower.assumptions0.get("is_molB")
        and lower.assumptions0.get("below_fermi")
        and upper.assumptions0.get("is_molB")
        and upper.assumptions0.get("below_fermi")
    ):  # A_bb
        symbol = "A"
        h_lines = 1
    elif (
        lower.assumptions0.get("is_molA")
        and lower.assumptions0.get("below_fermi")
        and upper.assumptions0.get("is_molA")
        and upper.assumptions0.get("below_fermi")
    ):  # B_aa
        symbol = "B"
        h_lines = 1
    elif (
        lower.assumptions0.get("is_molA")
        and lower.assumptions0.get("above_fermi")
        and upper.assumptions0.get("is_molA")
        and upper.assumptions0.get("above_fermi")
    ):  # C_rr
        symbol = "C"
        h_lines = 0
    elif (
        lower.assumptions0.get("is_molB")
        and lower.assumptions0.get("above_fermi")
        and upper.assumptions0.get("is_molB")
        and upper.assumptions0.get("above_fermi")
    ):  # D_ss
        symbol = "D"
        h_lines = 0
    elif (
        lower.assumptions0.get("is_molA")
        and lower.assumptions0.get("below_fermi")
        and upper.assumptions0.get("is_molA")
        and upper.assumptions0.get("above_fermi")
    ):  # E_ar
        symbol = "E"
        h_lines = 1
    elif (
        lower.assumptions0.get("is_molA")
        and lower.assumptions0.get("above_fermi")
        and upper.assumptions0.get("is_molA")
        and upper.assumptions0.get("below_fermi")
    ):  # E_ra
        symbol = "E"
        h_lines = 1
    elif (
        lower.assumptions0.get("is_molB")
        and lower.assumptions0.get("below_fermi")
        and upper.assumptions0.get("is_molB")
        and upper.assumptions0.get("above_fermi")
    ):  # F_bs
        symbol = "F"
        h_lines = 1
    elif (
        lower.assumptions0.get("is_molB")
        and lower.assumptions0.get("above_fermi")
        and upper.assumptions0.get("is_molB")
        and upper.assumptions0.get("below_fermi")
    ):  # F_sb
        symbol = "F"
        h_lines = 1
    elif (
        lower.assumptions0.get("is_molB")
        and lower.assumptions0.get("above_fermi")
        and upper.assumptions0.get("is_molA")
        and upper.assumptions0.get("above_fermi")
    ):  # G_sr
        symbol = "G"
        h_lines = 0
    elif (
        lower.assumptions0.get("is_molA")
        and lower.assumptions0.get("above_fermi")
        and upper.assumptions0.get("is_molB")
        and upper.assumptions0.get("above_fermi")
    ):  # G_rs
        symbol = "G"
        h_lines = 0
    elif (
        lower.assumptions0.get("is_molB")
        and lower.assumptions0.get("below_fermi")
        and upper.assumptions0.get("is_molA")
        and upper.assumptions0.get("below_fermi")
    ):  # H_ba
        symbol = "H"
        h_lines = 0
    elif (
        lower.assumptions0.get("is_molA")
        and lower.assumptions0.get("below_fermi")
        and upper.assumptions0.get("is_molB")
        and upper.assumptions0.get("below_fermi")
    ):  # H_ab
        symbol = "H"
        h_lines = 0
    elif (
        lower.assumptions0.get("is_molA")
        and lower.assumptions0.get("above_fermi")
        and upper.assumptions0.get("is_molB")
        and upper.assumptions0.get("below_fermi")
    ):  # I_rb
        symbol = "I"
        h_lines = 0
    elif (
        lower.assumptions0.get("is_molB")
        and lower.assumptions0.get("below_fermi")
        and upper.assumptions0.get("is_molA")
        and upper.assumptions0.get("above_fermi")
    ):  # I_br
        symbol = "I"
        h_lines = 0
    elif (
        lower.assumptions0.get("is_molB")
        and lower.assumptions0.get("above_fermi")
        and upper.assumptions0.get("is_molA")
        and upper.assumptions0.get("below_fermi")
    ):  # J_sa
        symbol = "J"
        h_lines = 0
    elif (
        lower.assumptions0.get("is_molA")
        and lower.assumptions0.get("below_fermi")
        and upper.assumptions0.get("is_molB")
        and upper.assumptions0.get("above_fermi")
    ):  # J_as
        symbol = "J"
        h_lines = 0
    else:  # temporary
        print("Still missing something!", lower, upper)
        symbol = "X"
        h_lines = 0

    return symbol, h_lines


def sinfinitizer(upper_tensors: tuple, lower_tensors: tuple) -> Expr:
    """
    Connect upper and lower tensors into a graph
    in every way possible (even not correct ways!)
    """
    idx_up_in = []
    idx_up_out = []
    idx_down_in = []
    idx_down_out = []

    for t in upper_tensors:
        idx_up_in += t.upper()
        idx_up_out += t.lower()
    for t in lower_tensors:
        idx_down_in += t.upper()
        idx_down_out += t.lower()

    idx_in = idx_up_in + idx_down_in
    idx_out = idx_up_out + idx_down_out

    # counting hole lines for sign eval
    h = count_hole_lines(idx_in + idx_out)

    n = len(idx_in)
    result = S.Zero
    for perm in permutations(idx_in):
        X = []
        hidden_holes = 0  # number of odd hole lines integrals
        for i in range(n):
            t, hid_h = _get_tensor_symbol(idx_out[i], perm[i])
            X.append(DoubleVacuumTensorSymbol(t, (idx_out[i],), (perm[i],)))
            hidden_holes += hid_h
        # create an expression for each possible connection
        expr = Mul(*upper_tensors, *X, *lower_tensors)

        # sign evaluation
        l = count_loops(expr)
        sign = (-1) ** (l + h + hidden_holes)

        # and append to result
        result = Add(result, sign * expr)

    return result
