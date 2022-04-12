import pytest

from sympy import latex
from sym2quantized_sapt.double_fermi_vac import wicks_double_vac
from sym2quantized_sapt.sapt_utils import (
    get_Pn_operator,
    get_P2_operator,
    get_P4_operator,
)


def test_can_recreate_P2():

    reference_latex = latex(wicks_double_vac(get_P2_operator()))

    tested_expr = latex(wicks_double_vac(get_Pn_operator(2)))

    assert reference_latex == tested_expr


def test_can_recreate_P4():

    reference_latex = latex(wicks_double_vac(get_P4_operator()))

    tested_expr = latex(wicks_double_vac(get_Pn_operator(4)))

    assert reference_latex == tested_expr
